"""
RetailIQ AI V2 ??? End-to-End Ingestion Pipeline

Executes the full data ingestion process:
1. Validates schemas of raw sources.
2. Checks join keys and holiday consistency.
3. Merges train, features, and stores tables without row drops or duplication.
4. Generates temporal features (Year, Month, WeekOfYear).
5. Exports processed analytical dataset as Parquet.
6. Saves detailed ingestion summary report for lineage tracking.
"""

import json
from pathlib import Path
from typing import Any
import pandas as pd

from src.retailiq.config import (
    ANALYTICAL_DATASET,
    REPORT_SUBDIRS,
    ensure_directories,
)
from src.retailiq.data.loader import (
    load_raw_features,
    load_raw_stores,
    load_raw_train,
)
from src.retailiq.data.validator import (
    check_duplicates,
    check_holiday_consistency,
    check_missing_values,
    check_sales_anomalies,
    validate_dates,
    validate_join_integrity,
    validate_schema,
)


def run_ingestion_pipeline(
    output_path: Path | str | None = None,
    save_summary_report: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute complete data ingestion pipeline.

    Args:
        output_path: Target path for analytical dataset. Defaults to ANALYTICAL_DATASET.
        save_summary_report: Whether to write JSON summary to reports/data_quality/.

    Returns:
        Tuple of (analytical_df, summary_dict).
    """
    ensure_directories()
    dest_path = Path(output_path) if output_path else ANALYTICAL_DATASET

    # ?????? 1. Load Raw Sources ??????
    train_df = load_raw_train()
    features_df = load_raw_features()
    stores_df = load_raw_stores()

    initial_train_rows = len(train_df)
    initial_features_rows = len(features_df)
    initial_stores_rows = len(stores_df)

    # ?????? 2. Schema Validation ??????
    schema_train = validate_schema(train_df, "train")
    schema_features = validate_schema(features_df, "features")
    schema_stores = validate_schema(stores_df, "stores")

    if not (schema_train["valid"] and schema_features["valid"] and schema_stores["valid"]):
        all_errors = schema_train["errors"] + schema_features["errors"] + schema_stores["errors"]
        raise ValueError(f"Schema validation failed: {all_errors}")

    # ?????? 3. Date Validation ??????
    date_val_train = validate_dates(train_df, "Date")
    date_val_features = validate_dates(features_df, "Date")

    # ?????? 4. Key Uniqueness and Join Integrity ??????
    train_dups = check_duplicates(train_df, subset=["Store", "Dept", "Date"])
    join_features_val = validate_join_integrity(
        train_df, features_df, ["Store", "Date"],
        left_name="train", right_name="features",
    )
    join_stores_val = validate_join_integrity(
        train_df, stores_df, ["Store"],
        left_name="train", right_name="stores",
    )
    holiday_val = check_holiday_consistency(train_df, features_df)

    # ?????? 5. Merge Pipeline ??????
    # Drop IsHoliday from features before merging to avoid duplicate columns
    # (after having verified 100% agreement between train and features above)
    features_clean = features_df.drop(columns=["IsHoliday"])

    # Merge train with features on Store and Date
    merged_df = pd.merge(
        train_df,
        features_clean,
        on=["Store", "Date"],
        how="left",
    )

    # Verify no row loss or duplication on first join
    if len(merged_df) != initial_train_rows:
        raise RuntimeError(
            f"Row count mismatch after features merge: expected {initial_train_rows}, got {len(merged_df)}"
        )

    # Merge with stores on Store
    analytical_df = pd.merge(
        merged_df,
        stores_df,
        on="Store",
        how="left",
    )

    # Verify row count after second join
    if len(analytical_df) != initial_train_rows:
        raise RuntimeError(
            f"Row count mismatch after stores merge: expected {initial_train_rows}, got {len(analytical_df)}"
        )

    # ?????? 6. Temporal Feature Engineering ??????
    analytical_df["Year"] = analytical_df["Date"].dt.year.astype("int32")
    analytical_df["Month"] = analytical_df["Date"].dt.month.astype("int32")
    analytical_df["WeekOfYear"] = analytical_df["Date"].dt.isocalendar().week.astype("int32")
    analytical_df["DayOfMonth"] = analytical_df["Date"].dt.day.astype("int32")

    # Reorder columns logically: Identifiers ??? Sales Target ??? Calendar ??? Features ??? Store Meta
    ordered_cols = [
        "Store", "Dept", "Date", "Year", "Month", "WeekOfYear", "DayOfMonth",
        "Weekly_Sales", "IsHoliday", "Type", "Size",
        "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment",
    ]
    analytical_df = analytical_df[ordered_cols]

    # ?????? 7. Save Processed Parquet ??????
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    analytical_df.to_parquet(dest_path, index=False, engine="pyarrow")

    # ?????? 8. Compile Comprehensive Ingestion Summary ??????
    sales_anomalies = check_sales_anomalies(analytical_df)
    missing_summary = check_missing_values(analytical_df).to_dict(orient="records")

    summary = {
        "pipeline": "Phase 01 ??? Data Ingestion & Transformation",
        "status": "SUCCESS",
        "output_file": str(dest_path),
        "raw_counts": {
            "train_rows": initial_train_rows,
            "train_cols": len(train_df.columns),
            "features_rows": initial_features_rows,
            "features_cols": len(features_df.columns),
            "stores_rows": initial_stores_rows,
            "stores_cols": len(stores_df.columns),
        },
        "processed_counts": {
            "analytical_rows": len(analytical_df),
            "analytical_cols": len(analytical_df.columns),
            "row_count_preservation_ratio": round(len(analytical_df) / initial_train_rows, 6),
        },
        "dimensions": {
            "unique_stores": int(analytical_df["Store"].nunique()),
            "unique_departments": int(analytical_df["Dept"].nunique()),
            "unique_dates": int(analytical_df["Date"].nunique()),
            "min_date": str(analytical_df["Date"].min().date()),
            "max_date": str(analytical_df["Date"].max().date()),
            "store_types": analytical_df["Type"].value_counts().to_dict(),
        },
        "validations": {
            "train_duplicates": train_dups["duplicate_rows_count"],
            "features_join_match_rate": join_features_val["match_rate"],
            "stores_join_match_rate": join_stores_val["match_rate"],
            "holiday_consistency": holiday_val["consistent"],
        },
        "sales_profile": sales_anomalies,
        "missing_values": missing_summary,
    }

    if save_summary_report:
        report_file = REPORT_SUBDIRS["data_quality"] / "01_ingestion_summary.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return analytical_df, summary

