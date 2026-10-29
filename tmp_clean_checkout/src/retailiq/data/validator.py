"""
RetailIQ AI V2 ??? Data Validator Module

Provides validation functions for schemas, column types, dates,
join integrity, missing values, duplicates, and anomaly reporting.
"""

from typing import Any
import pandas as pd
from src.retailiq.config import EXPECTED_SCHEMAS


class ValidationError(Exception):
    """Raised when data validation fails critical checks."""
    pass


def validate_schema(df: pd.DataFrame, schema_name: str) -> dict[str, Any]:
    """Validate that a DataFrame contains expected columns and valid basic types.

    Args:
        df: The DataFrame to validate.
        schema_name: Name of the schema in EXPECTED_SCHEMAS ('train', 'features', 'stores').

    Returns:
        Dictionary with validation results: {valid: bool, missing_cols: list, errors: list}.
    """
    if schema_name not in EXPECTED_SCHEMAS:
        raise ValueError(f"Unknown schema name: {schema_name}")

    expected = EXPECTED_SCHEMAS[schema_name]
    expected_cols = set(expected["columns"])
    actual_cols = set(df.columns)

    missing_cols = list(expected_cols - actual_cols)
    extra_cols = list(actual_cols - expected_cols)

    errors = []
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check dtypes for specified columns
    for col, expected_type in expected.get("dtypes", {}).items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type == "int" and not ("int" in actual_type or "Int" in actual_type):
                errors.append(f"Column '{col}' expected int-like type, got {actual_type}")
            elif expected_type == "float" and "float" not in actual_type:
                errors.append(f"Column '{col}' expected float-like type, got {actual_type}")

    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "schema_name": schema_name,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "errors": errors,
    }


def validate_dates(
    df: pd.DataFrame,
    date_col: str = "Date",
    expected_min: str | None = None,
    expected_max: str | None = None,
) -> dict[str, Any]:
    """Validate date column integrity: parsing, nulls, and date boundaries.

    Args:
        df: DataFrame containing the date column.
        date_col: Name of the date column.
        expected_min: Optional earliest expected date string (YYYY-MM-DD).
        expected_max: Optional latest expected date string (YYYY-MM-DD).

    Returns:
        Dictionary with validation metrics.
    """
    if date_col not in df.columns:
        raise ValidationError(f"Date column '{date_col}' not found in DataFrame.")

    col = df[date_col]
    nat_count = int(col.isna().sum())
    min_date = col.min()
    max_date = col.max()
    unique_dates = int(col.nunique())

    errors = []
    if nat_count > 0:
        errors.append(f"Found {nat_count} unparseable or NaT dates in '{date_col}'")

    if expected_min and min_date < pd.to_datetime(expected_min):
        errors.append(f"Min date {min_date} is earlier than expected {expected_min}")

    if expected_max and max_date > pd.to_datetime(expected_max):
        errors.append(f"Max date {max_date} is later than expected {expected_max}")

    return {
        "valid": len(errors) == 0,
        "date_column": date_col,
        "nat_count": nat_count,
        "min_date": str(min_date),
        "max_date": str(max_date),
        "unique_dates": unique_dates,
        "errors": errors,
    }


def validate_join_integrity(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_keys: list[str],
    left_name: str = "left",
    right_name: str = "right",
) -> dict[str, Any]:
    """Validate that every key in the left DataFrame has a matching record in the right DataFrame.

    Prevents silent data loss or unexpected null injection on joins.

    Args:
        left_df: The primary/left DataFrame.
        right_df: The reference/right DataFrame.
        join_keys: List of column names used as merge keys.
        left_name: Label for left table in reports.
        right_name: Label for right table in reports.

    Returns:
        Dictionary with join match statistics and unmapped key counts.
    """
    for key in join_keys:
        if key not in left_df.columns:
            raise ValidationError(f"Join key '{key}' missing from {left_name}")
        if key not in right_df.columns:
            raise ValidationError(f"Join key '{key}' missing from {right_name}")

    left_keys = left_df[join_keys].drop_duplicates()
    right_keys = right_df[join_keys].drop_duplicates()

    merged = pd.merge(left_keys, right_keys, on=join_keys, how="left", indicator=True)
    unmatched_count = int((merged["_merge"] == "left_only").sum())
    total_left_keys = len(left_keys)
    match_rate = (total_left_keys - unmatched_count) / total_left_keys if total_left_keys > 0 else 0.0

    errors = []
    if unmatched_count > 0:
        errors.append(
            f"{unmatched_count} of {total_left_keys} unique key combinations in {left_name} "
            f"have no match in {right_name} ({match_rate:.2%} match rate)."
        )

    return {
        "valid": unmatched_count == 0,
        "left_name": left_name,
        "right_name": right_name,
        "join_keys": join_keys,
        "total_unique_left_keys": total_left_keys,
        "unmatched_keys_count": unmatched_count,
        "match_rate": round(match_rate, 4),
        "errors": errors,
    }


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a structured breakdown of missing values per column.

    Returns:
        DataFrame with columns ['Column', 'Missing_Count', 'Missing_Pct', 'Dtype'].
    """
    total = len(df)
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total) * 100.0

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": [int(missing_counts[col]) for col in df.columns],
        "Missing_Pct": [round(float(missing_pcts[col]), 2) for col in df.columns],
        "Dtype": [str(df[col].dtype) for col in df.columns],
    })
    return report.sort_values(by="Missing_Count", ascending=False).reset_index(drop=True)


def check_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, Any]:
    """Check for duplicate rows on full DataFrame or key subset.

    Args:
        df: DataFrame to inspect.
        subset: Optional list of column names defining uniqueness.

    Returns:
        Dictionary with duplicate count and example duplicate rows.
    """
    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_count = int(dup_mask.sum())
    return {
        "has_duplicates": dup_count > 0,
        "duplicate_rows_count": dup_count,
        "subset": subset or "all_columns",
        "total_rows": len(df),
    }


def check_sales_anomalies(df: pd.DataFrame, sales_col: str = "Weekly_Sales") -> dict[str, Any]:
    """Report non-positive sales observations without modifying or dropping them.

    Documenting returns and adjustments preserves raw data fidelity.

    Args:
        df: DataFrame containing the sales column.
        sales_col: Name of sales column.

    Returns:
        Summary metrics of positive, zero, and negative sales observations.
    """
    if sales_col not in df.columns:
        raise ValidationError(f"Sales column '{sales_col}' not found.")

    col = df[sales_col]
    total = len(df)
    neg_count = int((col < 0).sum())
    zero_count = int((col == 0).sum())
    pos_count = int((col > 0).sum())

    return {
        "total_observations": total,
        "positive_count": pos_count,
        "positive_pct": round(pos_count / total * 100.0, 2),
        "zero_count": zero_count,
        "zero_pct": round(zero_count / total * 100.0, 3),
        "negative_count": neg_count,
        "negative_pct": round(neg_count / total * 100.0, 2),
        "min_sales": float(col.min()),
        "max_sales": float(col.max()),
        "median_sales": float(col.median()),
        "mean_sales": round(float(col.mean()), 2),
    }


def check_holiday_consistency(
    train_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict[str, Any]:
    """Verify that IsHoliday values match between train and features for identical (Store, Date).

    Args:
        train_df: Loaded train DataFrame.
        features_df: Loaded features DataFrame.

    Returns:
        Consistency report dictionary.
    """
    merged = pd.merge(
        train_df[["Store", "Date", "IsHoliday"]].drop_duplicates(),
        features_df[["Store", "Date", "IsHoliday"]].drop_duplicates(),
        on=["Store", "Date"],
        suffixes=("_train", "_features"),
    )
    mismatches = int((merged["IsHoliday_train"] != merged["IsHoliday_features"]).sum())
    return {
        "consistent": mismatches == 0,
        "compared_store_dates": len(merged),
        "mismatches": mismatches,
    }

