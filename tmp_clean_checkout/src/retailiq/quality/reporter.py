"""
RetailIQ AI V2 ??? Data Quality Reporter Module

Performs comprehensive statistical profiling, structural missingness classification,
merchandise return (negative sales) audit, and store-department coverage checks.
Generates reproducible CSV and JSON reports for dashboard analytics and modeling audit.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.retailiq.config import REPORT_SUBDIRS, ensure_directories
from src.retailiq.data.loader import load_analytical_dataset


def generate_column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate statistical distribution metrics for all numeric columns.

    Calculates central tendency, dispersion, quartiles, skewness, and kurtosis.

    Args:
        df: Input analytical DataFrame.

    Returns:
        DataFrame with column metrics.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifier and temporal index columns from pure statistical summary
    exclude_cols = {"Store", "Dept", "Year", "Month", "WeekOfYear", "DayOfMonth"}
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    stats_list = []
    for col in feature_cols:
        series = df[col].dropna()
        n = len(series)
        if n == 0:
            continue

        mean = float(series.mean())
        std = float(series.std())
        q25 = float(series.quantile(0.25))
        median = float(series.median())
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        skew = float(series.skew())
        kurt = float(series.kurtosis())

        stats_list.append({
            "Column": col,
            "Count": n,
            "Missing_Count": int(df[col].isna().sum()),
            "Missing_Pct": round(float(df[col].isna().sum() / len(df) * 100.0), 2),
            "Mean": round(mean, 2),
            "Std": round(std, 2),
            "Min": round(float(series.min()), 2),
            "Q25": round(q25, 2),
            "Median": round(median, 2),
            "Q75": round(q75, 2),
            "Max": round(float(series.max()), 2),
            "IQR": round(iqr, 2),
            "Skewness": round(skew, 2),
            "Kurtosis": round(kurt, 2),
        })

    return pd.DataFrame(stats_list)


def generate_missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Classify missingness patterns across all columns.

    Distinguishes structural business absence (e.g. MarkDown data collection start)
    from accidental transmission loss.

    Args:
        df: Analytical DataFrame.

    Returns:
        DataFrame detailing missingness counts, percentages, and classifications.
    """
    total_rows = len(df)
    profile = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(null_count / total_rows * 100.0, 2)

        if null_count == 0:
            classification = "Complete (0% missing)"
            first_non_null = str(df["Date"].min().date())
            last_non_null = str(df["Date"].max().date())
        elif col.startswith("MarkDown"):
            classification = "Structural (Not collected by Walmart prior to Nov 2011)"
            non_null_dates = df.loc[df[col].notna(), "Date"]
            first_non_null = str(non_null_dates.min().date()) if len(non_null_dates) > 0 else "N/A"
            last_non_null = str(non_null_dates.max().date()) if len(non_null_dates) > 0 else "N/A"
        elif col in ["CPI", "Unemployment"]:
            classification = "Complete within historical train period"
            first_non_null = str(df["Date"].min().date())
            last_non_null = str(df["Date"].max().date())
        else:
            classification = "Random / Sparse"
            first_non_null = "N/A"
            last_non_null = "N/A"

        profile.append({
            "Column": col,
            "Total_Rows": total_rows,
            "Missing_Count": null_count,
            "Missing_Pct": null_pct,
            "Classification": classification,
            "First_Observed_Date": first_non_null,
            "Last_Observed_Date": last_non_null,
        })

    return pd.DataFrame(profile)


def generate_negative_sales_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit non-positive sales records (returns and accounting reconciliations).

    Args:
        df: Analytical DataFrame.

    Returns:
        Tuple of (dept_audit_df, store_audit_df).
    """
    neg_mask = df["Weekly_Sales"] < 0
    neg_df = df[neg_mask]

    # Department-level return audit
    dept_audit = (
        neg_df.groupby("Dept", observed=True)
        .agg(
            Return_Count=("Weekly_Sales", "count"),
            Total_Return_Amount=("Weekly_Sales", "sum"),
            Min_Return_Record=("Weekly_Sales", "min"),
            Mean_Return_Record=("Weekly_Sales", "mean"),
            Affected_Stores=("Store", "nunique"),
        )
        .reset_index()
    )
    dept_audit["Total_Return_Amount"] = dept_audit["Total_Return_Amount"].round(2)
    dept_audit["Mean_Return_Record"] = dept_audit["Mean_Return_Record"].round(2)
    dept_audit = dept_audit.sort_values(by="Total_Return_Amount", ascending=True).reset_index(drop=True)

    # Store-level return audit
    store_audit = (
        neg_df.groupby(["Store", "Type"], observed=True)
        .agg(
            Return_Count=("Weekly_Sales", "count"),
            Total_Return_Amount=("Weekly_Sales", "sum"),
            Min_Return_Record=("Weekly_Sales", "min"),
            Affected_Depts=("Dept", "nunique"),
        )
        .reset_index()
    )
    store_audit["Total_Return_Amount"] = store_audit["Total_Return_Amount"].round(2)
    store_audit = store_audit.sort_values(by="Total_Return_Amount", ascending=True).reset_index(drop=True)

    return dept_audit, store_audit


def generate_store_department_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Audit the continuity of weekly observations for each store-department pair.

    Calculates active weeks out of 143 total weeks in the dataset.

    Args:
        df: Analytical DataFrame.

    Returns:
        DataFrame with coverage statistics per store-department pair.
    """
    total_dataset_weeks = int(df["Date"].nunique())

    coverage = (
        df.groupby(["Store", "Dept", "Type"], observed=True)
        .agg(
            Active_Weeks=("Date", "count"),
            Min_Date=("Date", "min"),
            Max_Date=("Date", "max"),
            Total_Sales=("Weekly_Sales", "sum"),
            Mean_Weekly_Sales=("Weekly_Sales", "mean"),
        )
        .reset_index()
    )

    coverage["Total_Dataset_Weeks"] = total_dataset_weeks
    coverage["Coverage_Pct"] = (coverage["Active_Weeks"] / total_dataset_weeks * 100.0).round(2)
    coverage["Status"] = np.where(
        coverage["Active_Weeks"] == total_dataset_weeks,
        "Full (143 Weeks)",
        "Partial (< 143 Weeks)",
    )
    coverage["Total_Sales"] = coverage["Total_Sales"].round(2)
    coverage["Mean_Weekly_Sales"] = coverage["Mean_Weekly_Sales"].round(2)
    coverage["Min_Date"] = coverage["Min_Date"].dt.date.astype(str)
    coverage["Max_Date"] = coverage["Max_Date"].dt.date.astype(str)

    return coverage.sort_values(by=["Store", "Dept"]).reset_index(drop=True)


def run_quality_audit(
    df: pd.DataFrame | None = None,
    save_reports: bool = True,
) -> dict[str, Any]:
    """Execute complete data quality and readiness audit.

    Generates and saves all quality reports.

    Args:
        df: Optional pre-loaded analytical DataFrame. Loads from Parquet if None.
        save_reports: Whether to write CSVs and summary JSON to reports/data_quality/.

    Returns:
        Dictionary of generated DataFrames and summary metrics.
    """
    ensure_directories()
    data = df if df is not None else load_analytical_dataset()

    stats_df = generate_column_statistics(data)
    missing_df = generate_missingness_profile(data)
    dept_returns_df, store_returns_df = generate_negative_sales_audit(data)
    coverage_df = generate_store_department_coverage(data)

    out_dir = REPORT_SUBDIRS["data_quality"]

    if save_reports:
        stats_df.to_csv(out_dir / "02_column_statistics.csv", index=False)
        missing_df.to_csv(out_dir / "03_missingness_profile.csv", index=False)
        dept_returns_df.to_csv(out_dir / "04_negative_sales_audit.csv", index=False)
        coverage_df.to_csv(out_dir / "05_store_department_coverage.csv", index=False)

        # Build high-level summary JSON
        summary = {
            "total_observations": len(data),
            "total_columns": len(data.columns),
            "date_range": {
                "min": str(data["Date"].min().date()),
                "max": str(data["Date"].max().date()),
                "total_weeks": int(data["Date"].nunique()),
            },
            "stores": {
                "count": int(data["Store"].nunique()),
                "type_breakdown": data["Type"].value_counts().to_dict(),
            },
            "departments": {
                "count": int(data["Dept"].nunique()),
                "active_pairs": len(coverage_df),
                "full_coverage_pairs": int((coverage_df["Status"] == "Full (143 Weeks)").sum()),
                "partial_coverage_pairs": int((coverage_df["Status"] != "Full (143 Weeks)").sum()),
            },
            "sales_audit": {
                "negative_rows": int((data["Weekly_Sales"] < 0).sum()),
                "zero_rows": int((data["Weekly_Sales"] == 0).sum()),
                "positive_rows": int((data["Weekly_Sales"] > 0).sum()),
                "total_return_amount": round(float(data.loc[data["Weekly_Sales"] < 0, "Weekly_Sales"].sum()), 2),
                "top_return_departments": dept_returns_df.head(5)[["Dept", "Total_Return_Amount", "Return_Count"]].to_dict(orient="records"),
            },
            "missingness": {
                "markdown_missing_pct": float(missing_df.loc[missing_df["Column"] == "MarkDown1", "Missing_Pct"].values[0]),
                "cpi_missing_count": int(missing_df.loc[missing_df["Column"] == "CPI", "Missing_Count"].values[0]),
                "unemployment_missing_count": int(missing_df.loc[missing_df["Column"] == "Unemployment", "Missing_Count"].values[0]),
            },
        }

        with open(out_dir / "06_data_quality_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {
        "column_statistics": stats_df,
        "missingness_profile": missing_df,
        "dept_returns": dept_returns_df,
        "store_returns": store_returns_df,
        "coverage": coverage_df,
    }
