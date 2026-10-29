"""
RetailIQ AI V2 ??? Phase 02: Data Quality & Readiness Tests

Validates:
- Statistical profiling calculations (central tendency, dispersion, shape)
- Structural missingness classification (MarkDown onset vs economic completeness)
- Merchandise return audit (negative sales aggregation and department impact)
- Store-department continuity and coverage metrics
- Report serialization and schema compliance
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.config import REPORT_SUBDIRS
from src.retailiq.data.loader import load_analytical_dataset
from src.retailiq.quality.reporter import (
    generate_column_statistics,
    generate_missingness_profile,
    generate_negative_sales_audit,
    generate_store_department_coverage,
    run_quality_audit,
)


@pytest.fixture(scope="module")
def analytical_df() -> pd.DataFrame:
    """Load analytical dataset once for all tests in module."""
    return load_analytical_dataset()


class TestStatisticalProfiling:
    """Verify statistical metrics calculation."""

    def test_column_statistics_shape_and_features(self, analytical_df: pd.DataFrame):
        stats_df = generate_column_statistics(analytical_df)
        assert len(stats_df) == 11, f"Expected 11 numerical features, got {len(stats_df)}"
        expected_cols = ["Column", "Mean", "Std", "Min", "Median", "Max", "Skewness", "Kurtosis"]
        for col in expected_cols:
            assert col in stats_df.columns, f"Missing stat column {col}"

    def test_weekly_sales_skewness(self, analytical_df: pd.DataFrame):
        stats_df = generate_column_statistics(analytical_df)
        sales_stat = stats_df[stats_df["Column"] == "Weekly_Sales"].iloc[0]
        assert sales_stat["Skewness"] > 2.0, "Expected positive sales skewness > 2.0"
        assert sales_stat["Min"] < 0, "Expected negative minimum sales (returns)"
        assert sales_stat["Max"] > 600000, "Expected peak holiday sales > $600k"


class TestMissingnessClassification:
    """Verify structural vs random missingness logic."""

    def test_economic_features_completeness(self, analytical_df: pd.DataFrame):
        missing_df = generate_missingness_profile(analytical_df)
        cpi_missing = missing_df.loc[missing_df["Column"] == "CPI", "Missing_Count"].values[0]
        unemp_missing = missing_df.loc[missing_df["Column"] == "Unemployment", "Missing_Count"].values[0]
        assert cpi_missing == 0, "CPI should have 0 missing values in train period"
        assert unemp_missing == 0, "Unemployment should have 0 missing values in train period"

    def test_raw_vs_analytical_cpi_reconciliation(self, analytical_df: pd.DataFrame):
        """Regression test for ISSUE-11-02: Reconcile 585 raw CPI nulls in future dates vs complete analytical dataset."""
        raw_feat_path = Path("data/raw/features.csv")
        assert raw_feat_path.exists()
        raw_df = pd.read_csv(raw_feat_path)

        # Raw features has 585 missing CPI and Unemployment
        assert raw_df["CPI"].isna().sum() == 585
        assert raw_df["Unemployment"].isna().sum() == 585

        # All missing CPI are in dates >= 2013-05-03 (future Kaggle evaluation dates)
        missing_dates = pd.to_datetime(raw_df.loc[raw_df["CPI"].isna(), "Date"])
        assert (missing_dates >= pd.to_datetime("2013-05-03")).all()

        # In historical labeled sales range (<= 2012-10-26), CPI and Unemployment are 100% complete
        hist_raw = raw_df[pd.to_datetime(raw_df["Date"]) <= pd.to_datetime("2012-10-26")]
        assert hist_raw["CPI"].isna().sum() == 0
        assert hist_raw["Unemployment"].isna().sum() == 0

        # Joined analytical dataset is 100% complete
        assert analytical_df["CPI"].isna().sum() == 0
        assert analytical_df["Unemployment"].isna().sum() == 0

    def test_markdown_structural_missingness(self, analytical_df: pd.DataFrame):
        missing_df = generate_missingness_profile(analytical_df)
        md1 = missing_df.loc[missing_df["Column"] == "MarkDown1"].iloc[0]
        assert md1["Missing_Pct"] > 50.0, "MarkDown1 should be > 50% missing"
        assert "Structural" in md1["Classification"]
        assert md1["First_Observed_Date"] == "2011-11-11"


class TestMerchandiseReturnsAudit:
    """Verify negative sales accounting."""

    def test_returns_counts_and_amounts(self, analytical_df: pd.DataFrame):
        dept_audit, store_audit = generate_negative_sales_audit(analytical_df)
        total_returns_count = dept_audit["Return_Count"].sum()
        assert total_returns_count == 1285, f"Expected 1,285 return records, got {total_returns_count}"
        total_amount = dept_audit["Total_Return_Amount"].sum()
        assert total_amount < -80000, f"Expected total return amount < -$80,000, got {total_amount}"
        assert len(dept_audit) == 50, f"Expected 50 affected departments, got {len(dept_audit)}"


class TestStoreDeptContinuity:
    """Verify store-department series continuity."""

    def test_coverage_pair_counts(self, analytical_df: pd.DataFrame):
        coverage_df = generate_store_department_coverage(analytical_df)
        assert len(coverage_df) == 3331, f"Expected 3,331 store-dept pairs, got {len(coverage_df)}"
        full_cov = (coverage_df["Status"] == "Full (143 Weeks)").sum()
        assert full_cov == 2660, f"Expected 2,660 full-coverage pairs, got {full_cov}"
        assert coverage_df["Active_Weeks"].max() == 143


class TestQualityAuditExecution:
    """Verify report export files exist and match schema."""

    def test_quality_reports_exist(self):
        out_dir = REPORT_SUBDIRS["data_quality"]
        expected_files = [
            "02_column_statistics.csv",
            "03_missingness_profile.csv",
            "04_negative_sales_audit.csv",
            "05_store_department_coverage.csv",
            "06_data_quality_summary.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            assert fpath.exists(), f"Missing quality report {fname}"
            assert fpath.stat().st_size > 0, f"Report {fname} is empty"

    def test_summary_json_integrity(self):
        summary_path = REPORT_SUBDIRS["data_quality"] / "06_data_quality_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_observations"] == 421570
        assert data["departments"]["active_pairs"] == 3331
        assert data["departments"]["full_coverage_pairs"] == 2660
        assert data["sales_audit"]["negative_rows"] == 1285
