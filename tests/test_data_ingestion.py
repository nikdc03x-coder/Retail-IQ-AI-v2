"""
RetailIQ AI V2 — Phase 01: Data Ingestion & Quality Tests

Validates:
- Raw file loading (train, features, stores)
- Schema validation and column checks
- Date parsing and temporal consistency
- Duplicate key validation
- Join integrity across tables
- Holiday alignment between train and features
- Missing value profiling
- Non-positive sales identification (without silent drops)
- Analytical Parquet export integrity and row count preservation
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.config import ANALYTICAL_DATASET, REPORT_SUBDIRS
from src.retailiq.data.loader import (
    load_analytical_dataset,
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


class TestRawDataLoading:
    """Verify raw CSV loaders handle schema, dtypes, and date parsing."""

    def test_load_raw_train(self):
        df = load_raw_train()
        assert len(df) == 421570, f"Expected 421,570 rows, got {len(df)}"
        assert list(df.columns) == ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert df["Store"].nunique() == 45
        assert df["Dept"].nunique() == 81

    def test_load_raw_features(self):
        df = load_raw_features()
        assert len(df) == 8190, f"Expected 8,190 rows, got {len(df)}"
        assert "Temperature" in df.columns
        assert "Fuel_Price" in df.columns
        assert "CPI" in df.columns
        assert "Unemployment" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_load_raw_stores(self):
        df = load_raw_stores()
        assert len(df) == 45, f"Expected 45 stores, got {len(df)}"
        assert list(df.columns) == ["Store", "Type", "Size"]
        assert set(df["Type"].unique()) == {"A", "B", "C"}


class TestDataValidation:
    """Verify schema, date, duplicate, and join validators."""

    def test_validate_schema_success(self):
        train = load_raw_train()
        res = validate_schema(train, "train")
        assert res["valid"] is True
        assert len(res["missing_columns"]) == 0

    def test_validate_schema_detects_missing(self):
        train = load_raw_train().drop(columns=["Weekly_Sales"])
        res = validate_schema(train, "train")
        assert res["valid"] is False
        assert "Weekly_Sales" in res["missing_columns"]

    def test_validate_dates_train(self):
        train = load_raw_train()
        res = validate_dates(train, "Date", expected_min="2010-02-05", expected_max="2012-10-26")
        assert res["valid"] is True
        assert res["nat_count"] == 0
        assert res["unique_dates"] == 143

    def test_check_duplicates_train(self):
        train = load_raw_train()
        res = check_duplicates(train, subset=["Store", "Dept", "Date"])
        assert res["has_duplicates"] is False
        assert res["duplicate_rows_count"] == 0

    def test_join_integrity_features(self):
        train = load_raw_train()
        features = load_raw_features()
        res = validate_join_integrity(train, features, ["Store", "Date"])
        assert res["valid"] is True
        assert res["match_rate"] == 1.0
        assert res["unmatched_keys_count"] == 0

    def test_join_integrity_stores(self):
        train = load_raw_train()
        stores = load_raw_stores()
        res = validate_join_integrity(train, stores, ["Store"])
        assert res["valid"] is True
        assert res["match_rate"] == 1.0
        assert res["unmatched_keys_count"] == 0

    def test_holiday_consistency(self):
        train = load_raw_train()
        features = load_raw_features()
        res = check_holiday_consistency(train, features)
        assert res["consistent"] is True
        assert res["mismatches"] == 0

    def test_sales_anomalies_reporting(self):
        train = load_raw_train()
        res = check_sales_anomalies(train)
        assert res["total_observations"] == 421570
        assert res["negative_count"] == 1285
        assert res["zero_count"] == 73
        assert res["positive_count"] == 420212
        assert res["min_sales"] < 0
        assert res["max_sales"] > 0


class TestAnalyticalDataset:
    """Verify processed Parquet dataset properties and lineage."""

    def test_analytical_parquet_exists(self):
        assert ANALYTICAL_DATASET.exists(), "Analytical Parquet file does not exist."

    def test_analytical_parquet_content(self):
        df = load_analytical_dataset()
        assert len(df) == 421570, "Row count changed after pipeline processing!"
        assert len(df.columns) == 20
        # Check added calendar columns
        for col in ["Year", "Month", "WeekOfYear", "DayOfMonth"]:
            assert col in df.columns, f"Missing temporal column {col}"
        # Target must have zero missing values
        assert df["Weekly_Sales"].isna().sum() == 0
        # Keys must have zero missing values
        assert df["Store"].isna().sum() == 0
        assert df["Dept"].isna().sum() == 0
        assert df["Date"].isna().sum() == 0

    def test_ingestion_summary_report_exists(self):
        report_path = REPORT_SUBDIRS["data_quality"] / "01_ingestion_summary.json"
        assert report_path.exists(), "Ingestion summary JSON not found."
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "SUCCESS"
        assert data["processed_counts"]["analytical_rows"] == 421570
        assert data["processed_counts"]["row_count_preservation_ratio"] == 1.0
