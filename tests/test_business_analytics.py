"""
RetailIQ AI V2 — Phase 03: Business Intelligence Analytics Tests

Validates:
- Store-week hierarchical aggregation integrity (45 stores x 143 weeks = 6,435 rows)
- Store performance ranking (Store 20 #1, Pareto shares, space productivity)
- Department contribution ranking (Dept 92 #1, cumulative shares)
- Store format comparison (Type A/B/C store counts and revenue shares)
- Holiday lift calculations (Thanksgiving peak surge > +35%)
- Monthly seasonality index
- Report file generation and schema integrity
"""

from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.config import REPORT_SUBDIRS
from src.retailiq.data.loader import load_analytical_dataset
from src.retailiq.analytics.aggregations import (
    aggregate_department_performance,
    aggregate_holiday_dynamics,
    aggregate_monthly_seasonality,
    aggregate_store_performance,
    aggregate_store_type_performance,
    aggregate_store_week_sales,
    aggregate_weekly_network_sales,
)


@pytest.fixture(scope="module")
def analytical_df() -> pd.DataFrame:
    """Load analytical dataset once for all tests in module."""
    return load_analytical_dataset()


@pytest.fixture(scope="module")
def store_week_df(analytical_df: pd.DataFrame) -> pd.DataFrame:
    """Compute store-week aggregation once for module."""
    return aggregate_store_week_sales(analytical_df)


class TestStoreWeekAggregation:
    """Verify department to store-week rollup integrity."""

    def test_store_week_shape_and_keys(self, store_week_df: pd.DataFrame, analytical_df: pd.DataFrame):
        # 45 stores * 143 weeks = 6,435 rows
        assert len(store_week_df) == 6435, f"Expected 6,435 store-weeks, got {len(store_week_df)}"
        assert store_week_df["Store"].nunique() == 45
        assert store_week_df["Date"].nunique() == 143

        # Revenue conservation: sum of store-weeks must match sum of dept-weeks
        total_store_week_sales = store_week_df["Weekly_Sales"].sum()
        total_dept_week_sales = analytical_df["Weekly_Sales"].sum()
        assert abs(total_store_week_sales - total_dept_week_sales) < 1.0, "Revenue leaked during store-week rollup!"


class TestStorePerformanceRankings:
    """Verify store-level ranking and productivity calculations."""

    def test_store_rankings_and_top_store(self, store_week_df: pd.DataFrame):
        store_perf = aggregate_store_performance(store_week_df)
        assert len(store_perf) == 45, f"Expected 45 stores, got {len(store_perf)}"
        assert list(store_perf["Sales_Rank"]) == list(range(1, 46))

        # Store 20 is the historical top revenue generator
        top_store = store_perf.iloc[0]
        assert top_store["Store"] == 20
        assert top_store["Total_Sales"] > 300_000_000

        # Share sum must be approximately 100%
        assert abs(store_perf["Sales_Share_Pct"].sum() - 100.0) < 0.5


class TestDepartmentPerformanceRankings:
    """Verify department contribution Pareto shares."""

    def test_department_rankings_and_top_dept(self, analytical_df: pd.DataFrame):
        dept_perf = aggregate_department_performance(analytical_df)
        assert len(dept_perf) == 81, f"Expected 81 departments, got {len(dept_perf)}"
        assert list(dept_perf["Dept_Rank"]) == list(range(1, 82))

        # Dept 92 is the top generator (~7% share)
        top_dept = dept_perf.iloc[0]
        assert top_dept["Dept"] == 92
        assert top_dept["Sales_Share_Pct"] > 6.0

        # Top 10 departments should account for over 45% of total sales
        top_10_share = dept_perf.iloc[9]["Cumulative_Share_Pct"]
        assert top_10_share > 45.0, f"Expected top 10 share > 45%, got {top_10_share}%"


class TestStoreTypePerformance:
    """Verify store type A, B, C breakdown."""

    def test_store_types_summary(self, store_week_df: pd.DataFrame):
        store_perf = aggregate_store_performance(store_week_df)
        type_perf = aggregate_store_type_performance(store_perf)

        assert len(type_perf) == 3
        assert set(type_perf["Type"]) == {"A", "B", "C"}

        type_a = type_perf[type_perf["Type"] == "A"].iloc[0]
        assert type_a["Store_Count"] == 22
        assert type_a["Network_Sales_Share_Pct"] > 60.0, "Type A should represent > 60% of network sales"


class TestHolidayDynamics:
    """Verify Walmart holiday surge and lift calculations."""

    def test_holiday_lift_and_thanksgiving(self, store_week_df: pd.DataFrame):
        holiday_df = aggregate_holiday_dynamics(store_week_df)

        # Baseline and All Holidays present
        names = holiday_df["Holiday_Name"].tolist()
        assert "Non-Holiday Baseline" in names
        assert "All Holidays Combined" in names
        assert "Thanksgiving" in names
        assert "Super Bowl" in names
        assert "Labor Day" in names
        assert "Christmas" in names

        # Thanksgiving must have large positive surge
        tg = holiday_df[holiday_df["Holiday_Name"] == "Thanksgiving"].iloc[0]
        assert tg["Lift_Vs_Baseline_Pct"] > 35.0, f"Expected Thanksgiving lift > 35%, got {tg['Lift_Vs_Baseline_Pct']}%"

        # Overall holidays must have positive lift
        all_hol = holiday_df[holiday_df["Holiday_Name"] == "All Holidays Combined"].iloc[0]
        assert all_hol["Lift_Vs_Baseline_Pct"] > 0, "Overall holiday lift should be positive"


class TestMonthlySeasonality:
    """Verify monthly aggregation and indices."""

    def test_monthly_seasonality_indices(self, store_week_df: pd.DataFrame):
        net_df = aggregate_weekly_network_sales(store_week_df)
        monthly_df = aggregate_monthly_seasonality(net_df)

        assert len(monthly_df) == 12, f"Expected 12 months, got {len(monthly_df)}"
        # December or November should have highest weekly average sales
        top_month = monthly_df.sort_values(by="Mean_Weekly_Sales", ascending=False).iloc[0]
        assert top_month["Month"] in [11, 12], "Peak sales should occur in holiday season (Nov/Dec)"


class TestBusinessAnalyticsReportsExist:
    """Verify all 7 pre-computed business intelligence reports exist."""

    def test_all_reports_present_on_disk(self):
        out_dir = REPORT_SUBDIRS["business_analytics"]
        expected_files = [
            "01_weekly_sales.csv",
            "02_store_performance.csv",
            "03_department_performance.csv",
            "04_store_week_sales.csv",
            "05_holiday_comparison.csv",
            "06_store_type_performance.csv",
            "07_monthly_seasonality.csv",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            assert fpath.exists(), f"Missing business analytics report {fname}"
            assert fpath.stat().st_size > 0, f"Report {fname} is empty"

    def test_canonical_store_type_report_schema(self):
        """Regression test for ISSUE-11-06: Verify canonical store type performance report schema."""
        out_dir = REPORT_SUBDIRS["business_analytics"]
        target = out_dir / "06_store_type_performance.csv"
        assert target.exists(), "06_store_type_performance.csv must exist as the canonical report"
        df = pd.read_csv(target)
        assert len(df) == 3, f"Expected 3 store types, got {len(df)}"
        assert set(df["Type"]) == {"A", "B", "C"}
        expected_cols = ["Type", "Store_Count", "Total_Sales", "Mean_Weekly_Store_Sales", "Network_Sales_Share_Pct"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in store type performance report"
