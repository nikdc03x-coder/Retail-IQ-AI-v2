"""
RetailIQ AI V2 — Phase 05: Forecast Reliability Tests

Validates:
- Store error aggregation across 45 locations
- Temporal error variations across 22 test weeks
- Holiday error degradation quantification
- Head-to-head store-level win rates vs Naive baseline
- 4-fold expanding-window rolling backtest consistency
- Reliability summary report schema and serialization
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.config import REPORT_SUBDIRS
from src.retailiq.reliability.error_analysis import (
    analyze_holiday_error_impact,
    analyze_store_errors,
    analyze_weekly_errors,
    compute_relative_baseline_errors,
)


@pytest.fixture(scope="module")
def test_preds() -> pd.DataFrame:
    """Load test predictions from Phase 04."""
    fpath = REPORT_SUBDIRS["sales_forecasting"] / "02_test_predictions.csv"
    return pd.read_csv(fpath, parse_dates=["Date"])


class TestStoreErrorAnalysis:
    """Verify store-by-store error dispersion."""

    def test_store_error_summary_shape(self, test_preds: pd.DataFrame):
        store_err = analyze_store_errors(test_preds)
        assert len(store_err) == 45, f"Expected 45 stores, got {len(store_err)}"
        assert list(store_err["Reliability_Rank"]) == list(range(1, 46))
        assert (store_err["WAPE_Pct"] > 0).all()

        # Best store WAPE should be < 3%, worst should be > 8%
        assert store_err.iloc[0]["WAPE_Pct"] < 3.0
        assert store_err.iloc[-1]["WAPE_Pct"] > 8.0


class TestWeeklyErrorAnalysis:
    """Verify temporal error profiling across test window."""

    def test_weekly_error_continuity(self, test_preds: pd.DataFrame):
        week_err = analyze_weekly_errors(test_preds)
        assert len(week_err) == 22, f"Expected 22 test weeks, got {len(week_err)}"
        assert (week_err["WAPE_Pct"] > 0).all()
        assert (week_err["MAE"] > 0).all()


class TestHolidayErrorDegradation:
    """Verify holiday error degradation hypothesis."""

    def test_holiday_error_increase(self, test_preds: pd.DataFrame):
        hol_err = analyze_holiday_error_impact(test_preds)
        assert len(hol_err) == 2

        non_hol = hol_err.loc[hol_err["Period_Type"] == "Non-Holiday Weeks", "WAPE_Pct"].values[0]
        hol = hol_err.loc[hol_err["Period_Type"] == "Holiday Weeks", "WAPE_Pct"].values[0]

        # Holiday WAPE should be higher than non-holiday WAPE
        assert hol > non_hol, f"Expected holiday WAPE ({hol}) > non-holiday WAPE ({non_hol})"


class TestBaselineHeadToHead:
    """Verify store-level outperformance vs Naive baseline."""

    def test_store_win_rate_threshold(self, test_preds: pd.DataFrame):
        store_rel = compute_relative_baseline_errors(test_preds)
        assert len(store_rel) == 45

        win_count = store_rel["XGB_Wins"].sum()
        win_rate = win_count / len(store_rel) * 100.0

        assert win_rate >= 75.0, f"Expected win rate >= 75%, got {win_rate}%"


class TestRollingBacktestReports:
    """Verify backtest and summary reports on disk."""

    def test_backtest_results_consistency(self):
        bt_path = REPORT_SUBDIRS["forecast_reliability"] / "05_rolling_backtest_results.csv"
        assert bt_path.exists()
        bt_df = pd.read_csv(bt_path)

        assert len(bt_df) == 4, f"Expected 4 folds, got {len(bt_df)}"
        # XGBoost should win all folds
        assert bt_df["XGB_Outperformed"].all(), "XGBoost should win 100% of backtest folds"
        # Mean error reduction should be > 15%
        assert bt_df["Error_Reduction_Pct"].mean() > 15.0

    def test_summary_json_metrics(self):
        sum_path = REPORT_SUBDIRS["forecast_reliability"] / "06_forecast_reliability_summary.json"
        assert sum_path.exists()
        with open(sum_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["stores_analyzed"] == 45
        assert data["store_win_rate_vs_naive_pct"] > 75.0
        assert data["backtest_summary"]["folds_outperformed"] == 4
