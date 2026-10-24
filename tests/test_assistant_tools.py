"""
RetailIQ AI V2 — Phase 09: Assistant Tools Unit Tests

Validates:
- Every typed tool in src/retailiq/assistant/tools.py returns valid schemas and numbers.
- Input validation and exception guards (invalid store numbers, negative counts).
- Concordance between tool output and underlying reports/ data.
"""

import pytest
from src.retailiq.assistant.tools import (
    get_dataset_overview,
    get_chain_sales_summary,
    get_store_rankings,
    get_department_rankings,
    compare_stores,
    get_holiday_performance,
    get_forecast_leaderboard,
    compare_forecasting_models,
    get_store_forecast_errors,
    get_anomaly_summary,
    get_anomaly_records,
    get_global_shap_importance,
    explain_store_prediction,
)


class TestDatasetAndSalesTools:
    """Verify dataset overview and chain sales summary tools."""

    def test_get_dataset_overview(self):
        res = get_dataset_overview()
        assert res["success"] is True
        assert res["data"]["total_rows"] == 421570
        assert res["data"]["unique_stores"] == 45
        assert res["data"]["unique_departments"] == 81
        assert "evidence_source" in res

    def test_get_chain_sales_summary(self):
        res = get_chain_sales_summary()
        assert res["success"] is True
        assert res["data"]["total_sales"] > 6_000_000_000
        assert res["data"]["mean_weekly_chain_sales"] > 40_000_000
        assert "evidence_source" in res


class TestRankingAndComparisonTools:
    """Verify store and department ranking and comparison tools."""

    def test_get_store_rankings(self):
        res = get_store_rankings(top_n=5, ascending=False)
        assert res["success"] is True
        assert len(res["data"]) == 5
        assert res["data"][0]["store"] == 20  # Store 20 is #1
        assert res["data"][0]["rank"] == 1

    def test_get_store_rankings_ascending(self):
        res = get_store_rankings(top_n=3, ascending=True)
        assert res["success"] is True
        assert len(res["data"]) == 3
        assert res["data"][0]["store"] == 33  # Store 33 is lowest

    def test_get_department_rankings(self):
        res = get_department_rankings(top_n=5)
        assert res["success"] is True
        assert len(res["data"]) == 5
        assert res["data"][0]["department"] == 92  # Dept 92 is #1

    def test_compare_stores_valid(self):
        res = compare_stores([4, 20])
        assert res["success"] is True
        assert len(res["data"]) == 2
        stores = [r["store"] for r in res["data"]]
        assert 4 in stores and 20 in stores

    def test_compare_stores_invalid_id(self):
        res = compare_stores([999])
        assert res["success"] is False
        assert "invalid" in res["error"].lower()


class TestHolidayAndForecastingTools:
    """Verify holiday lift and forecasting model evaluation tools."""

    def test_get_holiday_performance(self):
        res = get_holiday_performance()
        assert res["success"] is True
        assert "Thanksgiving" in res["summary"]
        assert "+41." in res["summary"]

    def test_get_forecast_leaderboard(self):
        res = get_forecast_leaderboard()
        assert res["success"] is True
        assert len(res["data"]) == 5
        champion = res["data"][0]
        assert "XGBoost" in champion["Model"]
        assert champion["WAPE_Pct"] < 4.0

    def test_compare_forecasting_models(self):
        res = compare_forecasting_models(model_a="XGBoost", model_b="Previous-Week Naive")
        assert res["success"] is True
        assert res["data"]["mae_reduction_pct"] > 20.0  # ~22.4% error reduction


class TestReliabilityAndAnomalyTools:
    """Verify forecast error spread and anomaly queue tools."""

    def test_get_store_forecast_errors(self):
        res = get_store_forecast_errors(top_n=3, best=True)
        assert res["success"] is True
        assert len(res["data"]) == 3
        assert res["data"][0]["WAPE_Pct"] < 3.0  # Store 37 is best (~2.06%)

    def test_get_anomaly_summary(self):
        res = get_anomaly_summary()
        assert res["success"] is True
        assert res["data"]["total_anomalies_flagged"] == 23
        assert res["data"]["consensus_flags_count"] == 14

    def test_get_anomaly_records(self):
        res = get_anomaly_records(top_n=5)
        assert res["success"] is True
        assert len(res["data"]) == 5
        assert res["data"][0]["Store"] == 14  # Store 14 on 2012-08-24 is #1 priority


class TestExplainabilityTools:
    """Verify global and local TreeSHAP attribution tools."""

    def test_get_global_shap_importance(self):
        res = get_global_shap_importance(top_n=5)
        assert res["success"] is True
        assert len(res["data"]) == 5
        assert res["data"][0]["Feature"] == "lag_52"  # Prior-year seasonality is #1

    def test_explain_store_prediction_valid(self):
        res = explain_store_prediction(store=20, date="2012-06-01")
        assert res["success"] is True
        assert res["data"]["store"] == 20
        assert res["data"]["model_prediction"] > 0
        assert len(res["data"]["top_drivers"]) > 0

    def test_explain_store_prediction_invalid_date(self):
        res = explain_store_prediction(store=20, date="1999-01-01")
        assert res["success"] is False
        assert "No prediction found" in res["error"]
