"""
RetailIQ AI V2 — Phase 09: Conversational Assistant Benchmark Suite

Evaluates the unified execution engine across 8 core business query categories:
- Data sizing and date range inquiries
- Chain sales volume and weekly averages
- Store and department rankings
- Holiday promotional dynamics
- Forecast model race and error reduction
- Outlier detection and investigation prioritization
- TreeSHAP feature attributions
- Guard validation and transparency metadata integrity
"""

import pytest
from src.retailiq.assistant.engine import RetailIQAssistantEngine


@pytest.fixture(scope="module")
def engine() -> RetailIQAssistantEngine:
    """Initialize offline/fallback assistant engine for deterministic benchmarking."""
    return RetailIQAssistantEngine(api_key="")


class TestAssistantBenchmark:
    """Benchmark tests validating accuracy of assistant engine responses."""

    def test_empty_query_guard(self, engine: RetailIQAssistantEngine):
        res = engine.ask("")
        assert res["route"] == "validation_guard"
        assert res["verified"] is False
        assert "specific business" in res["answer"].lower()

    def test_benchmark_dataset_sizing(self, engine: RetailIQAssistantEngine):
        res = engine.ask("How many rows and stores are in the dataset?")
        assert res["route"] == "deterministic_fallback"
        assert "get_dataset_overview" in res["tools_called"]
        assert "421,570" in res["answer"]
        assert "45 stores" in res["answer"]
        assert res["verified"] is True
        assert res["latency_seconds"] < 1.0

    def test_benchmark_store_rankings(self, engine: RetailIQAssistantEngine):
        res = engine.ask("Which are the top stores by sales?")
        assert res["route"] == "deterministic_fallback"
        assert "get_store_rankings" in res["tools_called"]
        assert "Store 20" in res["answer"]
        assert res["verified"] is True

    def test_benchmark_holiday_dynamics(self, engine: RetailIQAssistantEngine):
        res = engine.ask("How much sales lift does Thanksgiving produce?")
        assert res["route"] == "deterministic_fallback"
        assert "get_holiday_performance" in res["tools_called"]
        assert "Thanksgiving" in res["answer"]
        assert "+41." in res["answer"]
        assert res["verified"] is True

    def test_benchmark_forecasting_leaderboard(self, engine: RetailIQAssistantEngine):
        res = engine.ask("What is the best model on the forecasting leaderboard?")
        assert res["route"] == "deterministic_fallback"
        assert "get_forecast_leaderboard" in res["tools_called"]
        assert "XGBoost" in res["answer"]
        assert "3.83%" in res["answer"]
        assert res["verified"] is True

    def test_benchmark_model_head_to_head(self, engine: RetailIQAssistantEngine):
        res = engine.ask("How does XGBoost compare versus naive baseline?")
        assert res["route"] == "deterministic_fallback"
        assert "compare_forecasting_models" in res["tools_called"]
        assert "XGBoost" in res["answer"]
        assert "Previous-Week Naive" in res["answer"]
        assert "reduction in mean absolute error" in res["answer"]
        assert res["verified"] is True

    def test_benchmark_anomaly_queue(self, engine: RetailIQAssistantEngine):
        res = engine.ask("What are the priority anomaly records for review?")
        assert res["route"] == "deterministic_fallback"
        assert "get_anomaly_records" in res["tools_called"]
        assert "Store 14" in res["answer"]
        assert "2012-08-24" in res["answer"]
        assert res["verified"] is True

    def test_benchmark_shap_feature_importance(self, engine: RetailIQAssistantEngine):
        res = engine.ask("What are the top features driving the forecasting model in SHAP?")
        assert res["route"] == "deterministic_fallback"
        assert "get_global_shap_importance" in res["tools_called"]
        assert "lag_52" in res["answer"]
        assert "72%" in res["answer"]
        assert res["verified"] is True

    def test_transparency_schema_integrity(self, engine: RetailIQAssistantEngine):
        res = engine.ask("What are the total sales of the chain?")
        required_keys = ["route", "answer", "tools_called", "evidence_source", "latency_seconds", "provider", "verified"]
        for k in required_keys:
            assert k in res, f"Missing required transparency key: {k}"
        assert res["provider"] == "Deterministic Rule-Based Dispatcher (Offline Mode)"
        assert res["evidence_source"].startswith("reports/")
