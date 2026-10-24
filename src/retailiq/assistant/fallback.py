"""
RetailIQ AI V2 — Deterministic Fallback Dispatcher

Provides offline, rule-based question answering when the Google Gemini API key
is missing, expired, or network-constrained. Matches business questions to the
appropriate typed tools in src/retailiq/assistant/tools.py without compromising
numerical veracity or hallucinating answers.
"""

import re
from typing import Any
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


def _format_output(tool_name: str, args: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    """Helper to standardize fallback output schema."""
    success = res.get("success", False)
    answer = res.get("summary") if success else res.get("error", "Error executing analytical tool.")
    return {
        "route": "deterministic_fallback",
        "tool_called": tool_name,
        "arguments": args,
        "answer": answer,
        "data": res.get("data"),
        "evidence_source": res.get("evidence_source", "reports/"),
        "verified": success,
    }


def dispatch_fallback_query(query: str) -> dict[str, Any]:
    """Dispatch a natural language business question to approved tools via deterministic pattern matching."""
    q = query.lower().strip()

    # ── 1. Holiday Performance & Lifts ──
    if any(k in q for k in ["holiday", "thanksgiving", "christmas", "super bowl", "labor day", "holiday lift"]):
        res = get_holiday_performance()
        return _format_output("get_holiday_performance", {}, res)

    # ── 2. Anomaly Detection & Investigation ──
    if any(k in q for k in ["priority anomaly", "investigation queue", "audit queue", "unusual", "outlier record", "store 14"]):
        res = get_anomaly_records(top_n=5)
        return _format_output("get_anomaly_records", {"top_n": 5}, res)

    if any(k in q for k in ["anomaly", "anomalies", "outlier", "detector", "isolation forest", "z-score"]):
        res = get_anomaly_summary()
        return _format_output("get_anomaly_summary", {}, res)

    # ── 3. Explainable AI & SHAP ──
    if "explain" in q and "store" in q:
        store_match = re.search(r"store\s*(\d+)", q)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", q)
        store_id = int(store_match.group(1)) if store_match else 20
        date_str = date_match.group(1) if date_match else "2012-06-01"

        res = explain_store_prediction(store=store_id, date=date_str)
        return _format_output("explain_store_prediction", {"store": store_id, "date": date_str}, res)

    if any(k in q for k in ["shap", "feature importance", "driver", "what drives", "key features", "most important feature"]):
        res = get_global_shap_importance(top_n=5)
        return _format_output("get_global_shap_importance", {"top_n": 5}, res)

    # ── 4. Forecast Reliability & Spatial Errors ──
    if any(k in q for k in ["volatile store", "worst forecast", "highest error store", "most inaccurate"]):
        res = get_store_forecast_errors(top_n=5, best=False)
        return _format_output("get_store_forecast_errors", {"top_n": 5, "best": False}, res)

    if any(k in q for k in ["reliable store", "best forecast store", "lowest error store", "most accurate store", "win rate"]):
        res = get_store_forecast_errors(top_n=5, best=True)
        return _format_output("get_store_forecast_errors", {"top_n": 5, "best": True}, res)

    # ── 5. Forecasting Models & Leaderboard ──
    if any(k in q for k in ["vs naive", "versus naive", "compare model", "xgboost vs"]):
        res = compare_forecasting_models(model_a="XGBoost", model_b="Previous-Week Naive")
        return _format_output("compare_forecasting_models", {"model_a": "XGBoost", "model_b": "Previous-Week Naive"}, res)

    if any(k in q for k in ["best model", "leaderboard", "model performance", "forecast accuracy", "wape", "mae", "rmse", "r2", "forecasting", "model"]):
        res = get_forecast_leaderboard(split="Test")
        return _format_output("get_forecast_leaderboard", {"split": "Test"}, res)

    # ── 6. Department Performance ──
    if any(k in q for k in ["dept", "department"]):
        ascending = any(w in q for w in ["worst", "lowest", "bottom", "least"])
        res = get_department_rankings(top_n=5, ascending=ascending)
        return _format_output("get_department_rankings", {"top_n": 5, "ascending": ascending}, res)

    # ── 7. Store Rankings & Comparison ──
    store_matches = re.findall(r"store\s*(\d+)", q)
    if len(store_matches) >= 2:
        store_ids = [int(s) for s in store_matches]
        res = compare_stores(store_ids=store_ids)
        return _format_output("compare_stores", {"store_ids": store_ids}, res)

    if any(k in q for k in ["store rank", "top store", "best store", "highest store", "which store", "store sales"]):
        ascending = any(w in q for w in ["worst", "lowest", "bottom", "least"])
        res = get_store_rankings(top_n=5, ascending=ascending)
        return _format_output("get_store_rankings", {"top_n": 5, "ascending": ascending}, res)

    # ── 8. Chain Sales & Weekly Volumes ──
    if any(k in q for k in ["total sale", "total revenue", "overall sale", "average weekly sale", "chain revenue", "peak week", "sales"]):
        res = get_chain_sales_summary()
        return _format_output("get_chain_sales_summary", {}, res)

    # ── 9. Dataset Overview & Ingestion ──
    if any(k in q for k in ["dataset", "row", "how many observation", "overview", "raw data", "negative sale", "returns"]):
        res = get_dataset_overview()
        return _format_output("get_dataset_overview", {}, res)

    # ── 10. Default General Overview ──
    res = get_dataset_overview()
    output = _format_output("get_dataset_overview", {}, res)
    output["answer"] = (
        "I could not match your exact question to a specialized analytical tool. "
        "Here is the general dataset overview: " + str(output["answer"])
    )
    return output
