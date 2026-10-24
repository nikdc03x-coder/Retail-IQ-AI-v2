"""
RetailIQ AI V2 — Forecasting Experiment Orchestration Module

Runs reproducible, leakage-free 1-week-ahead store sales forecasting experiment:
1. Feature engineering (lags, rolling, cyclical calendar, store metadata).
2. Chronological Train / Validation / Test split.
3. Baseline evaluation (Previous-Week Naive, 52-Week Seasonal Naive).
4. Model training (Ridge, Random Forest, XGBoost).
5. Comprehensive metrics evaluation (MAE, RMSE, WAPE, R²).
6. Model artifact serialization (Joblib).
7. Report generation in reports/sales_forecasting/.
"""

import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd

from src.retailiq.config import MODELS_DIR, REPORT_SUBDIRS, ensure_directories
from src.retailiq.forecasting.evaluation import calculate_metrics
from src.retailiq.forecasting.features import build_forecasting_features, get_feature_columns
from src.retailiq.forecasting.models import (
    extract_feature_importance,
    predict_naive_previous_week,
    predict_seasonal_naive_52,
    train_random_forest_model,
    train_ridge_model,
    train_xgboost_model,
)
from src.retailiq.forecasting.splitter import chronological_split


def run_forecasting_experiment(
    store_week_df: pd.DataFrame | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute complete sales forecasting comparison experiment.

    Args:
        store_week_df: Optional pre-aggregated store-week DataFrame.
        save_artifacts: Whether to save models (.joblib) and reports (.csv/.json).

    Returns:
        Dictionary of trained models, comparison metrics, test predictions, and metadata.
    """
    ensure_directories()

    # Load store-week sales report if not provided
    if store_week_df is None:
        report_path = REPORT_SUBDIRS["business_analytics"] / "04_store_week_sales.csv"
        if not report_path.exists():
            raise FileNotFoundError(
                f"Store-week sales report not found at {report_path}. "
                "Please run scripts/03_business_analytics/run_analytics.py first."
            )
        store_week_df = pd.read_csv(report_path, parse_dates=["Date"])

    # ── 1. Feature Engineering ──
    features_df = build_forecasting_features(store_week_df)
    feature_cols = get_feature_columns()
    target_col = "Weekly_Sales"

    # ── 2. Chronological Split ──
    train_df, val_df, test_df, split_meta = chronological_split(
        features_df,
        val_start="2012-01-06",
        test_start="2012-06-01",
    )

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # ── 3. Evaluate Baseline Models ──
    # Baseline 1: Previous-Week Naive (lag_1)
    naive_val_pred = predict_naive_previous_week(val_df)
    naive_test_pred = predict_naive_previous_week(test_df)

    m_naive_val = calculate_metrics(y_val, naive_val_pred, "Previous-Week Naive", "Validation")
    m_naive_test = calculate_metrics(y_test, naive_test_pred, "Previous-Week Naive", "Test")

    # Baseline 2: 52-Week Seasonal Naive (lag_52)
    seasonal_val_pred = predict_seasonal_naive_52(val_df)
    seasonal_test_pred = predict_seasonal_naive_52(test_df)

    m_seasonal_val = calculate_metrics(y_val, seasonal_val_pred, "Seasonal Naive (52-Wk)", "Validation")
    m_seasonal_test = calculate_metrics(y_test, seasonal_test_pred, "Seasonal Naive (52-Wk)", "Test")

    # ── 4. Train & Evaluate Ridge Regression ──
    ridge_model = train_ridge_model(X_train, y_train, alpha=100.0)
    ridge_val_pred = ridge_model.predict(X_val)
    ridge_test_pred = ridge_model.predict(X_test)

    m_ridge_val = calculate_metrics(y_val, ridge_val_pred, "Ridge Linear", "Validation")
    m_ridge_test = calculate_metrics(y_test, ridge_test_pred, "Ridge Linear", "Test")

    # ── 5. Train & Evaluate Random Forest ──
    rf_model = train_random_forest_model(X_train, y_train, n_estimators=100, max_depth=12, random_state=42)
    rf_val_pred = rf_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test)

    m_rf_val = calculate_metrics(y_val, rf_val_pred, "Random Forest", "Validation")
    m_rf_test = calculate_metrics(y_test, rf_test_pred, "Random Forest", "Test")

    # ── 6. Train & Evaluate XGBoost ──
    xgb_model = train_xgboost_model(
        X_train, y_train,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_test_pred = xgb_model.predict(X_test)

    m_xgb_val = calculate_metrics(y_val, xgb_val_pred, "XGBoost", "Validation")
    m_xgb_test = calculate_metrics(y_test, xgb_test_pred, "XGBoost", "Test")

    # ── 7. Compile Model Comparison Table ──
    all_metrics = [
        m_naive_val, m_naive_test,
        m_seasonal_val, m_seasonal_test,
        m_ridge_val, m_ridge_test,
        m_rf_val, m_rf_test,
        m_xgb_val, m_xgb_test,
    ]
    metrics_df = pd.DataFrame(all_metrics)

    # ── 8. Assemble Test Predictions DataFrame ──
    test_pred_df = test_df[["Store", "Date", "IsHoliday", "Type", "Size"]].copy()
    test_pred_df["Actual_Sales"] = y_test.round(2)
    test_pred_df["Naive_Pred"] = naive_test_pred.round(2)
    test_pred_df["Seasonal_Pred"] = seasonal_test_pred.round(2)
    test_pred_df["Ridge_Pred"] = ridge_test_pred.round(2)
    test_pred_df["RF_Pred"] = rf_test_pred.round(2)
    test_pred_df["XGBoost_Pred"] = xgb_test_pred.round(2)

    # Calculate absolute residuals for primary champion model (XGBoost)
    test_pred_df["XGBoost_Residual"] = (test_pred_df["Actual_Sales"] - test_pred_df["XGBoost_Pred"]).round(2)
    test_pred_df["XGBoost_Abs_Error"] = test_pred_df["XGBoost_Residual"].abs()
    test_pred_df["XGBoost_Pct_Error"] = (
        test_pred_df["XGBoost_Abs_Error"] / test_pred_df["Actual_Sales"] * 100.0
    ).round(2)

    # ── 9. Feature Importance ──
    xgb_importance = extract_feature_importance(xgb_model, feature_cols, "XGBoost")
    rf_importance = extract_feature_importance(rf_model, feature_cols, "Random Forest")
    importance_df = pd.concat([xgb_importance, rf_importance], ignore_index=True)

    # ── 10. Save Artifacts ──
    out_dir = REPORT_SUBDIRS["sales_forecasting"]

    if save_artifacts:
        # Save Models
        joblib.dump(ridge_model, MODELS_DIR / "ridge_model.joblib")
        joblib.dump(rf_model, MODELS_DIR / "random_forest_model.joblib")
        joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.joblib")

        # Save Reports
        metrics_df.to_csv(out_dir / "01_model_comparison_metrics.csv", index=False)
        test_pred_df.to_csv(out_dir / "02_test_predictions.csv", index=False)
        importance_df.to_csv(out_dir / "03_feature_importance.csv", index=False)

        # Save Experiment Metadata JSON
        meta = {
            "experiment": "Phase 04 — Store-Level 1-Week-Ahead Sales Forecasting",
            "target": target_col,
            "horizon_weeks": 1,
            "split_dates": split_meta,
            "feature_columns": feature_cols,
            "test_summary": {
                "naive_mae": m_naive_test["MAE"],
                "naive_wape": m_naive_test["WAPE_Pct"],
                "seasonal_mae": m_seasonal_test["MAE"],
                "seasonal_wape": m_seasonal_test["WAPE_Pct"],
                "ridge_mae": m_ridge_test["MAE"],
                "ridge_wape": m_ridge_test["WAPE_Pct"],
                "rf_mae": m_rf_test["MAE"],
                "rf_wape": m_rf_test["WAPE_Pct"],
                "xgboost_mae": m_xgb_test["MAE"],
                "xgboost_wape": m_xgb_test["WAPE_Pct"],
                "champion_model": "XGBoost" if m_xgb_test["WAPE_Pct"] < m_rf_test["WAPE_Pct"] else "Random Forest",
            },
        }
        with open(out_dir / "04_experiment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return {
        "models": {
            "ridge": ridge_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
        },
        "metrics": metrics_df,
        "test_predictions": test_pred_df,
        "feature_importance": importance_df,
        "split_meta": split_meta,
    }
