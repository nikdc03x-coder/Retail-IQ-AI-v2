"""
RetailIQ AI V2 — Rolling Backtest Module

Executes expanding-window historical backtests to assess forecast stability
and verify that XGBoost consistently outperforms persistence baselines over time.
"""

from typing import Any
import pandas as pd
import numpy as np

from src.retailiq.forecasting.evaluation import calculate_metrics
from src.retailiq.forecasting.features import build_forecasting_features, get_feature_columns
from src.retailiq.forecasting.models import predict_naive_previous_week, train_xgboost_model


def run_rolling_backtest(
    features_df: pd.DataFrame,
    n_folds: int = 4,
    test_window_weeks: int = 8,
) -> pd.DataFrame:
    """Run an expanding-window rolling backtest over historical folds.

    Simulates periodic model retraining and deployment:
    - Fold 1: Train on initial period -> Test on next 8 weeks
    - Fold 2: Train expanded by 8 weeks -> Test on next 8 weeks
    - Fold 3: Train expanded by 16 weeks -> Test on next 8 weeks
    - Fold 4: Train expanded by 24 weeks -> Test on final 8 weeks

    Args:
        features_df: Pre-engineered feature DataFrame.
        n_folds: Number of backtest evaluation windows.
        test_window_weeks: Duration in weeks of each evaluation fold.

    Returns:
        DataFrame with fold metrics and outperformance delta.
    """
    df = features_df.copy()
    dates = sorted(df["Date"].unique())
    total_weeks = len(dates)

    # Reserve test windows from the end
    feature_cols = get_feature_columns()
    target_col = "Weekly_Sales"

    fold_results = []
    min_train_weeks = total_weeks - (n_folds * test_window_weeks)

    for fold in range(n_folds):
        train_end_idx = min_train_weeks + (fold * test_window_weeks)
        test_end_idx = train_end_idx + test_window_weeks

        train_dates = dates[:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        train_data = df[df["Date"].isin(train_dates)]
        test_data = df[df["Date"].isin(test_dates)]

        X_tr, y_tr = train_data[feature_cols], train_data[target_col]
        X_te, y_te = test_data[feature_cols], test_data[target_col]

        # Train model for this fold
        model = train_xgboost_model(
            X_tr, y_tr,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.06,
            random_state=42 + fold,
        )

        xgb_preds = model.predict(X_te)
        naive_preds = predict_naive_previous_week(test_data)

        m_xgb = calculate_metrics(y_te, xgb_preds, "XGBoost", f"Fold_{fold+1}")
        m_naive = calculate_metrics(y_te, naive_preds, "Naive", f"Fold_{fold+1}")

        wape_diff = round(m_naive["WAPE_Pct"] - m_xgb["WAPE_Pct"], 2)
        error_reduction = round((wape_diff / m_naive["WAPE_Pct"] * 100.0), 2) if m_naive["WAPE_Pct"] > 0 else 0.0

        fold_results.append({
            "Fold": fold + 1,
            "Train_Weeks": len(train_dates),
            "Test_Weeks": len(test_dates),
            "Test_Start_Date": str(pd.to_datetime(test_dates[0]).date()),
            "Test_End_Date": str(pd.to_datetime(test_dates[-1]).date()),
            "XGB_WAPE": m_xgb["WAPE_Pct"],
            "Naive_WAPE": m_naive["WAPE_Pct"],
            "WAPE_Improvement": wape_diff,
            "Error_Reduction_Pct": error_reduction,
            "XGB_MAE": m_xgb["MAE"],
            "Naive_MAE": m_naive["MAE"],
            "XGB_R2": m_xgb["R2"],
            "XGB_Outperformed": m_xgb["WAPE_Pct"] < m_naive["WAPE_Pct"],
        })

    return pd.DataFrame(fold_results)
