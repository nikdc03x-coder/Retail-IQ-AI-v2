"""
RetailIQ AI V2 — Statistical Anomaly Detection Module

Detects forecasting residual anomalies using standard score (Z-score) thresholding:
Z = (e - mu_e) / sigma_e
Flags observations where |Z| > 3.0 (extreme outliers beyond 3 standard deviations).
"""

from typing import Any
import numpy as np
import pandas as pd


def detect_statistical_anomalies(
    test_preds_df: pd.DataFrame,
    threshold_z: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect statistical residual anomalies on held-out test predictions.

    Args:
        test_preds_df: DataFrame containing Actual_Sales and XGBoost_Pred.
        threshold_z: Critical standard score cutoff (default 3.0).

    Returns:
        Tuple of (scored_df, metrics_summary).
    """
    df = test_preds_df.copy()
    residuals = df["Actual_Sales"] - df["XGBoost_Pred"]

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals, ddof=1))

    z_scores = (residuals - mean_res) / std_res if std_res > 0 else np.zeros_like(residuals)
    is_anomaly = np.abs(z_scores) > threshold_z

    df["Residual"] = residuals.round(2)
    df["Z_Score"] = z_scores.round(3)
    df["Is_Statistical_Anomaly"] = is_anomaly
    df["Residual_Direction"] = np.where(
        z_scores > threshold_z,
        "Extreme Under-Prediction (Actual >> Forecast)",
        np.where(
            z_scores < -threshold_z,
            "Extreme Over-Prediction (Forecast >> Actual)",
            "Expected Range",
        ),
    )

    total_obs = len(df)
    anomaly_count = int(is_anomaly.sum())

    summary = {
        "method": "Statistical Residual Z-Score",
        "threshold_z": threshold_z,
        "residual_mean": round(mean_res, 2),
        "residual_std": round(std_res, 2),
        "total_evaluated": total_obs,
        "anomalies_flagged": anomaly_count,
        "anomaly_rate_pct": round(anomaly_count / total_obs * 100.0, 2),
        "under_predictions": int((z_scores > threshold_z).sum()),
        "over_predictions": int((z_scores < -threshold_z).sum()),
    }

    return df, summary
