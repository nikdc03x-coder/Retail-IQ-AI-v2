"""
RetailIQ AI V2 — Isolation Forest Anomaly Detection Module

Detects multivariate anomalies using unsupervised tree isolation:
Partitions feature space to isolate abnormal observations with short path lengths.
Evaluates joint space of actual volume, predicted volume, absolute residuals, and percentage errors.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_isolation_forest_anomalies(
    test_preds_df: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect multivariate anomalies using Scikit-Learn IsolationForest.

    Args:
        test_preds_df: Test predictions DataFrame.
        contamination: Expected anomaly proportion (default 0.02 = ~2%).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (scored_df, metrics_summary).
    """
    df = test_preds_df.copy()

    # Calculate residual features if not present
    if "Residual" not in df.columns:
        df["Residual"] = df["Actual_Sales"] - df["XGBoost_Pred"]
    if "Abs_Error" not in df.columns:
        df["Abs_Error"] = df["Residual"].abs()
    if "Pct_Error" not in df.columns:
        df["Pct_Error"] = (df["Abs_Error"] / df["Actual_Sales"] * 100.0).round(2)

    # Multivariate feature matrix
    feature_matrix = df[[
        "Actual_Sales",
        "XGBoost_Pred",
        "Residual",
        "Abs_Error",
        "Pct_Error",
    ]].copy()

    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(feature_matrix)

    iforest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    raw_preds = iforest.fit_predict(scaled_feats)
    # Decision function: lower values mean more anomalous
    decision_scores = iforest.decision_function(scaled_feats)

    # Convert decision score to Anomaly Score in [0, 1] where 1 is most anomalous
    min_score = float(decision_scores.min())
    max_score = float(decision_scores.max())
    norm_anomaly_score = 1.0 - ((decision_scores - min_score) / (max_score - min_score))

    is_iforest_anomaly = (raw_preds == -1)

    df["IForest_Raw_Score"] = decision_scores.round(4)
    df["IForest_Anomaly_Score"] = norm_anomaly_score.round(4)
    df["Is_IForest_Anomaly"] = is_iforest_anomaly

    total_obs = len(df)
    flagged_count = int(is_iforest_anomaly.sum())

    summary = {
        "method": "Multivariate Isolation Forest",
        "contamination": contamination,
        "n_estimators": 100,
        "total_evaluated": total_obs,
        "anomalies_flagged": flagged_count,
        "anomaly_rate_pct": round(flagged_count / total_obs * 100.0, 2),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
    }

    return df, summary
