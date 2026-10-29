"""
RetailIQ AI V2 ??? Forecasting Evaluation Metrics Module

Provides industry-standard metrics for retail sales forecasting:
- MAE (Mean Absolute Error): Dollar-denominated scale of expected error.
- RMSE (Root Mean Squared Error): Penalizes larger forecast misses heavily.
- WAPE (Weighted Absolute Percentage Error): Total absolute error relative to total sales.
- R?? Score: Proportion of sales variance explained by the model.
"""

from typing import Any
import numpy as np


def calculate_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    model_name: str = "Model",
    split_name: str = "Test",
) -> dict[str, Any]:
    """Calculate comprehensive time-series evaluation metrics.

    Args:
        y_true: Ground truth actual sales array.
        y_pred: Model predicted sales array.
        model_name: Identifier for model (e.g. 'XGBoost', 'Naive-1').
        split_name: Dataset split label (e.g. 'Validation', 'Test').

    Returns:
        Dictionary with numeric and formatted metrics.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    residuals = y_t - y_p
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    # WAPE (Weighted Absolute Percentage Error)
    total_actual = float(np.sum(np.abs(y_t)))
    wape = float((np.sum(abs_errors) / total_actual) * 100.0) if total_actual > 0 else 0.0

    # R?? Score
    ss_res = float(np.sum(sq_errors))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Mean Percentage Error (signed bias)
    mpe = float(np.mean(residuals / np.where(y_t != 0, y_t, 1.0))) * 100.0

    return {
        "Model": model_name,
        "Split": split_name,
        "Observations": len(y_t),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE_Pct": round(wape, 2),
        "R2": round(r2, 4),
        "Bias_MPE_Pct": round(mpe, 2),
    }
