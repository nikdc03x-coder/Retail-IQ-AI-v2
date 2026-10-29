"""
RetailIQ AI V2 ??? SHAP Explainable AI Module

Provides model-agnostic and TreeSHAP explainability for the champion XGBoost model:
- Global feature importance (mean absolute SHAP impact in dollars).
- Local prediction attribution (waterfall additive contributions: f(x) = phi_0 + sum phi_i).
- Rigorous mathematical reconstruction checks within single-precision tolerance.
- Plain-English feature definitions distinguishing model influence from business causation.
"""

from typing import Any
import joblib
import numpy as np
import pandas as pd
import shap

from src.retailiq.config import MODELS_DIR, REPORT_SUBDIRS
from src.retailiq.forecasting.features import build_forecasting_features, get_feature_columns
from src.retailiq.forecasting.splitter import chronological_split


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "lag_1": "Previous week's sales (t-1). Provides baseline operational revenue momentum.",
    "rolling_mean_4": "4-week backward rolling sales average. Captures short-term monthly trend.",
    "Size": "Store physical square footage. Reflects structural sales capacity.",
    "lag_52": "Sales from 52 weeks ago (t-52). Anchors annual year-over-year seasonality.",
    "rolling_mean_8": "8-week backward rolling sales average. Smooths medium-term trajectory.",
    "lag_4": "Sales from 4 weeks prior (t-4). Reflects prior month demand level.",
    "lag_2": "Sales from 2 weeks prior (t-2). Intermediate autoregressive signal.",
    "rolling_std_4": "4-week rolling standard deviation. Measures recent sales volatility.",
    "WeekOfYear": "Calendar week number (1 to 52). Tracks seasonal retail calendar.",
    "Month": "Calendar month (1 to 12). Captures broad seasonal quarters.",
    "is_holiday_num": "Binary indicator for Walmart promotional holiday week.",
    "Type_A": "Indicator for Type A store (large supercenter format).",
    "Type_B": "Indicator for Type B store (medium regional format).",
    "Type_C": "Indicator for Type C store (small compact format).",
    "Temperature": "Average weekly temperature in store region (degrees Fahrenheit).",
    "Fuel_Price": "Regional fuel cost per gallon. Macroeconomic consumer cost indicator.",
    "CPI": "Consumer Price Index. Regional inflation and consumer purchasing power index.",
    "Unemployment": "Regional unemployment rate percentage.",
    "sin_week": "Cyclical sine transformation of week of year.",
    "cos_week": "Cyclical cosine transformation of week of year.",
}


def compute_shap_explanations(
    model: Any | None = None,
    test_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, np.ndarray]:
    """Compute TreeSHAP values, global importance, and local contributions.

    Args:
        model: Fitted XGBoost model. Loads models/xgboost_model.joblib if None.
        test_df: Held-out test DataFrame. Loads and splits analytical data if None.
        feature_cols: List of feature names.

    Returns:
        Tuple of (global_importance_df, local_contributions_df, base_value, raw_shap_matrix).
    """
    if model is None:
        model_path = MODELS_DIR / "xgboost_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run Phase 04 first.")
        model = joblib.load(model_path)

    if feature_cols is None:
        feature_cols = get_feature_columns()

    if test_df is None:
        store_week_path = REPORT_SUBDIRS["business_analytics"] / "04_store_week_sales.csv"
        if not store_week_path.exists():
            raise FileNotFoundError(f"Analytics report not found at {store_week_path}. Run Phase 03 first.")
        store_week = pd.read_csv(store_week_path, parse_dates=["Date"])
        feat_df = build_forecasting_features(store_week)
        _, _, test_df, _ = chronological_split(feat_df)

    X_test = test_df[feature_cols].copy()
    y_test = test_df["Weekly_Sales"].values
    preds = model.predict(X_test)

    # ?????? Compute TreeSHAP Explanations ??????
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_test)

    # Base value phi_0 (expected value across training distribution)
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_value = float(base_val[0])
    else:
        base_value = float(base_val)

    # ?????? 1. Global Feature Importance (Mean Absolute SHAP) ??????
    mean_abs_shap = np.mean(np.abs(raw_shap_values), axis=0)
    total_shap_mass = np.sum(mean_abs_shap)
    rel_pct = (mean_abs_shap / total_shap_mass * 100.0) if total_shap_mass > 0 else np.zeros_like(mean_abs_shap)

    global_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 2),
        "Relative_Impact_Pct": np.round(rel_pct, 2),
        "Feature_Description": [FEATURE_DESCRIPTIONS.get(f, "Engineered model feature") for f in feature_cols],
    })
    global_imp = global_imp.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    global_imp["Importance_Rank"] = range(1, len(global_imp) + 1)
    ordered_global = ["Importance_Rank", "Feature", "Mean_Abs_SHAP", "Relative_Impact_Pct", "Feature_Description"]
    global_imp = global_imp[ordered_global]

    # ?????? 2. Local Contributions DataFrame ??????
    local_df = test_df[["Store", "Date", "Type", "Size", "IsHoliday"]].copy().reset_index(drop=True)
    local_df["Date"] = pd.to_datetime(local_df["Date"]).dt.strftime("%Y-%m-%d")
    local_df["Actual_Sales"] = np.round(y_test, 2)
    local_df["Model_Prediction"] = np.round(preds, 2)
    local_df["Base_Value"] = round(base_value, 2)

    # Sum of SHAP contributions per observation
    shap_sums = np.sum(raw_shap_values, axis=1)
    local_df["Sum_SHAP_Contributions"] = np.round(shap_sums, 2)
    local_df["Reconstructed_Prediction"] = np.round(base_value + shap_sums, 2)
    local_df["Reconstruction_Diff"] = np.round(
        local_df["Model_Prediction"] - local_df["Reconstructed_Prediction"], 4
    )

    # Append individual feature SHAP columns: shap_<feature> and feature values: val_<feature>
    for idx, fcol in enumerate(feature_cols):
        local_df[f"shap_{fcol}"] = np.round(raw_shap_values[:, idx], 2)
        local_df[f"val_{fcol}"] = np.round(X_test[fcol].values, 4)

    return global_imp, local_df, base_value, raw_shap_values
def verify_shap_reconstruction(
    local_df: pd.DataFrame,
    max_tolerance_dollars: float = 5.0,
    max_tolerance_relative_pct: float = 0.01,
) -> dict[str, Any]:
    """Verify that reconstructed predictions equal model predictions within precision limits.

    Ensures the fundamental SHAP efficiency axiom holds:
    f(x) = phi_0 + sum_{i=1}^M phi_i

    Args:
        local_df: Output from compute_shap_explanations.
        max_tolerance_dollars: Allowed absolute error in dollars.
        max_tolerance_relative_pct: Allowed relative error percentage.

    Returns:
        Dictionary of validation results and summary metrics.
    """
    diffs = (local_df["Model_Prediction"] - local_df["Reconstructed_Prediction"]).abs()
    rel_diffs_pct = (diffs / local_df["Model_Prediction"].abs().replace(0, np.nan)) * 100.0

    max_abs_diff = float(diffs.max())
    mean_abs_diff = float(diffs.mean())
    max_rel_pct = float(rel_diffs_pct.max())
    mean_rel_pct = float(rel_diffs_pct.mean())

    all_passed = bool((max_abs_diff <= max_tolerance_dollars) and (max_rel_pct <= max_tolerance_relative_pct))

    return {
        "verified": all_passed,
        "sample_count": len(local_df),
        "max_abs_diff_dollars": round(max_abs_diff, 4),
        "mean_abs_diff_dollars": round(mean_abs_diff, 4),
        "max_rel_diff_pct": round(max_rel_pct, 6),
        "mean_rel_diff_pct": round(mean_rel_pct, 6),
        "max_tolerance_dollars": max_tolerance_dollars,
        "max_tolerance_relative_pct": max_tolerance_relative_pct,
        "efficiency_axiom_holds": all_passed,
    }


def extract_observation_waterfall(
    local_df: pd.DataFrame,
    store: int,
    date_str: str,
    feature_cols: list[str] | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Extract waterfall components for a specific store-week prediction.

    Aggregates lower-magnitude drivers into an 'Other features' category
    so the waterfall visualization is clean and readable for business users.

    Args:
        local_df: Explanations DataFrame.
        store: Store number.
        date_str: Date string ('YYYY-MM-DD').
        feature_cols: Feature columns list.
        top_n: Number of primary drivers to show individually.

    Returns:
        Dictionary with base_value, prediction, actual, and driver records.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    match = local_df[(local_df["Store"] == store) & (local_df["Date"] == date_str)]
    if match.empty:
        raise ValueError(f"No record found for Store {store} on {date_str}.")

    row = match.iloc[0]
    base_val = float(row["Base_Value"])
    pred = float(row["Model_Prediction"])
    actual = float(row["Actual_Sales"])

    drivers = []
    for f in feature_cols:
        shap_val = float(row[f"shap_{f}"])
        feat_val = float(row[f"val_{f}"])
        drivers.append({
            "feature": f,
            "shap_value": shap_val,
            "feature_value": feat_val,
            "abs_shap": abs(shap_val),
            "description": FEATURE_DESCRIPTIONS.get(f, ""),
        })

    # Sort by absolute impact descending
    drivers.sort(key=lambda d: d["abs_shap"], reverse=True)

    top_drivers = drivers[:top_n]
    remaining = drivers[top_n:]
    other_shap = sum(d["shap_value"] for d in remaining)

    waterfall_items = []
    for d in top_drivers:
        waterfall_items.append({
            "label": f"{d['feature']} ({d['feature_value']:,.1f})",
            "feature": d["feature"],
            "value": d["shap_value"],
            "description": d["description"],
        })

    if remaining and abs(other_shap) > 0.01:
        waterfall_items.append({
            "label": f"Other {len(remaining)} features",
            "feature": "Other",
            "value": round(other_shap, 2),
            "description": "Aggregated residual impact of remaining minor features",
        })

    return {
        "store": store,
        "date": date_str,
        "base_value": base_val,
        "prediction": pred,
        "actual": actual,
        "drivers": waterfall_items,
        "top_feature": top_drivers[0]["feature"],
        "top_impact": top_drivers[0]["shap_value"],
    }

