"""
RetailIQ AI V2 ??? Forecasting Model Training & Baselines Module

Implements:
1. Baseline 1: Previous-Week Naive (t-1)
2. Baseline 2: Seasonal 52-Week Naive (t-52)
3. Model 1: L2 Ridge Linear Regressor
4. Model 2: Random Forest Regressor
5. Model 3: XGBoost Regressor
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


def predict_naive_previous_week(df: pd.DataFrame) -> np.ndarray:
    """Predict sales using the previous week's observed sales (lag_1).

    Baseline formula: y_hat_t = y_{t-1}
    """
    return df["lag_1"].to_numpy(dtype=np.float64)


def predict_seasonal_naive_52(df: pd.DataFrame) -> np.ndarray:
    """Predict sales using the same week from 52 weeks ago (lag_52).

    Baseline formula: y_hat_t = y_{t-52}
    """
    return df["lag_52"].to_numpy(dtype=np.float64)


def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    alpha: float = 100.0,
) -> Pipeline:
    """Train Ridge regression with standard scaling pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.
        alpha: L2 penalty strength.

    Returns:
        Fitted Scikit-Learn Pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train Random Forest regression model.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of decision trees.
        max_depth: Maximum tree depth to control overfitting.
        random_state: Seed for reproducibility.

    Returns:
        Fitted RandomForestRegressor.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    random_state: int = 42,
) -> xgb.XGBRegressor:
    """Train XGBoost gradient boosted regression model.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of boosting rounds.
        max_depth: Tree depth.
        learning_rate: Shrinkage rate (eta).
        subsample: Row sampling ratio.
        colsample_bytree: Feature sampling ratio.
        random_state: Seed for reproducibility.

    Returns:
        Fitted XGBRegressor.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def extract_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str = "Model",
) -> pd.DataFrame:
    """Extract and rank feature importance metrics.

    Args:
        model: Fitted model (XGBoost, Random Forest, or Ridge Pipeline).
        feature_names: Ordered list of feature names.
        model_name: Model label.

    Returns:
        Sorted DataFrame with columns ['Feature', 'Importance', 'Relative_Pct', 'Model'].
    """
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
    elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
        raw_imp = np.abs(model.named_steps["ridge"].coef_)
    else:
        raw_imp = np.ones(len(feature_names))

    total = np.sum(raw_imp)
    rel_pct = (raw_imp / total * 100.0) if total > 0 else np.zeros_like(raw_imp)

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.round(raw_imp, 6),
        "Relative_Pct": np.round(rel_pct, 2),
        "Model": model_name,
    })
    return df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
