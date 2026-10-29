"""
RetailIQ AI V2 ??? Phase 04: Sales Forecasting & Model Evaluation Tests

Validates:
- Leakage-safe feature engineering (lags, rolling statistics, calendar cyclics)
- Chronological temporal partitions (zero future data leakage)
- Autoregressive metric calculation formulas (MAE, RMSE, WAPE, R??)
- Baseline and ML model predictions (Ridge, Random Forest, XGBoost)
- Model artifact serialization and inference integrity (.joblib)
- Benchmark leaderboard report structure
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.retailiq.config import MODELS_DIR, REPORT_SUBDIRS
from src.retailiq.forecasting.evaluation import calculate_metrics
from src.retailiq.forecasting.features import build_forecasting_features, get_feature_columns
from src.retailiq.forecasting.splitter import chronological_split


@pytest.fixture(scope="module")
def store_week_data() -> pd.DataFrame:
    """Load store-week sales data once for module."""
    fpath = REPORT_SUBDIRS["business_analytics"] / "04_store_week_sales.csv"
    return pd.read_csv(fpath, parse_dates=["Date"])


class TestFeatureEngineering:
    """Verify lag and rolling feature construction without leakage."""

    def test_feature_columns_created(self, store_week_data: pd.DataFrame):
        feat_df = build_forecasting_features(store_week_data)
        feature_cols = get_feature_columns()

        for col in feature_cols:
            assert col in feat_df.columns, f"Missing feature {col}"
            assert feat_df[col].isna().sum() == 0, f"Feature {col} contains unexpected NaNs"

    def test_lag_1_correctness(self, store_week_data: pd.DataFrame):
        feat_df = build_forecasting_features(store_week_data)
        # For Store 1, verify lag_1 matches previous week's actual sales
        s1 = feat_df[feat_df["Store"] == 1].sort_values("Date").reset_index(drop=True)
        for i in range(1, min(10, len(s1))):
            assert s1.loc[i, "lag_1"] == s1.loc[i - 1, "Weekly_Sales"]


class TestChronologicalSplit:
    """Verify strictly forward-in-time partitioning."""

    def test_no_temporal_leakage(self, store_week_data: pd.DataFrame):
        feat_df = build_forecasting_features(store_week_data)
        train_df, val_df, test_df, meta = chronological_split(feat_df)

        # Invariant: train dates < val dates < test dates
        assert train_df["Date"].max() < val_df["Date"].min()
        assert val_df["Date"].max() < test_df["Date"].min()

        # All 45 stores must be represented in test window
        assert test_df["Store"].nunique() == 45
        assert len(test_df) == 990  # 45 stores * 22 test weeks


class TestMetricsCalculation:
    """Verify mathematical calculation of MAE, RMSE, WAPE, and R??."""

    def test_metrics_exact_formula(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])

        res = calculate_metrics(y_true, y_pred, "TestModel", "Test")

        assert res["MAE"] == round((10.0 + 10.0 + 0.0) / 3, 2)
        assert res["WAPE_Pct"] == round((20.0 / 600.0) * 100.0, 2)
        assert res["RMSE"] > res["MAE"]
        assert res["R2"] > 0.95


class TestModelArtifactsAndLeaderboard:
    """Verify saved models, test inference, and metrics report."""

    def test_saved_models_can_load_and_predict(self, store_week_data: pd.DataFrame):
        feat_df = build_forecasting_features(store_week_data)
        _, _, test_df, _ = chronological_split(feat_df)
        X_test = test_df[get_feature_columns()]

        for mname in ["ridge_model.joblib", "random_forest_model.joblib", "xgboost_model.joblib"]:
            mpath = MODELS_DIR / mname
            assert mpath.exists(), f"Model artifact {mname} missing!"
            model = joblib.load(mpath)
            preds = model.predict(X_test)
            assert len(preds) == len(X_test)
            assert not np.isnan(preds).any()

    def test_leaderboard_accuracy_bounds(self):
        metrics_path = REPORT_SUBDIRS["sales_forecasting"] / "01_model_comparison_metrics.csv"
        assert metrics_path.exists()
        df = pd.read_csv(metrics_path)

        test_df = df[df["Split"] == "Test"]
        assert len(test_df) == 5, f"Expected 5 models on test split, got {len(test_df)}"

        # XGBoost champion WAPE must be < 4.5%
        xgb_wape = test_df.loc[test_df["Model"] == "XGBoost", "WAPE_Pct"].values[0]
        assert xgb_wape < 4.5, f"Expected XGBoost WAPE < 4.5%, got {xgb_wape}%"

        # Baseline check
        naive_wape = test_df.loc[test_df["Model"] == "Previous-Week Naive", "WAPE_Pct"].values[0]
        assert xgb_wape < naive_wape, "XGBoost should outperform Naive baseline!"

    def test_forecasting_horizon_and_tradeoff_metadata(self):
        """Regression test for ISSUE-11-01: Verify horizon definition and model trade-off documentation."""
        meta_path = REPORT_SUBDIRS["sales_forecasting"] / "04_experiment_metadata.json"
        assert meta_path.exists()
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["horizon_weeks"] == 1
        assert "Rolling 1-week-ahead" in meta.get("forecasting_paradigm", "")
        notes = meta.get("methodology_notes", {})
        assert "Rolling 1-step-ahead" in notes.get("evaluation_type", "")
        assert "XGBoost" in notes.get("model_tradeoff", "")
        assert "Ridge" in notes.get("model_tradeoff", "")
