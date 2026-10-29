"""
RetailIQ AI V2 ??? Phase 07: Explainable AI (SHAP) Tests

Validates:
- TreeSHAP global feature importance ranking, bounds, and relative impact mass.
- Local prediction attribution data structure and feature column alignment.
- Mathematical efficiency axiom reconstruction: f(x) = phi_0 + sum phi_i within single-precision limits.
- Observation waterfall extraction and driver aggregation.
- Report persistence and schema invariants in reports/explainable_ai/.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.config import REPORT_SUBDIRS
from src.retailiq.explainability.shap_analysis import (
    extract_observation_waterfall,
    verify_shap_reconstruction,
)


@pytest.fixture(scope="module")
def explainability_dir() -> Path:
    """Return reports directory for explainability."""
    return REPORT_SUBDIRS["explainable_ai"]


class TestGlobalFeatureImportance:
    """Verify global feature importance metrics and rankings."""

    def test_global_importance_structure(self, explainability_dir: Path):
        fpath = explainability_dir / "01_global_feature_importance.csv"
        assert fpath.exists(), "Global feature importance report must exist"

        df = pd.read_csv(fpath)
        expected_cols = ["Importance_Rank", "Feature", "Mean_Abs_SHAP", "Relative_Impact_Pct", "Feature_Description"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in global importance report"

        assert len(df) >= 15, "Should evaluate at least 15 engineered features"
        assert (df["Mean_Abs_SHAP"] >= 0).all(), "Mean absolute SHAP values must be non-negative"
        assert df["Importance_Rank"].tolist() == list(range(1, len(df) + 1)), "Ranks must be strictly 1..N"

        # Check relative impact percentage sums to approximately 100%
        total_rel_pct = df["Relative_Impact_Pct"].sum()
        assert 99.5 <= total_rel_pct <= 100.5, f"Relative impact sum was {total_rel_pct}, expected ~100%"

        # Primary seasonal driver (lag_52) should be #1
        assert df.iloc[0]["Feature"] == "lag_52", f"Expected lag_52 to be #1, got {df.iloc[0]['Feature']}"

    def test_top_feature_driver_attribution_shares(self, explainability_dir: Path):
        """Regression test for ISSUE-11-03: Verify exact shares for lag_52, rolling_mean_4, and lag_1."""
        df = pd.read_csv(explainability_dir / "01_global_feature_importance.csv")
        df["feat_clean"] = df["Feature"].str.strip()

        lag_52_share = float(df.loc[df["feat_clean"] == "lag_52", "Relative_Impact_Pct"].values[0])
        rm4_share = float(df.loc[df["feat_clean"] == "rolling_mean_4", "Relative_Impact_Pct"].values[0])
        lag_1_share = float(df.loc[df["feat_clean"] == "lag_1", "Relative_Impact_Pct"].values[0])

        assert 44.5 <= lag_52_share <= 45.5, f"lag_52 share expected ~45.01%, got {lag_52_share}%"
        assert 27.0 <= rm4_share <= 28.5, f"rolling_mean_4 share expected ~27.65%, got {rm4_share}%"
        assert 72.0 <= (lag_52_share + rm4_share) <= 73.5, f"Combined lag_52 + rm4 expected ~72.7%, got {lag_52_share + rm4_share}%"
        assert 1.5 <= lag_1_share <= 3.0, f"lag_1 share expected ~2.21%, got {lag_1_share}%"


class TestLocalPredictionAttribution:
    """Verify local prediction attributions for held-out test records."""

    def test_local_attributions_shape_and_columns(self, explainability_dir: Path):
        fpath = explainability_dir / "02_local_prediction_attributions.csv"
        assert fpath.exists(), "Local prediction attributions report must exist"

        df = pd.read_csv(fpath)
        assert len(df) == 990, f"Expected 990 test records, got {len(df)}"

        core_cols = [
            "Store", "Date", "Actual_Sales", "Model_Prediction",
            "Base_Value", "Sum_SHAP_Contributions", "Reconstructed_Prediction", "Reconstruction_Diff"
        ]
        for c in core_cols:
            assert c in df.columns, f"Missing core column {c}"

        assert not df["Model_Prediction"].isna().any(), "Model predictions must contain no NaNs"
        assert not df["Reconstructed_Prediction"].isna().any(), "Reconstructed predictions must contain no NaNs"


class TestEfficiencyAxiomReconstruction:
    """Verify the fundamental SHAP efficiency axiom holds."""

    def test_additive_reconstruction_tolerance(self, explainability_dir: Path):
        fpath = explainability_dir / "02_local_prediction_attributions.csv"
        df = pd.read_csv(fpath)

        verification = verify_shap_reconstruction(df, max_tolerance_dollars=5.0, max_tolerance_relative_pct=0.01)

        assert verification["verified"] is True, "Efficiency axiom verification failed"
        assert verification["efficiency_axiom_holds"] is True
        assert verification["max_abs_diff_dollars"] <= 5.0, f"Max diff ${verification['max_abs_diff_dollars']} exceeded $5.00"
        assert verification["max_rel_diff_pct"] < 0.001, f"Max rel error {verification['max_rel_diff_pct']}% exceeded 0.001%"


class TestObservationWaterfallExtraction:
    """Verify waterfall decomposition for an individual observation."""

    def test_waterfall_decomposition_keys_and_math(self, explainability_dir: Path):
        fpath = explainability_dir / "02_local_prediction_attributions.csv"
        df = pd.read_csv(fpath)

        sample_row = df.iloc[0]
        store = int(sample_row["Store"])
        date_str = str(sample_row["Date"])

        waterfall = extract_observation_waterfall(df, store=store, date_str=date_str, top_n=6)

        assert "base_value" in waterfall
        assert "prediction" in waterfall
        assert "actual" in waterfall
        assert "drivers" in waterfall
        assert len(waterfall["drivers"]) > 0

        # Sum of drivers plus base value should approximately equal prediction
        driver_sum = sum(d["value"] for d in waterfall["drivers"])
        reconstructed = waterfall["base_value"] + driver_sum
        diff = abs(reconstructed - waterfall["prediction"])
        assert diff <= 5.0, f"Waterfall reconstruction diff ${diff:.2f} exceeded $5.00"


class TestExplainabilityReportsExist:
    """Verify all expected reports exist and are non-empty."""

    @pytest.mark.parametrize("filename", [
        "01_global_feature_importance.csv",
        "02_local_prediction_attributions.csv",
        "03_reconstruction_verification.json",
        "04_top_drivers_summary.csv",
        "05_explainability_summary.json",
    ])
    def test_report_exists_and_non_empty(self, explainability_dir: Path, filename: str):
        target = explainability_dir / filename
        assert target.exists(), f"Missing report: {filename}"
        assert target.stat().st_size > 0, f"Empty report: {filename}"

    def test_summary_json_content(self, explainability_dir: Path):
        summary_path = explainability_dir / "05_explainability_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["model"] == "XGBoost Regressor (Champion)"
        assert summary["explainer"] == "TreeExplainer (TreeSHAP)"
        assert summary["total_test_observations"] == 990
        assert summary["top_global_feature"] == "lag_52"
        assert summary["reconstruction_check"]["verified"] is True

