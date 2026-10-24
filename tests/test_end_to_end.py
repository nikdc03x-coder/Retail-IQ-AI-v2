"""
RetailIQ AI V2 — Phase 10: End-to-End System Integration Tests

Validates:
- Complete analytical dataset Parquet persistence, dimensions, and types.
- Saved machine learning model artifacts in models/ and prediction capability.
- Full presence and non-emptiness of analytical reports across all 6 subdirectories.
- End-to-end integration of assistant engine with typed tools.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import pytest

from src.retailiq.config import (
    ANALYTICAL_DATASET,
    MODELS_DIR,
    REPORT_SUBDIRS,
)
from src.retailiq.assistant.engine import RetailIQAssistantEngine
from src.retailiq.forecasting.features import get_feature_columns


class TestAnalyticalDatasetIntegration:
    """Verify analytical Parquet dataset integrity."""

    def test_parquet_dataset_exists_and_dimensions(self):
        assert ANALYTICAL_DATASET.exists(), "Analytical Parquet dataset must exist"
        df = pd.read_parquet(ANALYTICAL_DATASET)
        assert len(df) == 421570, f"Expected 421,570 rows, found {len(df)}"
        assert df["Store"].nunique() == 45
        assert df["Dept"].nunique() == 81
        assert df["Date"].nunique() == 143


class TestTrainedModelsIntegration:
    """Verify trained forecasting model artifacts can be loaded and predict."""

    @pytest.mark.parametrize("model_file", [
        "ridge_model.joblib",
        "random_forest_model.joblib",
        "xgboost_model.joblib",
    ])
    def test_model_artifact_load_and_predict(self, model_file: str):
        path = MODELS_DIR / model_file
        assert path.exists(), f"Model artifact {model_file} must exist"
        model = joblib.load(path)

        feature_cols = get_feature_columns()
        dummy_input = pd.DataFrame([{f: 1.0 for f in feature_cols}])
        pred = model.predict(dummy_input)
        assert len(pred) == 1
        assert not pd.isna(pred[0])


class TestReportsCatalogCompleteness:
    """Verify all report subdirectories contain expected non-empty files."""

    def test_all_report_directories_populated(self):
        for name, directory in REPORT_SUBDIRS.items():
            assert directory.exists(), f"Report directory {name} must exist"
            files = [f for f in directory.glob("*.*") if f.name != ".gitkeep"]
            assert len(files) >= 3, f"Directory {name} should have at least 3 reports, found {len(files)}"
            for f in files:
                assert f.stat().st_size > 0, f"Report {f} is empty"



class TestAssistantEngineEndToEnd:
    """Verify assistant engine processes end-to-end queries."""

    def test_assistant_engine_end_to_end(self):
        engine = RetailIQAssistantEngine(api_key="")
        res = engine.ask("What is the champion model and its WAPE score?")
        assert res["verified"] is True
        assert "XGBoost" in res["answer"]
        assert "3.83%" in res["answer"]
        assert res["evidence_source"].startswith("reports/")
