"""
RetailIQ AI V2 — Phase 00 Foundation Smoke Tests

These tests verify the project structure, configuration, imports,
and essential infrastructure are working correctly.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectStructure:
    """Verify the project directory structure is correct."""

    def test_project_root_exists(self, project_root: Path):
        assert project_root.exists()
        assert project_root.is_dir()

    def test_source_package_exists(self, project_root: Path):
        pkg = project_root / "src" / "retailiq"
        assert pkg.exists()
        assert (pkg / "__init__.py").exists()

    def test_data_directories_exist(self, project_root: Path):
        assert (project_root / "data" / "raw").exists()
        assert (project_root / "data" / "processed").exists()

    def test_report_directories_exist(self, project_root: Path):
        subdirs = [
            "data_quality", "business_analytics", "sales_forecasting",
            "forecast_reliability", "anomaly_detection", "explainable_ai",
        ]
        for name in subdirs:
            assert (project_root / "reports" / name).exists(), f"Missing reports/{name}"

    def test_models_directory_exists(self, project_root: Path):
        assert (project_root / "models").exists()

    def test_tests_directory_exists(self, project_root: Path):
        assert (project_root / "tests").exists()

    def test_docs_directory_exists(self, project_root: Path):
        assert (project_root / "docs" / "phases").exists()

    def test_scripts_directories_exist(self, project_root: Path):
        scripts = project_root / "scripts"
        expected = [
            "01_data_ingestion", "02_data_quality", "03_business_analytics",
            "04_sales_forecasting", "05_forecast_reliability",
            "06_anomaly_detection", "07_explainability",
        ]
        for name in expected:
            assert (scripts / name).exists(), f"Missing scripts/{name}"

    def test_streamlit_config_exists(self, project_root: Path):
        config = project_root / ".streamlit" / "config.toml"
        assert config.exists()

    def test_env_example_exists(self, project_root: Path):
        assert (project_root / ".env.example").exists()

    def test_gitignore_exists(self, project_root: Path):
        assert (project_root / ".gitignore").exists()

    def test_requirements_exists(self, project_root: Path):
        assert (project_root / "requirements.txt").exists()

    def test_app_entry_exists(self, project_root: Path):
        assert (project_root / "app.py").exists()


class TestConfigImports:
    """Verify that the config module loads correctly."""

    def test_config_imports(self):
        from src.retailiq.config import (
            PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR,
            REPORTS_DIR, MODELS_DIR, Theme,
        )
        assert PROJECT_ROOT.exists()
        assert isinstance(Theme.VIOLET, str)

    def test_config_paths_are_absolute(self):
        from src.retailiq.config import PROJECT_ROOT, RAW_DATA_DIR
        assert PROJECT_ROOT.is_absolute()
        assert RAW_DATA_DIR.is_absolute()

    def test_theme_colors_are_valid_hex(self):
        from src.retailiq.config import Theme
        colors = [
            Theme.NAVY, Theme.INDIGO, Theme.VIOLET,
            Theme.TEAL, Theme.AMBER, Theme.CORAL,
        ]
        for color in colors:
            assert color.startswith("#"), f"Invalid hex: {color}"
            assert len(color) == 7, f"Expected 7-char hex: {color}"

    def test_expected_schemas_defined(self):
        from src.retailiq.config import EXPECTED_SCHEMAS
        assert "train" in EXPECTED_SCHEMAS
        assert "features" in EXPECTED_SCHEMAS
        assert "stores" in EXPECTED_SCHEMAS

    def test_ensure_directories_callable(self):
        from src.retailiq.config import ensure_directories
        # Should run without error (creates dirs if missing)
        ensure_directories()

    def test_gemini_model_has_default(self):
        from src.retailiq.config import GEMINI_MODEL
        assert isinstance(GEMINI_MODEL, str)
        assert len(GEMINI_MODEL) > 0


class TestSubpackageImports:
    """Verify all subpackages can be imported."""

    def test_import_data(self):
        import src.retailiq.data

    def test_import_quality(self):
        import src.retailiq.quality

    def test_import_analytics(self):
        import src.retailiq.analytics

    def test_import_forecasting(self):
        import src.retailiq.forecasting

    def test_import_reliability(self):
        import src.retailiq.reliability

    def test_import_anomalies(self):
        import src.retailiq.anomalies

    def test_import_explainability(self):
        import src.retailiq.explainability

    def test_import_assistant(self):
        import src.retailiq.assistant

    def test_import_dashboard(self):
        import src.retailiq.dashboard


class TestDashboardHelpers:
    """Verify dashboard helper functions work correctly."""

    def test_format_currency(self):
        from src.retailiq.dashboard.components import format_currency
        assert format_currency(1_500_000_000) == "$1.5B"
        assert format_currency(2_300_000) == "$2.3M"
        assert format_currency(45_000) == "$45.0K"
        assert format_currency(750) == "$750"

    def test_format_number(self):
        from src.retailiq.dashboard.components import format_number
        assert format_number(1_200_000) == "1.2M"
        assert format_number(45_000) == "45.0K"

    def test_format_percent(self):
        from src.retailiq.dashboard.components import format_percent
        assert format_percent(0.1523) == "15.2%"
        assert format_percent(0.05, precision=2) == "5.00%"

    def test_kpi_card_returns_html(self):
        from src.retailiq.dashboard.components import kpi_card
        html = kpi_card("Test Label", "$100K", "subtitle text")
        assert "Test Label" in html
        assert "$100K" in html
        assert "subtitle text" in html
        assert "kpi-card" in html


class TestGitignoreContent:
    """Verify .gitignore blocks sensitive files."""

    def test_env_blocked(self, project_root: Path):
        gitignore = (project_root / ".gitignore").read_text()
        assert ".env" in gitignore

    def test_venv_blocked(self, project_root: Path):
        gitignore = (project_root / ".gitignore").read_text()
        assert ".venv" in gitignore

    def test_pycache_blocked(self, project_root: Path):
        gitignore = (project_root / ".gitignore").read_text()
        assert "__pycache__" in gitignore

