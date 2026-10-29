"""
RetailIQ AI V2 ??? Test Configuration

Shared fixtures and helpers for the test suite.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def raw_data_dir(project_root: Path) -> Path:
    """Return the raw data directory."""
    return project_root / "data" / "raw"


@pytest.fixture
def processed_data_dir(project_root: Path) -> Path:
    """Return the processed data directory."""
    return project_root / "data" / "processed"


@pytest.fixture
def reports_dir(project_root: Path) -> Path:
    """Return the reports directory."""
    return project_root / "reports"

