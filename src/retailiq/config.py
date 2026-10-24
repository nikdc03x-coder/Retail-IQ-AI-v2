"""
RetailIQ AI V2 — Central Configuration

All project paths, constants, and environment variables are managed here.
This ensures every module uses the same directory structure and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if present ──
load_dotenv()

# ── Project root (two levels up from this file: src/retailiq/config.py → project root) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Directory paths ──
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Report subdirectories (one per analysis phase)
REPORT_SUBDIRS = {
    "data_quality": REPORTS_DIR / "data_quality",
    "business_analytics": REPORTS_DIR / "business_analytics",
    "sales_forecasting": REPORTS_DIR / "sales_forecasting",
    "forecast_reliability": REPORTS_DIR / "forecast_reliability",
    "anomaly_detection": REPORTS_DIR / "anomaly_detection",
    "explainable_ai": REPORTS_DIR / "explainable_ai",
    "strategy": REPORTS_DIR / "strategy",
}

# ── Raw data filenames ──
RAW_FILES = {
    "train": RAW_DATA_DIR / "train.csv",
    "features": RAW_DATA_DIR / "features.csv",
    "stores": RAW_DATA_DIR / "stores.csv",
}

# ── Processed data filenames ──
ANALYTICAL_DATASET = PROCESSED_DATA_DIR / "analytical_dataset.parquet"

# ── Expected schemas for validation ──
EXPECTED_SCHEMAS = {
    "train": {
        "columns": ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"],
        "dtypes": {
            "Store": "int",
            "Dept": "int",
            "Weekly_Sales": "float",
        },
    },
    "features": {
        "columns": [
            "Store", "Date", "Temperature", "Fuel_Price",
            "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
            "CPI", "Unemployment", "IsHoliday",
        ],
        "dtypes": {
            "Store": "int",
            "Temperature": "float",
            "Fuel_Price": "float",
        },
    },
    "stores": {
        "columns": ["Store", "Type", "Size"],
        "dtypes": {
            "Store": "int",
            "Size": "int",
        },
    },
}

# ── Gemini configuration ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Dashboard constants ──
class Theme:
    """Dashboard color palette and styling constants."""
    NAVY = "#0F1B2D"
    INDIGO = "#1E3A5F"
    VIOLET = "#6C63FF"
    TEAL = "#00D4AA"
    AMBER = "#FFB020"
    CORAL = "#FF6B6B"
    WHITE = "#FFFFFF"
    LIGHT_GRAY = "#E8ECF1"
    DARK_GRAY = "#8B95A5"

    # Plotly color sequence for charts
    CHART_COLORS = [VIOLET, TEAL, AMBER, CORAL, "#4ECDC4", "#C084FC", "#FB923C", "#38BDF8"]

    # Plotly chart template defaults
    PLOT_BG = "rgba(0,0,0,0)"
    PAPER_BG = "rgba(0,0,0,0)"
    GRID_COLOR = "rgba(255,255,255,0.08)"
    FONT_COLOR = LIGHT_GRAY
    FONT_FAMILY = "Inter, system-ui, sans-serif"


# ── Store / Department known ranges ──
STORE_RANGE = (1, 45)   # Will be validated from actual data
DEPARTMENT_MAX = 99      # Will be validated from actual data

# ── Forecasting constants ──
FORECAST_TARGET = "Weekly_Sales"
FORECAST_HORIZON = 1  # 1-week-ahead


def ensure_directories() -> None:
    """Create all required project directories if they don't exist."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for subdir in REPORT_SUBDIRS.values():
        subdir.mkdir(parents=True, exist_ok=True)

