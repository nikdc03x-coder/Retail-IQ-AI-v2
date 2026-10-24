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
    """Enterprise UI design system tokens, color palette, and chart styling."""
    # Base canvas & container surfaces (Tailwind Slate)
    NAVY = "#0B0F19"         # Primary canvas background (Slate-950)
    INDIGO = "#111827"       # Card & panel background (Slate-900)
    CARD_BG = "#111827"      # Container surface
    DARK_CARD = "#111827"    # Card surface alias
    CARD_BORDER = "#1F2937"  # Clean card border (Slate-800)
    BORDER_LIGHT = "#374151" # Hover & active border (Slate-700)

    # Core brand & interactive accents
    VIOLET = "#6366F1"       # Primary brand accent (Indigo-500)
    BRAND = "#4F46E5"        # Brand solid (Indigo-600)
    CYAN = "#06B6D4"         # Secondary highlight (Cyan-500)
    TEAL = "#10B981"         # Success & positive metrics (Emerald-500)
    GREEN = "#10B981"        # Success green
    SUCCESS = "#10B981"
    AMBER = "#F59E0B"        # Warning & caution (Amber-500)
    WARNING = "#F59E0B"
    CORAL = "#EF4444"        # Error & critical outliers (Rose-500)
    DANGER = "#EF4444"
    RED = "#EF4444"

    # High-contrast readable typography & neutrals
    WHITE = "#F9FAFB"        # Primary headings (Slate-50)
    LIGHT_GRAY = "#CBD5E1"   # Body text & primary labels (Slate-300)
    DARK_GRAY = "#9CA3AF"    # Secondary text, subtitles & captions (Slate-400)
    MUTED = "#6B7280"        # Inactive elements (Slate-500)
    GRAY = "#6B7280"         # Neutral gray alias (Slate-500)

    # Harmonized Plotly chart sequence (modern data visualization palette)
    CHART_COLORS = [
        "#6366F1",  # Indigo
        "#06B6D4",  # Cyan
        "#10B981",  # Emerald
        "#F59E0B",  # Amber
        "#EC4899",  # Pink
        "#8B5CF6",  # Purple
        "#3B82F6",  # Blue
        "#14B8A6",  # Teal
    ]

    # Plotly chart template defaults
    PLOT_BG = "rgba(0,0,0,0)"
    PAPER_BG = "rgba(0,0,0,0)"
    GRID_COLOR = "rgba(255, 255, 255, 0.06)"
    FONT_COLOR = "#CBD5E1"
    FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"


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

