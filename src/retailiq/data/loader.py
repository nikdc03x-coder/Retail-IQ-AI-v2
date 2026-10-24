"""
RetailIQ AI V2 — Data Loader Module

Provides standardized, type-safe data loading functions for raw CSVs
and processed Parquet datasets with date parsing and path validation.
"""

from pathlib import Path
import pandas as pd

from src.retailiq.config import (
    ANALYTICAL_DATASET,
    RAW_FILES,
)


def load_raw_train(path: Path | str | None = None) -> pd.DataFrame:
    """Load historical weekly sales observations (train.csv).

    Args:
        path: Optional explicit file path. Defaults to RAW_FILES['train'].

    Returns:
        DataFrame with parsed dates and typed columns.
    """
    file_path = Path(path) if path else RAW_FILES["train"]
    if not file_path.exists():
        raise FileNotFoundError(f"train.csv not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            "Store": "int32",
            "Dept": "int32",
            "Weekly_Sales": "float64",
            "IsHoliday": "bool",
        },
        parse_dates=["Date"],
    )
    return df


def load_raw_features(path: Path | str | None = None) -> pd.DataFrame:
    """Load external and economic features (features.csv).

    Args:
        path: Optional explicit file path. Defaults to RAW_FILES['features'].

    Returns:
        DataFrame with parsed dates and numeric columns.
    """
    file_path = Path(path) if path else RAW_FILES["features"]
    if not file_path.exists():
        raise FileNotFoundError(f"features.csv not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            "Store": "int32",
            "Temperature": "float64",
            "Fuel_Price": "float64",
            "MarkDown1": "float64",
            "MarkDown2": "float64",
            "MarkDown3": "float64",
            "MarkDown4": "float64",
            "MarkDown5": "float64",
            "CPI": "float64",
            "Unemployment": "float64",
            "IsHoliday": "bool",
        },
        parse_dates=["Date"],
    )
    return df


def load_raw_stores(path: Path | str | None = None) -> pd.DataFrame:
    """Load store metadata (stores.csv).

    Args:
        path: Optional explicit file path. Defaults to RAW_FILES['stores'].

    Returns:
        DataFrame with Store, Type, and Size.
    """
    file_path = Path(path) if path else RAW_FILES["stores"]
    if not file_path.exists():
        raise FileNotFoundError(f"stores.csv not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            "Store": "int32",
            "Type": "category",
            "Size": "int32",
        },
    )
    return df


def load_analytical_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the processed analytical dataset from Parquet.

    Args:
        path: Optional explicit path. Defaults to ANALYTICAL_DATASET.

    Returns:
        Full analytical DataFrame.
    """
    file_path = Path(path) if path else ANALYTICAL_DATASET
    if not file_path.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found at: {file_path}. "
            "Please run 'python scripts/01_data_ingestion/run_ingestion.py' first."
        )

    return pd.read_parquet(file_path)

