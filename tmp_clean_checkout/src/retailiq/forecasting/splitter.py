"""
RetailIQ AI V2 ??? Chronological Data Splitter Module

Enforces strictly forward-in-time train, validation, and test splits
to prevent data leakage in time-series evaluation.
"""

from typing import Any
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    val_start: str = "2012-01-06",
    test_start: str = "2012-06-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition dataset into chronological Train, Validation, and Test subsets.

    Split Design:
    - Train: All observations before val_start (approx 2011-02 to 2011-12)
    - Validation: Observations from val_start up to test_start (approx 2012-01 to 2012-05)
    - Test: Observations from test_start onwards (approx 2012-06 to 2012-10)

    Args:
        df: Feature-engineered DataFrame.
        val_start: Date string marking start of validation window.
        test_start: Date string marking start of test window.

    Returns:
        Tuple of (train_df, val_df, test_df, split_metadata).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    val_dt = pd.to_datetime(val_start)
    test_dt = pd.to_datetime(test_start)

    train_df = df[df["Date"] < val_dt].copy().reset_index(drop=True)
    val_df = df[(df["Date"] >= val_dt) & (df["Date"] < test_dt)].copy().reset_index(drop=True)
    test_df = df[df["Date"] >= test_dt].copy().reset_index(drop=True)

    # ?????? Strict Temporal Invariant Verification ??????
    if not train_df.empty and not val_df.empty:
        assert train_df["Date"].max() < val_df["Date"].min(), (
            f"Leakage detected: Train max date ({train_df['Date'].max()}) "
            f"not strictly before Val min date ({val_df['Date'].min()})"
        )

    if not val_df.empty and not test_df.empty:
        assert val_df["Date"].max() < test_df["Date"].min(), (
            f"Leakage detected: Val max date ({val_df['Date'].max()}) "
            f"not strictly before Test min date ({test_df['Date'].min()})"
        )

    metadata = {
        "train": {
            "rows": len(train_df),
            "start_date": str(train_df["Date"].min().date()),
            "end_date": str(train_df["Date"].max().date()),
            "weeks": int(train_df["Date"].nunique()),
            "stores": int(train_df["Store"].nunique()),
        },
        "validation": {
            "rows": len(val_df),
            "start_date": str(val_df["Date"].min().date()),
            "end_date": str(val_df["Date"].max().date()),
            "weeks": int(val_df["Date"].nunique()),
            "stores": int(val_df["Store"].nunique()),
        },
        "test": {
            "rows": len(test_df),
            "start_date": str(test_df["Date"].min().date()),
            "end_date": str(test_df["Date"].max().date()),
            "weeks": int(test_df["Date"].nunique()),
            "stores": int(test_df["Store"].nunique()),
        },
    }

    return train_df, val_df, test_df, metadata
