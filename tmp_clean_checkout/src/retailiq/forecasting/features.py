"""
RetailIQ AI V2 ??? Forecasting Feature Engineering Module

Constructs strictly backward-looking lag, rolling, calendar, and exogenous features
for 1-week-ahead store-level sales forecasting without future leakage.
"""

from typing import Any
import numpy as np
import pandas as pd


def build_forecasting_features(store_week_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer leakage-safe lag and rolling features on Store-Week series.

    Features generated per store (grouped by Store and sorted chronologically):
    - Lags: lag_1 (t-1), lag_2 (t-2), lag_4 (t-4), lag_52 (t-52 seasonal)
    - Rolling window statistics: rolling_mean_4, rolling_std_4, rolling_mean_8
      (strictly backward-looking, shifted by 1 to exclude current week t)
    - Calendar features: Month, WeekOfYear, DayOfMonth
    - Static metadata: Store Type (one-hot encoded), Size
    - Exogenous environmental/economic signals: Temperature, Fuel_Price, CPI, Unemployment, IsHoliday

    Args:
        store_week_df: Store-week level DataFrame with Date and Weekly_Sales.

    Returns:
        DataFrame with engineered features and NaN rows from lag_52 dropped.
    """
    df = store_week_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Store", "Date"]).reset_index(drop=True)

    # Group by Store for lag and rolling calculations
    grouped = df.groupby("Store", group_keys=False)

    # 1. Direct Lags (strictly observed in past)
    df["lag_1"] = grouped["Weekly_Sales"].shift(1)
    df["lag_2"] = grouped["Weekly_Sales"].shift(2)
    df["lag_4"] = grouped["Weekly_Sales"].shift(4)
    df["lag_52"] = grouped["Weekly_Sales"].shift(52)

    # 2. Rolling Window Statistics (shifted by 1 to prevent target leakage from week t)
    df["rolling_mean_4"] = (
        grouped["Weekly_Sales"]
        .apply(lambda s: s.shift(1).rolling(window=4, min_periods=4).mean())
    )
    df["rolling_std_4"] = (
        grouped["Weekly_Sales"]
        .apply(lambda s: s.shift(1).rolling(window=4, min_periods=4).std())
    )
    df["rolling_mean_8"] = (
        grouped["Weekly_Sales"]
        .apply(lambda s: s.shift(1).rolling(window=8, min_periods=8).mean())
    )

    # 3. Calendar & Cyclical features
    df["Month"] = df["Date"].dt.month.astype("int32")
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype("int32")
    df["DayOfMonth"] = df["Date"].dt.day.astype("int32")

    # Sin/Cos transformation of week of year for cyclical temporal representation
    df["sin_week"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52.0)

    # 4. Binary Holiday Flag
    df["is_holiday_num"] = df["IsHoliday"].astype("int32")

    # 5. One-Hot Encode Store Type (Type_A, Type_B, Type_C)
    type_dummies = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
    for col in ["Type_A", "Type_B", "Type_C"]:
        if col in type_dummies.columns:
            df[col] = type_dummies[col]
        else:
            df[col] = 0

    # 6. Filter to complete cases (dropping initial 52 weeks where lag_52 is NaN)
    complete_df = df.dropna(subset=["lag_52", "rolling_mean_8"]).copy().reset_index(drop=True)

    return complete_df


def get_feature_columns() -> list[str]:
    """Return canonical list of feature column names used for model training."""
    return [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_52",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_8",
        "Month",
        "WeekOfYear",
        "sin_week",
        "cos_week",
        "is_holiday_num",
        "Size",
        "Type_A",
        "Type_B",
        "Type_C",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
    ]
