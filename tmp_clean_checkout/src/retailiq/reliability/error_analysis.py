"""
RetailIQ AI V2 ??? Forecast Error Analysis Module

Analyzes where and when forecast errors occur:
- Store-by-store error disparity (high-error vs low-error locations).
- Temporal error volatility across historical test weeks.
- Holiday forecast degradation analysis (holiday vs non-holiday error lift).
- Store-level head-to-head win rate (XGBoost vs Naive persistence).
"""

from typing import Any
import numpy as np
import pandas as pd


def analyze_store_errors(test_preds_df: pd.DataFrame) -> pd.DataFrame:
    """Compute store-level error profiles on the held-out test set.

    Args:
        test_preds_df: Test predictions DataFrame containing Store, Actual_Sales, XGBoost_Pred.

    Returns:
        DataFrame sorted by WAPE (lowest to highest error).
    """
    df = test_preds_df.copy()
    df["Residual"] = df["Actual_Sales"] - df["XGBoost_Pred"]
    df["Abs_Error"] = df["Residual"].abs()

    store_errors = (
        df.groupby(["Store", "Type", "Size"], observed=True)
        .agg(
            Test_Weeks=("Date", "count"),
            Total_Actual_Sales=("Actual_Sales", "sum"),
            Mean_Weekly_Sales=("Actual_Sales", "mean"),
            MAE=("Abs_Error", "mean"),
            RMSE=("Residual", lambda r: float(np.sqrt(np.mean(r ** 2)))),
            Max_Abs_Error=("Abs_Error", "max"),
            Mean_Bias=("Residual", "mean"),
        )
        .reset_index()
    )

    # WAPE = sum(abs_errors) / sum(actuals)
    abs_sums = df.groupby("Store")["Abs_Error"].sum()
    act_sums = df.groupby("Store")["Actual_Sales"].sum()
    store_errors["WAPE_Pct"] = (abs_sums / act_sums * 100.0).round(2).values

    store_errors["MAE"] = store_errors["MAE"].round(2)
    store_errors["RMSE"] = store_errors["RMSE"].round(2)
    store_errors["Max_Abs_Error"] = store_errors["Max_Abs_Error"].round(2)
    store_errors["Mean_Bias"] = store_errors["Mean_Bias"].round(2)
    store_errors["Mean_Weekly_Sales"] = store_errors["Mean_Weekly_Sales"].round(2)
    store_errors["Total_Actual_Sales"] = store_errors["Total_Actual_Sales"].round(2)

    store_errors = store_errors.sort_values(by="WAPE_Pct", ascending=True).reset_index(drop=True)
    store_errors["Reliability_Rank"] = range(1, len(store_errors) + 1)

    return store_errors


def analyze_weekly_errors(test_preds_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze temporal error variations across individual test weeks.

    Args:
        test_preds_df: Test predictions DataFrame.

    Returns:
        DataFrame with weekly MAE, WAPE, and holiday indicator.
    """
    df = test_preds_df.copy()
    df["Residual"] = df["Actual_Sales"] - df["XGBoost_Pred"]
    df["Abs_Error"] = df["Residual"].abs()

    weekly = (
        df.groupby(["Date", "IsHoliday"], observed=True)
        .agg(
            Store_Count=("Store", "count"),
            Total_Actual_Sales=("Actual_Sales", "sum"),
            Total_Predicted_Sales=("XGBoost_Pred", "sum"),
            MAE=("Abs_Error", "mean"),
            RMSE=("Residual", lambda r: float(np.sqrt(np.mean(r ** 2)))),
            Max_Error=("Abs_Error", "max"),
        )
        .reset_index()
    )

    abs_sums = df.groupby("Date")["Abs_Error"].sum()
    act_sums = df.groupby("Date")["Actual_Sales"].sum()
    weekly["WAPE_Pct"] = (abs_sums / act_sums * 100.0).round(2).values

    weekly["Total_Actual_Sales"] = weekly["Total_Actual_Sales"].round(2)
    weekly["Total_Predicted_Sales"] = weekly["Total_Predicted_Sales"].round(2)
    weekly["MAE"] = weekly["MAE"].round(2)
    weekly["RMSE"] = weekly["RMSE"].round(2)
    weekly["Max_Error"] = weekly["Max_Error"].round(2)

    return weekly.sort_values(by="Date").reset_index(drop=True)


def analyze_holiday_error_impact(test_preds_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether forecast accuracy degrades during holiday periods.

    Args:
        test_preds_df: Test predictions DataFrame.

    Returns:
        DataFrame comparing holiday vs non-holiday test error metrics.
    """
    df = test_preds_df.copy()
    df["Residual"] = df["Actual_Sales"] - df["XGBoost_Pred"]
    df["Abs_Error"] = df["Residual"].abs()

    summary_list = []
    for is_hol in [False, True]:
        sub = df[df["IsHoliday"] == is_hol]
        label = "Holiday Weeks" if is_hol else "Non-Holiday Weeks"
        total_act = sub["Actual_Sales"].sum()
        total_err = sub["Abs_Error"].sum()
        wape = (total_err / total_act * 100.0) if total_act > 0 else 0.0

        summary_list.append({
            "Period_Type": label,
            "Weeks_Count": int(sub["Date"].nunique()),
            "Total_Observations": len(sub),
            "MAE": round(float(sub["Abs_Error"].mean()), 2),
            "RMSE": round(float(np.sqrt(np.mean(sub["Residual"] ** 2))), 2),
            "WAPE_Pct": round(wape, 2),
            "Mean_Weekly_Sales": round(float(sub["Actual_Sales"].mean()), 2),
        })

    hol_df = pd.DataFrame(summary_list)
    # Calculate error degradation percentage
    base_wape = hol_df.loc[hol_df["Period_Type"] == "Non-Holiday Weeks", "WAPE_Pct"].values[0]
    hol_df["WAPE_Change_Vs_NonHoliday_Pct"] = (
        (hol_df["WAPE_Pct"] - base_wape) / base_wape * 100.0
    ).round(2)

    return hol_df


def compute_relative_baseline_errors(test_preds_df: pd.DataFrame) -> pd.DataFrame:
    """Compute store-by-store comparison between XGBoost and Previous-Week Naive.

    Args:
        test_preds_df: Test predictions DataFrame.

    Returns:
        DataFrame with store-level win/loss indicator and error reduction percentage.
    """
    df = test_preds_df.copy()
    df["XGB_Abs_Error"] = (df["Actual_Sales"] - df["XGBoost_Pred"]).abs()
    df["Naive_Abs_Error"] = (df["Actual_Sales"] - df["Naive_Pred"]).abs()

    store_comp = (
        df.groupby(["Store", "Type"], observed=True)
        .agg(
            Total_Sales=("Actual_Sales", "sum"),
            XGB_MAE=("XGB_Abs_Error", "mean"),
            Naive_MAE=("Naive_Abs_Error", "mean"),
            XGB_Abs_Sum=("XGB_Abs_Error", "sum"),
            Naive_Abs_Sum=("Naive_Abs_Error", "sum"),
        )
        .reset_index()
    )

    store_comp["XGB_WAPE"] = (store_comp["XGB_Abs_Sum"] / store_comp["Total_Sales"] * 100.0).round(2)
    store_comp["Naive_WAPE"] = (store_comp["Naive_Abs_Sum"] / store_comp["Total_Sales"] * 100.0).round(2)
    store_comp["Error_Reduction_Pct"] = (
        (store_comp["Naive_WAPE"] - store_comp["XGB_WAPE"]) / store_comp["Naive_WAPE"] * 100.0
    ).round(2)

    store_comp["XGB_Wins"] = store_comp["XGB_WAPE"] < store_comp["Naive_WAPE"]
    store_comp["XGB_MAE"] = store_comp["XGB_MAE"].round(2)
    store_comp["Naive_MAE"] = store_comp["Naive_MAE"].round(2)

    ordered = [
        "Store", "Type", "XGB_WAPE", "Naive_WAPE", "Error_Reduction_Pct",
        "XGB_MAE", "Naive_MAE", "XGB_Wins",
    ]
    return store_comp[ordered].sort_values(by="Error_Reduction_Pct", ascending=False).reset_index(drop=True)
