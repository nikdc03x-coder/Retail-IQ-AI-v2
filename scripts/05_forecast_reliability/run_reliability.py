"""
RetailIQ AI V2 — Phase 05: Forecast Reliability Execution Script

Run with:
    python scripts/05_forecast_reliability/run_reliability.py

Evaluates:
- Store-by-store error dispersion (45 locations)
- Week-by-week temporal error patterns (22 test weeks)
- Holiday forecast degradation analysis
- Store-level win rate against baseline
- 4-fold expanding-window rolling backtest

Outputs reports to reports/forecast_reliability/.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from src.retailiq.config import REPORT_SUBDIRS, ensure_directories
from src.retailiq.forecasting.features import build_forecasting_features
from src.retailiq.reliability.backtest import run_rolling_backtest
from src.retailiq.reliability.error_analysis import (
    analyze_holiday_error_impact,
    analyze_store_errors,
    analyze_weekly_errors,
    compute_relative_baseline_errors,
)


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 05: Forecast Reliability & Backtesting")
    print("=" * 70)

    start_time = time.time()
    try:
        ensure_directories()
        out_dir = REPORT_SUBDIRS["forecast_reliability"]

        # Load held-out test predictions from Phase 04
        pred_path = REPORT_SUBDIRS["sales_forecasting"] / "02_test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Test predictions not found at {pred_path}")
        test_preds = pd.read_csv(pred_path, parse_dates=["Date"])

        # Load store-week sales for backtest
        store_week_path = REPORT_SUBDIRS["business_analytics"] / "04_store_week_sales.csv"
        store_week = pd.read_csv(store_week_path, parse_dates=["Date"])

        print("\n[1/5] Profiling store-by-store error disparity (45 stores)...")
        store_errors = analyze_store_errors(test_preds)

        print("[2/5] Evaluating temporal error variations across test weeks...")
        weekly_errors = analyze_weekly_errors(test_preds)

        print("[3/5] Assessing holiday forecast degradation impact...")
        holiday_errors = analyze_holiday_error_impact(test_preds)

        print("[4/5] Computing store-level head-to-head win rates vs Naive baseline...")
        store_relative = compute_relative_baseline_errors(test_preds)

        print("[5/5] Executing 4-fold expanding-window rolling backtest...")
        features_df = build_forecasting_features(store_week)
        backtest_results = run_rolling_backtest(features_df, n_folds=4, test_window_weeks=8)

        # ── Export Reports ──
        store_errors.to_csv(out_dir / "01_store_error_summary.csv", index=False)
        weekly_errors.to_csv(out_dir / "02_weekly_error_summary.csv", index=False)
        holiday_errors.to_csv(out_dir / "03_holiday_error_summary.csv", index=False)
        store_relative.to_csv(out_dir / "04_store_relative_errors.csv", index=False)
        backtest_results.to_csv(out_dir / "05_rolling_backtest_results.csv", index=False)

        # Build summary JSON
        win_count = int(store_relative["XGB_Wins"].sum())
        total_stores = len(store_relative)
        win_rate = (win_count / total_stores * 100.0)

        summary = {
            "experiment": "Phase 05 — Forecast Reliability & Error Breakdown",
            "stores_analyzed": total_stores,
            "store_win_rate_vs_naive_pct": round(win_rate, 2),
            "store_wape_range": {
                "min_wape": float(store_errors["WAPE_Pct"].min()),
                "min_store": int(store_errors.iloc[0]["Store"]),
                "max_wape": float(store_errors["WAPE_Pct"].max()),
                "max_store": int(store_errors.iloc[-1]["Store"]),
                "median_wape": float(store_errors["WAPE_Pct"].median()),
            },
            "holiday_degradation": {
                "non_holiday_wape": float(holiday_errors.loc[holiday_errors["Period_Type"] == "Non-Holiday Weeks", "WAPE_Pct"].values[0]),
                "holiday_wape": float(holiday_errors.loc[holiday_errors["Period_Type"] == "Holiday Weeks", "WAPE_Pct"].values[0]),
                "wape_change_pct": float(holiday_errors.loc[holiday_errors["Period_Type"] == "Holiday Weeks", "WAPE_Change_Vs_NonHoliday_Pct"].values[0]),
            },
            "backtest_summary": {
                "folds_evaluated": len(backtest_results),
                "folds_outperformed": int(backtest_results["XGB_Outperformed"].sum()),
                "mean_xgb_wape": round(float(backtest_results["XGB_WAPE"].mean()), 2),
                "mean_naive_wape": round(float(backtest_results["Naive_WAPE"].mean()), 2),
                "mean_error_reduction_pct": round(float(backtest_results["Error_Reduction_Pct"].mean()), 2),
            },
        }

        with open(out_dir / "06_forecast_reliability_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        elapsed = time.time() - start_time

        print("\n" + "-" * 70)
        print("  FORECAST RELIABILITY AUDIT COMPLETED SUCCESSFULLY")
        print("-" * 70)
        print(f"* Reports Directory:       {out_dir}")
        print(f"* Store Win Rate:          {win_rate:.1f}% ({win_count}/{total_stores} stores beat Naive baseline)")
        print(f"* Store WAPE Range:        {summary['store_wape_range']['min_wape']}% (Store {summary['store_wape_range']['min_store']}) to {summary['store_wape_range']['max_wape']}% (Store {summary['store_wape_range']['max_store']})")
        print(f"* Holiday Error Shift:     {summary['holiday_degradation']['holiday_wape']}% WAPE vs {summary['holiday_degradation']['non_holiday_wape']}% non-holiday ({summary['holiday_degradation']['wape_change_pct']:+.1f}%)")
        print(f"* Backtest Consistency:    {summary['backtest_summary']['folds_outperformed']}/{summary['backtest_summary']['folds_evaluated']} historical folds won by XGBoost")
        print(f"* Mean Backtest Reduction: -{summary['backtest_summary']['mean_error_reduction_pct']}% error reduction over Naive across all folds")
        print(f"* Execution Time:          {elapsed:.2f} seconds")
        print("-" * 70)

    except Exception as exc:
        print(f"\n[ERROR] Reliability analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
