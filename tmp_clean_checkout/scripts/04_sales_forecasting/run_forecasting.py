"""
RetailIQ AI V2 ??? Phase 04: Sales Forecasting Experiment Execution Script

Run with:
    python scripts/04_sales_forecasting/run_forecasting.py

Trains and evaluates:
- Baseline 1: Previous-Week Naive (t-1)
- Baseline 2: 52-Week Seasonal Naive (t-52)
- Model 1: L2 Ridge Linear Regressor
- Model 2: Random Forest Regressor
- Model 3: XGBoost Regressor

Outputs trained models to models/ and metrics reports to reports/sales_forecasting/.
"""

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

from src.retailiq.config import MODELS_DIR, REPORT_SUBDIRS
from src.retailiq.forecasting.experiment import run_forecasting_experiment


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 04: Sales Forecasting & Model Evaluation")
    print("=" * 70)

    start_time = time.time()
    try:
        print("\n[1/5] Engineering backward-looking lag, rolling, and cyclical features...")
        print("[2/5] Enforcing chronological Train / Validation / Test partitions...")
        print("[3/5] Evaluating persistence baselines (Naive-1 & Seasonal-52)...")
        print("[4/5] Training Ridge, Random Forest, and XGBoost regressor models...")
        print("[5/5] Computing MAE, RMSE, WAPE, R?? and serializing artifacts...")

        results = run_forecasting_experiment(save_artifacts=True)
        elapsed = time.time() - start_time

        metrics_df = results["metrics"]
        test_metrics = metrics_df[metrics_df["Split"] == "Test"].sort_values("WAPE_Pct")

        print("\n" + "-" * 70)
        print("  FORECASTING EXPERIMENT COMPLETED SUCCESSFULLY")
        print("-" * 70)
        print(f"* Models Saved In:         {MODELS_DIR}")
        print(f"* Reports Saved In:        {REPORT_SUBDIRS['sales_forecasting']}")
        print(f"* Execution Time:          {elapsed:.2f} seconds")
        print("-" * 70)
        print("\nHeld-Out Test Evaluation Leaderboard (Sorted by WAPE):")
        print("-" * 70)
        for _, row in test_metrics.iterrows():
            print(
                f"  {row['Model']:<25} | "
                f"MAE: ${row['MAE']:>10,.2f} | "
                f"RMSE: ${row['RMSE']:>11,.2f} | "
                f"WAPE: {row['WAPE_Pct']:>6.2f}% | "
                f"R2: {row['R2']:>6.4f}"
            )
        print("-" * 70)

    except Exception as exc:
        print(f"\n[ERROR] Forecasting experiment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
