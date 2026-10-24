"""
RetailIQ AI V2 — Phase 07: Explainable AI (SHAP) Execution Script

Run with:
    python scripts/07_explainability/run_shap_analysis.py

Executes:
- TreeSHAP explainer on champion XGBoost forecasting model
- Global feature importance (mean absolute SHAP in dollars and relative %)
- Local prediction attribution and waterfall additive decomposition
- Mathematical reconstruction verification (efficiency axiom audit)
- Top positive and negative driver distribution analysis

Outputs reports to reports/explainability/.
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

import numpy as np
import pandas as pd
from src.retailiq.config import REPORT_SUBDIRS, ensure_directories
from src.retailiq.explainability.shap_analysis import (
    compute_shap_explanations,
    verify_shap_reconstruction,
)
from src.retailiq.forecasting.features import get_feature_columns


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 07: Explainable AI (SHAP) Analysis")
    print("=" * 70)

    start_time = time.time()
    try:
        ensure_directories()
        out_dir = REPORT_SUBDIRS["explainable_ai"]

        print("\n[1/4] Fitting TreeSHAP explainer on champion XGBoost model...")
        global_imp, local_df, base_value, raw_shap = compute_shap_explanations()
        print(f"      Base Expected Value (phi_0): ${base_value:,.2f}")
        print(f"      Evaluated observations: {len(local_df):,} test records")

        print("\n[2/4] Verifying additive mathematical reconstruction (efficiency axiom)...")
        verification = verify_shap_reconstruction(local_df, max_tolerance_dollars=5.0)
        print(f"      Reconstruction Max Diff: ${verification['max_abs_diff_dollars']:.4f}")
        print(f"      Reconstruction Mean Diff: ${verification['mean_abs_diff_dollars']:.4f}")
        print(f"      Max Relative Error: {verification['max_rel_diff_pct']:.6f}%")
        print(f"      Efficiency Axiom Holds: {verification['efficiency_axiom_holds']}")

        print("\n[3/4] Analyzing top driver distributions across test records...")
        feature_cols = get_feature_columns()
        shap_cols = [f"shap_{f}" for f in feature_cols]
        shap_sub = local_df[shap_cols].copy()

        # Determine #1 positive driver and #1 negative driver per observation
        top_pos_features = []
        top_neg_features = []
        for _, row in shap_sub.iterrows():
            pos_series = row[row > 0]
            neg_series = row[row < 0]
            top_pos_features.append(pos_series.idxmax().replace("shap_", "") if not pos_series.empty else "None")
            top_neg_features.append(neg_series.idxmin().replace("shap_", "") if not neg_series.empty else "None")

        local_df["Top_Positive_Driver"] = top_pos_features
        local_df["Top_Negative_Driver"] = top_neg_features

        pos_dist = pd.Series(top_pos_features).value_counts().rename("Top_Positive_Count")
        neg_dist = pd.Series(top_neg_features).value_counts().rename("Top_Negative_Count")
        drivers_summary = pd.concat([pos_dist, neg_dist], axis=1).fillna(0).astype(int).reset_index()
        drivers_summary = drivers_summary.rename(columns={"index": "Feature"})

        print("\n[4/4] Exporting explainability artifacts...")
        global_imp.to_csv(out_dir / "01_global_feature_importance.csv", index=False)
        local_df.to_csv(out_dir / "02_local_prediction_attributions.csv", index=False)

        with open(out_dir / "03_reconstruction_verification.json", "w", encoding="utf-8") as f:
            json.dump(verification, f, indent=2)

        drivers_summary.to_csv(out_dir / "04_top_drivers_summary.csv", index=False)

        summary_full = {
            "model": "XGBoost Regressor (Champion)",
            "explainer": "TreeExplainer (TreeSHAP)",
            "base_value_dollars": round(base_value, 2),
            "total_test_observations": len(local_df),
            "features_analyzed": len(feature_cols),
            "top_global_feature": global_imp.iloc[0]["Feature"],
            "top_global_mean_abs_shap": float(global_imp.iloc[0]["Mean_Abs_SHAP"]),
            "top_global_impact_pct": float(global_imp.iloc[0]["Relative_Impact_Pct"]),
            "reconstruction_check": verification,
            "runtime_seconds": round(time.time() - start_time, 2),
        }

        with open(out_dir / "05_explainability_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_full, f, indent=2)

        print("\n" + "=" * 70)
        print("  Global Feature Importance Top 5:")
        for _, r in global_imp.head(5).iterrows():
            print(f"   {int(r['Importance_Rank'])}. {r['Feature']:<18} | ${r['Mean_Abs_SHAP']:>11,.2f} | {r['Relative_Impact_Pct']:>6.2f}%")

        print("\n  Phase 07 Explainability completed successfully.")
        print(f"  Execution time: {summary_full['runtime_seconds']:.2f} seconds.")
        print(f"  Artifacts saved to: {out_dir}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Phase 07 execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
