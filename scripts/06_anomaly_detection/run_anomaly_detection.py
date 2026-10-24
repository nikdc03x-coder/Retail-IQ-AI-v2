"""
RetailIQ AI V2 — Phase 06: Anomaly Detection & Investigation Queue Execution Script

Run with:
    python scripts/06_anomaly_detection/run_anomaly_detection.py

Executes:
- Statistical residual Z-score detector (|Z| > 3.0)
- Multivariate Isolation Forest detector (contamination = 2%)
- Cross-detector agreement synthesis
- Ranked operational investigation queue generation

Outputs reports to reports/anomaly_detection/.
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
from src.retailiq.anomalies.investigation import build_investigation_queue
from src.retailiq.anomalies.isolation_forest import detect_isolation_forest_anomalies
from src.retailiq.anomalies.statistical import detect_statistical_anomalies
from src.retailiq.config import REPORT_SUBDIRS, ensure_directories


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 06: Anomaly Detection & Investigation Queue")
    print("=" * 70)

    start_time = time.time()
    try:
        ensure_directories()
        out_dir = REPORT_SUBDIRS["anomaly_detection"]

        # Load held-out test predictions from Phase 04
        pred_path = REPORT_SUBDIRS["sales_forecasting"] / "02_test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Test predictions not found at {pred_path}")
        test_preds = pd.read_csv(pred_path, parse_dates=["Date"])

        print("\n[1/4] Calculating statistical residual Z-scores (|Z| > 3.0 cutoff)...")
        stat_df, stat_meta = detect_statistical_anomalies(test_preds, threshold_z=3.0)

        print("[2/4] Fitting multivariate Isolation Forest on volume, residuals, and errors...")
        ifor_df, ifor_meta = detect_isolation_forest_anomalies(test_preds, contamination=0.02, random_state=42)

        print("[3/4] Synthesizing detector agreement matrix (unanimous vs divergent)...")
        print("[4/4] Generating prioritized, evidence-backed operational investigation queue...")
        comparison_df, agree_df, queue_df, summary = build_investigation_queue(stat_df, ifor_df)

        # ── Export Reports ──
        stat_df.to_csv(out_dir / "01_statistical_scores.csv", index=False)
        ifor_df.to_csv(out_dir / "02_isolation_forest_scores.csv", index=False)
        comparison_df.to_csv(out_dir / "03_detector_comparison.csv", index=False)
        agree_df.to_csv(out_dir / "04_agreement_summary.csv", index=False)
        queue_df.to_csv(out_dir / "05_investigation_queue.csv", index=False)

        summary_full = {
            "experiment": "Phase 06 — Dual-Detector Anomaly Investigation",
            "statistical_summary": stat_meta,
            "isolation_forest_summary": ifor_meta,
            "synthesis_summary": summary,
        }
        with open(out_dir / "06_anomaly_detection_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_full, f, indent=2)

        elapsed = time.time() - start_time

        print("\n" + "-" * 70)
        print("  ANOMALY DETECTION & QUEUE GENERATION SUCCESSFUL")
        print("-" * 70)
        print(f"* Reports Directory:       {out_dir}")
        print(f"* Evaluated Records:       {summary['total_evaluated']:,} store-weeks")
        print(f"* Investigation Queue:     {summary['total_flagged']} records prioritized for review")
        print(f"  - Unanimous Flags:       {summary['unanimous_anomalies']} (confirmed by BOTH detectors)")
        print(f"  - Statistical Only:      {summary['statistical_only_anomalies']} (|Z| > 3.0 extreme residuals)")
        print(f"  - Isolation Forest Only: {summary['isolation_forest_only_anomalies']} (multivariate feature space)")
        print(f"* Detector Agreement:      {summary['detector_agreement_rate_pct']}% overall agreement")
        if not queue_df.empty:
            top_q = queue_df.iloc[0]
            print(f"* #1 Priority Investigation: Store {int(top_q['Store'])} on {str(pd.to_datetime(top_q['Date']).date())} (Priority Score: {top_q['Priority_Score']})")
        print(f"* Execution Time:          {elapsed:.2f} seconds")
        print("-" * 70)

    except Exception as exc:
        print(f"\n[ERROR] Anomaly detection failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
