"""
RetailIQ AI V2 ??? Master End-to-End Pipeline Runner

Executes all 7 analytical and machine learning pipeline stages in strict sequence:
1. Data Ingestion & Validation (CSV -> Parquet)
2. Data Quality & Statistical Profiling
3. Business Analytics & Aggregations
4. Sales Forecasting & Model Evaluation (Ridge, RF, XGBoost)
5. Forecast Reliability & Spatial Error Analysis
6. Anomaly Detection & Investigation Prioritization
7. Explainable AI (TreeSHAP Interpretability)

Run with:
    python scripts/run_all_pipelines.py
"""

import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PIPELINE_STAGES = [
    ("Phase 01: Data Ingestion", PROJECT_ROOT / "scripts" / "01_data_ingestion" / "run_ingestion.py"),
    ("Phase 02: Data Quality", PROJECT_ROOT / "scripts" / "02_data_quality" / "run_quality_checks.py"),
    ("Phase 03: Business Analytics", PROJECT_ROOT / "scripts" / "03_business_analytics" / "run_analytics.py"),
    ("Phase 04: Sales Forecasting", PROJECT_ROOT / "scripts" / "04_sales_forecasting" / "run_forecasting.py"),
    ("Phase 05: Forecast Reliability", PROJECT_ROOT / "scripts" / "05_forecast_reliability" / "run_reliability.py"),
    ("Phase 06: Anomaly Detection", PROJECT_ROOT / "scripts" / "06_anomaly_detection" / "run_anomaly_detection.py"),
    ("Phase 07: Explainable AI", PROJECT_ROOT / "scripts" / "07_explainability" / "run_shap_analysis.py"),
    ("Phase 08: Operational Strategy", PROJECT_ROOT / "scripts" / "08_strategy" / "run_strategy_engine.py"),
]


def main() -> None:
    print("=" * 75)
    print("   RetailIQ AI V2 -- End-to-End Master Pipeline Execution")
    print("=" * 75)

    python_executable = sys.executable
    total_start = time.time()
    results = []

    for idx, (stage_name, script_path) in enumerate(PIPELINE_STAGES, 1):
        print(f"\n[{idx}/8] Running {stage_name}...")
        print(f"      Script: {script_path.relative_to(PROJECT_ROOT)}")

        stage_start = time.time()
        cmd = [python_executable, str(script_path)]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8")
        duration = time.time() - stage_start

        if proc.returncode == 0:
            print(f"      [OK] Completed in {duration:.2f}s")
            results.append((stage_name, True, duration, ""))
        else:
            print(f"      [FAILED] Return code {proc.returncode} ({duration:.2f}s)")
            print(f"      Error: {proc.stderr[:300]}")
            results.append((stage_name, False, duration, proc.stderr))
            print("\nPipeline execution halted due to failure.")
            sys.exit(1)

    total_duration = time.time() - total_start
    print("\n" + "=" * 75)
    print("   Pipeline Execution Summary:")
    print("=" * 75)
    for name, success, dur, _ in results:
        status_str = "[PASSED]" if success else "[FAILED]"
        print(f"   {status_str:<8} | {name:<35} | {dur:>6.2f}s")

    print("-" * 75)
    print(f"   All {len(PIPELINE_STAGES)} pipeline stages executed successfully in {total_duration:.2f}s.")
    print("   Analytical dataset, models, reports, and SHAP artifacts are fully up-to-date.")
    print("=" * 75)


if __name__ == "__main__":
    main()
