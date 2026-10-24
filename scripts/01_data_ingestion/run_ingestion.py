"""
RetailIQ AI V2 — Phase 01: Data Ingestion Script

Run with:
    python scripts/01_data_ingestion/run_ingestion.py

Loads raw CSV files, validates schemas, executes join pipeline,
and exports the analytical Parquet dataset and summary metrics.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 stdout encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.retailiq.data.pipeline import run_ingestion_pipeline


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 01: Data Ingestion & Quality Validation")
    print("=" * 70)

    start_time = time.time()
    try:
        print("\n[1/4] Loading raw data files (train.csv, features.csv, stores.csv)...")
        print("[2/4] Validating schemas, dates, duplicate keys, and holiday alignment...")
        print("[3/4] Merging datasets and engineering temporal features...")
        print("[4/4] Writing analytical Parquet and summary reports...")

        df, summary = run_ingestion_pipeline()
        elapsed = time.time() - start_time

        print("\n" + "-" * 70)
        print("  INGESTION PIPELINE EXECUTION SUCCESSFUL")
        print("-" * 70)
        print(f"* Output Parquet:       {summary['output_file']}")
        print(f"* Total Rows:           {summary['processed_counts']['analytical_rows']:,}")
        print(f"* Total Columns:        {summary['processed_counts']['analytical_cols']}")
        print(f"* Row Preservation:     {summary['processed_counts']['row_count_preservation_ratio']:.6%}")
        print(f"* Unique Stores:        {summary['dimensions']['unique_stores']}")
        print(f"* Unique Departments:   {summary['dimensions']['unique_departments']}")
        print(f"* Unique Weeks:         {summary['dimensions']['unique_dates']}")
        print(f"* Date Range:           {summary['dimensions']['min_date']} to {summary['dimensions']['max_date']}")
        print(f"* Holiday Consistency:  {'100% Perfect' if summary['validations']['holiday_consistency'] else 'MISMATCH'}")
        print(f"* Features Match Rate:  {summary['validations']['features_join_match_rate']:.1%}")
        print(f"* Stores Match Rate:    {summary['validations']['stores_join_match_rate']:.1%}")
        print(f"* Total Elapsed Time:   {elapsed:.2f} seconds")
        print("-" * 70)

    except Exception as exc:
        print(f"\n[ERROR] Ingestion pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
