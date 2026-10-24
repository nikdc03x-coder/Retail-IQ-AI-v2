"""
RetailIQ AI V2 — Phase 02: Data Quality & Readiness Audit Script

Run with:
    python scripts/02_data_quality/run_quality_checks.py

Conducts comprehensive statistical profiling, structural missingness classification,
merchandise return (negative sales) audit, and store-department coverage checks.
Outputs reports to reports/data_quality/.
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

from src.retailiq.config import REPORT_SUBDIRS
from src.retailiq.quality.reporter import run_quality_audit


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 02: Data Quality & Readiness Audit")
    print("=" * 70)

    start_time = time.time()
    try:
        print("\n[1/4] Loading analytical dataset from Parquet...")
        print("[2/4] Computing statistical distributions and dispersion metrics...")
        print("[3/4] Profiling structural missingness and merchandise returns...")
        print("[4/4] Evaluating store-department temporal continuity & exporting...")

        results = run_quality_audit(save_reports=True)
        elapsed = time.time() - start_time

        out_dir = REPORT_SUBDIRS["data_quality"]
        stats_df = results["column_statistics"]
        missing_df = results["missingness_profile"]
        dept_returns = results["dept_returns"]
        coverage_df = results["coverage"]

        full_cov_count = int((coverage_df["Status"] == "Full (143 Weeks)").sum())
        total_pairs = len(coverage_df)

        print("\n" + "-" * 70)
        print("  DATA QUALITY AUDIT COMPLETED SUCCESSFULLY")
        print("-" * 70)
        print(f"* Reports Directory:       {out_dir}")
        print(f"* Features Profiled:       {len(stats_df)} numerical features")
        print(f"* Missingness Scanned:     {len(missing_df)} columns")
        print(f"* Active Store-Dept Pairs: {total_pairs:,} ({full_cov_count:,} full 143-week coverage)")
        print(f"* Return Departments:      {len(dept_returns)} departments with negative sales")
        print(f"* Total Returns Value:     ${dept_returns['Total_Return_Amount'].sum():,.2f}")
        print(f"* Execution Time:          {elapsed:.2f} seconds")
        print("-" * 70)
        print("\nGenerated Reports:")
        print(f"  1. {out_dir / '02_column_statistics.csv'}")
        print(f"  2. {out_dir / '03_missingness_profile.csv'}")
        print(f"  3. {out_dir / '04_negative_sales_audit.csv'}")
        print(f"  4. {out_dir / '05_store_department_coverage.csv'}")
        print(f"  5. {out_dir / '06_data_quality_summary.json'}")
        print("-" * 70)

    except Exception as exc:
        print(f"\n[ERROR] Quality checks failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
