"""
RetailIQ AI V2 — Phase 03: Business Analytics Execution Script

Run with:
    python scripts/03_business_analytics/run_analytics.py

Executes hierarchical aggregations across network, store, department,
store-type, holiday, and seasonality dimensions.
Exports 7 pre-computed business intelligence reports to reports/business_analytics/.
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

from src.retailiq.analytics.aggregations import run_business_analytics
from src.retailiq.config import REPORT_SUBDIRS


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 03: Business Intelligence Analytics Suite")
    print("=" * 70)

    start_time = time.time()
    try:
        print("\n[1/5] Loading analytical dataset and aggregating Store-Week observations...")
        print("[2/5] Ranking Store locations and calculating space productivity metrics...")
        print("[3/5] Calculating Department Pareto shares and contributions...")
        print("[4/5] Evaluating Walmart holiday surges (Thanksgiving, Super Bowl, etc.)...")
        print("[5/5] Profiling monthly seasonality indices and exporting reports...")

        results = run_business_analytics(save_reports=True)
        elapsed = time.time() - start_time

        out_dir = REPORT_SUBDIRS["business_analytics"]
        store_perf = results["store_performance"]
        dept_perf = results["department_performance"]
        holiday_df = results["holiday_comparison"]
        network_df = results["network_weekly"]
        store_type = results["store_type"]

        top_store = store_perf.iloc[0]
        top_dept = dept_perf.iloc[0]
        all_holidays = holiday_df[holiday_df["Holiday_Name"] == "All Holidays Combined"].iloc[0]
        thanksgiving = holiday_df[holiday_df["Holiday_Name"] == "Thanksgiving"].iloc[0]
        total_rev = network_df["Total_Network_Sales"].sum()

        print("\n" + "-" * 70)
        print("  BUSINESS ANALYTICS AGGREGATION COMPLETED SUCCESSFULLY")
        print("-" * 70)
        print(f"* Reports Directory:       {out_dir}")
        print(f"* Total Network Sales:     ${total_rev:,.2f} (across 143 recorded weeks)")
        print(f"* Mean Weekly Store Sales: ${store_perf['Mean_Weekly_Sales'].mean():,.2f}")
        print(f"* #1 Store:                Store {int(top_store['Store'])} (${top_store['Total_Sales']:,.2f} total, Type {top_store['Type']})")
        print(f"* #1 Department:           Dept {int(top_dept['Dept'])} (${top_dept['Total_Sales']:,.2f} total, {top_dept['Sales_Share_Pct']}% share)")
        print(f"* Overall Holiday Lift:    {all_holidays['Lift_Vs_Baseline_Pct']:+.2f}% vs non-holiday baseline")
        print(f"* Thanksgiving Surge:      {thanksgiving['Lift_Vs_Baseline_Pct']:+.2f}% peak weekly lift")
        print(f"* Store Type A Share:      {store_type.loc[store_type['Type'] == 'A', 'Network_Sales_Share_Pct'].values[0]}% of network sales")
        print(f"* Execution Time:          {elapsed:.2f} seconds")
        print("-" * 70)
        print("\nGenerated Reports:")
        print(f"  1. {out_dir / '01_weekly_sales.csv'}")
        print(f"  2. {out_dir / '02_store_performance.csv'}")
        print(f"  3. {out_dir / '03_department_performance.csv'}")
        print(f"  4. {out_dir / '04_store_week_sales.csv'}")
        print(f"  5. {out_dir / '05_holiday_comparison.csv'}")
        print(f"  6. {out_dir / '06_store_type_performance.csv'}")
        print(f"  7. {out_dir / '07_monthly_seasonality.csv'}")
        print("-" * 70)

    except Exception as exc:
        print(f"\n[ERROR] Business analytics failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
