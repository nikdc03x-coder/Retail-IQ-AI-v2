"""
RetailIQ AI V2 — Phase 08: Business Insights & Operational Decision Strategy Execution Script

Run with:
    python scripts/08_strategy/run_strategy_engine.py

Executes:
- Store strategic matrix formulation across 4 operational quadrants
- Dynamic safety stock & labor flex allocation playbook
- Department merchandising & markdown clearance optimization rules
- Executive decision summary & working capital ROI projections

Outputs reports to reports/strategy/.
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
from src.retailiq.strategy.decision_engine import (
    build_department_merchandising_rules,
    build_executive_decision_summary,
    build_resource_allocation_playbook,
    build_store_strategic_matrix,
)


def main() -> None:
    print("=" * 70)
    print("  RetailIQ AI V2 -- Phase 08: Operational Decision Strategy Engine")
    print("=" * 70)

    start_time = time.time()
    try:
        ensure_directories()
        out_dir = REPORT_SUBDIRS["strategy"]

        print("\n[1/4] Formulating 4-quadrant store strategic matrix (Volume vs WAPE)...")
        strategic_df = build_store_strategic_matrix()
        print(f"      Total stores classified: {len(strategic_df)} stores")
        tier_counts = strategic_df["Strategic_Tier"].value_counts()
        for tier_name, count in tier_counts.items():
            print(f"      - {tier_name:<28}: {count:>2} stores")

        print("\n[2/4] Generating supply chain & labor allocation playbook...")
        playbook_df = build_resource_allocation_playbook(strategic_df)
        print(f"      Total chain safety stock recommended: ${playbook_df['Total_Safety_Stock_Dollars'].sum():>12,.2f}")
        print(f"      Estimated annual capital optimization : ${playbook_df['Total_Annual_Optimization'].sum():>12,.2f}")
        print(f"      Weekly labor flex buffer hours        : {playbook_df['Total_Weekly_Flex_Hours'].sum():>5} hours")

        print("\n[3/4] Formulating department merchandising & markdown clearance rules...")
        merchandising_df = build_department_merchandising_rules()
        print(f"      Total departments analyzed: {len(merchandising_df)} departments")
        arch_counts = merchandising_df["Merchandising_Archetype"].value_counts()
        for arch_name, count in arch_counts.items():
            print(f"      - {arch_name:<28}: {count:>2} departments")

        print("\n[4/4] Synthesizing executive decision directives and ROI projections...")
        summary = build_executive_decision_summary(strategic_df, playbook_df, merchandising_df)
        summary["execution_runtime_seconds"] = round(time.time() - start_time, 2)

        # Export reports
        strategic_df.to_csv(out_dir / "01_store_strategic_matrix.csv", index=False)
        playbook_df.to_csv(out_dir / "02_resource_allocation_playbook.csv", index=False)
        merchandising_df.to_csv(out_dir / "03_department_markdown_rules.csv", index=False)

        with open(out_dir / "04_executive_decision_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("  Top 3 Strategic Directives:")
        for idx, directive in enumerate(summary["priority_executive_directives"][:3], 1):
            print(f"   {idx}. {directive['directive']}")
            print(f"      Target: {directive['target']}")
            print(f"      Action: {directive['action']}")

        print("\n  Phase 08 Operational Strategy completed successfully.")
        print(f"  Artifacts saved to: {out_dir}")
        print(f"  Execution time: {summary['execution_runtime_seconds']}s")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Phase 08 execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

