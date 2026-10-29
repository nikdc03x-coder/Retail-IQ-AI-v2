"""
RetailIQ AI V2 ??? Phase 08: Operational Strategy Tests

Validates:
- Store strategic matrix 4-quadrant classification (Volume vs WAPE).
- Supply chain safety stock and labor flex calculations.
- Department merchandising and markdown clearance rules.
- Executive decision summary JSON structure and financial impact bounds.
- Report persistence in reports/strategy/.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.config import REPORT_SUBDIRS
from src.retailiq.strategy.decision_engine import (
    build_department_merchandising_rules,
    build_executive_decision_summary,
    build_resource_allocation_playbook,
    build_store_strategic_matrix,
)


@pytest.fixture(scope="module")
def strategy_dir() -> Path:
    """Return reports directory for operational strategy."""
    return REPORT_SUBDIRS["strategy"]


class TestStoreStrategicMatrix:
    """Verify store quadrant segmentation and quantitative directives."""

    def test_store_strategic_matrix_shape_and_tiers(self):
        df = build_store_strategic_matrix()

        assert len(df) == 45, f"Expected 45 stores, got {len(df)}"
        assert "Strategic_Tier" in df.columns
        assert "Recommended_Safety_Stock_Dollars" in df.columns
        assert "Weekly_Labor_Flex_Hours" in df.columns
        assert "Annual_Working_Capital_Optimization" in df.columns

        tiers = set(df["Strategic_Tier"].unique())
        expected_tiers = {
            "Tier 1: Core Performer",
            "Tier 2: Volatile Powerhouse",
            "Tier 3: Stable Mid-Market",
            "Tier 4: Turnaround Target",
        }
        assert tiers == expected_tiers, f"Unexpected tiers: {tiers}"

        # Safety stocks and annual optimization must be non-negative
        assert (df["Recommended_Safety_Stock_Dollars"] > 0).all()
        assert (df["Annual_Working_Capital_Optimization"] > 0).all()
        assert (df["Weekly_Labor_Flex_Hours"] >= 15).all()

    def test_scenario_sensitivity_recalculation(self):
        """Regression test for ISSUE-11-04: Verify parametric scenario recalculation logic."""
        import numpy as np
        df = build_store_strategic_matrix()
        baseline_savings = float(df["Annual_Working_Capital_Optimization"].sum())
        assert 12_000_000 <= baseline_savings <= 12_500_000, f"Expected ~$12.3M baseline, got {baseline_savings}"

        # Test conservative scenario (2.5% holding cost, 0.8x buffer)
        sim_buffer = df["Mean_Weekly_Sales"] * (df["Safety_Buffer_Pct"] * 0.8)
        conservative_savings = float(np.round(sim_buffer * 52 * 0.025, 2).sum())
        assert 5_000_000 <= conservative_savings <= 6_000_000, f"Conservative scenario out of range: {conservative_savings}"

        # Test optimistic scenario (6.5% holding cost, 1.2x buffer)
        sim_buffer_opt = df["Mean_Weekly_Sales"] * (df["Safety_Buffer_Pct"] * 1.2)
        optimistic_savings = float(np.round(sim_buffer_opt * 52 * 0.065, 2).sum())
        assert 20_000_000 <= optimistic_savings <= 22_500_000, f"Optimistic scenario out of range: {optimistic_savings}"


class TestResourceAllocationPlaybook:
    """Verify aggregated supply chain and labor allocation playbook."""

    def test_playbook_aggregation_invariants(self):
        matrix_df = build_store_strategic_matrix()
        playbook_df = build_resource_allocation_playbook(matrix_df)

        assert len(playbook_df) == 4, f"Expected 4 strategic tiers in playbook, got {len(playbook_df)}"
        assert playbook_df["Store_Count"].sum() == 45, "Store counts must sum to 45"
        assert 99.9 <= playbook_df["Share_Of_Chain_Sales_Pct"].sum() <= 100.1, "Sales share must sum to ~100%"
        assert playbook_df["Total_Safety_Stock_Dollars"].sum() > 4_000_000, "Safety stock should exceed $4M across chain"
        assert playbook_df["Total_Weekly_Flex_Hours"].sum() > 1_000, "Flex hours should exceed 1000 weekly"


class TestDepartmentMerchandisingRules:
    """Verify department categorization and clearance directives."""

    def test_department_rules_coverage_and_archetypes(self):
        dept_df = build_department_merchandising_rules()

        assert len(dept_df) == 81, f"Expected 81 departments, got {len(dept_df)}"
        assert "Merchandising_Archetype" in dept_df.columns
        assert "Recommended_Markdown_Cadence" in dept_df.columns
        assert "Return_Rate_Pct" in dept_df.columns

        archetypes = set(dept_df["Merchandising_Archetype"].unique())
        assert "Core Revenue Anchor" in archetypes
        assert "High-Return Risk" in archetypes
        assert "Long-Tail Candidate" in archetypes

        # Dept 47 or 54 should be tagged as High-Return Risk
        dept_47_row = dept_df[dept_df["Dept"] == 47]
        if not dept_47_row.empty:
            assert dept_47_row.iloc[0]["Merchandising_Archetype"] == "High-Return Risk"


class TestStrategyReportsPersistence:
    """Verify all report files exist on disk with valid content."""

    @pytest.mark.parametrize("filename", [
        "01_store_strategic_matrix.csv",
        "02_resource_allocation_playbook.csv",
        "03_department_markdown_rules.csv",
        "04_executive_decision_summary.json",
    ])
    def test_report_exists_and_non_empty(self, strategy_dir: Path, filename: str):
        target = strategy_dir / filename
        assert target.exists(), f"Missing strategy report: {filename}"
        assert target.stat().st_size > 0, f"Empty strategy report: {filename}"

    def test_summary_json_integrity(self, strategy_dir: Path):
        summary_path = strategy_dir / "04_executive_decision_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_stores_classified"] == 45
        assert summary["total_departments_analyzed"] == 81
        assert "financial_impact_projections" in summary
        assert summary["financial_impact_projections"]["estimated_annual_working_capital_optimization"] > 5_000_000
        assert len(summary["priority_executive_directives"]) >= 3

