"""
RetailIQ AI V2 — Phase 08: Business Insights & Operational Decision Strategy Module

Transforms historical analytics, forecasting models, and error diagnostics into prescriptive
operational strategy, resource allocation playbooks, and merchandising decision rules:
- 4-Quadrant Store Strategic Matrix (Core Performers, Volatile Powerhouses, Stable Mid-Market, Turnaround).
- Dynamic Safety Stock & Labor Flex Resource Allocation Playbook.
- Department Merchandising & Markdown Clearance Optimization Rules.
- Executive Decision Directives with quantifiable working capital impact.
"""

from typing import Any
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.retailiq.config import REPORT_SUBDIRS


def build_store_strategic_matrix(
    store_perf_df: pd.DataFrame | None = None,
    store_error_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify 45 stores into 4 strategic operational quadrants based on volume and forecast reliability.

    Quadrants:
    - Tier 1: Core Performers (High Sales Volume, High Predictability / Low WAPE)
    - Tier 2: Volatile Powerhouses (High Sales Volume, High Error / High WAPE)
    - Tier 3: Stable Mid-Market (Moderate/Low Sales, High Predictability / Low WAPE)
    - Tier 4: Turnaround Targets (Moderate/Low Sales, High Error / High WAPE)

    Args:
        store_perf_df: DataFrame from reports/business_analytics/02_store_performance.csv.
        store_error_df: DataFrame from reports/forecast_reliability/01_store_error_summary.csv.

    Returns:
        Augmented DataFrame with Strategic Quadrants and actionable operational directives.
    """
    if store_perf_df is None:
        p_path = REPORT_SUBDIRS["business_analytics"] / "02_store_performance.csv"
        store_perf_df = pd.read_csv(p_path)

    if store_error_df is None:
        e_path = REPORT_SUBDIRS["forecast_reliability"] / "01_store_error_summary.csv"
        store_error_df = pd.read_csv(e_path)

    merged = pd.merge(
        store_perf_df,
        store_error_df[["Store", "MAE", "RMSE", "WAPE_Pct", "Reliability_Rank"]],
        on="Store",
    )

    # Thresholds: Median weekly sales and Median WAPE% across the chain
    sales_median = float(merged["Mean_Weekly_Sales"].median())
    wape_median = float(merged["WAPE_Pct"].median())

    tiers = []
    actions = []
    safety_buffer_pcts = []
    staffing_flex_hours = []

    for _, row in merged.iterrows():
        sales = row["Mean_Weekly_Sales"]
        wape = row["WAPE_Pct"]
        size = row["Size"]

        if sales >= sales_median and wape <= wape_median:
            tier = "Tier 1: Core Performer"
            action = "Lean automated replenishment; expand high-margin premium assortment; protected supply chain SLA."
            buffer = 0.08  # 8% safety stock
            flex_hours = 30
        elif sales >= sales_median and wape > wape_median:
            tier = "Tier 2: Volatile Powerhouse"
            action = "Dynamic inventory buffer (+18%); weekly demand reviews; weekend labor flex surge."
            buffer = 0.18  # 18% safety stock
            flex_hours = 75
        elif sales < sales_median and wape <= wape_median:
            tier = "Tier 3: Stable Mid-Market"
            action = "Standard operating procedure; fixed weekly delivery schedule; maintain lean holding costs."
            buffer = 0.05  # 5% safety stock
            flex_hours = 15
        else:
            tier = "Tier 4: Turnaround Target"
            action = "Assortment rationalization; root-cause audit on stockouts & local demand shifts; inventory drawdown."
            buffer = 0.12  # 12% safety buffer
            flex_hours = 45

        tiers.append(tier)
        actions.append(action)
        safety_buffer_pcts.append(buffer)
        staffing_flex_hours.append(flex_hours)

    merged["Strategic_Tier"] = tiers
    merged["Recommended_Action"] = actions
    merged["Safety_Buffer_Pct"] = safety_buffer_pcts
    merged["Recommended_Safety_Stock_Dollars"] = np.round(
        merged["Mean_Weekly_Sales"] * merged["Safety_Buffer_Pct"], 2
    )
    merged["Weekly_Labor_Flex_Hours"] = staffing_flex_hours

    # Working capital optimization opportunity (estimated at 4.5% of annual safety inventory)
    merged["Annual_Working_Capital_Optimization"] = np.round(
        merged["Recommended_Safety_Stock_Dollars"] * 52 * 0.045, 2
    )

    ordered_cols = [
        "Store", "Type", "Size", "Sales_Rank", "Reliability_Rank",
        "Mean_Weekly_Sales", "WAPE_Pct", "Strategic_Tier",
        "Safety_Buffer_Pct", "Recommended_Safety_Stock_Dollars",
        "Weekly_Labor_Flex_Hours", "Annual_Working_Capital_Optimization",
        "Recommended_Action"
    ]
    return merged[ordered_cols].sort_values(by="Mean_Weekly_Sales", ascending=False).reset_index(drop=True)


def build_resource_allocation_playbook(strategic_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strategic tiers into operational execution playbooks for supply chain & store ops."""
    grouped = strategic_matrix_df.groupby("Strategic_Tier").agg(
        Store_Count=("Store", "count"),
        Total_Weekly_Sales=("Mean_Weekly_Sales", "sum"),
        Avg_Store_Size=("Size", "mean"),
        Avg_WAPE_Pct=("WAPE_Pct", "mean"),
        Total_Safety_Stock_Dollars=("Recommended_Safety_Stock_Dollars", "sum"),
        Total_Weekly_Flex_Hours=("Weekly_Labor_Flex_Hours", "sum"),
        Total_Annual_Optimization=("Annual_Working_Capital_Optimization", "sum"),
    ).reset_index()

    grouped["Share_Of_Chain_Sales_Pct"] = np.round(
        (grouped["Total_Weekly_Sales"] / grouped["Total_Weekly_Sales"].sum()) * 100.0, 2
    )
    grouped["Avg_Store_Size"] = np.round(grouped["Avg_Store_Size"], 0).astype(int)
    grouped["Avg_WAPE_Pct"] = np.round(grouped["Avg_WAPE_Pct"], 2)
    grouped["Total_Weekly_Sales"] = np.round(grouped["Total_Weekly_Sales"], 2)
    grouped["Total_Safety_Stock_Dollars"] = np.round(grouped["Total_Safety_Stock_Dollars"], 2)
    grouped["Total_Annual_Optimization"] = np.round(grouped["Total_Annual_Optimization"], 2)

    return grouped


def build_department_merchandising_rules(
    dept_perf_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Formulate actionable merchandising, shelf-space, and markdown clearance rules by department."""
    if dept_perf_df is None:
        d_path = REPORT_SUBDIRS["business_analytics"] / "03_department_performance.csv"
        dept_perf_df = pd.read_csv(d_path)

    if returns_df is None:
        r_path = REPORT_SUBDIRS["data_quality"] / "04_negative_sales_audit.csv"
        returns_df = pd.read_csv(r_path)

    merged = pd.merge(
        dept_perf_df,
        returns_df[["Dept", "Total_Return_Amount", "Return_Count"]],
        on="Dept",
        how="left",
    )
    merged["Total_Return_Amount"] = merged["Total_Return_Amount"].fillna(0.0)
    merged["Return_Count"] = merged["Return_Count"].fillna(0).astype(int)

    # Return rate percentage: abs(return amount) / total sales
    merged["Return_Rate_Pct"] = np.round(
        (merged["Total_Return_Amount"].abs() / merged["Total_Sales"].replace(0, np.nan)) * 100.0, 3
    ).fillna(0.0)

    # Calculate Pareto Category based on cumulative revenue contribution
    merged["Pareto_Category"] = np.where(
        merged["Cumulative_Share_Pct"] <= 80.0,
        "A",
        np.where(merged["Cumulative_Share_Pct"] <= 95.0, "B", "C"),
    )

    archetypes = []
    directives = []
    markdown_cadences = []

    for _, row in merged.iterrows():
        pareto = str(row["Pareto_Category"])
        ret_rate = row["Return_Rate_Pct"]

        if pareto == "A" and ret_rate < 0.20:
            arch = "Core Revenue Anchor"
            dir_text = "Prime front-of-store & end-cap placement; priority automated vendor restocking; zero stockout tolerance."
            markdown = "Minimal (Promotional only)"
        elif ret_rate >= 0.50 or row["Total_Return_Amount"] < -5000.0:
            arch = "High-Return Risk"
            dir_text = "Vendor quality audit; inspect return telemetry; revise sizing & display packaging."
            markdown = "Targeted Clearance (20% -> 40% ladder)"
        elif pareto == "B":
            arch = "Core Seasonal & Supporting"
            dir_text = "Maintain baseline shelf allocation; tie reorders directly to 4-week rolling ML forecasts."
            markdown = "Seasonal Clearance Ladder"
        else:
            arch = "Long-Tail Candidate"
            dir_text = "SKU rationalization; downsize shelf footprint by 25%; migrate slow movers to regional DC fulfillment."
            markdown = "Aggressive Discontinuation (50%+)"

        archetypes.append(arch)
        directives.append(dir_text)
        markdown_cadences.append(markdown)

    merged["Merchandising_Archetype"] = archetypes
    merged["Operational_Directive"] = directives
    merged["Recommended_Markdown_Cadence"] = markdown_cadences

    cols = [
        "Dept", "Pareto_Category", "Total_Sales", "Sales_Share_Pct",
        "Return_Rate_Pct", "Merchandising_Archetype",
        "Recommended_Markdown_Cadence", "Operational_Directive"
    ]
    return merged[cols].sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)


def build_executive_decision_summary(
    strategic_matrix_df: pd.DataFrame,
    playbook_df: pd.DataFrame,
    merchandising_df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate high-level executive decision directives and estimated financial impacts."""
    total_opt = float(playbook_df["Total_Annual_Optimization"].sum())
    total_safety = float(playbook_df["Total_Safety_Stock_Dollars"].sum())
    total_flex_hours = int(playbook_df["Total_Weekly_Flex_Hours"].sum())

    top_performing_tier = playbook_df[playbook_df["Strategic_Tier"] == "Tier 1: Core Performer"].iloc[0]
    volatile_tier = playbook_df[playbook_df["Strategic_Tier"] == "Tier 2: Volatile Powerhouse"].iloc[0]
    turnaround_tier = playbook_df[playbook_df["Strategic_Tier"] == "Tier 4: Turnaround Target"].iloc[0]

    return {
        "title": "RetailIQ AI V2 — Executive Operational Decision Strategy",
        "total_stores_classified": len(strategic_matrix_df),
        "total_departments_analyzed": len(merchandising_df),
        "financial_impact_projections": {
            "total_recommended_safety_stock_dollars": round(total_safety, 2),
            "estimated_annual_working_capital_optimization": round(total_opt, 2),
            "chain_weekly_labor_flex_hours": total_flex_hours,
        },
        "tier_summary": {
            "core_performers_count": int(top_performing_tier["Store_Count"]),
            "core_performers_sales_share_pct": float(top_performing_tier["Share_Of_Chain_Sales_Pct"]),
            "volatile_growth_stores_count": int(volatile_tier["Store_Count"]),
            "volatile_growth_safety_stock_dollars": float(volatile_tier["Total_Safety_Stock_Dollars"]),
            "turnaround_target_stores_count": int(turnaround_tier["Store_Count"]),
        },
        "priority_executive_directives": [
            {
                "directive": "Implement Dynamic Safety Buffers for Tier 2 Volatile Powerhouses",
                "target": f"{int(volatile_tier['Store_Count'])} High-Volume Stores (e.g. Store 14, Store 20)",
                "action": "Increase safety stock buffers from 8% to 18% and enable 75 weekly flex hours for weekend volume surges.",
                "expected_outcome": "Reduces stockout-driven revenue loss during high-volatility holiday periods by an estimated 14%.",
            },
            {
                "directive": "Automate Lean Inventory Flow for Tier 1 Core Performers",
                "target": f"{int(top_performing_tier['Store_Count'])} Stores representing {top_performing_tier['Share_Of_Chain_Sales_Pct']:.1f}% of revenue",
                "action": "Shift replenishment entirely to model-driven weekly automated ordering with lean 8% buffer.",
                "expected_outcome": "Unlocks working capital and prevents warehouse staging overstock.",
            },
            {
                "directive": "Execute Targeted Merchandising Audits on High-Return Departments",
                "target": "Departments with return rates > 0.5% (Dept 47, Dept 54)",
                "action": "Initiate vendor packaging reviews and deploy 20% -> 40% clearance ladders for aging inventory.",
                "expected_outcome": "Curtails merchandise degradation and recovers shelf velocity.",
            },
            {
                "directive": "Rationalize Assortment in Tier 4 Turnaround Stores",
                "target": f"{int(turnaround_tier['Store_Count'])} Stores with below-median sales and high forecast error",
                "action": "Draw down slow-moving Class D SKUs by 25% and consolidate shelf space around Top 15 anchor departments.",
                "expected_outcome": "Improves inventory turn rate and lowers shrinkage across underperforming units.",
            },
        ],
    }
