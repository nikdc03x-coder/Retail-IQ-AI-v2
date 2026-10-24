"""
RetailIQ AI V2 — Page 08: Business Insights & Operational Decision Strategy

Prescriptive decision intelligence translating sales forecasting models, reliability metrics,
and historical analytics into actionable operational directives:
- 4-Quadrant Store Strategic Matrix (Volume vs Forecast Reliability).
- Safety Stock Buffers & Dynamic Labor Flex Resource Allocation Playbook.
- Department Merchandising & Markdown Clearance Optimization Rules.
- Executive Directives with quantifiable working capital impact.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.retailiq.config import REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_percent,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import apply_default_layout, bar_chart
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Business Insights & Operational Decision Strategy page."""
    page_header(
        "Operational Decision Strategy",
        "Prescriptive Intelligence: Store Quadrants, Dynamic Buffers & Merchandising Directives",
    )

    page_context(
        why="Predictive forecasts and explainability attributions only generate ROI when translated into concrete operational actions. This decision engine prescribes exact inventory buffers, labor staffing allocations, and markdown ladders.",
        business_problem="Answers: Which stores require dynamic inventory buffering vs lean replenishment? Where should flexible labor hours be allocated? Which departments need immediate packaging or markdown clearance?",
        data_source="Pre-computed decision reports in reports/strategy/ synthesized across analytical and model reliability modules.",
        what_you_learn="15 Core Performers generate 44.5% of chain revenue with low error; 8 Volatile Powerhouses require +18% safety stock; total working capital optimization opportunity exceeds $12.3M annually.",
    )

    out_dir = REPORT_SUBDIRS["strategy"]

    # Verify reports exist
    if not (out_dir / "01_store_strategic_matrix.csv").exists():
        st.warning(
            "⚠️ Strategy reports not found. Please run the execution script:\n\n"
            "`python scripts/08_strategy/run_strategy_engine.py`"
        )
        return

    @st.cache_data
    def load_strategy_data() -> dict:
        matrix = pd.read_csv(out_dir / "01_store_strategic_matrix.csv")
        playbook = pd.read_csv(out_dir / "02_resource_allocation_playbook.csv")
        merch = pd.read_csv(out_dir / "03_department_markdown_rules.csv")
        with open(out_dir / "04_executive_decision_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {
            "matrix": matrix,
            "playbook": playbook,
            "merch": merch,
            "summary": summary,
        }

    data = load_strategy_data()
    matrix_df = data["matrix"]
    playbook_df = data["playbook"]
    merch_df = data["merch"]
    summary = data["summary"]

    fin = summary["financial_impact_projections"]
    tier_meta = summary["tier_summary"]

    # ── Top KPI Row ──
    render_kpi_row([
        {"label": "Core Performers (Tier 1)", "value": f"{tier_meta['core_performers_count']} Stores", "subtitle": f"{tier_meta['core_performers_sales_share_pct']:.1f}% chain volume"},
        {"label": "Volatile Powerhouses", "value": f"{tier_meta['volatile_growth_stores_count']} Stores", "subtitle": f"+18% dynamic buffer recommended"},
        {"label": "Turnaround Targets", "value": f"{tier_meta['turnaround_target_stores_count']} Stores", "subtitle": "Priority SKU rationalization"},
        {"label": "Working Capital Optimization", "value": format_currency(fin['estimated_annual_working_capital_optimization']), "subtitle": "Annual holding cost savings"},
        {"label": "Chain Weekly Labor Flex", "value": f"{fin['chain_weekly_labor_flex_hours']:,} Hours", "subtitle": "Dynamic weekend capacity surge"},
    ])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Store Strategic Quadrants",
        "📦 Inventory & Labor Playbook",
        "🏷️ Department Merchandising Rules",
        "📋 Executive Action Plan",
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: Store Strategic Quadrants
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        section_header("Store Strategic Matrix", "Classification by Weekly Revenue Volume vs. Forecast Error (WAPE %)")

        c_filter, _ = st.columns([1, 2])
        with c_filter:
            tier_options = ["All Tiers"] + sorted(matrix_df["Strategic_Tier"].unique().tolist())
            selected_tier = st.selectbox("Filter Strategic Tier", options=tier_options, index=0)

        # Plotly 2D Scatter Matrix
        color_map = {
            "Tier 1: Core Performer": Theme.CYAN,
            "Tier 2: Volatile Powerhouse": Theme.CORAL,
            "Tier 3: Stable Mid-Market": Theme.VIOLET,
            "Tier 4: Turnaround Target": Theme.AMBER,
        }

        fig_matrix = px.scatter(
            matrix_df,
            x="WAPE_Pct",
            y="Mean_Weekly_Sales",
            color="Strategic_Tier",
            size="Size",
            hover_data=["Store", "Type", "Sales_Rank", "Reliability_Rank"],
            labels={
                "WAPE_Pct": "Forecast Error (WAPE %)",
                "Mean_Weekly_Sales": "Mean Weekly Sales ($)",
                "Strategic_Tier": "Strategic Quadrant",
            },
            color_discrete_map=color_map,
        )

        # Add median boundary lines
        sales_med = float(matrix_df["Mean_Weekly_Sales"].median())
        wape_med = float(matrix_df["WAPE_Pct"].median())

        fig_matrix.add_hline(y=sales_med, line_dash="dot", line_color=Theme.GRAY, opacity=0.6)
        fig_matrix.add_vline(x=wape_med, line_dash="dot", line_color=Theme.GRAY, opacity=0.6)

        fig_matrix = apply_default_layout(fig_matrix, "Store Strategic Positioning (Size reflects store sq footage)")
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Filtered Table
        view_df = matrix_df if selected_tier == "All Tiers" else matrix_df[matrix_df["Strategic_Tier"] == selected_tier]
        display_cols = [
            "Store", "Type", "Size", "Sales_Rank", "Mean_Weekly_Sales",
            "WAPE_Pct", "Strategic_Tier", "Safety_Buffer_Pct",
            "Recommended_Safety_Stock_Dollars", "Weekly_Labor_Flex_Hours",
            "Recommended_Action"
        ]
        formatted_dataframe(
            view_df[display_cols],
            title=f"Store Strategy Directory ({len(view_df)} Stores)",
            description="Operational recommendations and dynamic inventory buffer requirements per store.",
        )

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: Resource Allocation & Safety Buffer Calculator
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        section_header("Supply Chain & Labor Resource Playbook", "Aggregate Requirements Across Operational Tiers")

        # Playbook summary table
        formatted_dataframe(
            playbook_df,
            title="Operational Playbook Matrix by Strategic Tier",
            description="Aggregated sales volume, safety stock capital, and labor flex hours.",
        )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        section_header("Interactive Store Safety Stock & Labor Calculator", "Store-Specific Resource Allocator")

        calc_store = st.selectbox(
            "Select Store for Operational Prescription",
            options=sorted(matrix_df["Store"].unique()),
            index=13,  # Store 14 default
        )
        s_row = matrix_df[matrix_df["Store"] == calc_store].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Store Tier", s_row["Strategic_Tier"].split(":")[1].strip())
        with c2:
            st.metric("Recommended Safety Buffer", f"{s_row['Safety_Buffer_Pct']*100:.0f}%", f"{format_currency(s_row['Recommended_Safety_Stock_Dollars'])}")
        with c3:
            st.metric("Weekly Staffing Flex", f"{s_row['Weekly_Labor_Flex_Hours']} Hours", "Surge capacity")
        with c4:
            st.metric("Capital Optimization", format_currency(s_row["Annual_Working_Capital_Optimization"]), "Estimated annual gain")

        context_box(
            f"📋 Strategic Directive for Store {calc_store}: {s_row['Recommended_Action']}"
        )

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: Department Merchandising Rules
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        section_header("Department Merchandising & Markdown Clearance Rules", "Category Archetypes & Shelf Strategy")

        c_arch, _ = st.columns([1, 2])
        with c_arch:
            arch_opts = ["All Archetypes"] + sorted(merch_df["Merchandising_Archetype"].unique().tolist())
            selected_arch = st.selectbox("Filter Merchandising Archetype", options=arch_opts, index=0)

        # Bar chart of archetype distribution
        arch_dist = merch_df["Merchandising_Archetype"].value_counts().reset_index()
        arch_dist.columns = ["Archetype", "Count"]

        fig_arch = px.bar(
            arch_dist,
            x="Count",
            y="Archetype",
            orientation="h",
            color="Archetype",
            color_discrete_sequence=[Theme.CYAN, Theme.VIOLET, Theme.AMBER, Theme.CORAL],
            text="Count",
        )
        fig_arch.update_traces(textposition="outside")
        fig_arch.update_layout(showlegend=False)
        fig_arch = apply_default_layout(fig_arch, "Department Breakdown by Merchandising Archetype")
        st.plotly_chart(fig_arch, use_container_width=True)

        m_view = merch_df if selected_arch == "All Archetypes" else merch_df[merch_df["Merchandising_Archetype"] == selected_arch]
        formatted_dataframe(
            m_view,
            title=f"Department Merchandising Rules ({len(m_view)} Departments)",
            description="Clearance cadences, return rates, and shelf-space directives.",
        )

    # ══════════════════════════════════════════════════════════════════
    # TAB 4: Executive Action Plan
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        section_header("Executive Strategic Action Plan", "Top Immediate Leadership Directives")

        directives = summary["priority_executive_directives"]
        for idx, d in enumerate(directives, 1):
            st.markdown(f"""
            <div style="background-color: {Theme.DARK_CARD}; border: 1px solid {Theme.CARD_BORDER};
                        border-left: 4px solid {Theme.CYAN}; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4 style="color: {Theme.WHITE}; margin: 0; font-size: 1.05rem;">
                        {idx}. {d['directive']}
                    </h4>
                    <span style="background-color: rgba(99, 102, 241, 0.2); color: {Theme.CYAN};
                                 padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                        {d['target']}
                    </span>
                </div>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem; line-height: 1.5; margin: 0 0 0.5rem 0;">
                    <b>Prescribed Action:</b> {d['action']}
                </p>
                <p style="color: {Theme.GREEN}; font-size: 0.82rem; font-weight: 500; margin: 0;">
                    🎯 <b>Expected Outcome:</b> {d['expected_outcome']}
                </p>
            </div>
            """, unsafe_allow_html=True)
