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
        category="Prescriptive Strategy & Decision Engine",
    )

    page_context(
        why="Predictive forecasts and explainability attributions only generate ROI when translated into concrete operational actions. This decision engine prescribes inventory buffers, labor staffing allocations, and markdown ladders.",
        business_problem="Answers: Which stores require dynamic inventory buffering vs lean replenishment? Where should flexible labor hours be allocated? How do holding cost assumptions impact simulated working capital optimization?",
        data_source="Pre-computed decision reports in reports/strategy/ synthesized across analytical and model reliability modules.",
        what_you_learn="15 Core Performers generate 44.5% of chain revenue with low error; 8 Volatile Powerhouses require +18% safety stock; simulated working capital optimization reaches $12.3M under baseline 4.5% holding cost assumptions.",
    )

    out_dir = REPORT_SUBDIRS["strategy"]

    # Verify reports exist
    if not (out_dir / "01_store_strategic_matrix.csv").exists():
        st.warning(
            "Strategy reports not found. Please run the execution script:\n\n"
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
        {"label": "Simulated Savings (Baseline)", "value": format_currency(fin['estimated_annual_working_capital_optimization']), "subtitle": "Scenario estimate (4.5% holding cost)"},
        {"label": "Chain Weekly Labor Flex", "value": f"{fin['chain_weekly_labor_flex_hours']:,} Hours", "subtitle": "Dynamic weekend surge capacity"},
    ])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab_scen, tab2, tab3, tab4 = st.tabs([
        "Store Strategic Quadrants",
        "Business Scenario Analysis",
        "Inventory & Labor Playbook",
        "Department Merchandising Rules",
        "Executive Action Plan",
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
    # TAB: Business Scenario Analysis (Interactive Simulator)
    # ══════════════════════════════════════════════════════════════════
    with tab_scen:
        section_header(
            "Simulated Potential Savings — Scenario Estimate",
            "Parametric working capital sensitivity model across inventory holding cost rates and buffer policies",
        )
        context_box(
            "<b>Transparency & Audit Qualification (ISSUE-11-04):</b> This model produces a <i>simulated heuristic projection</i> based on parameterized inventory holding cost rates and safety stock buffers applied to historical mean weekly sales. "
            "It does <b>not</b> represent confirmed, realized, or balance-sheet cash savings. The actual Walmart Kaggle dataset contains sales and promotions, not real-time warehouse inventory ledgers. "
            "Use this tool to stress-test how cost-of-capital fluctuations impact theoretical working capital optimization.",
            title="Model Nature & Limitations",
        )

        col_preset, col_ctrl = st.columns([1, 2])
        with col_preset:
            preset = st.radio(
                "Select Strategy Scenario Preset",
                options=[
                    "Baseline Audited (4.5% Cost, 1.0x Buffer)",
                    "Conservative (2.5% Cost, 0.8x Buffer)",
                    "Optimistic (6.5% Cost, 1.2x Buffer)",
                    "Custom Parameter Mode",
                ],
                index=0,
                key="scenario_preset_sel",
            )

        if "Baseline" in preset:
            default_holding_cost = 4.5
            default_buffer_mult = 1.0
            slider_disabled = True
        elif "Conservative" in preset:
            default_holding_cost = 2.5
            default_buffer_mult = 0.8
            slider_disabled = True
        elif "Optimistic" in preset:
            default_holding_cost = 6.5
            default_buffer_mult = 1.2
            slider_disabled = True
        else:
            default_holding_cost = 4.5
            default_buffer_mult = 1.0
            slider_disabled = False

        with col_ctrl:
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                active_cost = st.slider(
                    "Annual Holding Cost Rate (%)",
                    min_value=1.0,
                    max_value=12.0,
                    value=float(default_holding_cost),
                    step=0.5,
                    disabled=slider_disabled,
                    help="Assumed annual carrying cost of safety inventory (cost of capital, shrinkage, warehouse holding). Default baseline: 4.5%.",
                    key="slider_holding_cost",
                )
            with c_s2:
                active_mult = st.slider(
                    "Safety Buffer Intensity Multiplier",
                    min_value=0.5,
                    max_value=2.0,
                    value=float(default_buffer_mult),
                    step=0.1,
                    disabled=slider_disabled,
                    help="Scales recommended safety buffer percentages (Tier 1: 8%*M, Tier 2: 18%*M, Tier 3: 5%*M, Tier 4: 12%*M).",
                    key="slider_buffer_mult",
                )

        # Dynamic Recalculation
        sim_df = matrix_df.copy()
        sim_df["Sim_Buffer_Pct"] = sim_df["Safety_Buffer_Pct"] * active_mult
        sim_df["Sim_Safety_Stock_Dollars"] = np.round(sim_df["Mean_Weekly_Sales"] * sim_df["Sim_Buffer_Pct"], 2)
        sim_df["Sim_Annual_Optimization"] = np.round(sim_df["Sim_Safety_Stock_Dollars"] * 52 * (active_cost / 100.0), 2)

        total_sim_savings = float(sim_df["Sim_Annual_Optimization"].sum())
        total_sim_buffer = float(sim_df["Sim_Safety_Stock_Dollars"].sum())
        baseline_savings = float(fin["estimated_annual_working_capital_optimization"])
        savings_delta = total_sim_savings - baseline_savings
        delta_pct = (savings_delta / baseline_savings * 100.0) if baseline_savings > 0 else 0.0

        # Scenario KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Simulated Potential Savings",
                format_currency(total_sim_savings),
                delta=f"{savings_delta:+,.0f} ({delta_pct:+.1f}%)" if abs(savings_delta) > 100 else "Baseline Level",
                delta_color="normal",
            )
        with m2:
            st.metric(
                "Audited Baseline Estimate",
                format_currency(baseline_savings),
                "Heuristic benchmark (4.5%)",
                delta_color="off",
            )
        with m3:
            st.metric(
                "Total Recommended Safety Stock",
                format_currency(total_sim_buffer),
                f"Across {len(sim_df)} store networks",
                delta_color="off",
            )
        with m4:
            st.metric(
                "Effective Holding Cost Rate",
                f"{active_cost:.1f}%",
                f"Buffer mult: {active_mult:.1f}x",
                delta_color="off",
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Plotly Sensitivity Curve
        col_chart, col_assump = st.columns([3, 2])
        with col_chart:
            cost_range = np.linspace(1.0, 12.0, 23)
            curve_savings = [total_sim_buffer * 52 * (c / 100.0) for c in cost_range]
            sens_df = pd.DataFrame({"Holding_Cost_Pct": cost_range, "Simulated_Savings": curve_savings})

            fig_sens = px.line(
                sens_df,
                x="Holding_Cost_Pct",
                y="Simulated_Savings",
                title="Working Capital Optimization Sensitivity to Holding Cost Rate",
                labels={"Holding_Cost_Pct": "Holding Cost Rate (%)", "Simulated_Savings": "Simulated Potential Savings ($)"},
            )
            fig_sens.add_vline(x=active_cost, line_dash="dash", line_color=Theme.CORAL, annotation_text=f"Active: {active_cost:.1f}%")
            fig_sens.add_hline(y=total_sim_savings, line_dash="dot", line_color=Theme.CYAN, annotation_text=f"${total_sim_savings/1e6:.2f}M")
            fig_sens = apply_default_layout(fig_sens, "Holding Cost Sensitivity Curve ($1M - $35M)")
            st.plotly_chart(fig_sens, use_container_width=True)

        with col_assump:
            st.markdown(f"""
            <div class="section-card">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Scenario Assumptions & Governance</div>
                <h3>Economic Model Parameters</h3>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.82rem; margin-bottom: 0.35rem;">
                    <b>Holding Cost Formula:</b> Savings = &sum;(Mean Weekly Sales &times; Buffer Pct &times; 52 &times; Holding Cost Rate).
                </p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.82rem; margin-bottom: 0.35rem;">
                    <b>Linear Sensitivity:</b> Because savings scale linearly with cost rate, every 1% increase in holding cost adds approx $2.73M to projected optimization.
                </p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.82rem; margin-bottom: 0.35rem;">
                    <b>Is Higher or Lower Preferred?</b> Lower holding costs reflect leaner operational expense; higher savings projections indicate greater theoretical upside from buffer discipline.
                </p>
                <p style="color: {Theme.DARK_GRAY}; font-size: 0.78rem; margin: 0;">
                    <b>Limitations:</b> This simulation assumes uniform carrying costs across departments and does not model vendor lead-time variability or stockout penalty costs.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Assumptions Table
        assumptions_data = pd.DataFrame([
            {"Parameter": "Annual Inventory Holding Cost Rate", "Baseline Value": "4.5%", "Active Value": f"{active_cost:.1f}%", "Domain Rationale": "Represents cost of capital, insurance, shrinkage, and warehouse holding delta."},
            {"Parameter": "Tier 1 (Core Performer) Buffer", "Baseline Value": "8.0%", "Active Value": f"{8.0*active_mult:.1f}%", "Domain Rationale": "High sales, low forecast error (predictable replenishment)."},
            {"Parameter": "Tier 2 (Volatile Powerhouse) Buffer", "Baseline Value": "18.0%", "Active Value": f"{18.0*active_mult:.1f}%", "Domain Rationale": "High sales, high forecast error (requires dynamic buffer against surges)."},
            {"Parameter": "Tier 3 (Stable Mid-Market) Buffer", "Baseline Value": "5.0%", "Active Value": f"{5.0*active_mult:.1f}%", "Domain Rationale": "Low sales, low forecast error (lean standard replenishment)."},
            {"Parameter": "Tier 4 (Turnaround Target) Buffer", "Baseline Value": "12.0%", "Active Value": f"{12.0*active_mult:.1f}%", "Domain Rationale": "Low sales, high forecast error (drawdown buffer pending assortment cleanup)."},
        ])
        formatted_dataframe(
            assumptions_data,
            title="Active Scenario Parameter Assumptions Table",
            description="Comparison of baseline audited assumptions against currently simulated scenario parameters.",
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
            f"Strategic Directive for Store {calc_store}: {s_row['Recommended_Action']}"
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
            <div style="background-color: {Theme.CARD_BG}; border: 1px solid {Theme.CARD_BORDER};
                        border-left: 4px solid {Theme.BRAND}; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div style="color: {Theme.WHITE}; font-size: 1rem; font-weight: 600;">
                        {idx}. {d['directive']}
                    </div>
                    <span style="background-color: rgba(99, 102, 241, 0.15); color: {Theme.BRAND}; border: 1px solid rgba(99, 102, 241, 0.3);
                                 padding: 0.2rem 0.65rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                        {d['target']}
                    </span>
                </div>
                <div style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem; line-height: 1.5; margin: 0 0 0.6rem 0;">
                    <strong style="color: {Theme.WHITE};">Prescribed Action:</strong> {d['action']}
                </div>
                <div style="color: {Theme.GREEN}; font-size: 0.85rem; font-weight: 500;">
                    <strong>Expected Outcome:</strong> {d['expected_outcome']}
                </div>
            </div>
            """, unsafe_allow_html=True)

