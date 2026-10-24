"""
RetailIQ AI V2 — Page 03: Business Analytics & Exploratory Data Analysis

Exploration of store performance rankings, department Pareto distributions,
promotional holiday demand lifts, and multi-year network revenue trajectories.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from src.retailiq.config import REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_percent,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import bar_chart, line_chart, scatter_chart
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Business Analytics page."""
    page_header(
        "Business Analytics & Commercial Intelligence",
        "Store Rankings, Department Pareto Dynamics, Holiday Surge Lifts, and Multi-Year Trajectories",
        category="Commercial Analytics",
    )

    page_context(
        why="Enterprise forecasting requires deep empirical knowledge of store-level hierarchies, department contribution concentration, and calendar seasonality.",
        business_problem="Identifies top revenue generating locations, Pareto concentration in product categories, and promotional uplift across holidays.",
        data_source="Pre-computed analytical aggregations in reports/business_analytics/ generated from 421,570 observations.",
        what_you_learn="Store 20 and Store 4 lead network revenue, top 10 departments generate 48.1% of sales, and Thanksgiving drives a +41.3% promotional surge.",
    )

    out_dir = REPORT_SUBDIRS["business_analytics"]

    if not (out_dir / "02_store_performance.csv").exists():
        st.warning(
            "Analytics reports not found. Please run the analytics pipeline:\n\n"
            "`python scripts/03_business_analytics/run_analytics.py`"
        )
        return

    @st.cache_data
    def load_analytics_data() -> dict:
        network = pd.read_csv(out_dir / "01_weekly_sales.csv", parse_dates=["Date"])
        stores = pd.read_csv(out_dir / "02_store_performance.csv")
        depts = pd.read_csv(out_dir / "03_department_performance.csv")
        store_week = pd.read_csv(out_dir / "04_store_week_sales.csv", parse_dates=["Date"])
        holidays = pd.read_csv(out_dir / "05_holiday_comparison.csv")
        store_types = pd.read_csv(out_dir / "06_store_type_summary.csv")
        monthly = pd.read_csv(out_dir / "07_monthly_seasonality.csv")
        return {
            "network": network,
            "stores": stores,
            "depts": depts,
            "store_week": store_week,
            "holidays": holidays,
            "store_types": store_types,
            "monthly": monthly,
        }

    data = load_analytics_data()
    network_df = data["network"]
    stores_df = data["stores"]
    depts_df = data["depts"]
    store_week_df = data["store_week"]
    holidays_df = data["holidays"]
    store_types_df = data["store_types"]
    monthly_df = data["monthly"]

    top_store = stores_df.iloc[0]
    top_dept = depts_df.iloc[0]
    thanksgiving_row = holidays_df[holidays_df["Holiday_Name"] == "Thanksgiving"].iloc[0]

    # ── Top KPI Row ──
    render_kpi_row([
        {"label": "Total Network Revenue", "value": format_currency(network_df["Total_Network_Sales"].sum()), "subtitle": "143 recorded weeks (2010-2012)"},
        {"label": "Avg Store-Week Revenue", "value": format_currency(stores_df["Mean_Weekly_Sales"].mean()), "subtitle": "Across 45 store locations"},
        {"label": "Top Store Location", "value": f"Store {int(top_store['Store'])}", "subtitle": f"{format_currency(top_store['Total_Sales'])} total (Type {top_store['Type']})"},
        {"label": "Top Revenue Category", "value": f"Dept {int(top_dept['Dept'])}", "subtitle": f"{top_dept['Sales_Share_Pct']}% network share"},
        {"label": "Peak Thanksgiving Lift", "value": f"+{thanksgiving_row['Lift_Vs_Baseline_Pct']:.1f}%", "subtitle": "Surge vs non-holiday baseline"},
    ])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Store Performance & Formats",
        "Department Contribution",
        "Holiday Dynamics & Seasonality",
        "Network Time Series",
    ])

    # ── Tab 1: Store Performance ──
    with tab1:
        section_header("Store Revenue Rankings & Space Productivity", "Comparative performance across 45 stores and 3 operational formats")
        context_box(
            "Store-Week aggregation isolates true operational capacity. Type A supercenters account for 64.3% of total network revenue, driven by higher square footage and broader assortment.",
            title="Format Productivity",
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            top_n = st.slider("Display Store Count", min_value=5, max_value=45, value=15, step=5, key="store_top_n")
            top_stores = stores_df.head(top_n).sort_values("Total_Sales", ascending=True)
            top_stores["Store_Label"] = "Store " + top_stores["Store"].astype(str)

            fig_stores = bar_chart(
                top_stores,
                x="Total_Sales",
                y="Store_Label",
                title=f"Top {top_n} Stores by Cumulative Revenue",
                orientation="h",
                color="Type",
            )
            st.plotly_chart(fig_stores, use_container_width=True)

        with c2:
            st.markdown("#### Store Format Benchmarks")
            formatted_dataframe(
                store_types_df,
                title="Format Economics (Type A, B, C)",
                description="Comparative metrics for supercenter, regional, and compact store models.",
            )

            fig_scatter = scatter_chart(
                stores_df,
                x="Size",
                y="Total_Sales",
                color="Type",
                title="Physical Square Footage vs Cumulative Revenue",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        formatted_dataframe(
            stores_df,
            title="Complete 45-Store Performance Ranking",
            description="Rankings, weekly averages, square footage productivity, and sales coefficient of variation.",
        )

    # ── Tab 2: Department Contribution ──
    with tab2:
        section_header("Department Revenue Pareto Shares", "Category concentration and revenue Pareto distribution")
        context_box(
            "Department 92 (Groceries) is the top revenue generator ($483.9M), followed by Dept 95 ($449.3M) and Dept 38 ($393.1M). The top 10 departments generate 48.1% of all Walmart sales.",
            title="Pareto Concentration",
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            top_depts = depts_df.head(15).sort_values("Total_Sales", ascending=True)
            top_depts["Dept_Label"] = "Dept " + top_depts["Dept"].astype(str)
            fig_dept = bar_chart(
                top_depts,
                x="Total_Sales",
                y="Dept_Label",
                title="Top 15 Departments by Cumulative Revenue ($)",
                orientation="h",
            )
            st.plotly_chart(fig_dept, use_container_width=True)

        with c2:
            top_10 = depts_df.head(10)[["Dept_Rank", "Dept", "Total_Sales", "Sales_Share_Pct", "Cumulative_Share_Pct"]]
            formatted_dataframe(
                top_10,
                title="Top 10 Anchor Categories",
                description="Cumulative share metrics for the primary commercial revenue drivers.",
            )
            context_box(
                "Retail inventory and replenishment strategies should prioritize core high-contribution departments while tracking return friction in volatile categories.",
                title="Merchandising Rationale",
            )

        formatted_dataframe(
            depts_df,
            title="All 81 Departments Performance Directory",
            description="Complete department sales, mean weekly sales, and cumulative share percentages.",
        )

    # ── Tab 3: Holiday Dynamics & Seasonality ──
    with tab3:
        section_header("Walmart Holiday Surges & Annual Seasonality", "Empirical sales uplift across promotional holiday weeks")
        context_box(
            "Holiday weeks generate a +7.84% aggregate sales lift. Thanksgiving delivers the highest single promotional surge (+41.30% lift), while Christmas week records a minor -2.61% dip due to holiday store closures and shopping pull-forward into mid-December.",
            title="Promotional Calendar",
        )

        c1, c2 = st.columns(2)
        with c1:
            holiday_plot = holidays_df[holidays_df["Holiday_Name"] != "Non-Holiday Baseline"].copy()
            fig_holidays = bar_chart(
                holiday_plot,
                x="Holiday_Name",
                y="Lift_Vs_Baseline_Pct",
                title="Sales Lift Percentage vs Non-Holiday Baseline",
            )
            st.plotly_chart(fig_holidays, use_container_width=True)

        with c2:
            fig_month = line_chart(
                monthly_df,
                x="Month_Name",
                y="Mean_Weekly_Sales",
                title="Average Weekly Network Sales by Month ($)",
            )
            st.plotly_chart(fig_month, use_container_width=True)

        formatted_dataframe(
            holidays_df,
            title="Promotional Holiday Surge Metrics",
            description="Mean weekly store sales, median sales, recorded weeks, and lift vs non-holiday benchmark.",
        )

    # ── Tab 4: Network Time Series ──
    with tab4:
        section_header("143-Week Network Revenue Trajectory (2010 - 2012)", "Macro time-series sales patterns across 45 stores")
        context_box(
            "Network weekly sales aggregated across all 45 stores. Clear annual dual-spikes appear in late November (Thanksgiving / Black Friday) and mid-December (pre-Christmas rush).",
            title="Macro Seasonality",
        )

        fig_net = line_chart(
            network_df,
            x="Date",
            y="Total_Network_Sales",
            title="Total Walmart Weekly Sales (Chain-Wide)",
        )
        st.plotly_chart(fig_net, use_container_width=True)

        st.markdown("#### Compare Individual Store Trajectories")
        selected_stores_ts = st.multiselect(
            "Select Store Locations to Compare",
            options=sorted(store_week_df["Store"].unique()),
            default=[1, 20, 33],
            key="ts_stores_sel",
        )

        if selected_stores_ts:
            ts_filtered = store_week_df[store_week_df["Store"].isin(selected_stores_ts)].copy()
            ts_filtered["Store"] = "Store " + ts_filtered["Store"].astype(str)
            fig_stores_ts = line_chart(
                ts_filtered,
                x="Date",
                y="Weekly_Sales",
                color="Store",
                title="Weekly Sales Trajectory by Store Location",
            )
            st.plotly_chart(fig_stores_ts, use_container_width=True)
