"""
RetailIQ AI V2 — Page 03: Business Intelligence Analytics

Interactive business analytics exploring store rankings, department Pareto shares,
Walmart holiday surges, and 143-week network time series trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.retailiq.config import REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_number,
    format_percent,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import (
    apply_default_layout,
    bar_chart,
    line_chart,
    scatter_chart,
)
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Business Intelligence Analytics page."""
    page_header(
        "Business Intelligence Analytics",
        "Empirical Store Rankings, Department Contribution, Holiday Surges & Network Trends",
    )

    page_context(
        why="Enterprise sales data must be aggregated at unambiguous hierarchical levels (network, store, department) to inform operational planning and resource allocation.",
        business_problem="Answers: Which stores and departments drive revenue? How large is the holiday demand surge? How does store format impact space productivity?",
        data_source="Pre-computed analytical reports in reports/business_analytics/ aggregated from historical Walmart records.",
        what_you_learn="Store 20 is the #1 revenue driver ($301M); Department 92 accounts for 7.2% of sales; Thanksgiving delivers a +41.3% demand surge.",
    )

    out_dir = REPORT_SUBDIRS["business_analytics"]

    # Verify reports exist
    if not (out_dir / "02_store_performance.csv").exists():
        st.warning(
            "⚠️ Business analytics reports not found. Please run the analytics pipeline:\n\n"
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
        store_types = pd.read_csv(out_dir / "06_store_type_performance.csv")
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
        {"label": "#1 Ranked Store", "value": f"Store {int(top_store['Store'])}", "subtitle": f"{format_currency(top_store['Total_Sales'])} total (Type {top_store['Type']})"},
        {"label": "#1 Ranked Department", "value": f"Dept {int(top_dept['Dept'])}", "subtitle": f"{top_dept['Sales_Share_Pct']}% network share"},
        {"label": "Peak Thanksgiving Lift", "value": f"+{thanksgiving_row['Lift_Vs_Baseline_Pct']:.1f}%", "subtitle": "Surge vs non-holiday average"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏬 Store Performance & Formats",
        "📦 Department Contribution",
        "🎉 Holiday Surges & Seasonality",
        "📈 Network Time Series",
    ])

    # ── Tab 1: Store Performance ──
    with tab1:
        section_header("Store Revenue Rankings & Space Productivity", "🏬")
        context_box(
            "**Aggregation Level: Store-Week.** We evaluate total store revenue, mean weekly revenue, and space productivity (sales per square foot). "
            "Type A supercenters account for 64.3% of total revenue."
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            top_n = st.slider("Number of stores to display", min_value=5, max_value=45, value=15, step=5, key="store_top_n")
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
            st.markdown("#### Store Format Benchmarks (Type A, B, C)")
            st.dataframe(store_types_df, use_container_width=True, hide_index=True)

            fig_scatter = scatter_chart(
                stores_df,
                x="Size",
                y="Total_Sales",
                color="Type",
                title="Store Size (Sq Ft) vs Cumulative Sales ($)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        formatted_dataframe(
            stores_df,
            title="Complete 45-Store Performance Ranking",
            description="Rankings, weekly averages, square footage productivity, and sales coefficient of variation.",
        )

    # ── Tab 2: Department Contribution ──
    with tab2:
        section_header("Department Revenue Pareto Shares", "📦")
        context_box(
            "**Aggregation Level: Department.** Department 92 (Groceries) is the top revenue generator ($483.9M), followed by Dept 95 ($449.3M) and Dept 38 ($393.1M). "
            "The top 10 departments generate **48.1% of all Walmart sales**."
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            top_depts = depts_df.head(15).sort_values("Total_Sales", ascending=True)
            top_depts["Dept_Label"] = "Dept " + top_depts["Dept"].astype(str)
            fig_dept = bar_chart(
                top_depts,
                x="Total_Sales",
                y="Dept_Label",
                title="Top 15 Departments by Revenue ($)",
                orientation="h",
            )
            st.plotly_chart(fig_dept, use_container_width=True)

        with c2:
            st.markdown("#### Top 10 Revenue Generators (Pareto Analysis)")
            top_10 = depts_df.head(10)[["Dept_Rank", "Dept", "Total_Sales", "Sales_Share_Pct", "Cumulative_Share_Pct"]]
            st.dataframe(top_10, use_container_width=True, hide_index=True)
            context_box(
                "**Business Takeaway:** Retail inventory and merchandising strategies should prioritize core high-contribution departments while tracking returns in volatile categories."
            )

        formatted_dataframe(
            depts_df,
            title="All 81 Departments Performance Table",
            description="Complete department sales, mean weekly sales, and cumulative share percentages.",
        )

    # ── Tab 3: Holiday Dynamics & Seasonality ──
    with tab3:
        section_header("Walmart Holiday Surges & Annual Seasonality", "🎉")
        context_box(
            "**Key Finding:** Holiday weeks generate a **+7.84% aggregate sales lift**. "
            "Thanksgiving delivers the highest surge (**+41.30% lift**), while Christmas week records a **-2.61% dip** (due to store closures on Dec 25 and early shopping pull-forward into mid-December)."
        )

        c1, c2 = st.columns(2)
        with c1:
            # Holiday lift bar chart
            holiday_plot = holidays_df[holidays_df["Holiday_Name"] != "Non-Holiday Baseline"].copy()
            fig_holidays = bar_chart(
                holiday_plot,
                x="Holiday_Name",
                y="Lift_Vs_Baseline_Pct",
                title="Sales Lift Percentage vs Non-Holiday Baseline",
            )
            st.plotly_chart(fig_holidays, use_container_width=True)

        with c2:
            # Monthly Seasonality curve
            fig_month = line_chart(
                monthly_df,
                x="Month_Name",
                y="Mean_Weekly_Sales",
                title="Average Weekly Network Sales by Month ($)",
            )
            st.plotly_chart(fig_month, use_container_width=True)

        st.markdown("#### Holiday Surge Breakdown Table")
        formatted_dataframe(
            holidays_df,
            title="Walmart Holiday Surge Metrics",
            description="Mean weekly store sales, median sales, recorded weeks, and lift vs non-holiday benchmark.",
        )

    # ── Tab 4: Network Time Series ──
    with tab4:
        section_header("143-Week Network Revenue Trajectory (2010 - 2012)", "📈")
        context_box(
            "**Network Weekly Total:** Aggregated across all 45 stores. "
            "Notice clear annual dual-spikes in late November (Thanksgiving / Black Friday) and mid-December (pre-Christmas shopping)."
        )

        # Interactive Network Line Chart
        fig_net = line_chart(
            network_df,
            x="Date",
            y="Total_Network_Sales",
            title="Total Walmart Weekly Sales (All Stores)",
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Multi-store comparative view
        st.markdown("#### Compare Individual Store Trajectories")
        selected_stores_ts = st.multiselect(
            "Select Stores to Compare",
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
                title="Weekly Sales Trajectory by Store",
            )
            st.plotly_chart(fig_stores_ts, use_container_width=True)
