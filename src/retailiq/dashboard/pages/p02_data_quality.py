"""
RetailIQ AI V2 — Page 02: Data Quality & Readiness Audit

Empirical health audit, missingness taxonomy, merchandise return profiling,
and time-series observation continuity metrics.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from src.retailiq.config import REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_number,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import bar_chart
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Data Quality & Readiness page."""
    page_header(
        "Data Quality & Readiness Audit",
        "Empirical Telemetry Health, Structural Missingness Taxonomy, and Returns Accounting",
        category="Data Quality & Integrity",
    )

    page_context(
        why="Reliable forecasting and decision intelligence require transparent verification of data completeness, return patterns, and feature distributions.",
        business_problem="Prevents garbage-in garbage-out modeling by isolating structural missingness, auditing returns, and quantifying continuity.",
        data_source="Pre-computed reports in reports/data_quality/ generated from analytical_dataset.parquet.",
        what_you_learn="100% economic feature completeness, structural onset of MarkDown tracking, return prevalence by department, and store-department observation continuity.",
    )

    out_dir = REPORT_SUBDIRS["data_quality"]

    if not (out_dir / "06_data_quality_summary.json").exists():
        st.warning(
            "Data quality reports not found. Please run the quality script:\n\n"
            "`python scripts/02_data_quality/run_quality_checks.py`"
        )
        return

    @st.cache_data
    def load_reports() -> dict:
        with open(out_dir / "06_data_quality_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        stats = pd.read_csv(out_dir / "02_column_statistics.csv")
        missing = pd.read_csv(out_dir / "03_missingness_profile.csv")
        returns = pd.read_csv(out_dir / "04_negative_sales_audit.csv")
        coverage = pd.read_csv(out_dir / "05_store_department_coverage.csv")
        return {
            "summary": summary,
            "stats": stats,
            "missing": missing,
            "returns": returns,
            "coverage": coverage,
        }

    data = load_reports()
    summary = data["summary"]
    stats_df = data["stats"]
    missing_df = data["missing"]
    returns_df = data["returns"]
    coverage_df = data["coverage"]

    # ── Top KPI Row ──
    full_pairs = summary["departments"]["full_coverage_pairs"]
    total_pairs = summary["departments"]["active_pairs"]
    full_pct = (full_pairs / total_pairs) if total_pairs > 0 else 0.0

    render_kpi_row([
        {"label": "Economic Completeness", "value": "100.0%", "subtitle": "CPI & Unemployment (0 nulls)"},
        {"label": "MarkDown Sparsity", "value": f"{summary['missingness']['markdown_missing_pct']:.1f}%", "subtitle": "Structural: pre-Nov 2011"},
        {"label": "Merchandise Returns", "value": f"{summary['sales_audit']['negative_rows']:,}", "subtitle": f"{format_currency(summary['sales_audit']['total_return_amount'])} cumulative"},
        {"label": "Continuous Series", "value": f"{full_pct:.1%}", "subtitle": f"{full_pairs:,} of {total_pairs:,} pairs (143 wks)"},
        {"label": "Return Departments", "value": str(len(returns_df)), "subtitle": "Across 45 store locations"},
    ])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Missingness Profiling",
        "Feature Distributions",
        "Merchandise Returns Audit",
        "Series Continuity",
    ])

    # ── Tab 1: Missingness ──
    with tab1:
        section_header("Missing Value Audit & Classification", "Structural classification of nullity patterns across analytical features")
        context_box(
            "Missing values are NOT random corruption. CPI and Unemployment are 100% complete across historical observations. MarkDown features were structurally uncollected by Walmart prior to November 11, 2011.",
            title="Completeness Audit",
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            missing_chart_df = missing_df[missing_df["Missing_Count"] > 0].sort_values("Missing_Pct", ascending=True)
            if not missing_chart_df.empty:
                fig_miss = bar_chart(
                    missing_chart_df,
                    x="Missing_Pct",
                    y="Column",
                    title="Missingness Percentage by Feature",
                    orientation="h",
                )
                st.plotly_chart(fig_miss, use_container_width=True)
        with c2:
            st.markdown(f"""
            <div class="section-card">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Audit Telemetry</div>
                <h3>Classification Assessment</h3>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>MarkDown1 - 5:</b> Structural missingness (~64% missing). First recorded on 2011-11-11.</p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>CPI & Unemployment:</b> 0% missing in train window. Clean continuous signals.</p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Weekly_Sales:</b> 0% missing. Complete historical target.</p>
                <p style="color: {Theme.DARK_GRAY}; font-size: 0.8rem; margin: 0;"><b>Store Metadata:</b> 0% missing. Complete format and area metrics.</p>
            </div>
            """, unsafe_allow_html=True)

        formatted_dataframe(
            missing_df,
            title="Complete Feature Missingness Profile",
            description="Audit classification and earliest observed non-null dates per feature.",
        )

    # ── Tab 2: Feature Statistics ──
    with tab2:
        section_header("Numerical Feature Distribution Metrics", "Dispersion parameters, central tendency, and distribution shapes")
        context_box(
            "Central tendency (Mean, Median), dispersion (Std, IQR), and distribution shape (Skewness, Kurtosis). High positive skewness in Weekly_Sales (3.26) indicates heavy holiday demand tails.",
            title="Dispersion Metrics",
        )

        formatted_dataframe(
            stats_df,
            title="Statistical Dispersion & Shape Parameters",
            description="Summary metrics for all 11 continuous features in analytical dataset.",
        )

    # ── Tab 3: Merchandise Returns ──
    with tab3:
        section_header("Merchandise Returns & Accounting Adjustments", "Analysis of negative sales transactions and return concentrations")
        context_box(
            "1,285 records (0.30%) contain negative Weekly_Sales values. These represent legitimate customer returns and accounting reconciliations. Deleting them would bias sales aggregations upward; they are preserved with full lineage.",
            title="Accounting Policy",
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            top_returns = returns_df.head(10).sort_values("Total_Return_Amount", ascending=True)
            fig_ret = bar_chart(
                top_returns,
                x="Total_Return_Amount",
                y="Dept",
                title="Top 10 Departments by Cumulative Return Amount ($)",
                orientation="h",
            )
            st.plotly_chart(fig_ret, use_container_width=True)
        with c2:
            st.markdown(f"""
            <div class="section-card">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Returns Profile</div>
                <h3>Audit Summary</h3>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Total Return Records:</b> {summary['sales_audit']['negative_rows']:,}</p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Cumulative Dollar Value:</b> {format_currency(summary['sales_audit']['total_return_amount'], 2)}</p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Top Return Dept:</b> Dept {returns_df.iloc[0]['Dept']} ({format_currency(returns_df.iloc[0]['Total_Return_Amount'], 2)})</p>
                <p style="color: {Theme.DARK_GRAY}; font-size: 0.8rem; margin: 0;"><b>Affected Stores:</b> Up to {returns_df['Affected_Stores'].max()} stores per department</p>
            </div>
            """, unsafe_allow_html=True)

        formatted_dataframe(
            returns_df,
            title="Department Returns Breakdown",
            description="Detailed return counts, cumulative dollar amounts, and minimum single-week adjustments.",
        )

    # ── Tab 4: Continuity ──
    with tab4:
        section_header("Store-Department Series Continuity", "Observation longevity across 143 historical weeks")
        context_box(
            "79.9% of store-department pairs are active across all 143 historical weeks. Remaining pairs represent seasonal departments (e.g. holiday decor) or newly opened departments with fewer historical weeks.",
            title="Temporal Continuity",
        )

        cov_summary = coverage_df["Status"].value_counts().reset_index()
        cov_summary.columns = ["Continuity Status", "Store-Dept Pairs"]
        formatted_dataframe(
            cov_summary,
            title="Continuity Classification Summary",
            description="Overview of full vs partial historical coverage across 3,331 store-department pairs.",
        )

        store_sel = st.selectbox(
            "Filter Continuity by Store Location",
            options=sorted(coverage_df["Store"].unique()),
            index=0,
            key="cov_store_sel",
        )
        store_cov = coverage_df[coverage_df["Store"] == store_sel]
        formatted_dataframe(
            store_cov,
            title=f"Department Coverage for Store {store_sel}",
            description=f"Showing recorded weeks out of 143 total for Store {store_sel}.",
        )
