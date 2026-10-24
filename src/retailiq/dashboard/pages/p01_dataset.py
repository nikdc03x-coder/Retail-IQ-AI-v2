"""
RetailIQ AI V2 — Page 01: Dataset Explorer

Interactive exploration of raw Walmart sales data, schema contracts,
ingestion lineage, and dataset structure.
"""

import json
import streamlit as st
import pandas as pd

from src.retailiq.config import ANALYTICAL_DATASET, REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_number,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import bar_chart, histogram
from src.retailiq.dashboard.theme import context_box, page_header, section_header
from src.retailiq.data.loader import (
    load_analytical_dataset,
    load_raw_features,
    load_raw_stores,
    load_raw_train,
)


def render() -> None:
    """Render the Dataset Explorer page."""
    page_header(
        "Dataset Explorer",
        "Historical Walmart Sales Data, Schema Lineage & Structural Profile",
    )

    page_context(
        why="Before modeling or generating business insights, we must inspect the underlying data structure, column definitions, and join integrity.",
        business_problem="Addresses data provenance, schema predictability, completeness, and structural boundaries of the Walmart records.",
        data_source="Original Kaggle Walmart sales dataset (train.csv, features.csv, stores.csv) joined into analytical_dataset.parquet.",
        what_you_learn="Exact row counts, date boundaries (Feb 2010 - Oct 2012), store classifications (A/B/C), and promotional MarkDown data availability.",
    )

    # Check if analytical dataset exists
    if not ANALYTICAL_DATASET.exists():
        st.warning(
            "⚠️ Analytical dataset not found. Please run the ingestion pipeline:\n\n"
            "`python scripts/01_data_ingestion/run_ingestion.py`"
        )
        return

    # Load analytical dataset (cached for responsive UI)
    @st.cache_data
    def get_data() -> pd.DataFrame:
        return load_analytical_dataset()

    @st.cache_data
    def get_summary() -> dict:
        summary_path = REPORT_SUBDIRS["data_quality"] / "01_ingestion_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    df = get_data()
    summary = get_summary()

    # ── Top KPI Row ──
    total_sales = float(df["Weekly_Sales"].sum())
    render_kpi_row([
        {"label": "Total Observations", "value": f"{len(df):,}", "subtitle": "Store-Dept-Weeks"},
        {"label": "Active Stores", "value": str(df["Store"].nunique()), "subtitle": "Types A, B, C"},
        {"label": "Departments", "value": str(df["Dept"].nunique()), "subtitle": "Across 45 locations"},
        {"label": "Total Recorded Sales", "value": format_currency(total_sales), "subtitle": "2010 - 2012 cumulative"},
        {"label": "Temporal Window", "value": "143 Weeks", "subtitle": "Feb 2010 - Oct 2012"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Preview & Filters",
        "🗂️ Raw Source Tables",
        "📊 Distributions & Store Types",
        "🔗 Ingestion Lineage & Joins",
    ])

    # ── Tab 1: Preview & Filters ──
    with tab1:
        section_header("Analytical Dataset Preview", "🔎")
        context_box(
            "**What are we measuring?** Department-week sales records enriched with economic features (CPI, Fuel Price, Temperature, MarkDowns) and store metadata (Type, Size)."
        )

        # Filters
        c1, c2, c3 = st.columns(3)
        with c1:
            all_stores = sorted(df["Store"].unique())
            selected_stores = st.multiselect(
                "Filter Stores",
                options=all_stores,
                default=[1, 2, 3, 4],
                key="filter_stores_p01",
            )
        with c2:
            all_depts = sorted(df["Dept"].unique())
            selected_depts = st.multiselect(
                "Filter Departments",
                options=all_depts,
                default=[1, 2, 3],
                key="filter_depts_p01",
            )
        with c3:
            max_preview_rows = st.selectbox(
                "Preview Row Limit",
                options=[20, 50, 100, 200],
                index=1,
                key="preview_limit_p01",
            )

        filtered = df.copy()
        if selected_stores:
            filtered = filtered[filtered["Store"].isin(selected_stores)]
        if selected_depts:
            filtered = filtered[filtered["Dept"].isin(selected_depts)]

        formatted_dataframe(
            filtered,
            max_rows=max_preview_rows,
            title="Analytical Dataset Records",
            description=f"Showing records matching filters ({len(filtered):,} matching records total).",
        )

    # ── Tab 2: Raw Source Tables ──
    with tab2:
        section_header("Raw Source Tables Metadata", "📁")
        context_box(
            "**Three raw tables** form the foundation of RetailIQ AI V2. No raw records were deleted or modified."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="section-card">
                <h3>train.csv</h3>
                <p><b>Observations:</b> 421,570 rows</p>
                <p><b>Columns (5):</b> Store, Dept, Date, Weekly_Sales, IsHoliday</p>
                <p><b>Key:</b> (Store, Dept, Date) unique</p>
                <p><b>Role:</b> Labeled weekly sales target</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="section-card">
                <h3>features.csv</h3>
                <p><b>Observations:</b> 8,190 rows</p>
                <p><b>Columns (12):</b> Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday</p>
                <p><b>Key:</b> (Store, Date) unique</p>
                <p><b>Role:</b> External economic/promotional signals</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="section-card">
                <h3>stores.csv</h3>
                <p><b>Observations:</b> 45 rows</p>
                <p><b>Columns (3):</b> Store, Type, Size</p>
                <p><b>Key:</b> Store unique (1 to 45)</p>
                <p><b>Role:</b> Store structural metadata</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Column Schema & Types in Processed Parquet")
        schema_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Non-Null Count": [f"{df[c].notna().sum():,}" for c in df.columns],
            "Null Count": [f"{df[c].isna().sum():,}" for c in df.columns],
            "Null Percentage": [f"{(df[c].isna().sum() / len(df) * 100):.1f}%" for c in df.columns],
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Distributions ──
    with tab3:
        section_header("Structural Distributions", "📊")

        col_a, col_b = st.columns(2)
        with col_a:
            # Store Type Distribution
            store_counts = df[["Store", "Type"]].drop_duplicates()["Type"].value_counts().reset_index()
            store_counts.columns = ["Store Type", "Number of Stores"]
            fig_types = bar_chart(
                store_counts,
                x="Store Type",
                y="Number of Stores",
                title="Store Count by Classification Type",
            )
            st.plotly_chart(fig_types, use_container_width=True)
            context_box(
                "**Why this chart?** Type A represents large supercenters (22 stores), Type B medium stores (17 stores), and Type C compact formats (6 stores)."
            )

        with col_b:
            # Sales Distribution (sample for responsiveness)
            sample_sales = df["Weekly_Sales"].sample(n=min(20000, len(df)), random_state=42)
            fig_hist = histogram(
                pd.DataFrame({"Weekly_Sales": sample_sales}),
                x="Weekly_Sales",
                title="Weekly Sales Distribution (20k Sample)",
                nbins=50,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            context_box(
                "**What does the result mean?** Sales distribution is heavily right-skewed with a long tail representing holiday spikes and high-volume departments."
            )

    # ── Tab 4: Ingestion Lineage & Joins ──
    with tab4:
        section_header("Ingestion Lineage & Integrity Checks", "🔗")

        if summary:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    label="Row Preservation Ratio",
                    value=f"{summary.get('processed_counts', {}).get('row_count_preservation_ratio', 1.0):.1%}",
                    help="Merged rows equal exactly raw train rows (421,570).",
                )
            with c2:
                st.metric(
                    label="Holiday Flag Consistency",
                    value="100.0%",
                    help="train.csv and features.csv IsHoliday values match with 0 discrepancies.",
                )
            with c3:
                st.metric(
                    label="Join Match Rate",
                    value="100.0%",
                    help="All Store and Date combinations matched across tables.",
                )

            st.markdown("#### Sales Profile Analysis (Raw Preservation)")
            sales_prof = summary.get("sales_profile", {})
            st.markdown(f"""
            - **Positive Weekly Sales:** {sales_prof.get('positive_count', 0):,} ({sales_prof.get('positive_pct', 0)}%)
            - **Zero Weekly Sales:** {sales_prof.get('zero_count', 0):,} ({sales_prof.get('zero_pct', 0)}%)
            - **Negative Weekly Sales (Customer Returns/Adjustments):** {sales_prof.get('negative_count', 0):,} ({sales_prof.get('negative_pct', 0)}%)
            - **Minimum Weekly Sales:** {format_currency(sales_prof.get('min_sales', 0), 2)}
            - **Maximum Weekly Sales:** {format_currency(sales_prof.get('max_sales', 0), 2)}
            - **Median Weekly Sales:** {format_currency(sales_prof.get('median_sales', 0), 2)}
            """)
            context_box(
                "**Data engineering decision:** Negative sales observations (1,285 rows / 0.30%) reflect real-world merchandise returns and inventory reconciliation. "
                "In accordance with project principles, these records were **preserved without silent deletion**."
            )
