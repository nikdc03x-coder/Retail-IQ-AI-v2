"""
RetailIQ AI V2 — Page 06: Anomaly Detection & Investigation Queue

Interactive audit of unusual sales observations detected by Statistical Residual Z-scores
and Multivariate Isolation Forest with prioritized operational investigation evidence.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
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
from src.retailiq.dashboard.charts import apply_default_layout, bar_chart, histogram, scatter_chart
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Anomaly Detection & Investigation Queue page."""
    page_header(
        "Anomaly Detection & Investigation Queue",
        "Dual-Detector Synthesis: Statistical Z-Scores (|Z| > 3.0) & Isolation Forest",
    )

    page_context(
        why="Retail managers cannot review thousands of sales records manually. An automated dual-detector pipeline isolates high-priority operational discrepancies for targeted investigation.",
        business_problem="Answers: Which historical store-weeks deviated severely from expectations? Did detectors agree? What is the operational hypothesis for each flagged record?",
        data_source="Pre-computed anomaly reports in reports/anomaly_detection/ evaluated across 990 held-out test records.",
        what_you_learn="23 records prioritized (2.3% of test set); 14 are unanimous flags confirmed by both detectors; Store 14 on 2012-08-24 is the #1 investigation priority.",
    )

    out_dir = REPORT_SUBDIRS["anomaly_detection"]

    # Verify reports exist
    if not (out_dir / "05_investigation_queue.csv").exists():
        st.warning(
            "⚠️ Anomaly detection reports not found. Please run the anomaly script:\n\n"
            "`python scripts/06_anomaly_detection/run_anomaly_detection.py`"
        )
        return

    @st.cache_data
    def load_anomaly_data() -> dict:
        queue = pd.read_csv(out_dir / "05_investigation_queue.csv", parse_dates=["Date"])
        comparison = pd.read_csv(out_dir / "03_detector_comparison.csv", parse_dates=["Date"])
        agreement = pd.read_csv(out_dir / "04_agreement_summary.csv")
        with open(out_dir / "06_anomaly_detection_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {
            "queue": queue,
            "comparison": comparison,
            "agreement": agreement,
            "summary": summary,
        }

    data = load_anomaly_data()
    queue_df = data["queue"]
    comparison_df = data["comparison"]
    agreement_df = data["agreement"]
    summary = data["summary"]["synthesis_summary"]

    top_record = queue_df.iloc[0] if not queue_df.empty else None

    # ── Top KPI Row ──
    render_kpi_row([
        {"label": "Investigation Queue", "value": f"{len(queue_df)} Records", "subtitle": "Prioritized for operational audit"},
        {"label": "Unanimous Flags", "value": f"{summary['unanimous_anomalies']} Records", "subtitle": "Confirmed by BOTH detectors"},
        {"label": "Detector Agreement", "value": f"{summary['detector_agreement_rate_pct']}%", "subtitle": "967 normal + 14 unanimous"},
        {"label": "#1 Priority Flag", "value": f"Store {int(top_record['Store']) if top_record is not None else 'N/A'}", "subtitle": f"Priority Score: {top_record['Priority_Score'] if top_record is not None else 'N/A'}"},
        {"label": "Peak Residual Miss", "value": format_currency(float(queue_df['Residual'].abs().max())), "subtitle": "Maximum single-week discrepancy"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cautionary Note on Anomaly Labels ──
    context_box(
        "⚠️ **Operational Interpretation Note:** Anomaly flags indicate statistical divergence between actual sales and model expectations. "
        "They are **operational review hypotheses**, NOT confirmed fraud, data corruption, or system errors. "
        "Real-world causes typically include unmodeled local events, extreme weather, competitor actions, or promotional timing shifts."
    )

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Prioritized Investigation Queue",
        "🤝 Dual-Detector Agreement",
        "📐 Statistical Z-Score Profile",
        "🌲 Isolation Forest Space",
    ])

    # ── Tab 1: Investigation Queue ──
    with tab1:
        section_header("Ranked Operational Investigation Queue", "📋")
        context_box(
            "**Priority Score [0 - 100]:** Combines normalized residual magnitude (50% weight) and Isolation Forest anomaly score (50% weight). "
            "Observations are ranked from highest to lowest review urgency with automated business evidence notes."
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            agree_filter = st.multiselect(
                "Filter by Detector Agreement",
                options=sorted(queue_df["Agreement_Classification"].unique()),
                default=sorted(queue_df["Agreement_Classification"].unique()),
                key="queue_agree_filter",
            )
            min_priority = st.slider("Minimum Priority Score", min_value=0.0, max_value=100.0, value=50.0, step=5.0, key="min_priority_slider")

        filtered_queue = queue_df[
            (queue_df["Agreement_Classification"].isin(agree_filter)) &
            (queue_df["Priority_Score"] >= min_priority)
        ]

        with c2:
            st.markdown(f"**Showing {len(filtered_queue)} of {len(queue_df)} queued records.**")
            if not filtered_queue.empty:
                st.info(f"💡 **Top Investigation Target:** {filtered_queue.iloc[0]['Investigation_Evidence']}")

        formatted_dataframe(
            filtered_queue,
            max_rows=30,
            title="Prioritized Investigation Queue Records",
            description="Store, date, actuals, forecast, residual, priority score, and automated investigation hypothesis.",
        )

    # ── Tab 2: Detector Agreement ──
    with tab2:
        section_header("Dual-Detector Agreement & Divergence Analysis", "🤝")
        context_box(
            "**Why use two detectors?** Statistical Z-scores catch massive univariate forecasting errors (|Z| > 3.0). "
            "Isolation Forest captures complex multidimensional outliers (e.g., normal percentage error but abnormal volume conjunction). "
            "Where both agree, investigation priority is highest."
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            fig_agree = bar_chart(
                agreement_df,
                x="Record_Count",
                y="Classification",
                title="Observations by Detector Agreement Classification",
                orientation="h",
            )
            st.plotly_chart(fig_agree, use_container_width=True)

        with c2:
            st.markdown("#### Agreement Classification Matrix")
            st.dataframe(agreement_df, use_container_width=True, hide_index=True)
            st.markdown(f"""
            <div class="section-card">
                <h3>Agreement Highlights</h3>
                <p><b>Unanimous (Both Detectors):</b> {summary['unanimous_anomalies']} records</p>
                <p><b>Statistical Only:</b> {summary['statistical_only_anomalies']} records</p>
                <p><b>Isolation Forest Only:</b> {summary['isolation_forest_only_anomalies']} records</p>
                <p><b>Total Divergence:</b> 9 records where detectors disagreed</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: Statistical Z-Scores ──
    with tab3:
        section_header("Statistical Residual Standard Score (|Z| > 3.0)", "📐")
        context_box(
            "**Standard Threshold |Z| > 3.0:** Under a Gaussian residual assumption, observations beyond 3 standard deviations represent less than 0.27% of probability mass. "
            "Records outside the dashed red cutoff lines represent severe under- or over-predictions."
        )

        # Z-score distribution with cutoff lines
        fig_z = px.histogram(
            comparison_df,
            x="Z_Score",
            nbins=50,
            title="Residual Z-Score Distribution (Threshold: |Z| = 3.0)",
            color_discrete_sequence=[Theme.VIOLET],
        )
        fig_z.add_vline(x=3.0, line_dash="dash", line_color=Theme.CORAL, annotation_text="+3.0 Sigma")
        fig_z.add_vline(x=-3.0, line_dash="dash", line_color=Theme.CORAL, annotation_text="-3.0 Sigma")
        fig_z = apply_default_layout(fig_z, "Residual Z-Score Distribution (Threshold: |Z| = 3.0)")
        st.plotly_chart(fig_z, use_container_width=True)

        fig_z_scatter = px.scatter(
            comparison_df,
            x="Actual_Sales",
            y="Z_Score",
            color="Agreement_Classification",
            title="Actual Sales ($) vs Residual Z-Score",
            color_discrete_sequence=Theme.CHART_COLORS,
        )
        fig_z_scatter.add_hline(y=3.0, line_dash="dash", line_color=Theme.CORAL)
        fig_z_scatter.add_hline(y=-3.0, line_dash="dash", line_color=Theme.CORAL)
        fig_z_scatter = apply_default_layout(fig_z_scatter, "Actual Sales ($) vs Residual Z-Score")
        st.plotly_chart(fig_z_scatter, use_container_width=True)

    # ── Tab 4: Isolation Forest Space ──
    with tab4:
        section_header("Multivariate Isolation Forest Partition Space", "🌲")
        context_box(
            "**Feature Space:** Isolation Forest isolates points in the joint space of Actual Sales, Predicted Sales, Absolute Residual, and Percentage Error. "
            "Brighter color indicates higher anomaly score."
        )

        fig_ifor = px.scatter(
            comparison_df,
            x="Actual_Sales",
            y="XGBoost_Pred",
            color="IForest_Anomaly_Score",
            hover_data=["Store", "Date", "Z_Score", "Agreement_Classification"],
            title="Actual Sales vs XGBoost Forecast (Colored by Isolation Forest Score)",
            color_continuous_scale="Viridis",
        )
        # Add 45-degree parity
        max_v = max(comparison_df["Actual_Sales"].max(), comparison_df["XGBoost_Pred"].max())
        fig_ifor.add_trace(go.Scatter(
            x=[0, max_v],
            y=[0, max_v],
            mode="lines",
            name="Parity (y = x)",
            line=dict(dash="dash", color=Theme.DARK_GRAY),
        ))
        fig_ifor = apply_default_layout(fig_ifor, "Actual Sales vs XGBoost Forecast (Colored by Isolation Forest Score)")
        st.plotly_chart(fig_ifor, use_container_width=True)
