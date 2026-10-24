"""
RetailIQ AI V2 — Page 00: Project Overview

Executive platform overview introducing system capabilities, data architecture,
and operational workflows.
"""

import streamlit as st
from src.retailiq.dashboard.theme import page_header, section_header, context_box
from src.retailiq.config import Theme


def render() -> None:
    """Render the Project Overview page."""
    page_header(
        "RetailIQ AI Enterprise Platform",
        "Explainable Retail Sales Forecasting, Anomaly Triage & Decision Strategy Platform",
        category="Executive Overview",
    )

    st.markdown("""
    <div style="font-size: 0.95rem; line-height: 1.6; color: #CBD5E1; margin-bottom: 1.5rem;">
        RetailIQ AI transforms historical Walmart retail network data into verified, evidence-grounded
        decision intelligence. Spanning seven specialized analytical disciplines—from ingestion telemetry
        and multi-model benchmarking to TreeSHAP attribution and conversational AI—the platform delivers
        statistically sound, mathematically verified operational guidance for merchandisers and store directors.
    </div>
    """, unsafe_allow_html=True)

    # ── Platform Capabilities Grid ──
    section_header("Platform Capabilities", "Core analytical modules accessible via the navigation sidebar")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="section-card">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Module 01 & 02</div>
            <h3>Data Quality & Engineering</h3>
            <p style="color: {Theme.DARK_GRAY}; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">
                Automated schema validation, missingness profiling, return transaction audits,
                and continuity tracking across 421,570 historical observations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Data Quality Audit", key="btn_open_quality", use_container_width=True):
            st.session_state["nav_page"] = "Data Quality Audit"
            st.rerun()

        st.markdown(f"""
        <div class="section-card" style="margin-top: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Module 04 & 05</div>
            <h3>Sales Forecasting & Reliability</h3>
            <p style="color: {Theme.DARK_GRAY}; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">
                XGBoost champion model ($R^2 = 0.9416$) evaluated with temporal splitting,
                spatial error distributions, holiday degradation metrics, and 4-fold rolling backtests.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Sales Forecasting", key="btn_open_forecast", use_container_width=True):
            st.session_state["nav_page"] = "Sales Forecasting"
            st.rerun()

        st.markdown(f"""
        <div class="section-card" style="margin-top: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Module 07</div>
            <h3>Explainable AI (TreeSHAP)</h3>
            <p style="color: {Theme.DARK_GRAY}; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">
                Global feature importance rankings and local prediction waterfall decompositions
                strictly verified against the cooperative game theory efficiency axiom ($<0.0001\\%$ relative error).
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Explainable AI", key="btn_open_shap", use_container_width=True):
            st.session_state["nav_page"] = "Explainable AI"
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="section-card">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Module 03</div>
            <h3>Business Analytics</h3>
            <p style="color: {Theme.DARK_GRAY}; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">
                Hierarchical sales aggregations, store revenue rankings, department Pareto curves,
                promotional holiday demand lifts, and multi-year seasonal indices.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Business Analytics", key="btn_open_analytics", use_container_width=True):
            st.session_state["nav_page"] = "Business Analytics"
            st.rerun()

        st.markdown(f"""
        <div class="section-card" style="margin-top: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Module 06</div>
            <h3>Anomaly Triage & Investigation</h3>
            <p style="color: {Theme.DARK_GRAY}; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">
                Dual-detector synthesis combining Statistical Z-Score ($|Z| > 3.0$) and Multivariate
                Isolation Forest, prioritized into an evidence-backed operational audit queue.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Anomaly Investigation", key="btn_open_anomalies", use_container_width=True):
            st.session_state["nav_page"] = "Anomaly Investigation"
            st.rerun()

        st.markdown(f"""
        <div class="section-card" style="margin-top: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Module 08 & 09</div>
            <h3>Operational Strategy & AI Assistant</h3>
            <p style="color: {Theme.DARK_GRAY}; font-size: 0.85rem; line-height: 1.5; margin: 0 0 1rem 0;">
                Prescriptive 4-quadrant store matrix with dynamic safety stock allocations, plus an
                evidence-grounded Gemini conversational analyst with 13 verified query tools.
            </p>
        </div>
        """, unsafe_allow_html=True)
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("Operational Strategy", key="btn_open_strategy", use_container_width=True):
                st.session_state["nav_page"] = "Operational Strategy"
                st.rerun()
        with c_act2:
            if st.button("AI Business Assistant", key="btn_open_assistant", use_container_width=True):
                st.session_state["nav_page"] = "AI Business Assistant"
                st.rerun()

    # ── Dataset Overview ──
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    section_header("Dataset Architecture", "Historical retail benchmark telemetry spanning 45 store locations")

    st.markdown("""
    | Source File | Temporal Scope | Observations | Key Fields & Signal Content |
    | :--- | :--- | :--- | :--- |
    | `train.csv` | 2010-02-05 to 2012-10-26 | 421,570 rows | Store, Department, Date, Weekly Sales, Holiday Indicator |
    | `features.csv` | 2010-02-05 to 2013-07-26 | 8,190 rows | CPI, Unemployment, Fuel Price, Temperature, MarkDown1-5 |
    | `stores.csv` | Static Metadata | 45 stores | Store Format (Type A, B, C), Physical Square Footage |
    """)

    context_box(
        "Use the left sidebar navigation to explore individual analytical modules. "
        "Every page provides interactive controls, empirical evidence, and methodology details.",
        title="Navigation Guidance",
    )
