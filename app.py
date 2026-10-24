"""
RetailIQ AI V2 — Streamlit Application Entry Point

This is the main entry point for the Streamlit dashboard.
Run with: python -m streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure the project root is on the Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retailiq.dashboard.theme import apply_theme
from src.retailiq.config import Theme

# ── Page configuration (must be the first Streamlit command) ──
st.set_page_config(
    page_title="RetailIQ AI V2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply custom theme ──
apply_theme()

# ── Sidebar navigation ──
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h2 style="color: {Theme.WHITE}; font-size: 1.3rem; font-weight: 700;
                   margin: 0; letter-spacing: -0.02em;">
            📊 RetailIQ AI
        </h2>
        <p style="color: {Theme.VIOLET}; font-size: 0.75rem; font-weight: 500;
                  letter-spacing: 0.1em; margin: 0.2rem 0 0 0;">
            VERSION 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Page navigation
    page = st.radio(
        "Navigation",
        options=[
            "🏠 Project Overview",
            "🗄️ Dataset Explorer",
            "✅ Data Quality",
            "📊 Business Analytics",
            "📈 Sales Forecasting",
            "🎯 Forecast Reliability",
            "⚠️ Anomaly Investigation",
            "🔬 Explainable AI",
            "🤖 AI Business Assistant",
        ],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.divider()

    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="color: {Theme.DARK_GRAY}; font-size: 0.7rem; margin: 0;">
            Built with Python · Streamlit · Plotly
        </p>
        <p style="color: {Theme.DARK_GRAY}; font-size: 0.7rem; margin: 0.2rem 0 0 0;">
            Scikit-learn · XGBoost · SHAP · Gemini
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Page routing ──
if page == "🏠 Project Overview":
    from src.retailiq.dashboard.pages.p00_overview import render
    render()

elif page == "🗄️ Dataset Explorer":
    from src.retailiq.dashboard.pages.p01_dataset import render
    render()

elif page == "✅ Data Quality":
    from src.retailiq.dashboard.pages.p02_data_quality import render
    render()

elif page == "📊 Business Analytics":
    st.info("📦 Business Analytics will be available after Phase 03.")

elif page == "📈 Sales Forecasting":
    st.info("📦 Sales Forecasting will be available after Phase 04.")

elif page == "🎯 Forecast Reliability":
    st.info("📦 Forecast Reliability will be available after Phase 05.")

elif page == "⚠️ Anomaly Investigation":
    st.info("📦 Anomaly Investigation will be available after Phase 06.")

elif page == "🔬 Explainable AI":
    st.info("📦 Explainable AI will be available after Phase 07.")

elif page == "🤖 AI Business Assistant":
    st.info("📦 AI Business Assistant will be available after Phase 09.")

