"""
RetailIQ AI V2 — Streamlit Enterprise Application Entry Point

Main entry point for the RetailIQ decision intelligence platform.
Run with: python -m streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retailiq.dashboard.theme import apply_theme
from src.retailiq.config import Theme

# ── Page configuration (must be the first Streamlit command) ──
st.set_page_config(
    page_title="RetailIQ AI — Enterprise Decision Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply enterprise design system theme ──
apply_theme()

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 1.1rem 0.5rem 0.8rem 0.5rem; border-bottom: 1px solid #1F2937;">
        <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.35rem;">
            <div style="background: linear-gradient(135deg, #4F46E5, #6366F1); width: 28px; height: 28px;
                        border-radius: 6px; display: flex; align-items: center; justify-content: center;
                        font-weight: 800; font-size: 0.85rem; color: #FFFFFF;">
                R
            </div>
            <div>
                <div style="color: #F9FAFB; font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;">
                    RetailIQ AI
                </div>
                <div style="color: #818CF8; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;">
                    Enterprise V2.0
                </div>
            </div>
        </div>
        <div style="color: #64748B; font-size: 0.73rem; margin-top: 0.4rem; line-height: 1.35;">
            Explainable Sales Forecasting & Decision Strategy Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Clean navigation options without generic decorative emojis
    page = st.radio(
        "Navigation",
        options=[
            "Project Overview",
            "Dataset Explorer",
            "Data Quality Audit",
            "Business Analytics",
            "Sales Forecasting",
            "Forecast Reliability",
            "Anomaly Investigation",
            "Explainable AI",
            "Operational Strategy",
            "AI Business Assistant",
        ],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Sidebar Footer
    st.markdown(f"""
    <div style="border-top: 1px solid #1F2937; padding: 0.9rem 0.5rem 0.5rem 0.5rem; margin-top: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.35rem;">
            <div style="width: 7px; height: 7px; border-radius: 50%; background-color: #10B981;"></div>
            <span style="color: #9CA3AF; font-size: 0.72rem; font-weight: 500;">System Online · All Models Verified</span>
        </div>
        <div style="color: #64748B; font-size: 0.68rem;">
            Python 3.12 · XGBoost · SHAP · Gemini
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Page Routing ──
if page == "Project Overview":
    from src.retailiq.dashboard.pages.p00_overview import render
    render()

elif page == "Dataset Explorer":
    from src.retailiq.dashboard.pages.p01_dataset import render
    render()

elif page == "Data Quality Audit":
    from src.retailiq.dashboard.pages.p02_data_quality import render
    render()

elif page == "Business Analytics":
    from src.retailiq.dashboard.pages.p03_analytics import render
    render()

elif page == "Sales Forecasting":
    from src.retailiq.dashboard.pages.p04_forecasting import render
    render()

elif page == "Forecast Reliability":
    from src.retailiq.dashboard.pages.p05_reliability import render
    render()

elif page == "Anomaly Investigation":
    from src.retailiq.dashboard.pages.p06_anomalies import render
    render()

elif page == "Explainable AI":
    from src.retailiq.dashboard.pages.p07_explainability import render
    render()

elif page == "Operational Strategy":
    from src.retailiq.dashboard.pages.p08_strategy import render
    render()

elif page == "AI Business Assistant":
    from src.retailiq.dashboard.pages.p08_assistant import render
    render()
