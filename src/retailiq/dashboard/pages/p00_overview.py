"""
RetailIQ AI V2 — Page 00: Project Overview

Landing page that introduces the platform, explains the business problem,
and provides an overview of all six capabilities.
"""

import streamlit as st
from src.retailiq.dashboard.theme import page_header, section_header, context_box
from src.retailiq.dashboard.components import render_kpi_row
from src.retailiq.config import Theme


def render() -> None:
    """Render the Project Overview page."""

    page_header(
        "RetailIQ AI V2",
        "Explainable Retail Sales Forecasting & Decision Intelligence Platform",
    )

    # ── Welcome message ──
    st.markdown("""
    RetailIQ AI V2 transforms historical Walmart retail sales data into actionable
    business intelligence through **six integrated capabilities** — from data quality
    monitoring to an AI-powered conversational assistant.

    Built with Python, Scikit-learn, XGBoost, SHAP, Plotly, and Google Gemini.
    """)

    # ── Platform capability cards ──
    section_header("Platform Capabilities", "🎯")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="section-card">
            <h3>🔍 Data Engineering & Quality</h3>
            <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem;">
            Validated, reproducible data pipeline with quality monitoring,
            lineage tracking, and documented transformations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card">
            <h3>📈 Sales Forecasting</h3>
            <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem;">
            One-week-ahead store-level forecasts using ML models compared
            against naive baselines with proper chronological evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card">
            <h3>🔬 Explainable AI</h3>
            <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem;">
            SHAP-based global and local explanations showing which features
            influence each prediction and by how much.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="section-card">
            <h3>📊 Business Analytics</h3>
            <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem;">
            Interactive store rankings, department analysis, holiday
            comparisons, and sales trend exploration.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card">
            <h3>⚠️ Anomaly Investigation</h3>
            <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem;">
            Statistical and Isolation Forest anomaly detectors with
            agreement analysis and a ranked investigation queue.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card">
            <h3>🤖 AI Business Assistant</h3>
            <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.88rem;">
            Gemini-powered conversational analyst that answers free-form
            business questions grounded in verified data evidence.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Dataset overview ──
    st.divider()
    section_header("Dataset", "🗄️")
    st.markdown("""
    **Source:** Public Walmart historical sales dataset
    ([Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast))

    | File | Contents | Role |
    |------|----------|------|
    | `train.csv` | Store-department weekly sales | Historical labeled observations |
    | `features.csv` | Economic and promotional features | External signals |
    | `stores.csv` | Store type and size | Static metadata |

    > **Note:** This is a historical dataset (~2010–2012), not a live operational database.
    > Analyses reflect past patterns, not current Walmart operations.
    """)

    # ── Architecture note ──
    st.divider()
    section_header("Architecture", "🏗️")
    st.markdown(f"""
    ```
    Raw CSVs → Validated Pipeline → Analytical Dataset (Parquet)
        ↓                                    ↓
    Quality Reports              Analytics / Forecasting / SHAP
        ↓                                    ↓
    Streamlit Dashboard  ←──── Pre-computed Report CSVs
        ↓
    Gemini Assistant  ←──── Approved Python Tools
    ```

    - **Pipeline scripts** in `scripts/` produce reports and model artifacts
    - **Dashboard** reads pre-computed reports — no heavy computation at render time
    - **Gemini assistant** uses controlled tool functions, not arbitrary code execution
    """)

    context_box(
        "Navigate using the sidebar to explore each capability. "
        "Each page explains what it measures, why, and what the results mean."
    )

