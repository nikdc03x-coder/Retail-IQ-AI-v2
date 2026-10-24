"""
RetailIQ AI V2 — Dashboard Theme & Styling

Centralized CSS, color utilities, and layout helpers for the Streamlit dashboard.
Uses the project's design system tokens from config.Theme.
"""

import streamlit as st
from src.retailiq.config import Theme


def apply_theme() -> None:
    """Inject custom CSS into the Streamlit page for a polished BI-style look."""
    st.markdown(f"""
    <style>
        /* ── Import professional font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global typography ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}

        /* ── Main header gradient ── */
        .main-header {{
            background: linear-gradient(135deg, {Theme.NAVY} 0%, {Theme.INDIGO} 50%, {Theme.VIOLET} 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(108, 99, 255, 0.2);
        }}
        .main-header h1 {{
            color: {Theme.WHITE};
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
        }}
        .main-header p {{
            color: {Theme.LIGHT_GRAY};
            font-size: 0.95rem;
            margin: 0;
            opacity: 0.9;
        }}

        /* ── KPI metric cards ── */
        .kpi-card {{
            background: {Theme.INDIGO};
            border: 1px solid rgba(108, 99, 255, 0.15);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(108, 99, 255, 0.15);
        }}
        .kpi-label {{
            color: {Theme.DARK_GRAY};
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            color: {Theme.WHITE};
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .kpi-subtitle {{
            color: {Theme.DARK_GRAY};
            font-size: 0.75rem;
        }}

        /* ── Section cards ── */
        .section-card {{
            background: {Theme.INDIGO};
            border: 1px solid rgba(108, 99, 255, 0.12);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        .section-card h3 {{
            color: {Theme.WHITE};
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
        }}

        /* ── Context boxes for chart interpretation ── */
        .context-box {{
            background: rgba(108, 99, 255, 0.08);
            border-left: 3px solid {Theme.VIOLET};
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1rem;
            margin: 0.8rem 0;
            font-size: 0.85rem;
            color: {Theme.LIGHT_GRAY};
        }}

        /* ── Sidebar styling ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {Theme.NAVY} 0%, #0a1220 100%);
            border-right: 1px solid rgba(108, 99, 255, 0.15);
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            color: {Theme.LIGHT_GRAY} !important;
            font-weight: 400;
        }}

        /* ── Expander styling ── */
        .streamlit-expanderHeader {{
            font-weight: 500;
            color: {Theme.LIGHT_GRAY};
        }}

        /* ── Hide Streamlit branding ── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* ── Data table tweaks ── */
        .stDataFrame {{
            border-radius: 8px;
            overflow: hidden;
        }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, description: str) -> None:
    """Render a gradient page header with title and description."""
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 {title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, icon: str = "📌") -> None:
    """Render a styled section heading."""
    st.markdown(f"### {icon} {title}")


def context_box(text: str) -> None:
    """Render a context/interpretation box next to a visualization."""
    st.markdown(f'<div class="context-box">{text}</div>', unsafe_allow_html=True)

