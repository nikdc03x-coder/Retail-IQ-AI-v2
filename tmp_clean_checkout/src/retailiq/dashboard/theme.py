"""
RetailIQ AI V2 ??? Dashboard Theme & Design System

Centralized styling, typography, and reusable layout helpers for the Streamlit dashboard.
Adheres to a clean, professional, human-centered enterprise UI/UX design language.
"""

from typing import Any
import streamlit as st
from src.retailiq.config import Theme


def apply_theme() -> None:
    """Inject production-grade enterprise CSS into the Streamlit application."""
    st.markdown(f"""
    <style>
        /* ?????? Import Inter & Google Material Symbols ?????? */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

        /* ?????? Global typography & canvas ?????? */
        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color: {Theme.LIGHT_GRAY};
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        .stApp {{
            background-color: {Theme.NAVY};
        }}

        /* ?????? Material Symbols icon class ?????? */
        .material-symbols-outlined {{
            font-family: 'Material Symbols Outlined';
            font-weight: normal;
            font-style: normal;
            font-size: 1.25rem;
            line-height: 1;
            display: inline-block;
            vertical-align: middle;
            letter-spacing: normal;
            text-transform: none;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-feature-settings: 'liga';
            -webkit-font-smoothing: antialiased;
        }}

        /* ?????? Executive Page Header ?????? */
        .main-header {{
            background: {Theme.INDIGO};
            padding: 1.5rem 1.75rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            border: 1px solid {Theme.CARD_BORDER};
            border-left: 4px solid {Theme.VIOLET};
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        }}
        .main-header .header-category {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {Theme.VIOLET};
            margin-bottom: 0.25rem;
        }}
        .main-header h1 {{
            color: {Theme.WHITE};
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.02em;
            line-height: 1.25;
        }}
        .main-header p {{
            color: {Theme.DARK_GRAY};
            font-size: 0.88rem;
            margin: 0;
            line-height: 1.5;
        }}

        /* ?????? KPI Metric Cards ?????? */
        .kpi-card {{
            background: {Theme.INDIGO};
            border: 1px solid {Theme.CARD_BORDER};
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            text-align: left;
            transition: border-color 0.15s ease, transform 0.15s ease;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
            height: 100%;
        }}
        .kpi-card:hover {{
            border-color: {Theme.BORDER_LIGHT};
            transform: translateY(-1px);
        }}
        .kpi-label {{
            color: {Theme.DARK_GRAY};
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }}
        .kpi-value {{
            color: {Theme.WHITE};
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.15;
            font-variant-numeric: tabular-nums;
        }}
        .kpi-subtitle {{
            color: {Theme.MUTED};
            font-size: 0.75rem;
            margin-top: 0.35rem;
            line-height: 1.35;
        }}

        /* ?????? Section Cards & Containers ?????? */
        .section-card {{
            background: {Theme.INDIGO};
            border: 1px solid {Theme.CARD_BORDER};
            border-radius: 8px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
            transition: border-color 0.15s ease;
        }}
        .section-card:hover {{
            border-color: {Theme.BORDER_LIGHT};
        }}
        .section-card h3 {{
            color: {Theme.WHITE};
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.01em;
        }}

        /* ?????? Section Headings ?????? */
        .section-header-wrap {{
            margin: 1.75rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {Theme.CARD_BORDER};
        }}
        .section-header-wrap h3 {{
            color: {Theme.WHITE};
            font-size: 1.15rem;
            font-weight: 600;
            margin: 0 0 0.25rem 0;
            letter-spacing: -0.01em;
        }}
        .section-header-wrap p {{
            color: {Theme.DARK_GRAY};
            font-size: 0.82rem;
            margin: 0;
        }}

        /* ?????? Informational Context Callouts ?????? */
        .context-box {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid {Theme.CARD_BORDER};
            border-left: 3px solid {Theme.VIOLET};
            border-radius: 6px;
            padding: 0.9rem 1.15rem;
            margin: 0.85rem 0;
            font-size: 0.84rem;
            line-height: 1.55;
            color: {Theme.LIGHT_GRAY};
        }}
        .context-box strong, .context-box b {{
            color: {Theme.WHITE};
        }}

        /* ?????? Sidebar Navigation ?????? */
        section[data-testid="stSidebar"] {{
            background-color: #0A0E17 !important;
            border-right: 1px solid {Theme.CARD_BORDER} !important;
        }}
        section[data-testid="stSidebar"] .stRadio > div {{
            gap: 2px;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            padding: 0.55rem 0.85rem !important;
            border-radius: 6px !important;
            transition: all 0.15s ease !important;
            color: {Theme.DARK_GRAY} !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            background-color: {Theme.INDIGO} !important;
            color: {Theme.WHITE} !important;
        }}

        /* ?????? Tabs Styling ?????? */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {Theme.CARD_BORDER};
            padding-bottom: 2px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px 6px 0 0;
            padding: 0.6rem 1.1rem;
            font-size: 0.85rem;
            font-weight: 500;
            color: {Theme.DARK_GRAY};
            background-color: transparent;
            border: none;
            transition: color 0.15s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {Theme.WHITE};
        }}
        .stTabs [aria-selected="true"] {{
            color: {Theme.WHITE} !important;
            border-bottom: 2px solid {Theme.VIOLET} !important;
            font-weight: 600 !important;
        }}

        /* ?????? Button Styling ?????? */
        .stButton > button {{
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.84rem !important;
            border: 1px solid {Theme.CARD_BORDER} !important;
            background-color: {Theme.INDIGO} !important;
            color: {Theme.LIGHT_GRAY} !important;
            padding: 0.45rem 1rem !important;
            transition: all 0.15s ease !important;
        }}
        .stButton > button:hover {{
            border-color: {Theme.VIOLET} !important;
            color: {Theme.WHITE} !important;
            background-color: #1E293B !important;
        }}
        .stButton > button:active {{
            transform: scale(0.99);
        }}

        /* ?????? Form Inputs & Selectboxes ?????? */
        div[data-baseweb="select"] > div {{
            background-color: {Theme.INDIGO} !important;
            border-color: {Theme.CARD_BORDER} !important;
            border-radius: 6px !important;
            color: {Theme.WHITE} !important;
            font-size: 0.85rem !important;
        }}
        div[data-baseweb="select"] > div:hover {{
            border-color: {Theme.BORDER_LIGHT} !important;
        }}

        /* ?????? Expanders ?????? */
        .streamlit-expanderHeader {{
            background-color: {Theme.INDIGO} !important;
            border: 1px solid {Theme.CARD_BORDER} !important;
            border-radius: 6px !important;
            color: {Theme.LIGHT_GRAY} !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            padding: 0.75rem 1rem !important;
        }}

        /* ?????? Dataframes ?????? */
        .stDataFrame {{
            border: 1px solid {Theme.CARD_BORDER};
            border-radius: 6px;
            overflow: hidden;
        }}

        /* ?????? Chrome / Header visibility ?????? */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        [data-testid="stSidebarCollapseButton"] {{
            visibility: visible !important;
            display: block !important;
            color: {Theme.WHITE} !important;
        }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, description: str, category: str | None = None) -> None:
    """Render an executive enterprise page header without tacky emoji icons.

    Args:
        title: Clean, professional page title.
        description: Informative description of the page's operational purpose.
        category: Optional domain category pill (e.g., 'Forecasting & Benchmarks').
    """
    category_html = f'<div class="header-category">{category}</div>' if category else ""
    st.markdown(f"""
    <div class="main-header">
        {category_html}
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a clean, standardized section heading without decorative emojis.

    Args:
        title: Section title.
        subtitle: Optional secondary description or caption.
    """
    subtitle_html = f"<p>{subtitle}</p>" if subtitle and not subtitle.startswith(":") else ""
    st.markdown(f"""
    <div class="section-header-wrap">
        <h3>{title}</h3>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def context_box(text: str, title: str | None = None) -> None:
    """Render an informational context callout box.

    Args:
        text: Main explanation or analytical interpretation.
        title: Optional callout title.
    """
    title_html = f"<div style='font-weight: 600; color: #F9FAFB; margin-bottom: 0.3rem;'>{title}</div>" if title else ""
    st.markdown(f"""
    <div class="context-box">
        {title_html}
        <div>{text}</div>
    </div>
    """, unsafe_allow_html=True)
