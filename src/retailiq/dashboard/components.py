"""
RetailIQ AI V2 — Reusable Dashboard Components

KPI metric cards, data table wrappers, download handlers, and contextual guidance components.
"""

from typing import Any
import streamlit as st
import pandas as pd
from src.retailiq.config import Theme


def kpi_card(label: str, value: str, subtitle: str = "") -> str:
    """Return HTML for a single KPI metric card.

    Args:
        label: Short metric name (e.g., 'Total Sales')
        value: Formatted value string (e.g., '$1.2B')
        subtitle: Optional context line (e.g., '45 stores, 143 weeks')
    """
    subtitle_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {subtitle_html}
    </div>
    """


def render_kpi_row(kpis: list[dict[str, Any]]) -> None:
    """Render a row of standardized KPI cards.

    Args:
        kpis: List of dicts with keys 'label', 'value', and optional 'subtitle'.
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            st.markdown(
                kpi_card(kpi["label"], kpi["value"], kpi.get("subtitle", "")),
                unsafe_allow_html=True,
            )


def formatted_dataframe(
    df: pd.DataFrame,
    max_rows: int = 50,
    title: str | None = None,
    description: str | None = None,
) -> None:
    """Display a clean, formatted dataframe with header information and CSV export.

    Args:
        df: DataFrame to display.
        max_rows: Maximum rows to show (prevents browser overload).
        title: Optional section title above the table.
        description: Optional explanation below the title.
    """
    if title:
        st.markdown(f"<div style='font-size: 0.95rem; font-weight: 600; color: #F9FAFB; margin-bottom: 0.2rem;'>{title}</div>", unsafe_allow_html=True)
    if description:
        st.caption(description)

    display_df = df.head(max_rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df):,} rows.")

    # CSV export
    csv = df.to_csv(index=False)
    file_slug = (title or "dataset").lower().replace(" ", "_").replace("(", "").replace(")", "")
    st.download_button(
        label="Export CSV",
        data=csv,
        file_name=f"{file_slug}.csv",
        mime="text/csv",
        key=f"dl_{file_slug}_{abs(hash(str(df.shape) + str(title)))}",
    )


def format_currency(value: float, precision: int = 0) -> str:
    """Format a number as USD currency string."""
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:,.{precision}f}"


def format_number(value: float, precision: int = 1) -> str:
    """Format a number with thousands separators."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{precision}f}K"
    else:
        return f"{value:,.{precision}f}"


def format_percent(value: float, precision: int = 1) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{precision}f}%"


def page_context(
    why: str,
    business_problem: str,
    data_source: str,
    what_you_learn: str,
) -> None:
    """Render a structured methodology and business context card.

    Args:
        why: Operational rationale for this page.
        business_problem: Concrete business decisions addressed.
        data_source: Source datasets and reports utilized.
        what_you_learn: Actionable takeaways for decision-makers.
    """
    with st.expander("Methodology & Business Context", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.6rem;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Operational Purpose</div>
                <div style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45;">{why}</div>
            </div>
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Business Decision Addressed</div>
                <div style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45;">{business_problem}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.6rem;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Data Artifacts & Evidence</div>
                <div style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45;"><code>{data_source}</code></div>
            </div>
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Key Takeaways</div>
                <div style="font-size: 0.83rem; color: #CBD5E1; line-height: 1.45;">{what_you_learn}</div>
            </div>
            """, unsafe_allow_html=True)
