"""
RetailIQ AI V2 — Reusable Dashboard Components

KPI cards, formatted tables, CSV export buttons, and other reusable UI elements.
"""

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


def render_kpi_row(kpis: list[dict]) -> None:
    """Render a row of KPI cards.

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
    """Display a clean, formatted dataframe with optional title and CSV download.

    Args:
        df: DataFrame to display.
        max_rows: Maximum rows to show (prevents browser overload).
        title: Optional section title above the table.
        description: Optional explanation below the title.
    """
    if title:
        st.markdown(f"**{title}**")
    if description:
        st.caption(description)

    display_df = df.head(max_rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df):,} rows.")

    # CSV download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{(title or 'data').lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_{title}_{id(df)}",
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
    """Render a standard page context section explaining why this page exists.

    Args:
        why: Why this page was built.
        business_problem: Which business problem it addresses.
        data_source: Which dataset or report file it uses.
        what_you_learn: What the user can learn from this page.
    """
    with st.expander("ℹ️ About this page", expanded=False):
        st.markdown(f"""
**Why this page exists:** {why}

**Business problem:** {business_problem}

**Data source:** {data_source}

**What you can learn:** {what_you_learn}
        """)

