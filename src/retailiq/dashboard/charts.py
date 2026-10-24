"""
RetailIQ AI V2 — Plotly Chart Factory

Consistent chart styling using the project's design system.
All chart functions return a plotly Figure object for Streamlit rendering.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.retailiq.config import Theme


def apply_default_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the project's standard chart layout to a Plotly figure.

    Args:
        fig: Plotly figure to style.
        title: Chart title.

    Returns:
        The styled figure.
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=Theme.WHITE)),
        plot_bgcolor=Theme.PLOT_BG,
        paper_bgcolor=Theme.PAPER_BG,
        font=dict(family=Theme.FONT_FAMILY, color=Theme.FONT_COLOR, size=12),
        xaxis=dict(
            gridcolor=Theme.GRID_COLOR,
            zerolinecolor=Theme.GRID_COLOR,
        ),
        yaxis=dict(
            gridcolor=Theme.GRID_COLOR,
            zerolinecolor=Theme.GRID_COLOR,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=Theme.LIGHT_GRAY),
        ),
        margin=dict(l=60, r=30, t=60, b=50),
        hoverlabel=dict(
            bgcolor=Theme.CARD_BG,
            bordercolor=Theme.CARD_BORDER,
            font_size=12,
            font_color=Theme.WHITE,
        ),
    )
    return fig


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str | None = None,
    labels: dict | None = None,
) -> go.Figure:
    """Create a styled line chart.

    Args:
        df: Source data.
        x: Column name for x-axis.
        y: Column name for y-axis.
        title: Chart title.
        color: Column for color grouping.
        labels: Axis label overrides.
    """
    fig = px.line(
        df, x=x, y=y, color=color, labels=labels or {},
        color_discrete_sequence=Theme.CHART_COLORS,
    )
    return apply_default_layout(fig, title)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    orientation: str = "v",
    color: str | None = None,
    labels: dict | None = None,
) -> go.Figure:
    """Create a styled bar chart.

    Args:
        df: Source data.
        x: Column for x-axis (or values if horizontal).
        y: Column for y-axis (or categories if horizontal).
        title: Chart title.
        orientation: 'v' for vertical, 'h' for horizontal.
        color: Column for color grouping.
        labels: Axis label overrides.
    """
    fig = px.bar(
        df, x=x, y=y, orientation=orientation, color=color,
        labels=labels or {},
        color_discrete_sequence=Theme.CHART_COLORS,
    )
    return apply_default_layout(fig, title)


def histogram(
    df: pd.DataFrame,
    x: str,
    title: str = "",
    nbins: int = 50,
    labels: dict | None = None,
) -> go.Figure:
    """Create a styled histogram."""
    fig = px.histogram(
        df, x=x, nbins=nbins, labels=labels or {},
        color_discrete_sequence=[Theme.VIOLET],
    )
    return apply_default_layout(fig, title)


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str | None = None,
    labels: dict | None = None,
) -> go.Figure:
    """Create a styled scatter plot."""
    fig = px.scatter(
        df, x=x, y=y, color=color, labels=labels or {},
        color_discrete_sequence=Theme.CHART_COLORS,
    )
    return apply_default_layout(fig, title)


def box_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    labels: dict | None = None,
) -> go.Figure:
    """Create a styled box plot."""
    fig = px.box(
        df, x=x, y=y, labels=labels or {},
        color_discrete_sequence=[Theme.VIOLET],
    )
    return apply_default_layout(fig, title)


def heatmap(
    df: pd.DataFrame,
    title: str = "",
    colorscale: str = "Viridis",
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Create a styled heatmap from a DataFrame."""
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
        )
    )
    return apply_default_layout(fig, title)


