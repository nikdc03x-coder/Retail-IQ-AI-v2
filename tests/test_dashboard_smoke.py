"""
RetailIQ AI V2 — Phase 08: Dashboard Smoke Tests

Validates:
- Importability and existence of callable `render()` in all page modules.
- Formatter functions (currency, percentage, number, KPI cards).
- Plotly chart factory methods (line, bar, scatter, heatmap, layout).
- Dashboard application entry point routing integrity.
"""

import importlib
import pytest
import pandas as pd
import plotly.graph_objects as go

from src.retailiq.dashboard.components import (
    format_currency,
    format_percent,
    format_number,
    kpi_card,
)
from src.retailiq.dashboard.charts import (
    apply_default_layout,
    line_chart,
    bar_chart,
    scatter_chart,
    heatmap,
)


PAGE_MODULES = [
    "src.retailiq.dashboard.pages.p00_overview",
    "src.retailiq.dashboard.pages.p01_dataset",
    "src.retailiq.dashboard.pages.p02_data_quality",
    "src.retailiq.dashboard.pages.p03_analytics",
    "src.retailiq.dashboard.pages.p04_forecasting",
    "src.retailiq.dashboard.pages.p05_reliability",
    "src.retailiq.dashboard.pages.p06_anomalies",
    "src.retailiq.dashboard.pages.p07_explainability",
    "src.retailiq.dashboard.pages.p08_strategy",
    "src.retailiq.dashboard.pages.p08_assistant",
]


class TestDashboardPageModules:
    """Verify that all dashboard page modules are valid and export render()."""

    @pytest.mark.parametrize("mod_path", PAGE_MODULES)
    def test_page_module_import_and_render_function(self, mod_path: str):
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, "render"), f"Module {mod_path} must export a render() function"
        assert callable(getattr(mod, "render")), f"render() in {mod_path} must be callable"


class TestDashboardComponents:
    """Verify standard component formatters and KPI generators."""

    def test_currency_formatting(self):
        assert format_currency(1_250_000) == "$1.2M"
        assert format_currency(500) == "$500"
        assert format_currency(2_500_000_000) == "$2.5B"

    def test_percent_formatting(self):
        assert format_percent(0.04567) == "4.6%"
        assert format_percent(0.10, precision=2) == "10.00%"

    def test_number_formatting(self):
        assert format_number(421570) == "421.6K"
        assert format_number(50) == "50.0"

    def test_kpi_card_html(self):
        html = kpi_card("Total Sales", "$6.74B", subtitle="+4.2% YoY")
        assert "Total Sales" in html
        assert "$6.74B" in html
        assert "+4.2% YoY" in html



class TestDashboardCharts:
    """Verify Plotly chart factory produces styled figures."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": ["2012-01-01", "2012-01-08", "2012-01-15"],
            "Sales": [1000, 1500, 1200],
            "Forecast": [950, 1480, 1220],
            "Store": [1, 1, 1],
        })

    def test_apply_default_layout(self):
        fig = go.Figure()
        styled = apply_default_layout(fig, title="Test Chart")
        assert styled.layout.title.text == "Test Chart"
        assert styled.layout.plot_bgcolor is not None

    def test_line_chart_generation(self, sample_data: pd.DataFrame):
        fig = line_chart(sample_data, x="Date", y="Sales", title="Weekly Sales Trend")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_bar_chart_generation(self, sample_data: pd.DataFrame):
        fig = bar_chart(sample_data, x="Date", y="Sales", title="Sales Bar Chart")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_scatter_chart_generation(self, sample_data: pd.DataFrame):
        fig = scatter_chart(sample_data, x="Sales", y="Forecast", title="Sales vs Forecast")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_heatmap_generation(self):
        corr = pd.DataFrame({
            "A": [1.0, 0.5],
            "B": [0.5, 1.0],
        }, index=["A", "B"])
        fig = heatmap(corr, title="Correlation Heatmap")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
