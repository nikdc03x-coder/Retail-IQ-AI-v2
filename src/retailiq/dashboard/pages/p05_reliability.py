"""
RetailIQ AI V2 — Page 05: Forecast Reliability Analysis

Interactive analysis of model reliability across locations, time periods,
holiday volatility, and rolling historical backtests.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.retailiq.config import REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_percent,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import apply_default_layout, bar_chart, line_chart, scatter_chart
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Forecast Reliability page."""
    page_header(
        "Forecast Reliability & Error Diagnostics",
        "Spatial Disparity, Holiday Volatility, Baseline Win Rates & Rolling Backtests",
    )

    page_context(
        why="A single aggregate error number hides operational risks. Managers must understand which stores are difficult to predict and whether accuracy degrades during promotional surges.",
        business_problem="Answers: Do errors vary by store location? Are holiday forecasts riskier? Does XGBoost consistently beat the baseline over multiple historical periods?",
        data_source="Pre-computed reports in reports/forecast_reliability/ evaluated across 45 stores and 4 backtest folds.",
        what_you_learn="Store 37 is the most reliable (2.06% WAPE); Store 14 is the most volatile (10.59% WAPE); XGBoost beats the baseline in 77.8% of stores and 100% of backtest folds.",
    )

    out_dir = REPORT_SUBDIRS["forecast_reliability"]

    # Verify reports exist
    if not (out_dir / "01_store_error_summary.csv").exists():
        st.warning(
            "⚠️ Reliability reports not found. Please run the reliability audit:\n\n"
            "`python scripts/05_forecast_reliability/run_reliability.py`"
        )
        return

    @st.cache_data
    def load_reliability_data() -> dict:
        store_err = pd.read_csv(out_dir / "01_store_error_summary.csv")
        week_err = pd.read_csv(out_dir / "02_weekly_error_summary.csv", parse_dates=["Date"])
        hol_err = pd.read_csv(out_dir / "03_holiday_error_summary.csv")
        store_rel = pd.read_csv(out_dir / "04_store_relative_errors.csv")
        backtest = pd.read_csv(out_dir / "05_rolling_backtest_results.csv")
        with open(out_dir / "06_forecast_reliability_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        return {
            "store_err": store_err,
            "week_err": week_err,
            "hol_err": hol_err,
            "store_rel": store_rel,
            "backtest": backtest,
            "summary": summary,
        }

    data = load_reliability_data()
    store_err = data["store_err"]
    week_err = data["week_err"]
    hol_err = data["hol_err"]
    store_rel = data["store_rel"]
    backtest = data["backtest"]
    summary = data["summary"]

    best_store = store_err.iloc[0]
    worst_store = store_err.iloc[-1]
    hol_wape = summary["holiday_degradation"]["holiday_wape"]
    non_hol_wape = summary["holiday_degradation"]["non_holiday_wape"]
    hol_change = summary["holiday_degradation"]["wape_change_pct"]

    # ── Top KPI Row ──
    render_kpi_row([
        {"label": "Store Win Rate", "value": f"{summary['store_win_rate_vs_naive_pct']}%", "subtitle": f"35 of 45 stores beat Naive"},
        {"label": "Most Reliable Store", "value": f"Store {int(best_store['Store'])}", "subtitle": f"{best_store['WAPE_Pct']}% WAPE ({format_currency(best_store['MAE'])})"},
        {"label": "Highest Volatility Store", "value": f"Store {int(worst_store['Store'])}", "subtitle": f"{worst_store['WAPE_Pct']}% WAPE ({format_currency(worst_store['MAE'])})"},
        {"label": "Holiday Error Shift", "value": f"{hol_change:+.1f}%", "subtitle": f"{hol_wape}% vs {non_hol_wape}% non-holiday"},
        {"label": "Backtest Consistency", "value": "100.0%", "subtitle": f"{summary['backtest_summary']['folds_outperformed']}/{summary['backtest_summary']['folds_evaluated']} folds won by XGBoost"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏬 Store Error Disparity",
        "📅 Weekly & Holiday Error",
        "⚔️ Model vs Baseline Head-to-Head",
        "🔄 Rolling Backtest Stability",
    ])

    # ── Tab 1: Store Disparity ──
    with tab1:
        section_header("Store-by-Store Error Disparity (45 Stores)", "🏬")
        context_box(
            "**Key Finding:** Forecasting error varies by a factor of 5x across locations. "
            "High-volume, stable locations like Store 37 achieve a **2.06% WAPE**, whereas smaller or more volatile locations like Store 14 exhibit **10.59% WAPE**."
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            sorted_stores = store_err.sort_values("WAPE_Pct", ascending=True).copy()
            sorted_stores["Store_Label"] = "Store " + sorted_stores["Store"].astype(str)
            fig_store_wape = bar_chart(
                sorted_stores,
                x="WAPE_Pct",
                y="Store_Label",
                title="WAPE by Store (Lowest to Highest Error)",
                orientation="h",
                color="Type",
            )
            st.plotly_chart(fig_store_wape, use_container_width=True)

        with c2:
            st.markdown("#### Top 5 Most Predictable Stores")
            top_5_rel = store_err.head(5)[["Reliability_Rank", "Store", "Type", "WAPE_Pct", "MAE", "Mean_Weekly_Sales"]]
            st.dataframe(top_5_rel, use_container_width=True, hide_index=True)

            st.markdown("#### Top 5 Most Challenging Stores")
            bot_5_rel = store_err.tail(5).sort_values("WAPE_Pct", ascending=False)[["Reliability_Rank", "Store", "Type", "WAPE_Pct", "MAE", "Mean_Weekly_Sales"]]
            st.dataframe(bot_5_rel, use_container_width=True, hide_index=True)

        formatted_dataframe(
            store_err,
            title="Complete 45-Store Reliability Table",
            description="MAE, RMSE, WAPE, maximum single-week absolute error, and directional bias per store.",
        )

    # ── Tab 2: Weekly & Holiday Error ──
    with tab2:
        section_header("Temporal Error Dynamics & Holiday Vulnerability", "📅")
        context_box(
            "**Holiday Error Assessment:** During holiday weeks (e.g. Labor Day week), average WAPE increases from **3.79% to 4.58%** (+20.8% relative error shift). "
            "Promotional swings make holiday demand inherently more volatile to predict."
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_week = line_chart(
                week_err,
                x="Date",
                y="WAPE_Pct",
                title="Weekly Forecast WAPE Across 22 Test Weeks (%)",
            )
            st.plotly_chart(fig_week, use_container_width=True)

        with c2:
            st.markdown("#### Holiday vs Non-Holiday Error Breakdown")
            st.dataframe(hol_err, use_container_width=True, hide_index=True)
            context_box(
                "**Business Recommendation:** Retail inventory planners should maintain safety stock buffers during holiday windows to absorb higher forecast variance."
            )

        formatted_dataframe(
            week_err,
            title="Weekly Test Error Time Series",
            description="Week-by-week actuals, predictions, MAE, and WAPE across the held-out evaluation window.",
        )

    # ── Tab 3: Baseline Head-to-Head ──
    with tab3:
        section_header("XGBoost vs Previous-Week Naive Head-to-Head", "⚔️")
        context_box(
            "**Store-Level Dominance:** XGBoost beats the naive persistence baseline in **35 out of 45 stores (77.8% win rate)**. "
            "In the scatter plot below, points lying below the 45-degree parity line represent locations where XGBoost delivers superior accuracy."
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            # Scatter plot: Naive WAPE vs XGB WAPE
            fig_head = px.scatter(
                store_rel,
                x="Naive_WAPE",
                y="XGB_WAPE",
                color="Type",
                hover_data=["Store", "Error_Reduction_Pct"],
                title="XGBoost WAPE vs Naive WAPE by Store",
                color_discrete_sequence=Theme.CHART_COLORS,
            )
            # Add 45-degree parity line
            max_val = max(store_rel["Naive_WAPE"].max(), store_rel["XGB_WAPE"].max()) + 1.0
            fig_head.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Parity (y = x)",
                line=dict(dash="dash", color=Theme.DARK_GRAY),
            ))
            fig_head = apply_default_layout(fig_head, "XGBoost WAPE vs Naive WAPE by Store")
            st.plotly_chart(fig_head, use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="section-card">
                <h3>Head-to-Head Summary</h3>
                <p><b>XGBoost Wins:</b> 35 stores (77.8%)</p>
                <p><b>Naive Wins:</b> 10 stores (22.2%)</p>
                <p><b>Average Error Reduction:</b> -18.7% across all 45 stores</p>
                <p><b>Insight:</b> Stores where Naive outperformed were primarily ultra-stable Type C stores with near-flat weekly demand.</p>
            </div>
            """, unsafe_allow_html=True)

        formatted_dataframe(
            store_rel,
            title="Store-by-Store Win Rate Audit",
            description="Direct comparison of XGBoost WAPE vs Naive WAPE and percentage error reduction.",
        )

    # ── Tab 4: Rolling Backtest ──
    with tab4:
        section_header("Expanding-Window Rolling Backtest (4 Folds)", "🔄")
        context_box(
            "**Methodology:** An expanding-window backtest retrains the model on expanding historical slices and evaluates on consecutive 8-week test windows. "
            "XGBoost outperformed the baseline in **100% of backtest folds**, demonstrating temporal stability."
        )

        # Bar chart comparing fold WAPEs
        melted_bt = backtest[["Fold", "XGB_WAPE", "Naive_WAPE"]].melt(
            id_vars=["Fold"], var_name="Model", value_name="WAPE_Pct"
        )
        fig_bt = bar_chart(
            melted_bt,
            x="Fold",
            y="WAPE_Pct",
            color="Model",
            title="Rolling Backtest WAPE by Fold (Lower is Better)",
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        formatted_dataframe(
            backtest,
            title="Historical Backtest Evaluation Results",
            description="Fold periods, training weeks, test windows, and error reduction percentage.",
        )
