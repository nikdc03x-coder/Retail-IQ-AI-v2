"""
RetailIQ AI V2 — Page 04: Sales Forecasting & Model Evaluation

Rigorous chronological benchmark comparing baseline heuristics against
Ridge Regression, Random Forest, and champion XGBoost forecasting models.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from src.retailiq.config import REPORT_SUBDIRS, Theme
from src.retailiq.dashboard.components import (
    format_currency,
    format_percent,
    formatted_dataframe,
    page_context,
    render_kpi_row,
)
from src.retailiq.dashboard.charts import bar_chart, histogram, line_chart, scatter_chart
from src.retailiq.dashboard.theme import context_box, page_header, section_header


def render() -> None:
    """Render the Sales Forecasting page."""
    page_header(
        "Sales Forecasting & Model Benchmark Leaderboard",
        "Chronological Benchmark: Naive Persistence Baselines vs Ridge, Random Forest & XGBoost",
        category="Predictive Modeling",
    )

    page_context(
        why="Retail managers need predictive foresight to plan staffing, logistics, and stock replenishment one week in advance.",
        business_problem="Answers: Can ML models reliably beat simple persistence baselines? What is the expected error margin? Which features drive predictive power?",
        data_source="Pre-computed experiment reports in reports/sales_forecasting/ evaluated on 990 held-out test observations.",
        what_you_learn="XGBoost is the champion model with 3.83% WAPE ($39.8k MAE), delivering a 22.3% error reduction over the Previous-Week Naive baseline.",
    )

    out_dir = REPORT_SUBDIRS["sales_forecasting"]

    if not (out_dir / "01_model_comparison_metrics.csv").exists():
        st.warning(
            "Forecasting reports not found. Please run the experiment script:\n\n"
            "`python scripts/04_sales_forecasting/run_forecasting.py`"
        )
        return

    @st.cache_data
    def load_forecasting_data() -> dict:
        metrics = pd.read_csv(out_dir / "01_model_comparison_metrics.csv")
        preds = pd.read_csv(out_dir / "02_test_predictions.csv", parse_dates=["Date"])
        importance = pd.read_csv(out_dir / "03_feature_importance.csv")
        with open(out_dir / "04_experiment_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "metrics": metrics,
            "preds": preds,
            "importance": importance,
            "meta": meta,
        }

    data = load_forecasting_data()
    metrics_df = data["metrics"]
    preds_df = data["preds"]
    importance_df = data["importance"]
    meta = data["meta"]

    test_metrics = metrics_df[metrics_df["Split"] == "Test"].sort_values("WAPE_Pct").reset_index(drop=True)
    champ = test_metrics.iloc[0]
    naive_test = test_metrics[test_metrics["Model"] == "Previous-Week Naive"].iloc[0]
    error_reduction = ((naive_test["WAPE_Pct"] - champ["WAPE_Pct"]) / naive_test["WAPE_Pct"] * 100.0)

    # ── Top KPI Row ──
    render_kpi_row([
        {"label": "Champion Model", "value": champ["Model"], "subtitle": f"WAPE: {champ['WAPE_Pct']}% on test set"},
        {"label": "Champion Test MAE", "value": format_currency(champ["MAE"]), "subtitle": f"RMSE: {format_currency(champ['RMSE'])}"},
        {"label": "Baseline WAPE (Naive)", "value": f"{naive_test['WAPE_Pct']}%", "subtitle": f"MAE: {format_currency(naive_test['MAE'])}"},
        {"label": "Error Reduction", "value": f"{error_reduction:.1f}%", "subtitle": "Improvement over Naive persistence"},
        {"label": "Forecast Horizon", "value": "1-Week-Ahead", "subtitle": "Strictly backward-looking features"},
    ])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Interactive Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Benchmark Leaderboard",
        "Predictions vs Actuals",
        "Engineered Feature Importance",
        "Residual Diagnostics",
    ])

    # ── Tab 1: Leaderboard ──
    with tab1:
        section_header("Chronological Test Set Leaderboard", "Standardized out-of-sample evaluation across all 45 store locations")
        context_box(
            "Fair Benchmark Protocol: All models were trained on historical records and evaluated on the exact same 990 test observations (June 2012 - October 2012) spanning all 45 stores. Lower MAE, RMSE, and WAPE are strictly preferred.",
            title="Benchmark Methodology",
        )

        c1, c2 = st.columns([3, 2])
        with c1:
            fig_lead = bar_chart(
                test_metrics,
                x="WAPE_Pct",
                y="Model",
                title="Model Test WAPE Comparison (Lower is Better)",
                orientation="h",
            )
            st.plotly_chart(fig_lead, use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="section-card">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #818CF8; margin-bottom: 0.25rem;">Evaluation Synthesis</div>
                <h3>Model Performance Rankings</h3>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>1. XGBoost:</b> Top performer (3.83% WAPE) capturing nonlinear interactions between store capacity and recent momentum.</p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>2. Ridge Linear:</b> Highly competitive (3.91% WAPE), demonstrating strong linearity in weekly store-level aggregation.</p>
                <p style="color: {Theme.LIGHT_GRAY}; font-size: 0.84rem; margin-bottom: 0.4rem;"><b>3. Random Forest:</b> Solid performance (4.00% WAPE), handily outperforming baselines.</p>
                <p style="color: {Theme.DARK_GRAY}; font-size: 0.8rem; margin: 0;"><b>4. Naive Baselines:</b> Previous-week (4.93%) and seasonal (5.12%) exhibit clear accuracy degradation.</p>
            </div>
            """, unsafe_allow_html=True)

        formatted_dataframe(
            metrics_df,
            title="Complete Model Evaluation Metrics (Validation & Test Sets)",
            description="MAE, RMSE, WAPE, R², and Mean Percentage Error (bias) across experimental partitions.",
        )

    # ── Tab 2: Predictions vs Actuals ──
    with tab2:
        section_header("Forecast Trajectory by Store Location", "Granular actual vs predicted sales series over test horizon")
        context_box(
            "Select an individual store to observe how closely model predictions track ground-truth actual sales during the held-out test window (June - Oct 2012).",
            title="Trajectory Inspection",
        )

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            selected_store = st.selectbox(
                "Select Store Location",
                options=sorted(preds_df["Store"].unique()),
                index=19,  # Default to Store 20
                key="fc_store_sel",
            )
            models_to_plot = st.multiselect(
                "Models to Overlay",
                options=["XGBoost_Pred", "RF_Pred", "Ridge_Pred", "Naive_Pred", "Seasonal_Pred"],
                default=["XGBoost_Pred", "Naive_Pred"],
                key="fc_models_sel",
            )

        with col_s2:
            store_sub = preds_df[preds_df["Store"] == selected_store].copy()
            store_mae = (store_sub["Actual_Sales"] - store_sub["XGBoost_Pred"]).abs().mean()
            store_wape = ((store_sub["Actual_Sales"] - store_sub["XGBoost_Pred"]).abs().sum() / store_sub["Actual_Sales"].sum()) * 100.0
            st.metric(
                label=f"Store {selected_store} XGBoost Performance",
                value=f"{store_wape:.2f}% WAPE",
                delta=f"MAE: {format_currency(store_mae)}",
                delta_color="off",
            )

        plot_cols = ["Date", "Actual_Sales"] + models_to_plot
        melted = store_sub[plot_cols].melt(id_vars=["Date"], var_name="Series", value_name="Sales")
        fig_ts = line_chart(
            melted,
            x="Date",
            y="Sales",
            color="Series",
            title=f"Store {selected_store} — 1-Week-Ahead Forecast vs Actuals (Test Period)",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Tab 3: Feature Importance ──
    with tab3:
        section_header("Predictive Feature Importance & Drivers", "Model reliance across engineered lag, rolling, and calendar signals")
        context_box(
            "Feature importance indicates which variables most frequently split decision trees to minimize mean squared error. Note: High predictive importance indicates correlation and model reliance, not real-world causation.",
            title="Model Interpretation",
        )

        model_imp_sel = st.radio("Select Model", options=["XGBoost", "Random Forest"], horizontal=True, key="imp_model_sel")
        imp_sub = importance_df[importance_df["Model"] == model_imp_sel].head(12).sort_values("Importance", ascending=True)

        fig_imp = bar_chart(
            imp_sub,
            x="Importance",
            y="Feature",
            title=f"Top 12 Predictive Features — {model_imp_sel}",
            orientation="h",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        formatted_dataframe(
            importance_df[importance_df["Model"] == model_imp_sel],
            title=f"Full Feature Importance Table ({model_imp_sel})",
            description="Relative percentage contribution and raw impurity/gain reduction.",
        )

    # ── Tab 4: Residual Diagnostics ──
    with tab4:
        section_header("Forecast Residual Analysis & Error Dispersion", "Diagnostic checks for systematic model bias")
        context_box(
            "Residual Formula: e = Actual - Predicted. A well-calibrated forecasting model exhibits residuals centered near zero with symmetrical dispersion.",
            title="Residual Calibration",
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_res_hist = histogram(
                preds_df,
                x="XGBoost_Residual",
                title="XGBoost Forecast Residual Distribution",
                nbins=40,
            )
            st.plotly_chart(fig_res_hist, use_container_width=True)

        with c2:
            fig_scatter_err = scatter_chart(
                preds_df,
                x="Actual_Sales",
                y="XGBoost_Residual",
                color="Type",
                title="Actual Sales Volume vs Forecast Residual",
            )
            st.plotly_chart(fig_scatter_err, use_container_width=True)

        formatted_dataframe(
            preds_df,
            max_rows=50,
            title="Held-Out Test Predictions Audit Table",
            description="Individual store-week actuals, model predictions, absolute error, and percentage error.",
        )
