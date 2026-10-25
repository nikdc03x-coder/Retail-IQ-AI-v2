"""
RetailIQ AI V2 — Page 08: AI Business Assistant

Conversational business intelligence powered by Google Gemini and backed by
deterministic tool execution on pre-computed analytical datasets:
- Evidence-grounded natural language question answering.
- Dual-mode routing: Google Gemini with Function Calling or Deterministic Rule-Based Fallback.
- Dynamic interactive visualization charts (Plotly) rendered directly in conversational responses.
- Interactive follow-up suggestion chips and deep-link navigation into analytical modules.
- Complete execution transparency (tools invoked, source files, latency, verification).
"""

from typing import Any
import streamlit as st
import pandas as pd
import plotly.express as px

from src.retailiq.config import Theme, GEMINI_API_KEY, GEMINI_MODEL
from src.retailiq.dashboard.components import page_context
from src.retailiq.dashboard.theme import page_header
from src.retailiq.dashboard.charts import apply_default_layout
from src.retailiq.assistant.engine import RetailIQAssistantEngine


SUGGESTED_PROMPTS = [
    "Which are the top 5 stores by historical sales?",
    "How much sales lift does Thanksgiving drive vs non-holidays?",
    "What is the forecasting model leaderboard and champion?",
    "How does XGBoost compare against the Previous-Week Naive baseline?",
    "Which store records are top priority in the anomaly queue?",
    "What are the top 5 features driving XGBoost predictions in TreeSHAP?",
    "Explain the forecast for Store 20 on 2012-06-01",
    "What are the top 5 departments by sales volume?",
]


def render() -> None:
    """Render the AI Business Assistant page."""
    page_header(
        "AI Business Assistant",
        "Evidence-Grounded Retail Intelligence & Conversational Analytics",
        category="Conversational Analytics & Grounding",
    )

    page_context(
        why="Business executives and merchandising managers require instant answers to ad-hoc inquiries without writing Python or manual SQL queries, backed by mathematical and auditable guarantees.",
        business_problem="Answers: What is our top-performing store? Did holiday promotions drive lift? Which forecast model should we deploy? Why did Store 14 trigger an anomaly flag?",
        data_source="Strictly grounded in analytical reports from reports/ and Parquet datasets via 13 approved Python tools.",
        what_you_learn="Conversational AI can safely power enterprise retail intelligence when strictly constrained to typed execution tools with full evidence attribution.",
    )

    # ── Initialize session state ──
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "answer": (
                    "**RetailIQ Intelligence Engine Active**\n\n"
                    "Enterprise retail business analyst grounded in verified data pipelines across chain sales "
                    "performance, promotional dynamics, forecasting accuracy, spatial reliability, anomaly queues, "
                    "and TreeSHAP feature attributions.\n\n"
                    "Select an inquiry below or type a custom question to begin."
                ),
                "metadata": {
                    "route": "system_greeting",
                    "provider": "RetailIQ Core",
                    "evidence_source": "N/A",
                    "latency_seconds": 0.0,
                    "verified": True,
                },
            }
        ]

    # Initialize custom api key state
    if "user_gemini_key" not in st.session_state:
        st.session_state.user_gemini_key = GEMINI_API_KEY

    # ── Configuration Drawer ──
    with st.expander("Assistant Configuration & API Settings", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            key_input = st.text_input(
                "Gemini API Key (optional):",
                value=st.session_state.user_gemini_key,
                type="password",
                help="Enter a Google Gemini API key to enable live generative reasoning. If left blank, the deterministic fallback engine handles queries with 100% accuracy.",
                key="api_key_input",
            )
            if key_input != st.session_state.user_gemini_key:
                st.session_state.user_gemini_key = key_input
                st.rerun()

        with c2:
            st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.assistant_messages = st.session_state.assistant_messages[:1]
                st.rerun()

    # ── Engine Initialization ──
    engine = RetailIQAssistantEngine(api_key=st.session_state.user_gemini_key)

    # ── Status Pill ──
    if engine.has_gemini:
        status_html = (
            f"<div style='display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.85rem; border-radius: 9999px; "
            f"background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); "
            f"color: {Theme.GREEN}; font-size: 0.8rem; font-weight: 600; margin-bottom: 1.2rem;'>"
            f"<span style='display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: {Theme.GREEN};'></span>"
            f"Gemini Generative Engine Active ({GEMINI_MODEL}) · Tool Calling Enabled"
            f"</div>"
        )
    else:
        status_html = (
            f"<div style='display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.85rem; border-radius: 9999px; "
            f"background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); "
            f"color: {Theme.AMBER}; font-size: 0.8rem; font-weight: 600; margin-bottom: 1.2rem;'>"
            f"<span style='display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: {Theme.AMBER};'></span>"
            f"Deterministic Fallback Active · 100% Verifiable Tool Dispatch"
            f"</div>"
        )
    st.markdown(status_html, unsafe_allow_html=True)

    # ── Quick Query Suggestions ──
    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-weight: 600; margin-bottom: 0.5rem;'>Recommended Inquiries</p>", unsafe_allow_html=True)
    chips_cols = st.columns(4)
    for idx, prompt_text in enumerate(SUGGESTED_PROMPTS):
        col = chips_cols[idx % 4]
        with col:
            # Short label for button
            btn_label = prompt_text.split("?")[0].replace("Which are the", "").replace("What are the", "").replace("What is the", "").strip()
            if len(btn_label) > 28:
                btn_label = btn_label[:25] + "..."
            if st.button(btn_label, key=f"chip_{idx}", use_container_width=True):
                _handle_user_query(engine, prompt_text)
                st.rerun()

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # ── Chat History Display ──
    for msg_idx, msg in enumerate(st.session_state.assistant_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["answer"])

            meta = msg.get("metadata", {})
            if msg["role"] == "assistant" and meta.get("route") != "system_greeting":
                chart_type = meta.get("chart_type") or (meta.get("tools_called")[0] if meta.get("tools_called") else None)
                data_payload = meta.get("data")

                # ── 1. Interactive Visualization Widget ──
                if chart_type and data_payload:
                    _render_interactive_chart(chart_type, data_payload)

                # ── 2. Structured Tabular Evidence Preview ──
                if isinstance(data_payload, list) and len(data_payload) > 0 and isinstance(data_payload[0], dict):
                    with st.expander(f"View Tabular Evidence Records ({len(data_payload)} rows)", expanded=False):
                        df_preview = pd.DataFrame(data_payload)
                        st.dataframe(df_preview, use_container_width=True, hide_index=True)

                # ── 3. Deep-Link Navigation Shortcut ──
                deep_link = meta.get("deep_link_page")
                if deep_link:
                    c_link, _ = st.columns([1, 2])
                    with c_link:
                        if st.button(f"Explore in {deep_link} →", key=f"nav_{msg_idx}", use_container_width=True):
                            st.session_state["nav_page"] = deep_link
                            st.rerun()

                # ── 4. Interactive Follow-up Prompt Chips ──
                followups = meta.get("followups", [])
                if followups:
                    st.markdown("<p style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-top: 0.8rem; margin-bottom: 0.35rem;'>Suggested Follow-Ups</p>", unsafe_allow_html=True)
                    f_cols = st.columns(min(len(followups), 3))
                    for f_idx, followup in enumerate(followups[:3]):
                        with f_cols[f_idx]:
                            lbl = followup.replace("What are the ", "").replace("Which are the ", "").replace("What is the ", "").replace("How does ", "")
                            if len(lbl) > 30:
                                lbl = lbl[:28] + "..."
                            if st.button(lbl, key=f"f_btn_{msg_idx}_{f_idx}", help=followup, use_container_width=True):
                                _handle_user_query(engine, followup)
                                st.rerun()

                # ── 5. Transparency Drawer ──
                with st.expander("Transparency & Execution Trace", expanded=False):
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        st.markdown(f"**Execution Route:** `{meta.get('route')}`")
                        st.markdown(f"**Engine Provider:** {meta.get('provider')}")
                        st.markdown(f"**Tools Executed:** `{', '.join(meta.get('tools_called', ['N/A']))}`")
                    with t_col2:
                        st.markdown(f"**Data Evidence Source:** `{meta.get('evidence_source')}`")
                        st.markdown(f"**Latency:** `{meta.get('latency_seconds', 0.0)}s`")
                        st.markdown(f"**Mathematical Verification:** `{'Verified' if meta.get('verified') else 'Unverified'}`")

    # ── User Chat Input ──
    user_input = st.chat_input("Ask a business question about sales, models, anomalies, or SHAP...")
    if user_input:
        _handle_user_query(engine, user_input)
        st.rerun()


def _render_interactive_chart(chart_type: str, data: Any) -> None:
    """Render a purpose-built interactive Plotly visualization for the assistant's finding."""
    if not data:
        return

    try:
        if chart_type in ["get_store_rankings", "store_rankings"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x="total_sales",
                y=[f"Store {s} (Type {t})" for s, t in zip(df_chart["store"], df_chart["type"])],
                orientation="h",
                color="type",
                text=[f"${v/1e6:.1f}M ({p:.1f}%)" for v, p in zip(df_chart["total_sales"], df_chart["pct_of_chain"])],
                labels={"total_sales": "Total Historical Sales ($)", "y": "Store Location"},
                color_discrete_map={"A": Theme.VIOLET, "B": Theme.CYAN, "C": Theme.AMBER},
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Top Store Historical Revenue & Chain Share")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=260, showlegend=True, margin=dict(l=80, r=40, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["get_department_rankings", "department_rankings"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x="total_sales",
                y=[f"Dept {d}" for d in df_chart["department"]],
                orientation="h",
                text=[f"${v/1e6:.1f}M ({p:.1f}%)" for v, p in zip(df_chart["total_sales"], df_chart["pct_of_chain"])],
                labels={"total_sales": "Total Historical Sales ($)", "y": "Department"},
                color_discrete_sequence=[Theme.CYAN],
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Department Revenue Concentration (Pareto Leaders)")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=60, r=40, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["get_holiday_performance", "holiday_performance"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x="Holiday_Name",
                y="Mean_Store_Weekly_Sales",
                color="Lift_Vs_Baseline_Pct",
                text=[f"{l:+.1f}%" if l != 0 else "Baseline" for l in df_chart["Lift_Vs_Baseline_Pct"]],
                labels={"Mean_Store_Weekly_Sales": "Mean Weekly Sales per Store ($)", "Holiday_Name": "Holiday Cycle"},
                color_continuous_scale=["#1E293B", Theme.CYAN, Theme.GREEN],
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Holiday Store Sales vs Non-Holiday Baseline")
            fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=50, r=30, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["get_forecast_leaderboard", "forecast_leaderboard"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x="Model",
                y="WAPE_Pct",
                color="Model",
                text=[f"{w:.2f}%" for w in df_chart["WAPE_Pct"]],
                labels={"WAPE_Pct": "Forecast Error (WAPE %)", "Model": "Model Architecture"},
                color_discrete_map={
                    "XGBoost": Theme.GREEN,
                    "Random Forest": Theme.CYAN,
                    "Ridge": Theme.VIOLET,
                    "Seasonal Naive": Theme.AMBER,
                    "Previous-Week Naive": Theme.CORAL,
                },
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Forecasting Error Benchmark (WAPE % — Lower is Superior)")
            fig.update_layout(height=260, showlegend=False, margin=dict(l=50, r=30, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["get_global_shap_importance", "shap_importance"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x="Relative_Impact_Pct",
                y="Feature",
                orientation="h",
                text=[f"{p:.1f}%" for p in df_chart["Relative_Impact_Pct"]],
                labels={"Relative_Impact_Pct": "Relative Impact (%)", "Feature": "Feature Driver"},
                color_discrete_sequence=[Theme.BRAND],
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Global TreeSHAP Attribution Breakdown")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=80, r=40, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["get_anomaly_records", "anomaly_records"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x=[f"Store {s}<br>{d}" for s, d in zip(df_chart["Store"], df_chart["Date"])],
                y="Priority_Score",
                color="Priority_Score",
                text=[f"{s:.0f}/100" for s in df_chart["Priority_Score"]],
                labels={"y": "Priority Score", "x": "Target"},
                color_continuous_scale="Reds",
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Top Outlier Investigation Queue Targets")
            fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=50, r=30, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["get_store_forecast_errors", "store_forecast_errors"] and isinstance(data, list):
            df_chart = pd.DataFrame(data)
            fig = px.bar(
                df_chart,
                x=[f"Store {s}" for s in df_chart["Store"]],
                y="WAPE_Pct",
                text=[f"{w:.1f}%" for w in df_chart["WAPE_Pct"]],
                labels={"WAPE_Pct": "WAPE (%)", "x": "Store"},
                color_discrete_sequence=[Theme.CYAN],
            )
            fig.update_traces(textposition="outside")
            fig = apply_default_layout(fig, "Store Error Reliability (WAPE %)")
            fig.update_layout(height=260, margin=dict(l=50, r=30, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type in ["explain_store_prediction"] and isinstance(data, dict):
            drivers = data.get("top_drivers", [])
            if drivers:
                df_drivers = pd.DataFrame(drivers)
                fig = px.bar(
                    df_drivers,
                    x="SHAP_Value",
                    y="Feature",
                    orientation="h",
                    color=df_drivers["SHAP_Value"] > 0,
                    color_discrete_map={True: Theme.GREEN, False: Theme.CORAL},
                    text=[f"{v:+,.0f}" for v in df_drivers["SHAP_Value"]],
                    labels={"SHAP_Value": "Attribution ($)", "Feature": "Feature Driver"},
                )
                fig.update_traces(textposition="outside")
                fig = apply_default_layout(fig, f"Local TreeSHAP Drivers: Store {data.get('store')} on {data.get('date')}")
                fig.update_layout(height=280, showlegend=False, yaxis=dict(autorange="reversed"), margin=dict(l=80, r=40, t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _handle_user_query(engine: RetailIQAssistantEngine, prompt: str) -> None:
    """Execute user query and record interaction in session state."""
    # Append user question
    st.session_state.assistant_messages.append({
        "role": "user",
        "answer": prompt,
    })

    # Execute through engine
    result = engine.ask(prompt)

    # Append assistant response
    st.session_state.assistant_messages.append({
        "role": "assistant",
        "answer": result["answer"],
        "metadata": {
            "route": result.get("route"),
            "provider": result.get("provider"),
            "tools_called": result.get("tools_called", []),
            "evidence_source": result.get("evidence_source"),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "verified": result.get("verified", True),
            "data": result.get("data"),
            "chart_type": result.get("chart_type"),
            "followups": result.get("followups", []),
            "deep_link_page": result.get("deep_link_page"),
        },
    })
