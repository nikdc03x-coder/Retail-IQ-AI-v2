"""
RetailIQ AI V2 — Page 08: AI Business Assistant

Conversational business intelligence powered by Google Gemini and backed by
deterministic tool execution on pre-computed analytical datasets:
- Evidence-grounded natural language question answering.
- Dual-mode routing: Google Gemini with Function Calling or Deterministic Rule-Based Fallback.
- Complete execution transparency (tools invoked, source files, latency, verification).
- Interactive quick-query suggestion chips.
"""

from typing import Any
import streamlit as st
import pandas as pd

from src.retailiq.config import Theme, GEMINI_API_KEY, GEMINI_MODEL
from src.retailiq.dashboard.components import page_context
from src.retailiq.dashboard.theme import context_box, page_header
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
    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["answer"])

            # Transparency drawer for assistant messages
            meta = msg.get("metadata", {})
            if msg["role"] == "assistant" and meta.get("route") != "system_greeting":
                with st.expander("Transparency & Evidence Grounding", expanded=False):
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        st.markdown(f"**Execution Route:** `{meta.get('route')}`")
                        st.markdown(f"**Engine Provider:** {meta.get('provider')}")
                        st.markdown(f"**Tools Executed:** `{', '.join(meta.get('tools_called', ['N/A']))}`")
                    with t_col2:
                        st.markdown(f"**Data Evidence Source:** `{meta.get('evidence_source')}`")
                        st.markdown(f"**Latency:** `{meta.get('latency_seconds', 0.0)}s`")
                        st.markdown(f"**Mathematical Verification:** `{'Verified' if meta.get('verified') else 'Unverified'}`")

                    # If data is present as tabular records, display a mini table
                    data_payload = meta.get("data")
                    if isinstance(data_payload, list) and len(data_payload) > 0 and isinstance(data_payload[0], dict):
                        st.markdown("**Underlying Structured Evidence:**")
                        df_preview = pd.DataFrame(data_payload)
                        st.dataframe(df_preview, use_container_width=True, hide_index=True)

    # ── User Chat Input ──
    user_input = st.chat_input("Ask a business question about sales, models, anomalies, or SHAP...")
    if user_input:
        _handle_user_query(engine, user_input)
        st.rerun()


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
        },
    })
