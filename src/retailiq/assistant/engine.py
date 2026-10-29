"""
RetailIQ AI V2 — Unified Assistant Execution Engine

Directs incoming user queries through either the Gemini Function Calling client
or the deterministic fallback dispatcher based on API key availability and connectivity.
Ensures 100% reliable responses with full transparency on data sources and execution route.
"""

from typing import Any
import os
import time

from src.retailiq.assistant.fallback import dispatch_fallback_query
from src.retailiq.assistant.gemini_client import GeminiAssistantClient


class RetailIQAssistantEngine:
    """Unified conversational assistant engine with dual-route architecture."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.gemini_client = GeminiAssistantClient(api_key=api_key, model=model)

    @property
    def has_gemini(self) -> bool:
        """Return True if a Gemini API client is active and configured."""
        return self.gemini_client.is_available

    def ask(self, query: str, conversation_history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Process a natural language business inquiry.

        The UI can control whether live Gemini calls are allowed via the
        Streamlit session state flag ``use_live_gemini``. When the flag is
        ``False`` (default) the engine will always fall back to the deterministic
        dispatcher even if a valid API key is present.
        """
        start_time = time.time()
        q_clean = query.strip()
        if not q_clean:
            return {
                "route": "validation_guard",
                "answer": "Please ask a specific business or data science question about the Walmart dataset.",
                "tools_called": [],
                "evidence_source": "N/A",
                "latency_seconds": 0.0,
                "provider": "RetailIQ Guard",
                "verified": False,
            }

        # Input length guardrail to prevent buffer abuse or denial-of-service
        if len(q_clean) > 2000:
            return {
                "route": "validation_guard",
                "answer": "Input query exceeds maximum allowed length of 2,000 characters. Please provide a more concise inquiry.",
                "tools_called": [],
                "evidence_source": "Input Length Guardrail",
                "latency_seconds": 0.0,
                "provider": "RetailIQ Guard",
                "verified": False,
            }

        gemini_fallback_reason: str | None = None

        # ── Determine whether live Gemini is allowed ──
        allow_live = False
        try:
            import streamlit as st
            # In a Streamlit UI the user explicitly toggles the flag; default to False when not set.
            allow_live = bool(st.session_state.get("use_live_gemini", True))
        except Exception:
            # Not running inside Streamlit – assume live Gemini is allowed.
            allow_live = True


        # Determine Gemini availability and set fallback reason if unavailable
        if not self.has_gemini:
            gemini_fallback_reason = "Gemini credentials unavailable; using verified Python analytics."
        # Determine whether live Gemini is allowed and set fallback reason if disabled
        if not allow_live:
            gemini_fallback_reason = "Live Gemini disabled; using verified Python analytics."
        # ── Route 1: Gemini with tool calling if API key is active and live is allowed ──
        if self.has_gemini and allow_live:
            try:
                gemini_res = self.gemini_client.generate_response(q_clean, conversation_history=conversation_history)
                latency = round(time.time() - start_time, 2)
                return {
                    "route": "gemini_function_calling",
                    "answer": gemini_res["answer"],
                    "tools_called": gemini_res.get("tools_called", []),
                    "tool_details": gemini_res.get("tool_details", []),
                    "evidence_source": "Dynamic Tool Grounding via reports/",
                    "latency_seconds": latency,
                    "provider": f"Google Gemini ({self.gemini_client.model})",
                    "verified": True,
                }
            except Exception as e:
                # Capture failure reason and fallback gracefully without crashing
                gemini_fallback_reason = "Live Gemini API unavailable; using verified Python analytics."

        # ── Route 2: Deterministic Fallback Dispatcher ──
        fallback_res = dispatch_fallback_query(q_clean, conversation_history=conversation_history)
        latency = round(time.time() - start_time, 3)

        return {
            "route": "deterministic_fallback",
            "answer": fallback_res["answer"],
            "tools_called": [fallback_res["tool_called"]],
            "arguments": fallback_res.get("arguments", {}),
            "data": fallback_res.get("data"),
            "evidence_source": fallback_res.get("evidence_source", "reports/"),
            "latency_seconds": latency,
            "provider": "Deterministic Rule-Based Dispatcher (Offline Mode)",
            "verified": fallback_res.get("verified", True),
            "chart_type": fallback_res.get("chart_type"),
            "followups": fallback_res.get("followups", []),
            "deep_link_page": fallback_res.get("deep_link_page"),
            "fallback_reason": gemini_fallback_reason,
        }

