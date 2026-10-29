"""
RetailIQ AI V2 — Google Gemini Client with Function Calling

Connects to the Google Gemini API using the official google-genai SDK.
Exposes approved analytical tools to Gemini via typed function declarations
and provides strict system instructions preventing hallucinations or unauthorized code execution.
"""

from typing import Any
import os
from google import genai
from google.genai import types

from src.retailiq.config import GEMINI_API_KEY, GEMINI_MODEL
from src.retailiq.assistant.tools import (
    get_dataset_overview,
    get_chain_sales_summary,
    get_store_rankings,
    get_department_rankings,
    compare_stores,
    get_holiday_performance,
    get_forecast_leaderboard,
    compare_forecasting_models,
    get_store_forecast_errors,
    get_anomaly_summary,
    get_anomaly_records,
    get_global_shap_importance,
    explain_store_prediction,
    get_operational_strategy_summary,
)

SYSTEM_INSTRUCTION = """
You are the RetailIQ AI Business Analyst — an expert retail data science and analytics assistant.
You provide executive-ready, evidence-grounded insights based on historical Walmart retail sales,
machine learning forecast evaluations, error reliability diagnostics, anomaly audits, and TreeSHAP explainability.

CRITICAL OPERATIONAL RULES:
1. ONLY state facts and numbers directly returned by your tools. NEVER invent, extrapolate, or hallucinate figures.
2. If asked about metrics not covered by your tools, state clearly that the historical analytical dataset does not contain that information.
3. Distinguish between store-week sales, department-week sales, and total chain revenue. Never conflate different aggregation levels.
4. When discussing forecasting accuracy, cite exact metrics (WAPE %, MAE $, RMSE $, R²).
5. Frame anomaly flags strictly as operational review targets, not confirmed employee fraud or accounting malpractice.
6. Frame SHAP values as statistical feature influences on model predictions, not direct operational causes.
7. Keep responses concise, clear, and professional, using markdown bolding for key numerical figures.
"""

TOOLS_LIST = [
    get_dataset_overview,
    get_chain_sales_summary,
    get_store_rankings,
    get_department_rankings,
    compare_stores,
    get_holiday_performance,
    get_forecast_leaderboard,
    compare_forecasting_models,
    get_store_forecast_errors,
    get_anomaly_summary,
    get_anomaly_records,
    get_global_shap_importance,
    explain_store_prediction,
    get_operational_strategy_summary,
]


class GeminiAssistantClient:
    """Client wrapper for Gemini conversational model with tool calling, including simple rate‑limiting and file‑based caching.
    
    The cache stores the full JSON response for a given prompt + optional conversation history.
    Cached entries live indefinitely (they are small) and are looked‑up before any API call.
    Rate limiting enforces a minimum interval between live requests (default 1 s) to avoid hitting per‑minute quotas.
    """
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # Use provided API key if explicitly passed (including empty string)
        # Only fall back to environment/config when api_key is None
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        self.model = model or os.getenv("GEMINI_MODEL") or GEMINI_MODEL

        self.model = model or os.getenv("GEMINI_MODEL") or GEMINI_MODEL
        self._client: genai.Client | None = None

        # ---------- rate‑limiting & caching state ----------
        import time, hashlib, json
        from pathlib import Path
        self._last_call: float = 0.0
        self._min_interval: float = 1.0  # seconds between live calls
        # Cache directory: <repo_root>/cache/gemini
        self._cache_dir = Path(__file__).resolve().parents[2] / "cache" / "gemini"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def is_available(self) -> bool:
        """Check if client is initialized with a valid API key."""
        return self._client is not None and bool(self.api_key)

    def _cache_path(self, key: str) -> "Path":
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if path.is_file():
            try:
                import json
                return json.loads(path.read_text())
            except Exception:
                return None
        return None

    def _save_to_cache(self, key: str, data: dict) -> None:
        path = self._cache_path(key)
        try:
            import json
            path.write_text(json.dumps(data))
        except Exception:
            pass

    def _rate_limit(self) -> None:
        import time
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def generate_response(self, user_prompt: str, conversation_history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Generate an evidence‑grounded answer using Gemini tool calling.

        This method first checks the cache. If a cached response is found it is returned
        immediately without contacting the Gemini API. Otherwise a live request is made
        respecting the rate‑limiter.
        """
        if not self.is_available or self._client is None:
            raise RuntimeError("Gemini client is not configured with an API key.")

        # ---------- cache lookup ----------
        import hashlib, json
        cache_key_input = {
            "prompt": user_prompt,
            "history": conversation_history,
        }
        cache_key = hashlib.sha256(json.dumps(cache_key_input, sort_keys=True).encode()).hexdigest()
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        # ---------- rate limiting ----------
        self._rate_limit()

        try:
            config = types.GenerateContentConfig(
                tools=TOOLS_LIST,
                temperature=0.2,
                system_instruction=SYSTEM_INSTRUCTION,
            )

            # Build multi‑turn contents if history exists
            contents: Any = user_prompt
            if conversation_history:
                formatted_contents = []
                for msg in conversation_history[-6:]:
                    role = "user" if msg.get("role") == "user" else "model"
                    text = msg.get("answer") or msg.get("content") or ""
                    if text:
                        formatted_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
                if formatted_contents:
                    formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
                    contents = formatted_contents

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            answer_text = response.text if getattr(response, "text", None) else "No textual response generated."
            tools_used = []
            if hasattr(response, "automatic_function_calling_history") and response.automatic_function_calling_history:
                for content in response.automatic_function_calling_history:
                    if hasattr(content, "parts") and content.parts:
                        for part in content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                tools_used.append({
                                    "tool": getattr(part.function_call, "name", "unknown_tool"),
                                    "args": dict(part.function_call.args) if getattr(part.function_call, "args", None) else {},
                                })
            if not tools_used and getattr(response, "function_calls", None):
                for fc in response.function_calls:
                    tools_used.append({
                        "tool": getattr(fc, "name", "unknown_tool"),
                        "args": dict(fc.args) if getattr(fc, "args", None) else {},
                    })

            result = {
                "route": "gemini_function_calling",
                "model": self.model,
                "answer": answer_text,
                "tools_called": [t["tool"] for t in tools_used] if tools_used else [],
                "tool_details": tools_used,
                "verified": True,
            }
            # Save to cache before returning
            self._save_to_cache(cache_key, result)
            return result

        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}")

    """Client wrapper for Gemini conversational model with tool calling."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # Use provided API key if explicitly passed (including empty string)
        # Only fall back to environment/config when api_key is None
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        self.model = model or os.getenv("GEMINI_MODEL") or GEMINI_MODEL

        self.model = model or os.getenv("GEMINI_MODEL") or GEMINI_MODEL
        self._client: genai.Client | None = None

        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def is_available(self) -> bool:
        """Check if client is initialized with a valid API key."""
        return self._client is not None and bool(self.api_key)

    def generate_response(self, user_prompt: str, conversation_history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Generate an evidence-grounded answer using Gemini tool calling.

        Args:
            user_prompt: Natural language question from the user.
            conversation_history: Optional list of past chat messages to resolve multi-turn dialogue.

        Returns:
            Dictionary with response text, tools executed, and execution metadata.
        """
        if not self.is_available or self._client is None:
            raise RuntimeError("Gemini client is not configured with an API key.")

        try:
            config = types.GenerateContentConfig(
                tools=TOOLS_LIST,
                temperature=0.2,
                system_instruction=SYSTEM_INSTRUCTION,
            )

            # Build multi-turn contents if history exists
            contents: Any = user_prompt
            if conversation_history:
                formatted_contents = []
                for msg in conversation_history[-6:]:
                    role = "user" if msg.get("role") == "user" else "model"
                    text = msg.get("answer") or msg.get("content") or ""
                    if text:
                        formatted_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
                if formatted_contents:
                    formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
                    contents = formatted_contents

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Extract response text
            answer_text = response.text if getattr(response, "text", None) else "No textual response generated."

            # Inspect function calls from automatic_function_calling_history and candidates
            tools_used = []
            if hasattr(response, "automatic_function_calling_history") and response.automatic_function_calling_history:
                for content in response.automatic_function_calling_history:
                    if hasattr(content, "parts") and content.parts:
                        for part in content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                tools_used.append({
                                    "tool": getattr(part.function_call, "name", "unknown_tool"),
                                    "args": dict(part.function_call.args) if getattr(part.function_call, "args", None) else {},
                                })

            if not tools_used and getattr(response, "function_calls", None):
                for fc in response.function_calls:
                    tools_used.append({
                        "tool": getattr(fc, "name", "unknown_tool"),
                        "args": dict(fc.args) if getattr(fc, "args", None) else {},
                    })

            return {
                "route": "gemini_function_calling",
                "model": self.model,
                "answer": answer_text,
                "tools_called": [t["tool"] for t in tools_used] if tools_used else [],
                "tool_details": tools_used,
                "verified": True,
            }

        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}")
