"""
RetailIQ AI V2 — Assistant Conversational Context Resolver

Extracts entity context (stores, models, topics, compared pairs) from prior conversation
turns to support multi-turn dialogue, anaphoric references ("it", "their", "that"),
follow-up comparisons, and chart generation.
"""

from typing import Any
import re


def extract_context_from_history(conversation_history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Parse previous conversation messages to resolve referents for the current turn.

    Args:
        conversation_history: List of chat messages with role, answer/content, and metadata.

    Returns:
        Dictionary with extracted last_stores, last_compared_stores, last_model, last_topic, etc.
    """
    ctx: dict[str, Any] = {
        "last_stores": [],
        "last_compared_stores": [],
        "last_model": None,
        "last_topic": None,
        "last_tool_called": None,
        "last_chart_type": None,
        "last_data": None,
    }

    if not conversation_history:
        return ctx

    # Traverse history backwards to find the most recent entities
    for msg in reversed(conversation_history):
        role = msg.get("role", "")
        text = (msg.get("answer") or msg.get("content") or "").lower()
        meta = msg.get("metadata", {})

        # Extract stores from metadata or text
        if meta and "arguments" in meta and "store_ids" in meta["arguments"]:
            store_ids = meta["arguments"]["store_ids"]
            if store_ids and not ctx["last_compared_stores"]:
                ctx["last_compared_stores"] = [int(s) for s in store_ids]
            if store_ids and not ctx["last_stores"]:
                ctx["last_stores"] = [int(s) for s in store_ids]

        if not ctx["last_stores"]:
            store_matches = re.findall(r"store\s*(\d+)", text)
            if store_matches:
                ctx["last_stores"] = [int(s) for s in store_matches]

        # Extract models
        if not ctx["last_model"]:
            if "ridge" in text:
                ctx["last_model"] = "Ridge Linear"
            elif "xgboost" in text:
                ctx["last_model"] = "XGBoost"
            elif "random forest" in text:
                ctx["last_model"] = "Random Forest"

        # Extract tool and chart
        if not ctx["last_tool_called"] and meta.get("tools_called"):
            tools = meta["tools_called"]
            if isinstance(tools, list) and len(tools) > 0:
                first_tool = tools[0]
                ctx["last_tool_called"] = first_tool.get("tool") if isinstance(first_tool, dict) else str(first_tool)

        if not ctx["last_chart_type"] and meta.get("chart_type"):
            ctx["last_chart_type"] = meta.get("chart_type")

        if ctx["last_data"] is None and meta.get("data"):
            ctx["last_data"] = meta.get("data")

        # Stop early if we have resolved the main entities
        if ctx["last_stores"] and ctx["last_model"]:
            break

    return ctx
