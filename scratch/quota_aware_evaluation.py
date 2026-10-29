"""
quota_aware_evaluation.py
--------------------------
Implements a quota‑aware wrapper around the Gemini client for Phase 15 evaluation.
The script can be executed directly (`python -m scratch.quota_aware_evaluation`) or imported by CI.
"""
import os
import json
import time
import logging
from typing import Any, Dict, List

from src.retailiq.assistant.gemini_client import GeminiAssistantClient
from src.retailiq.assistant.engine import RetailIQAssistantEngine

# ---------------------------------------------------------------------------
# Configuration – can be overridden via environment variables
# ---------------------------------------------------------------------------
MAX_LIVE_REQUESTS = int(os.getenv("GEMINI_MAX_LIVE", "5"))
MAX_LIVE_REQUESTS = max(0, MAX_LIVE_REQUESTS)

# ---------------------------------------------------------------------------
# Global quota state (module level – reset on each run)
# ---------------------------------------------------------------------------
_live_requests_made = 0
_daily_quota_exhausted = False

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _record_quota_state() -> Dict[str, Any]:
    """Expose the current quota counters for diagnostics."""
    return {"live_requests_made": _live_requests_made, "daily_quota_exhausted": _daily_quota_exhausted}


def _safe_gemini_call(client: GeminiAssistantClient, prompt: str) -> Dict[str, Any]:
    """Attempt a Gemini request respecting the configured quota.

    Returns a dictionary with either a ``"response"`` key containing the Gemini
    payload or a ``"fallback"`` key describing why the live call was not made.
    """
    global _live_requests_made, _daily_quota_exhausted

    if _daily_quota_exhausted:
        return {"fallback": True, "reason": "Daily quota already exhausted"}
    if _live_requests_made >= MAX_LIVE_REQUESTS:
        return {"fallback": True, "reason": f"Live request limit reached ({MAX_LIVE_REQUESTS})"}

    try:
        resp = client.generate_response(prompt)
        _live_requests_made += 1
        return {"response": resp, "fallback": False}
    except Exception as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg:
            if "daily" in msg.lower():
                _daily_quota_exhausted = True
            return {"fallback": True, "reason": "Quota exhausted", "error": msg}
        # Transient errors – up to two retries with exponential back‑off
        for attempt in range(2):
            backoff = 2 ** attempt
            log.warning("Gemini call failed (attempt %d). Retrying in %ds … (%s)", attempt + 1, backoff, msg)
            time.sleep(backoff)
            try:
                resp = client.generate_response(prompt)
                _live_requests_made += 1
                return {"response": resp, "fallback": False}
            except Exception as e2:
                msg = str(e2)
                if "RESOURCE_EXHAUSTED" in msg:
                    if "daily" in msg.lower():
                        _daily_quota_exhausted = True
                    return {"fallback": True, "reason": "Quota exhausted during retry", "error": msg}
                continue
        return {"fallback": True, "reason": "Transient error after retries", "error": msg}


def evaluate_questions(question_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Run the evaluation batch, mixing live Gemini calls with deterministic fallback.

    ``question_list`` is a list of ``{"id": <str>, "prompt": <str>}`` dictionaries.
    The function returns a list of per‑question result dictionaries suitable for JSON
    serialisation.
    """
    client = GeminiAssistantClient()
    engine = RetailIQAssistantEngine()  # deterministic fallback engine

    results: List[Dict[str, Any]] = []
    for q in question_list:
        qid = q.get("id", "unknown")
        prompt = q.get("prompt", "")
        entry: Dict[str, Any] = {
            "question_id": qid,
            "prompt": prompt,
            "live_attempted": not _daily_quota_exhausted and _live_requests_made < MAX_LIVE_REQUESTS,
        }
        gem_res = _safe_gemini_call(client, prompt)
        if not gem_res.get("fallback"):
            resp = gem_res["response"]
            entry.update({
                "provider": "Gemini",
                "model": client.model,
                "live_success": True,
                "answer": resp.get("answer", ""),
                "tools_called": resp.get("tools_called", []),
                "tool_details": resp.get("tool_details", []),
                "fallback": False,
                "numeric_correct": None,
                "quota_used": _live_requests_made,
            })
        else:
            fallback_reason = gem_res.get("reason", "unknown")
            fallback_resp = engine.ask(prompt)
            entry.update({
                "provider": "DeterministicEngine",
                "model": "N/A",
                "live_success": False,
                "fallback": True,
                "fallback_reason": fallback_reason,
                "answer": fallback_resp,
                "tools_called": [],
                "tool_details": [],
                "numeric_correct": None,
                "quota_used": _live_requests_made,
            })
        results.append(entry)
    return results


def main() -> None:
    # Example question set – replace with the real benchmark list as needed.
    sample_questions = [
        {"id": "q1", "prompt": "Which store has the highest cumulative historical sales?"},
        {"id": "q2", "prompt": "Give the top 3 stores by sales."},
        {"id": "q3", "prompt": "What is the total sales for Store 20?"},
    ]
    report = evaluate_questions(sample_questions)
    out_path = os.path.join(os.getcwd(), "quota_aware_evaluation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Evaluation finished – %d entries written to %s", len(report), out_path)
    log.info("Final quota state: %s", _record_quota_state())


if __name__ == "__main__":
    main()

