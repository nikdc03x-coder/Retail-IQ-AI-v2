# Phase 15 – Live Gemini Integration Validation

## 1. Gemini Call History Reconciliation

The project previously reported two contradictory statements:
1. *A live Gemini call `get_store_rankings(top_n=1, ascending=False)` succeeded and returned Store 20's verified sales.*
2. *All live Gemini calls later failed due to free‑tier quota exhaustion.*

### Evidence Search
- **`live_gemini_evaluation_results.json`** – contains three textual answers about store rankings, but the `tool` field is `null` for every entry, meaning **no function‑calling payload was returned**.
- **Background task logs** (`task‑2489.log`) – show repeated `429 RESOURCE_EXHAUSTED` responses from the Gemini API; no successful HTTP 200 responses are present.
- **Git history** – no commit adds a log entry with a successful function‑call payload.

### Conclusion
No verifiable artifact confirms a successful function‑calling execution of `get_store_rankings`. The earlier claim cannot be independently substantiated and must be treated as **unverified**.

---

## 2. Quota‑Aware Evaluation Implementation

- **Script:** `scratch/quota_aware_evaluation.py`
- **Key behaviours:**
  - Configurable maximum live requests via `GEMINI_MAX_LIVE` (default 5).
  - Global counters `_live_requests_made` and `_daily_quota_exhausted`.
  - Detects `RESOURCE_EXHAUSTED` and marks daily quota exhausted.
  - Provides up‑to‑two exponential‑back‑off retries for transient errors.
  - Falls back to `RetailIQAssistantEngine` when live calls are blocked.
  - Generates `quota_aware_evaluation_report.json` with per‑question metadata.

Running the script with `GEMINI_MAX_LIVE=0` (as done in the verification step) produced a report containing three entries, all routed through the deterministic fallback and clearly labeled with `fallback_reason: "Live request limit reached (0)"`.

---

## 3. Offline Benchmark Results

The offline run (zero live calls) produced `quota_aware_evaluation_report.json`. Highlights:
- **All three sample questions** were answered by the deterministic engine.
- **Tool usage:** `get_store_rankings` (for the first two questions) and `get_chain_sales_summary` (for the third) were invoked locally.
- **Evidence source** points to CSV reports under `reports/business_analytics/`.
- **Numeric correctness** is pending validation against ground truth; the report includes the field `numeric_correct: null` to be filled by downstream analysis.
- **Quota consumption:** `quota_used` remains `0` because no live Gemini requests were made.

---

## 4. Remaining Live‑Integration Checks

1. **When the free‑tier quota refreshes**, run a single controlled Gemini request (e.g., `"Which store has the highest cumulative historical sales?"`). Capture the full response object, verifying that:
   - The HTTP status is 200.
   - The response includes a `function_calls` section with `tool: "get_store_rankings"` and proper arguments.
   - Evidence source and verification flags are present.
2. **If the call succeeds**, update the audit with timestamps, provider/model, and tool execution details.
3. **If quota remains exhausted**, the audit notes that live verification is pending.

---

## 5. Deployment Readiness

- All **194 existing tests** pass (`197` after adding the new quota‑handling tests).
- The **pipeline** (`scripts/run_all_pipelines.py`) runs all eight stages successfully.
- **Documentation** (`docs/phases/15_gemini_migration_and_release.md`) reflects current status and next steps.
- **No secret leakage** – the Gemini API key is only read from environment/`.env` and never logged.

**Blockers:** Free‑tier Gemini quota exhaustion prevents live validation of function‑calling. No paid usage is enabled.

---

*All edits are on branch `phase-15-gemini-migration-and-release`; no pushes or merges have been performed.*
