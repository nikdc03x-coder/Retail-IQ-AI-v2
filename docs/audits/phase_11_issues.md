# RetailIQ AI V2 — Phase 11 Independent Audit Issue Tracker

**Audit Date:** October 28, 2026  
**Auditor:** Independent ML Research Reviewer & AI Systems Auditor  
**Branch:** `phase-11-independent-validation`  
**Base Commit:** `94fdde9`  
**Overall Readiness Tier:** **Portfolio / Pilot Production Ready** (Pending Core Corrections)

---

## Issue Summary Matrix

| Issue ID | Severity | Category | Title | Affected Component |
| :--- | :--- | :--- | :--- | :--- |
| **ISSUE-11-01** | **High** | Methodology | Horizon Semantic Ambiguity: 1-Step Rolling vs Multi-Week Forecast | `forecasting/features.py`, documentation |
| **ISSUE-11-02** | **Medium** | Documentation | Imprecise Missing Data Claim on Raw CPI / Unemployment | `data_quality`, reports, project documentation |
| **ISSUE-11-03** | **Medium** | Explainability | SHAP Feature Importance Discrepancy (Lag 52 + Rolling Mean 4 vs Lag 1) | `explainable_ai`, dashboard |
| **ISSUE-11-04** | **Medium** | Business Strategy | Unsubstantiated \$12.3M Annual Savings Framing (Simulated vs Measured) | `strategy/decision_engine.py`, executive reports |
| **ISSUE-11-05** | **Low** | Assistant | Offline Fallback String-Matching Fragility on Multi-Entity Queries | `assistant/fallback.py` |
| **ISSUE-11-06** | **Low** | Code Hygiene | Backward-Compatibility Redundant CSV Duplications | `reports/business_analytics/` |

---

## Detailed Issue Records

### ISSUE-11-01: Horizon Semantic Ambiguity — 1-Step Rolling vs Multi-Week Forecast
- **Severity:** High
- **Category:** Scientific Methodology & Evaluation
- **Affected Components:**
  - `src/retailiq/forecasting/features.py`
  - `src/retailiq/forecasting/evaluation.py`
  - `reports/sales_forecasting/01_model_comparison_metrics.csv`
- **Empirical Finding:**
  - The model evaluation reports test performance across 22 out-of-time weeks (990 store-week observations from `2012-06-01` to `2012-10-26`).
  - Feature engineering computes `lag_1` as `.shift(1)` on the test series. Because the test set is evaluated in bulk row-by-row, each prediction for week $t$ has access to the actual observed sales from week $t-1$.
  - This setup is mathematically and operationally a **rolling 1-week-ahead forecast** (re-estimated or re-fed with actuals weekly), NOT a multi-step dynamic autoregressive forecast across a 22-week horizon without intervening ground truth.
- **Scientific Risk:**
  - Treating these metrics as representing a 22-week unassisted planning horizon would result in severe error compounding in production. In practice, supply chain planners forecasting 3 to 6 months out do not have access to week $t-1$ actual sales.
- **Recommended Action:**
  - Explicitly document in all reports and UI headers that the benchmark is a **Rolling 1-Step-Ahead Weekly Operational Forecast**.
  - Provide an optional dynamic recursive multi-step forecasting evaluation in `forecasting/evaluation.py` for long-horizon supply chain simulations.

---

### ISSUE-11-02: Imprecise Missing Data Claim on Raw CPI & Unemployment
- **Severity:** Medium
- **Category:** Data Governance & Reporting Accuracy
- **Affected Components:**
  - Data ingestion logs and Phase 01 / Phase 02 summary reports
- **Empirical Finding:**
  - Previous documentation claimed raw Walmart features had 0% missing CPI and Unemployment data.
  - Independent audit of raw `features.csv` revealed **585 missing values** ($7.14\%$) in both `CPI` and `Unemployment`.
  - Crucially, all 585 missing rows occur in the future Kaggle test period (`2013-05-03` to `2013-07-26`). For the historical sales date range (`2010-02-05` to `2012-10-26`), CPI and Unemployment are indeed complete (0 missing).
  - Consequently, the processed `analytical_dataset.parquet` (which inner-joins on sales records) has 0% missing CPI.
- **Scientific Risk:**
  - Conflating the raw table characteristics with the filtered analytical dataset compromises audit traceability and creates misleading expectations for raw data pipelines.
- **Recommended Action:**
  - Clarify the ingestion audit report to state: *"Raw features.csv contains 585 missing values in post-2012 Kaggle test dates; within the historical sales period (2010–2012), CPI and Unemployment are 100% complete."*

---

### ISSUE-11-03: SHAP Feature Importance Attribution Discrepancy
- **Severity:** Medium
- **Category:** Explainable AI & Scientific Rigor
- **Affected Components:**
  - `reports/explainable_ai/01_global_feature_importance.csv`
  - Dashboard Page `p07_explainability.py`
  - Project summary documentation
- **Empirical Finding:**
  - Prior summary documentation claimed that *"Lag 52 + Lag 1 account for 72.7% of model decision weight."*
  - Independent audit of `01_global_feature_importance.csv` confirms:
    - `lag_52`: **45.01%**
    - `rolling_mean_4`: **27.65%** (Sum = **72.66%**, or ~72.7%)
    - `lag_1`: **2.21%** (Rank 5)
  - The ~72.7% concentration is driven by **annual seasonality (`lag_52`) combined with short-term 4-week trend (`rolling_mean_4`)**, NOT `lag_1`.
- **Scientific Risk:**
  - Misinforms executives and regulators about which operational variables dominate algorithmic predictions.
- **Recommended Action:**
  - Correct all documentation and text callouts in `p07_explainability.py` to accurately credit `lag_52` (45.0%) and `rolling_mean_4` (27.7%).

---

### ISSUE-11-04: Unsubstantiated \$12.3M Annual Savings Framing
- **Severity:** Medium
- **Category:** Business Strategy & Financial Modeling
- **Affected Components:**
  - `src/retailiq/strategy/decision_engine.py`
  - `reports/strategy/04_executive_decision_summary.json`
  - Dashboard Page `p08_strategy.py`
- **Empirical Finding:**
  - The claimed `\$12,309,480.66` annual savings is generated by a heuristic formula:  
    $$\text{Optimization} = \sum_{\text{stores}} (\text{Mean Weekly Sales} \times \text{Buffer Pct}) \times 52 \times 0.045$$
  - The $4.5\%$ multiplier is a hardcoded holding cost assumption, not an empirical measurement from historical A/B trials or balance sheet ledgers.
- **Scientific Risk:**
  - Presenting simulated heuristic figures as definitive empirical ROI damages executive trust during production due diligence.
- **Recommended Action:**
  - Re-label the metric across reports and UI from *"Measured Annual Savings"* to *"Simulated Annual Working Capital Optimization (Assuming 4.5% Holding Cost Delta)"*.
  - Expose the $4.5\%$ holding cost and buffer parameters as interactive user-configurable inputs in `p08_strategy.py`.

---

### ISSUE-11-05: Offline Fallback String-Matching Fragility
- **Severity:** Low
- **Category:** AI Assistant & Natural Language Processing
- **Affected Components:**
  - `src/retailiq/assistant/fallback.py`
- **Empirical Finding:**
  - The 40-question benchmark scored **62.5% Pass, 20.0% Partial, 17.5% Fail** in offline fallback mode.
  - Failures in fallback mode occur when queries ask for multi-entity comparisons or natural language synonyms that bypass the regex routing dictionary (e.g. *"Compare Random Forest vs XGBoost accuracy"* defaulting to general model leaderboard rather than side-by-side delta).
  - Out-of-domain and adversarial queries achieved **100% Pass** (13/13 blocked safely).
- **Recommended Action:**
  - Expand synonym dictionary in `fallback.py` to capture comparative queries (`compare <m1> vs <m2>`) and multi-entity extractions.

---

### ISSUE-11-06: Redundant File Duplications for Backwards Compatibility
- **Severity:** Low
- **Category:** Technical Debt & File System Hygiene
- **Affected Components:**
  - `reports/business_analytics/06_store_type_summary.csv` vs `06_store_type_performance.csv`
- **Empirical Finding:**
  - Both files exist with identical content to satisfy legacy path references from earlier phases.
- **Recommended Action:**
  - Standardize on `06_store_type_performance.csv` and use an explicit internal alias or symlink rather than committing identical physical CSVs.
