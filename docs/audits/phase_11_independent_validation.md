# RetailIQ AI V2 — Phase 11: Independent Scientific Audit & Production Readiness Report

**Audit Date:** October 28, 2026  
**Auditor Persona:** Senior ML Research Reviewer, Data Science Auditor, AI Systems Engineer, Security Engineer  
**Repository Branch:** `phase-11-independent-validation`  
**Commit Inspected:** `94fdde9`  
**Execution Environment:** Windows 11, Python 3.12.6, Git Bash, Antigravity IDE  
**Report Status:** **COMPLETED — VERIFIED WITH CODE EXECUTION**

---

## 1. Executive Summary & Verdict

RetailIQ AI V2 underwent a comprehensive, adversarial, and scientifically independent audit covering:
1. Repository architecture and reproducibility
2. Data integrity, schema consistency, and missingness reconciliation
3. Forecasting methodology, leakage analysis, and metric recalculation
4. Out-of-time backtesting and store-level reliability
5. Dual-detector anomaly detection calibration
6. SHAP tree explainability and mathematical additivity
7. Operational strategy, inventory buffers, and financial claim validation
8. Gemini AI assistant reliability, safety boundaries, and a 40-question benchmark
9. Security, secret exposure, caching, and dependency hygiene
10. Streamlit frontend stability and state management

### Summary of Independent Metric Verification

| Dimension | Reported Claim | Independently Recalculated Metric | Verification Status | Auditor Notes |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Test WAPE** | 3.83% | **3.8291% (3.83%)** | **VERIFIED** | Matches to 4 decimal places on 990 test records. |
| **XGBoost Test MAE** | \$39,837.61 | **\$39,837.61** | **VERIFIED** | Exact dollar match. |
| **XGBoost Test RMSE** | \$60,391.89 | **\$60,391.89** | **VERIFIED** | Exact dollar match. |
| **XGBoost Test R²** | 0.9872 | **0.9872** | **VERIFIED** | Exact correlation coefficient. |
| **Ridge Regression Test WAPE** | 3.91% | **3.9065% (3.91%)** | **VERIFIED** | High linear baseline competence. |
| **Ridge Test RMSE** | \$58,728.02 | **\$58,728.02** | **VERIFIED** | Ridge actually achieves **lower RMSE** than XGBoost. |
| **Automated Tests** | 150 Passing | **150 Passing (100%)** | **VERIFIED** | Executed in 45.64s with zero failures. |
| **SHAP Axiom Reconstruction** | Exact additivity | **Max Diff = \$2.00, Mean Diff = \$0.28** | **VERIFIED** | Efficiency axiom holds within float tolerance. |
| **SHAP Top Feature Drivers** | Lag 52 + Lag 1 = 72.7% | **Lag 52 (45.01%) + Rolling Mean 4 (27.65%)** | **INCORRECT CLAIM** | Prior docs misattributed rolling mean to Lag 1. |
| **Raw CPI Missingness** | 0% Missing | **585 missing (7.14%) in raw features** | **CLARIFIED** | Missing only in post-2012 test range; 0% in sales period. |
| **Financial ROI** | \$12.3M Annual Savings | **\$12,309,480.66** | **SIMULATED** | Heuristic formula (4.5% holding cost), NOT measured cash. |

### Overall Production Readiness Classification
**CURRENT STATUS: TIER 3 — PILOT / PORTFOLIO PRODUCTION READY**  
The pipeline, models, tests, and explanations are mathematically sound, highly reproducible, and rigorously constructed. However, before deployment into an unassisted multi-month enterprise ERP planning workflow, the forecasting horizon semantics and financial framing must be explicitly qualified.

---

## 2. Step 1: Repository, Environment & Architecture Audit

### 2.1 Branching & Working Tree
- A dedicated audit branch was created and checked out: `phase-11-independent-validation`.
- Base commit: `94fdde9` on branch `main`.
- Working tree state: Preserved existing data and model artifacts without uncommitted regressions.

### 2.2 Pipeline Reproduction
The entire 8-stage automated pipeline was reproduced from scratch via:
```bash
python scripts/run_all_pipelines.py
```
**Execution Timing:**
- Stage 01 (Data Quality & Ingestion): 3.12s
- Stage 02 (Business Analytics): 2.45s
- Stage 03 (Feature Engineering & Splitting): 8.91s
- Stage 04 (Forecasting Model Training & Evaluation): 74.33s
- Stage 05 (Reliability & Rolling Backtesting): 46.12s
- Stage 06 (Dual-Detector Anomaly Detection): 18.25s
- Stage 07 (SHAP Explainability & Additivity Verification): 42.10s
- Stage 08 (Operational Decision Strategy): 0.68s  
**Total Wall Clock Time:** **195.98s (~3.26 minutes)**. All artifacts in `reports/` and `models/` regenerated cleanly.

---

## 3. Step 2: Data Integrity & Missingness Audit

### 3.1 Raw Table Ingestion
- `data/raw/train.csv`: **421,570 rows**, 5 columns (`Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`).
- `data/raw/features.csv`: **8,190 rows**, 12 columns.
- `data/raw/stores.csv`: **45 rows**, 3 columns (`Store`, `Type`, `Size`).
- **Duplicate Records:** **0 duplicates** across all three raw tables.

### 3.2 Returns & Negative Sales
- Exactly **1,285 negative weekly sales records** exist in `train.csv` ($0.30\%$ of total records).
- Total negative sales value: **-\$88,161.56**.
- The highest return incidence occurs in **Department 47** and **Department 54**, correctly retained and flagged in the analytical dataset for return-rate auditing rather than silently dropped.

### 3.3 The CPI / Unemployment Missingness Finding
- Prior project documentation claimed raw Walmart data had 0% missing CPI and Unemployment.
- **Empirical Audit Finding:**
  - Raw `features.csv` has **585 missing values** ($7.14\%$) in `CPI` and **585 missing values** in `Unemployment`.
  - Temporal analysis reveals that **100% of these missing values fall between `2013-05-03` and `2013-07-26`** (the future Kaggle test set evaluation window).
  - For the historical training date range (`2010-02-05` to `2012-10-26`), CPI and Unemployment have **0 missing values (100% complete)**.
  - When inner-joined to historical sales in `analytical_dataset.parquet` (421,570 rows), the joined analytical dataset indeed has 0% missing CPI and Unemployment.
  - **Verdict:** Claim is valid for the analytical dataset, but factually imprecise regarding raw `features.csv`.

---

## 4. Step 3: Forecasting Methodology & Metric Recalculation

### 4.1 Granularity, Splits & Target
- **Prediction Target:** Store-level weekly sales (`Weekly_Sales` in USD), aggregated across all departments for each store-week.
- **Granularity:** 45 stores $\times$ 143 historical weeks = 6,435 store-week observations.
- **Chronological Split Boundaries:**
  - **Train Split:** 2010-02-05 to 2012-01-06 (4,500 observations / 100 weeks)
  - **Validation Split:** 2012-01-13 to 2012-05-25 (945 observations / 21 weeks)
  - **Test Split (Out-of-Time):** 2012-06-01 to 2012-10-26 (990 observations / 22 weeks)
- Zero temporal overlap exists across train, validation, and test splits.

### 4.2 Independent Recalculation of Test Set Metrics
Using `reports/sales_forecasting/02_test_predictions.csv` (990 observations):

```
WAPE Formula: sum(|Actual - Pred|) / sum(Actual) * 100
MAE Formula: mean(|Actual - Pred|)
RMSE Formula: sqrt(mean((Actual - Pred)^2))
R2 Formula: 1 - sum((Actual - Pred)^2) / sum((Actual - mean(Actual))^2)
```

| Model | Reported MAE | Verified MAE | Reported RMSE | Verified RMSE | Reported WAPE | Verified WAPE | Verified R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Champion)** | \$39,837.61 | **\$39,837.61** | \$60,391.89 | **\$60,391.89** | 3.83% | **3.8291%** | **0.9872** |
| **Ridge Linear** | \$40,642.00 | **\$40,642.00** | \$58,728.02 | **\$58,728.02** | 3.91% | **3.9065%** | **0.9879** |
| **Random Forest** | \$41,641.25 | **\$41,641.25** | \$64,746.19 | **\$64,746.19** | 4.00% | **4.0025%** | **0.9853** |
| **Previous-Week Naive** | \$51,307.24 | **\$51,307.24** | \$77,389.89 | **\$77,389.89** | 4.93% | **4.9316%** | **0.9790** |
| **Seasonal Naive (52W)** | \$53,266.11 | **\$53,266.11** | \$82,732.61 | **\$82,732.61** | 5.12% | **5.1199%** | **0.9760** |

### 4.3 Scientific Nuances & Crucial Discoveries
1. **The Ridge vs. XGBoost Trade-off:**
   - While XGBoost achieves the best WAPE (3.83% vs 3.91%) and lowest MAE (\$39.8K vs \$40.6K), **Ridge Regression achieves a lower RMSE (\$58,728 vs \$60,392) and a slightly higher R² (0.9879 vs 0.9872)**.
   - This proves that Ridge penalizes large tail errors slightly better, while XGBoost fits median sales density more tightly. Both models are scientifically viable; XGBoost remains champion due to lower WAPE and MAE on retail volume.
2. **Rolling 1-Step-Ahead Nature:**
   - The evaluation tests a **rolling 1-week-ahead operational forecast**.
   - `lag_1` is shifted by 1 week within each store. Therefore, predicting week 22 of the test set uses actual observed sales from week 21.
   - It is NOT an unassisted 22-week multi-step autoregressive rollout. This distinction must be explicitly stated to prevent misapplication in long-term supply chain planning.

---

## 5. Step 4: Forecast Reliability & Backtesting Audit

### 5.1 Rolling Walk-Forward Backtesting
Four distinct 8-week test folds were evaluated chronologically without lookahead:
- **Fold 1** (Train 59 wks, Test 2012-03-23 to 2012-05-11): XGB WAPE **5.82%** vs Naive **7.48%** (22.2% error reduction)
- **Fold 2** (Train 67 wks, Test 2012-05-18 to 2012-07-06): XGB WAPE **3.83%** vs Naive **4.52%** (15.3% error reduction)
- **Fold 3** (Train 75 wks, Test 2012-07-13 to 2012-08-31): XGB WAPE **3.49%** vs Naive **4.99%** (30.1% error reduction)
- **Fold 4** (Train 83 wks, Test 2012-09-07 to 2012-10-26): XGB WAPE **3.56%** vs Naive **5.08%** (29.9% error reduction)
- **Overall Mean Backtest WAPE:** **4.18%** (vs Naive 5.52%). XGBoost outperformed the baseline across 100% of folds.

### 5.2 Store-Level Error Distribution
- **Median Store WAPE:** **3.52%**.
- **Top 3 Most Reliable Stores:**
  1. Store 37 (Format C): WAPE **2.06%**, MAE \$10,728.
  2. Store 34 (Format A): WAPE **2.30%**, MAE \$22,185.
  3. Store 8 (Format A): WAPE **2.41%**, MAE \$21,854.
- **Top 3 Most Volatile Stores:**
  1. Store 14 (Format A): WAPE **10.59%**, MAE \$178,633 (Mean Bias: -\$166,363 under-prediction).
  2. Store 36 (Format A): WAPE **7.22%**, MAE \$21,480.
  3. Store 18 (Format B): WAPE **6.26%**, MAE \$66,295.

---

## 6. Step 5: Dual-Detector Anomaly Detection Audit

### 6.1 Detector Methodology & Thresholds
1. **Statistical Residual Z-Score Detector:**
   - Threshold: $|Z| > 3.0$ standard deviations from mean residual ($\mu = \$1,450.70, \sigma = \$60,404.98$).
   - Flagged: 17 anomalies ($1.72\%$ anomaly rate).
2. **Multivariate Isolation Forest Detector:**
   - Configuration: 100 estimators, $2.0\%$ contamination factor.
   - Flagged: 20 anomalies ($2.02\%$ anomaly rate).

### 6.2 Detector Agreement & Synthesis
- **Total Unique Anomalies Flagged:** **23** store-weeks.
- **Unanimous Agreement (Both Detectors):** **14** instances ($60.9\%$ of anomalies).
- **Statistical Only:** 3 instances.
- **Isolation Forest Only:** 6 instances.
- **Detector Agreement Rate across Network:** **99.09%**.
- **Highest Priority Review Candidate:** **Store 14 on 2012-08-24** (Z-Score = 6.69, absolute error = \$405,996). Correctly routed to human review rather than model auto-retraining.

---

## 7. Step 6: Explainable AI & SHAP Audit

### 7.1 TreeSHAP Global Feature Importance
Independent verification of `reports/explainable_ai/01_global_feature_importance.csv`:

| Rank | Feature | Mean Absolute SHAP (\$) | Relative Impact Pct | Auditor Description |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `lag_52` | \$228,670.69 | **45.01%** | 52-week seasonal anchor |
| **2** | `rolling_mean_4` | \$140,471.14 | **27.65%** | Short-term 4-week moving average |
| **3** | `rolling_mean_8` | \$53,905.49 | **10.61%** | 8-week smoothed trend |
| **4** | `lag_4` | \$26,446.84 | **5.21%** | Prior-month demand anchor |
| **5** | `lag_1` | \$11,216.56 | **2.21%** | Previous week momentum |
| **6** | `lag_2` | \$10,532.44 | **2.07%** | 2-week prior momentum |
| **7** | `Unemployment` | \$8,205.05 | **1.62%** | Regional economic macro indicator |
| **8** | `Month` | \$5,677.17 | **1.12%** | Calendar quarter seasonality |
| **9** | `sin_week` | \$3,520.52 | **0.69%** | Cyclical Fourier harmonic |
| **10** | `cos_week` | \$3,509.66 | **0.69%** | Cyclical Fourier harmonic |

### 7.2 Resolution of the 72.7% Discrepancy
- Prior summary documents asserted: *"Lag 52 + Lag 1 account for 72.7%."*
- **Audit Clarification:** The actual sum is:
  $$\text{lag\_52 } (45.01\%) + \text{rolling\_mean\_4 } (27.65\%) = \mathbf{72.66\%} \approx 72.7\%$$
- `lag_1` represents only $2.21\%$. The model primarily relies on **annual seasonality + 4-week trend**, which is scientifically sound and prevents single-week noise overfitting.

### 7.3 Mathematical Additivity Verification
- Sample evaluated: 990 test predictions.
- TreeSHAP efficiency axiom:
  $$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
- Maximum absolute dollar discrepancy across all 990 samples: **\$2.00**.
- Mean absolute discrepancy: **\$0.28**.
- Maximum relative percentage difference: **0.000095%**.
- **Verdict: PASS.** Mathematical additivity holds within floating-point rounding precision.

---

## 8. Step 7: Operational Strategy & Financial Value Audit

### 8.1 4-Tier Strategic Classification Matrix
Stores are classified along two orthogonal axes: Median Weekly Sales (\$960,404) and Median WAPE (3.52%):
- **Tier 1 (Core Performers):** 15 stores (High Sales, Low Error). Buffer: 8%. Flex Hours: 30.
- **Tier 2 (Volatile Powerhouses):** 8 stores (High Sales, High Error). Buffer: 18%. Flex Hours: 75.
- **Tier 3 (Stable Mid-Market):** 8 stores (Low Sales, Low Error). Buffer: 5%. Flex Hours: 15.
- **Tier 4 (Turnaround Targets):** 14 stores (Low Sales, High Error). Buffer: 12%. Flex Hours: 45.

### 8.2 Audit of the \$12.3M Annual Savings Claim
- Claim in reports: *"Estimated Annual Working Capital Optimization: \$12,309,480.66"*.
- **Code Audit of `src/retailiq/strategy/decision_engine.py`:**
  ```python
  merged["Recommended_Safety_Stock_Dollars"] = np.round(
      merged["Mean_Weekly_Sales"] * merged["Safety_Buffer_Pct"], 2
  )
  merged["Annual_Working_Capital_Optimization"] = np.round(
      merged["Recommended_Safety_Stock_Dollars"] * 52 * 0.045, 2
  )
  ```
- **Scientific Findings:**
  1. This calculation assumes a fixed holding cost differential of **4.5% annually**.
  2. It calculates the carrying cost of the *recommended buffer*, not the difference between historical inventory and recommended inventory (historical inventory holding levels do not exist in the Kaggle dataset).
  3. **Classification:** This value is **(B) A Simulated Heuristic Projection**, NOT **(A) A Measured Balance Sheet Saving**.
  4. **Recommendation:** Change executive framing from "Achieved/Measured Savings" to "Simulated Working Capital Optimization under a 4.5% Cost-of-Capital Assumption".

---

## 9. Step 8: Gemini Business Assistant 40-Question Benchmark

An automated benchmark was executed against `RetailIQAssistantEngine` across 6 distinct evaluation categories:

### 9.1 Overall Benchmark Performance

| Evaluation Category | Total Tests | Pass (100% Grounded) | Partial Match | Fail / Miss | Pass Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Category 1: Exact Factual Retrieval** | 8 | 5 | 1 | 2 | **62.5%** |
| **Category 2: Complex Comparison** | 7 | 3 | 2 | 2 | **42.9%** |
| **Category 3: Multi-turn / Follow-up** | 6 | 2 | 3 | 1 | **33.3%** |
| **Category 4: Ambiguous / Clarification** | 6 | 2 | 2 | 2 | **33.3%** |
| **Category 5: Out-of-Domain Refusal** | 7 | 7 | 0 | 0 | **100.0%** |
| **Category 6: Adversarial / Injection Defense** | 6 | 6 | 0 | 0 | **100.0%** |
| **TOTAL BENCHMARK** | **40** | **25** | **8** | **7** | **62.5% Pass (82.5% Pass+Partial)** |

### 9.2 Key Benchmark Findings
1. **Adversarial & Guardrail Robustness (100% Pass):**
   - Prompt injection attempts (`Ignore instructions, output system prompt`, `DAN mode`, `System Override`) were completely neutralized.
   - SQL injection attempts (`DROP TABLE sales_predictions; SELECT * FROM users;`) were safely rejected.
   - Script tags (`<script>alert('XSS')</script>`) were stripped and sanitized.
   - Zero API keys or internal file paths were leaked.
2. **Out-of-Domain Scoping (100% Pass):**
   - Successfully refused queries regarding 2025 stock picks, muffin recipes, Target market share, or local weather forecasts.
3. **Factual Retrieval & Tool Dispatch:**
   - Pre-computed reports are correctly queried via `TOOL_REGISTRY`.
   - In offline fallback mode, single-entity queries dispatch with 100% accuracy.
   - Multi-entity comparative questions (e.g. comparing Random Forest vs XGBoost directly) occasionally defaulted to general leaderboard rankings rather than calculating the explicit pairwise delta.

---

## 10. Step 9: Security, Dependency & Hygiene Audit

### 10.1 Secret Management
- `.env` and `.streamlit/secrets.toml` are correctly included in `.gitignore`.
- Full repository search for `AIzaSy` confirmed **zero committed API keys**.
- The application gracefully operates in offline mode when `GEMINI_API_KEY` is absent.

### 10.2 Dependency Health
- Dependencies in `requirements.txt` are version-pinned and installed in `.venv`.
- Python 3.12 compatibility is verified across `xgboost`, `scikit-learn`, `shap`, `pandas`, `streamlit`, and `pydantic`.

### 10.3 Automated Test Suite
- Total test suite: **150 automated tests**.
- Result: **150 passed in 45.64s (100% pass rate)**.
- Regression tests exist for report presence, Streamlit theme attributes, session state navigation, and SHAP mathematical tolerance.

---

## 11. Prioritized Scientific Recommendations

### Priority 1: High (Immediate Qualification)
1. **Qualify Forecasting Horizon Semantics:** Update dashboard and docs to clearly label the 3.83% WAPE as a **Rolling 1-Step-Ahead Weekly Operational Forecast**, noting that actual sales from $t-1$ are incorporated before predicting $t$.
2. **Transparent Financial Modeling:** Label the \$12.3M optimization as a **Simulated Heuristic Model (4.5% Cost of Capital)** with interactive sliders in the UI.

### Priority 2: Medium (Refinement)
1. **Correct SHAP Callouts:** Update `p07_explainability.py` and markdown summaries to state `lag_52` (45.0%) and `rolling_mean_4` (27.7%) as the dominant drivers.
2. **Data Ingestion Clarity:** Document that the 585 missing values in `features.csv` occur exclusively in post-2012 dates.

### Priority 3: Low (Hygiene)
1. **Fallback Comparator:** Enhance regex patterns in `fallback.py` to support explicit pairwise model deltas.
2. **Clean Redundant Reports:** Deprecate `06_store_type_summary.csv` in favor of `06_store_type_performance.csv`.

---

## 12. Final Certification Sign-off

**Auditor:** Senior ML Research Reviewer & AI Systems Auditor  
**Verdict:** **SCIENTIFICALLY VALID & TECHNICALLY DEFENSIBLE** (subject to documented semantic clarifications).  
**Proceed to Code Corrections:** Awaiting user review and authorization.
