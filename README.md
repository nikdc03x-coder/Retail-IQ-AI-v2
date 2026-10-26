# RetailIQ AI V2

**An Explainable Retail Sales Forecasting & Decision Intelligence Platform**

---

## 1. Executive Summary

**RetailIQ AI V2** is a portfolio- and research-grade enterprise retail intelligence platform built from the ground up on the historical Walmart weekly sales dataset. It bridges the gap between raw retail data engineering, predictive machine learning, reliability diagnostics, unsupervised anomaly auditing, cooperative game-theory interpretability (TreeSHAP), and evidence-grounded generative AI.

The platform provides an end-to-end reproducible pipeline, an executive 10-page dark Slate-950/900 Streamlit dashboard, a dual-route conversational business analyst powered by Google Gemini with deterministic fallback and interactive Plotly visuals, and an automated 150-test verification suite.

---

## 2. Core Capabilities & Architecture

| # | Capability | Technical Methodology | Key Outcome |
|:---:|:---|:---|:---|
| **01** | **Data Engineering & Lineage** | Schema validation, duplicate checks, join integrity, Parquet storage | 421,570 rows preserved (100.0%); 0 duplicate records; 0 holiday mismatches |
| **02** | **Data Quality & Audit** | Statistical profiling, missingness breakdown, returns audit | CPI/Unemployment 0% missing; 1,285 customer returns (-$88.2K) preserved |
| **03** | **Business Analytics** | Hierarchical store-week rollups, Pareto shares, holiday lift analysis | Total sales: $6.74B; #1 Store 20 ($301.4M); Thanksgiving lift: +41.34% |
| **04** | **Sales Forecasting** | 18 autoregressive features, chronological split, ML vs Naive baselines | Champion XGBoost: **3.83% WAPE** (-22.3% error reduction vs Naive) |
| **05** | **Forecast Reliability** | Spatial error spread, holiday error shifts, 4-fold expanding backtests | 77.8% store win rate vs Naive; 100% backtest consistency across all folds |
| **06** | **Anomaly Detection** | Residual Z-score (|Z|>3) + multivariate Isolation Forest consensus | 23 prioritized audit targets (2.3%); 14 unanimous flags; 99.09% agreement |
| **07** | **Explainable AI (TreeSHAP)** | Global dollar attribution, local waterfall decomposition, efficiency audit | Top 2 features explain 72.66% power; 100% predictions reconstruct exactly |
| **08** | **Operational Decision Strategy** | 4-Quadrant Store Matrix (Volume vs WAPE), Dynamic Buffers & Markdown Rules | $12.3M annual holding cost optimization; 1,800 weekly labor flex hours |
| **09** | **Enterprise Dashboard** | Multi-page Streamlit app (`app.py`), Material Symbols, Inter typography | Sub-0.5s load times; 10 operational pages; zero casual emojis |
| **10** | **AI Business Assistant** | Dual-route: Gemini (`gemini-2.0-flash`) tool calling + deterministic fallback | 13 typed Python tools; interactive Plotly charts; dynamic follow-up chips |
| **11** | **Testing & Verification** | Automated master runner (`run_all_pipelines.py`) and Pytest test suite | 150 passing tests; full end-to-end pipeline executes in <40 seconds |

---

## 3. Key Empirical Findings & Leaderboard

### Forecasting Model Leaderboard (June–October 2012 Held-Out Test Split, $N = 990$)

| Rank | Model Architecture | WAPE (%) | MAE ($) | RMSE ($) | $R^2$ | Error vs. Naive |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **XGBoost Regressor (Champion)** | **3.83%** | **$39,837.61** | **$60,391.89** | **0.9872** | **-22.3%** |
| 2 | Ridge Linear Regressor | 3.91% | $40,642.00 | $58,728.02 | 0.9879 | -20.8% |
| 3 | Random Forest Regressor | 4.00% | $41,641.25 | $64,746.19 | 0.9853 | -18.8% |
| 4 | Previous-Week Naive Baseline | 4.93% | $51,307.24 | $77,389.89 | 0.9790 | Baseline |
| 5 | Seasonal Naive (52-Week) Baseline | 5.12% | $53,266.11 | $82,732.61 | 0.9760 | +3.8% |

### Key Empirical Takeaways
- **Annual Seasonality & Trailing Momentum Dominate:** TreeSHAP reveals that prior-year same-week sales (`lag_52`, mean absolute impact $\$228.7\text{K}$) and trailing 4-week momentum (`rolling_mean_4`, mean impact $\$140.5\text{K}$) account for **$72.66\%$** of all predictive power.
- **Holiday Volatility:** Thanksgiving drives a **$+41.34\%$** chain-wide sales lift, but increases forecast error volatility by **$+20.8\%$** (4.58% WAPE vs 3.79% non-holiday).
- **Spatial Reliability Disparity:** Store forecast accuracy varies by a factor of $5\times$ across locations (Store 37 leads with $2.06\%$ WAPE; Store 14 exhibits $10.59\%$ WAPE).
- **Outlier Audit Prioritization:** Exactly 23 observations are flagged for operational review (led by Store 14 on 2012-08-24 with a $+\$421\text{K}$ residual miss).

---

## 4. Technology Stack

- **Data Processing:** Python 3.12, Pandas, NumPy, PyArrow (Parquet)
- **Machine Learning:** Scikit-learn, XGBoost, Joblib
- **Interpretability & Explainability:** SHAP (TreeExplainer)
- **Generative AI & Tool Calling:** Google Gemini API (`gemini-2.0-flash` via `google-genai` SDK)
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Dashboard Application:** Streamlit
- **Testing & Quality Control:** Pytest

---

## 5. Quick Start & Reproducibility Guide

### Instant Start via Node / npm (Recommended)
You can launch the full application in a single command without manually activating your virtual environment:

```bash
# Start Streamlit application immediately (port 8501)
npm run dev

# Run automated test suite
npm test
```

---

### Python Environment Setup

```bash
# 1. Clone repository and navigate to root
cd ~/Documents/retailiq-ai-v2

# 2. Activate Python virtual environment
source .venv/Scripts/activate  # On Windows Git Bash
# or .venv\Scripts\activate   # On Windows CMD/PowerShell

# 3. Verify dependencies
pip install -r requirements.txt

# 4. Optional: Configure Gemini API Key in .env (Deterministic fallback works offline without it)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY="your-api-key"
```

### Running the Entire Analytical Pipeline
To execute all 8 pipeline stages sequentially from raw CSV data to final SHAP explainability and operational decision artifacts in under 40 seconds:

```bash
python scripts/run_all_pipelines.py
```

Or execute individual pipeline modules:
```bash
python scripts/01_data_ingestion/run_ingestion.py
python scripts/02_data_quality/run_quality_checks.py
python scripts/03_business_analytics/run_analytics.py
python scripts/04_sales_forecasting/run_forecasting.py
python scripts/05_forecast_reliability/run_reliability.py
python scripts/06_anomaly_detection/run_anomaly_detection.py
python scripts/07_explainability/run_shap_analysis.py
python scripts/08_strategy/run_strategy_engine.py
```

### Running Automated Test Suite
To execute all 150 unit, integration, benchmark, and smoke tests:

```bash
pytest -v
```

### Launching the Interactive Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser to access the 10 interactive operational pages.

---

## 6. Directory Structure

```
retailiq-ai-v2/
├── app.py                             # Streamlit application entry point & router
├── pyproject.toml                     # Project packaging and tool configuration
├── requirements.txt                   # Frozen production dependencies
├── .env.example                       # Template for environment variables
├── data/
│   ├── raw/                           # Immutable source files (train, features, stores)
│   └── processed/                     # Analytical dataset (Parquet, 421.5k rows)
├── models/                            # Trained models (Ridge, Random Forest, XGBoost)
├── reports/                           # Output artifacts per analysis phase
│   ├── data_quality/                  # Statistical profiles & missingness audits
│   ├── business_analytics/            # Store/dept rankings, holiday lift reports
│   ├── sales_forecasting/             # Leaderboard metrics, predictions, importance
│   ├── forecast_reliability/          # Store error spread, rolling backtest results
│   ├── anomaly_detection/             # Z-score, isolation forest, investigation queue
│   └── explainable_ai/                # Global SHAP importance, local attributions
├── src/retailiq/                      # Source Python package
│   ├── config.py                      # Central configuration, schemas, theme tokens
│   ├── data/                          # Loader, schema validator, ingestion pipeline
│   ├── quality/                       # Data profiling and returns auditing
│   ├── analytics/                     # Store-week aggregations, rankings, holiday lift
│   ├── forecasting/                   # Features, chronological splitter, models, evaluation
│   ├── reliability/                   # Spatial error distributions, backtesting
│   ├── anomalies/                     # Statistical & Isolation Forest outlier detectors
│   ├── explainability/                # TreeSHAP explainer, efficiency axiom verification
│   ├── assistant/                     # 13 typed analytics tools, fallback, Gemini engine
│   └── dashboard/                     # Theme styles, components, chart factory, 8 pages
├── scripts/                           # Reproducible CLI pipeline runners
│   ├── run_all_pipelines.py           # Master end-to-end pipeline orchestrator
│   ├── 01_data_ingestion/             # run_ingestion.py
│   ├── 02_data_quality/               # run_quality_checks.py
│   ├── 03_business_analytics/         # run_analytics.py
│   ├── 04_sales_forecasting/          # run_forecasting.py
│   ├── 05_forecast_reliability/       # run_reliability.py
│   ├── 06_anomaly_detection/          # run_anomaly_detection.py
│   └── 07_explainability/             # run_shap_analysis.py
├── tests/                             # 138-test automated pytest verification suite
└── docs/                              # Comprehensive research documentation
    ├── architecture.md                # System layers, components, dependency map
    ├── data_flow.md                   # Full data transformation lifecycle
    ├── phases/                        # 16-section reports for Phases 00 through 10
    └── research/                      # Progress logs and research milestones
```

---

## 7. Ethical AI & Scientific Boundaries

To uphold software quality and research integrity, RetailIQ AI V2 explicitly states the following operational boundaries:
1. **Influence vs. Causation:** SHAP values quantify internal model attribution weights, not physical real-world levers. High historical sales influence future predictions because of consumer habits, not because prior revenue physically causes future revenue.
2. **Audit Targets vs. Confirmed Fraud:** Anomaly flags highlight observations with unusually large forecast residuals or multidimensional distance. They serve as operational review targets, not confirmed employee theft, accounting defects, or system failure.
3. **Historical Scope:** Historical sales reflect consumer demand at recorded prices and inventory levels. They do not equal operational profit or margin, and cannot simulate live inventory stockouts without dedicated supply chain telemetry.
4. **Controlled Generative AI:** Google Gemini is strictly constrained to executing pre-approved typed analytical functions. It does not execute arbitrary code or query live external databases unassisted.

---

## 8. Research & Scholarship Defense Positioning

For third-year B.Tech project presentations, technical interviews, and postgraduate scholarship applications, this platform provides concrete proof of:
- **Data Engineering Discipline:** 100% row preservation, zero silent dropping, and structured Parquet engineering.
- **Machine Learning Rigor:** Fair chronological time-series splitting, leakage-free lag construction, and benchmarking ML against naive baselines.
- **Reliability Engineering:** Auditing error distributions across store geographies and expanding historical horizons.
- **Explainability:** Auditing the cooperative game theory efficiency axiom ($f(x) = \phi_0 + \sum \phi_i$) within $<0.0001\%$ precision.
- **Software Quality:** Clean package modularity, 138 automated tests, and sub-second UI rendering.

---

## 9. License

Academic research and demonstration project — not licensed for commercial operations.
