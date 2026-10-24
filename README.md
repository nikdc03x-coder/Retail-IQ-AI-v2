# RetailIQ AI V2

**Explainable Retail Sales Forecasting & Decision Intelligence Platform**

---

## Overview

RetailIQ AI V2 transforms historical Walmart retail sales data into actionable business intelligence through six integrated capabilities:

| # | Capability | Description |
|---|-----------|-------------|
| 1 | **Data Engineering & Quality** | Validated, reproducible pipeline with quality monitoring |
| 2 | **Business Analytics** | Interactive store rankings, trends, holiday analysis |
| 3 | **Sales Forecasting** | ML models vs baselines with chronological evaluation |
| 4 | **Forecast Reliability** | Error analysis by store, time period, and holiday |
| 5 | **Anomaly Investigation** | Statistical + Isolation Forest detectors with agreement analysis |
| 6 | **Explainable AI** | SHAP global/local explanations with reconstruction checks |
| — | **AI Business Assistant** | Gemini-powered conversational analyst grounded in evidence |

## Dataset

Public Walmart historical sales forecasting dataset ([Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)):
- ~421K store-department-week observations
- 45 stores, ~81 departments
- Period: approximately 2010–2012

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy, PyArrow |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Conversational AI | Google Gemini (`google-genai`) |
| Testing | Pytest |

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd retailiq-ai-v2

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the data pipeline (in order)
python scripts/01_data_ingestion/run_ingestion.py
python scripts/02_data_quality/run_quality_checks.py
python scripts/03_business_analytics/run_analytics.py
python scripts/04_sales_forecasting/run_forecasting.py
python scripts/05_forecast_reliability/run_reliability.py
python scripts/06_anomaly_detection/run_anomaly_detection.py
python scripts/07_explainable_ai/run_explainability.py

# Run tests
pytest

# Launch the dashboard
python -m streamlit run app.py
```

## Project Status

| Phase | Status |
|-------|--------|
| Phase 00 — Foundation | ✅ Complete |
| Phase 01 — Data Ingestion | 🔲 Pending |
| Phase 02 — Data Quality | 🔲 Pending |
| Phase 03 — Business Analytics | 🔲 Pending |
| Phase 04 — Sales Forecasting | 🔲 Pending |
| Phase 05 — Forecast Reliability | 🔲 Pending |
| Phase 06 — Anomaly Detection | 🔲 Pending |
| Phase 07 — Explainable AI | 🔲 Pending |
| Phase 08 — Dashboard | 🔲 Pending |
| Phase 09 — Business Assistant | 🔲 Pending |
| Phase 10 — Testing & Docs | 🔲 Pending |

## Documentation

- [Architecture](docs/architecture.md)
- [Phase Documentation](docs/phases/)
- [Progress Log](docs/research/progress_log.md)

## License

Academic project — not for commercial use.

