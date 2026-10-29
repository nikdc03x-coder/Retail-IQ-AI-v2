# Phase 00 ??? Foundation

## Problem Addressed

A data science project requires a well-organized, reproducible foundation before any analysis begins. Without consistent directory structure, dependency management, configuration, and testing infrastructure, later phases become fragile and difficult to maintain.

## Why This Phase Was Needed

- Establish a clean, importable Python package structure
- Set up version control from the very first file
- Define a consistent configuration system (paths, constants, environment variables)
- Create a professionally styled Streamlit shell ready for page-by-page development
- Ensure all imports, paths, and directory references work before adding complexity
- Establish testing infrastructure early so every phase can be verified

## Input Data

No data is processed in this phase. The directory structure for `data/raw/` and `data/processed/` is created to receive files in Phase 01.

## Methods and Libraries Used

| Tool | Purpose |
|------|---------|
| Git | Version control initialization |
| Python 3.12 | Runtime environment |
| venv | Isolated virtual environment |
| pip | Dependency installation |
| Streamlit | Dashboard shell with custom theme |
| Plotly | Chart factory setup |
| python-dotenv | Environment variable management |
| Pytest | Test framework configuration |

## Why Each Method Was Selected

- **Git branching model:** Phase-oriented branches enable incremental review and clean history
- **venv over conda:** Simpler, standard Python tooling, no external dependency manager needed
- **Streamlit:** Python-native dashboard framework ??? no frontend build toolchain required
- **python-dotenv:** Standard pattern for secret management without hardcoding
- **Pytest:** Industry-standard Python test framework with rich fixture support

## Work Completed

1. Created the project directory structure with all subdirectories
2. Initialized Git repository on `phase-00-foundation` branch
3. Created `.gitignore` blocking `.env`, `.venv`, `__pycache__`, generated data
4. Created `requirements.txt` with pinned dependency ranges
5. Created `pyproject.toml` with pytest and ruff configuration
6. Created `.env.example` template (no secrets)
7. Created `.streamlit/config.toml` with dark theme and run-on-save
8. Created `src/retailiq/config.py` ??? central configuration with paths, schemas, theme
9. Created all subpackage `__init__.py` files (data, quality, analytics, forecasting, reliability, anomalies, explainability, assistant, dashboard)
10. Created dashboard theme (`theme.py`), components (`components.py`), charts (`charts.py`)
11. Created Project Overview page (`p00_overview.py`)
12. Created `app.py` Streamlit entry point with sidebar navigation
13. Created `tests/conftest.py` with shared fixtures
14. Created `tests/test_foundation.py` with 24 smoke tests
15. Created this documentation
16. Set up Python virtual environment and installed all dependencies

## Validation and Tests

35 automated tests in `tests/test_foundation.py` covering:
- Project structure (13 directory/file existence checks)
- Configuration imports and path validity (6 tests)
- Package and subpackage imports (9 tests)
- Dashboard helper functions (4 tests)
- `.gitignore` rule checks (3 tests)

## Actual Results

All 35 tests pass cleanly in 4.52s. Streamlit shell and configuration validate without error.

## Limitations

- No data has been loaded or processed yet
- Dashboard pages 2???9 show placeholder messages
- No ML models, reports, or analysis artifacts exist yet

## Outputs Produced

| Output | Location |
|--------|----------|
| Project structure | `retailiq-ai-v2/` |
| Configuration module | `src/retailiq/config.py` |
| Dashboard theme | `src/retailiq/dashboard/theme.py` |
| Dashboard components | `src/retailiq/dashboard/components.py` |
| Chart factory | `src/retailiq/dashboard/charts.py` |
| Streamlit app | `app.py` |
| Foundation tests | `tests/test_foundation.py` |
| This document | `docs/phases/00_foundation.md` |

## How This Connects to the Next Phase

Phase 01 (Data Ingestion) will:
1. Copy the raw Walmart CSV files into `data/raw/`
2. Validate schemas against `EXPECTED_SCHEMAS` from config
3. Parse dates, join datasets, and produce `data/processed/analytical_dataset.parquet`
4. Document data lineage and row-count tracking

## Git Branch and Commit

- **Branch:** `phase-00-foundation`
- **Commit:** Initial foundation ??? project structure, config, Streamlit shell, tests

