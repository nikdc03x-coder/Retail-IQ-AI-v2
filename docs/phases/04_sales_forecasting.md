# Phase 04 — Sales Forecasting & Model Evaluation

## 1. Phase Title
**Phase 04: Leakage-Safe Time-Series Feature Engineering, Chronological Splitting, and Multi-Model Evaluation Benchmark**

---

## 2. Problem Addressed
Retail sales forecasting is prone to methodological flaws that invalidate benchmark claims:
- Random cross-validation shuffling across temporal sequences, which leaks future information into past predictions.
- Over-promising forecast horizons (claiming multi-step forecasting when models rely on one-step lagged targets).
- Declaring complex ML algorithms (like XGBoost) superior without rigorous comparison against naive and seasonal persistence baselines.
- Evaluating competing models on differing subsets of observations.

Phase 04 establishes a reproducible, chronological 1-week-ahead ($t+1$) store-level forecasting experiment that fairly compares persistence baselines with linear and tree-based machine learning models.

---

## 3. Why the Phase Was Needed
- **Strict Leakage Prevention:** Feature transformations (lags, rolling averages, std) must be backward-looking ($t-k$) and shifted to preclude target leakage.
- **Fair Benchmark Testing:** All models (Previous-Week Naive, Seasonal 52-Week Naive, Ridge Regression, Random Forest, XGBoost) are trained and tested on the exact same temporal subsets.
- **Standardized Multi-Metric Evaluation:** Evaluate models on both scale-dependent (MAE, RMSE) and scale-independent (WAPE, R²) criteria across validation and held-out test splits.
- **Model Serialization:** Save production-ready trained model artifacts (.joblib) for downstream explainability (Phase 07) and real-time inference.

---

## 4. Input Data
- `reports/business_analytics/04_store_week_sales.csv` (6,435 store-week observations across 45 stores and 143 dates).

---

## 5. Data Understanding
- **Target Variable:** Store-week `Weekly_Sales` ($y_{s, t}$), representing total weekly store revenue.
- **Forecast Horizon:** Exactly **1-week-ahead ($t+1$)** using previously observed sales and known calendar signals.
- **Temporal Partitions:**
  - **Training Split:** Dates before `2012-01-06` (48 modeled weeks after dropping initial 52-week lag warmup, 2,160 observations).
  - **Validation Split:** `2012-01-06` to `2012-05-25` (21 weeks, 945 observations).
  - **Held-Out Test Split:** `2012-06-01` to `2012-10-26` (22 weeks, 990 observations).
- **Features Formulated (20):**
  - Autoregressive Lags: `lag_1`, `lag_2`, `lag_4`, `lag_52` (year-over-year seasonal lag).
  - Rolling Windows (shifted by 1): `rolling_mean_4`, `rolling_std_4`, `rolling_mean_8`.
  - Exogenous/Environmental: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `is_holiday_num`.
  - Static Store Attributes: `Size`, `Type_A`, `Type_B`, `Type_C`.
  - Calendar/Cyclical: `Month`, `WeekOfYear`, `sin_week`, `cos_week`.

---

## 6. Methods and Libraries Used
- **Scikit-Learn 1.9+:** Standard scaling pipeline, `Ridge(alpha=100.0)`, and `RandomForestRegressor(n_estimators=100, max_depth=12)`.
- **XGBoost 3.4+:** `XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85)`.
- **Joblib 1.6+:** High-performance model serialization.
- **Pandas & NumPy:** Feature engineering and metric calculation.
- **Streamlit & Plotly:** Model comparison leaderboard, trajectory plots, feature importance, and residual diagnostics.
- **Pytest 9.1+:** 6 dedicated tests verifying feature alignment, absence of leakage, and metric tolerances.

---

## 7. Why Each Method Was Selected
- **Chronological Split over K-Fold:** Shuffling time series generates unrealistically optimistic test scores due to auto-correlated leakage. Forward-in-time splitting simulates genuine operational deployment.
- **WAPE (Weighted Absolute Percentage Error) as Primary Metric:** Unlike MAPE, which divides by individual actuals and explodes when actuals are small or negative, WAPE ($\sum |y - \hat{y}| / \sum |y|$) provides a robust, volume-weighted percentage error suitable for retail portfolio assessment.
- **Inclusion of Naive Baselines:** A complex ML model is only valuable if it statistically beats simple rules of thumb (e.g. predicting last week's sales).

---

## 8. Work Completed
1. **Feature Engineering Engine (`src/retailiq/forecasting/features.py`):**
   - Built `build_forecasting_features()` constructing strictly backward-looking lags and rolling stats.
   - Built cyclical Fourier encodings (`sin_week`, `cos_week`) and store one-hot encodings.
2. **Chronological Splitter (`src/retailiq/forecasting/splitter.py`):**
   - Built `chronological_split()` with automated temporal assertions ensuring train max < val min < test min.
3. **Evaluation Engine (`src/retailiq/forecasting/evaluation.py`):**
   - Built `calculate_metrics()` computing MAE, RMSE, WAPE, R², and Bias MPE.
4. **Model Architecture (`src/retailiq/forecasting/models.py`):**
   - Implemented persistence baselines (`predict_naive_previous_week`, `predict_seasonal_naive_52`).
   - Implemented `train_ridge_model()`, `train_random_forest_model()`, and `train_xgboost_model()`.
   - Implemented `extract_feature_importance()` for comparative model interpretability.
5. **Experiment Orchestrator (`src/retailiq/forecasting/experiment.py`):**
   - Pipeline executing training, baseline evaluation, model serialization (`.joblib`), and report generation.
6. **Execution Script (`scripts/04_sales_forecasting/run_forecasting.py`):**
   - Executable CLI running full experiment in 2.56 seconds.
7. **Interactive Forecasting Dashboard (`src/retailiq/dashboard/pages/p04_forecasting.py`):**
   - 4-tab UI with Leaderboard, Store Trajectory Overlays, Feature Importance, and Residual Diagnostics.
8. **Automated Test Suite (`tests/test_sales_forecasting.py`):**
   - 6 tests asserting mathematical metric correctness, zero leakage, model inference, and accuracy bounds.

---

## 9. Validation and Tests
Executed via Pytest:
- `TestFeatureEngineering`: Verified 20 feature columns generated with zero NaNs on modeled subset, and verified `lag_1` exactly matches $t-1$ observed sales.
- `TestChronologicalSplit`: Verified strict temporal separation (train < val < test) and 45 stores across all 990 test rows.
- `TestMetricsCalculation`: Verified mathematical calculation against synthetic ground truth.
- `TestModelArtifactsAndLeaderboard`: Verified `.joblib` model artifacts load and run inference on test features.
- Verified XGBoost test WAPE < 4.5% and outperformance over Naive baseline.
- **Overall Result:** 70 tests passed across project in 17.24s.

---

## 10. Actual Results
### Held-Out Test Evaluation Benchmark (June 2012 – October 2012, 990 observations)
| Rank | Model | MAE ($) | RMSE ($) | WAPE (%) | R² Score | Error Reduction vs Naive |
|---|---|---|---|---|---|---|
| **1 (Champion)** | **XGBoost** | **\$39,837.61** | **\$60,391.89** | **3.83%** | **0.9872** | **-22.3%** |
| 2 | Ridge Linear | \$40,642.00 | \$58,728.02 | 3.91% | 0.9879 | -20.7% |
| 3 | Random Forest | \$41,641.25 | \$64,746.19 | 4.00% | 0.9853 | -18.9% |
| 4 | Previous-Week Naive | \$51,307.24 | \$77,389.89 | 4.93% | 0.9790 | Baseline |
| 5 | Seasonal Naive (52-Wk) | \$53,266.11 | \$82,732.61 | 5.12% | 0.9760 | +3.9% (worse) |

- **Top Predictive Features (XGBoost):**
  1. `lag_1` (Recent revenue baseline)
  2. `rolling_mean_4` (Short-term operational momentum)
  3. `Size` (Store square-footage capacity)
  4. `lag_52` (Year-over-year seasonality)
  5. `rolling_mean_8` (Medium-term trend)

---

## 11. Interpretation of Results
- **ML Delivers Genuine Value:** XGBoost reduces absolute forecasting error by **22.3%** relative to the standard persistence baseline (\$39.8k MAE vs \$51.3k MAE).
- **Linear Model Competitiveness:** Ridge linear regression achieved a strong 3.91% WAPE, nearly matching XGBoost. In weekly store-level retail aggregations, autoregressive lags and store size explain a dominant fraction of variance linearly.
- **Why Seasonal Naive (52-Wk) Underperformed:** While 52-week seasonality captures annual patterns, it fails to account for store-level secular growth, local inflation, or calendar shifts (e.g. floating holiday dates). Combining lag_1, lag_52, and rolling means within XGBoost captures both short-term momentum and seasonal anchors.

---

## 12. Limitations
- **Forecast Horizon:** This benchmark strictly evaluates **1-week-ahead ($t+1$) forecasting**. It must not be presented as a multi-week recursive forecast without acknowledging autoregressive error compounding.
- **Store-Level Aggregation:** Errors are evaluated at total store weekly revenue. Department-level forecasting would naturally exhibit higher relative percentage error (due to smaller revenue denominators).

---

## 13. Problems Encountered and Resolutions
- **Issue:** The first 52 weeks of the dataset cannot compute `lag_52` without artificial padding.
- **Resolution:** Dropped the initial 52 warmup weeks (February 2010 to January 2011) so that every single model was evaluated on identical feature sets and identical observation periods without synthetic imputation.

---

## 14. Outputs Produced
| Output Artifact | Location | Type |
|---|---|---|
| Model Comparison Metrics | `reports/sales_forecasting/01_model_comparison_metrics.csv` | CSV Report |
| Test Predictions Audit | `reports/sales_forecasting/02_test_predictions.csv` | CSV Report |
| Feature Importance Report | `reports/sales_forecasting/03_feature_importance.csv` | CSV Report |
| Experiment Metadata JSON | `reports/sales_forecasting/04_experiment_metadata.json` | JSON Metric Report |
| Ridge Model Artifact | `models/ridge_model.joblib` | Serialized Model (2.1 KB) |
| Random Forest Artifact | `models/random_forest_model.joblib` | Serialized Model (7.1 MB) |
| XGBoost Model Artifact | `models/xgboost_model.joblib` | Serialized Model (382 KB) |
| Forecasting Features Module | `src/retailiq/forecasting/features.py` | Python Module |
| Chronological Splitter | `src/retailiq/forecasting/splitter.py` | Python Module |
| Evaluation Metrics Module | `src/retailiq/forecasting/evaluation.py` | Python Module |
| Model Training Module | `src/retailiq/forecasting/models.py` | Python Module |
| Experiment Orchestrator | `src/retailiq/forecasting/experiment.py` | Python Module |
| Forecasting Execution Script | `scripts/04_sales_forecasting/run_forecasting.py` | Python Script |
| Sales Forecasting Page | `src/retailiq/dashboard/pages/p04_forecasting.py` | Streamlit Page |
| Forecasting Tests | `tests/test_sales_forecasting.py` | Pytest Suite |
| Phase 04 Documentation | `docs/phases/04_sales_forecasting.md` | Research Document |

---

## 15. How This Connects to the Next Phase
With the champion model (XGBoost) and predictions established, the project moves to **Phase 05 (Forecast Reliability Analysis)**:
- Investigate whether forecasting errors vary systematically across the 45 stores (high-error vs low-error locations).
- Analyze error volatility across weeks and evaluate whether holiday periods degrade forecast reliability.
- Conduct a rolling backtest over historical folds to determine if XGBoost consistently outperforms the baseline over time.

---

## 16. Git Branch and Commit Reference
- **Branch:** `phase-04-sales-forecasting`
- **Commit:** `feat(phase-04): implement sales forecasting models, evaluation benchmark, and dashboard`
