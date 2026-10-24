# Phase 05 — Forecast Reliability & Error Diagnostics

## 1. Phase Title
**Phase 05: Spatial Error Disparity, Temporal Volatility, Holiday Risk Assessment, and Rolling Backtest Stability**

---

## 2. Problem Addressed
A single global test score (e.g. 3.83% WAPE) conceals localized operational vulnerabilities:
- Forecast accuracy may vary dramatically by location, masking high-risk stores.
- Promotional holiday periods may suffer higher forecast variance than non-holiday weeks.
- A model might outperform a naive baseline on a single arbitrary test split while underperforming over other historical periods.
- Retail managers cannot plan safety stock or labor buffers without store-specific and temporal reliability intervals.

Phase 05 moves beyond aggregate accuracy to conduct diagnostic reliability analysis across stores, weeks, holidays, and expanding-window historical folds.

---

## 3. Why the Phase Was Needed
- **Spatial Error Breakdown:** Profile WAPE and MAE individually across all 45 Walmart locations.
- **Holiday Risk Quantification:** Test whether forecast accuracy systematically degrades during holiday promotional surges.
- **Head-to-Head Win Rate:** Verify what percentage of stores show statistically lower error with XGBoost compared to naive persistence.
- **Rolling Backtest Assessment:** Retrain and evaluate across 4 expanding historical folds to confirm temporal stability and guard against split-selection bias.

---

## 4. Input Data
- `reports/sales_forecasting/02_test_predictions.csv` (990 store-week test predictions).
- `reports/business_analytics/04_store_week_sales.csv` (6,435 store-week series used for backtesting).

---

## 5. Data Understanding
Empirical reliability diagnostics revealed:
1. **Store Error Disparity (5x Spread):**
   - **Most Reliable Store:** Store 37 achieved a **2.06% WAPE** (\$10,750 MAE on \$522,763 mean weekly sales).
   - **Most Volatile Store:** Store 14 exhibited a **10.59% WAPE** (\$197,357 MAE on \$1,863,227 mean weekly sales).
   - Median store WAPE across the portfolio is **3.46%**.
2. **Holiday Error Shift:**
   - Non-holiday test weeks average **3.79% WAPE** (\$39,076 MAE).
   - Holiday test weeks average **4.58% WAPE** (\$55,830 MAE).
   - This represents a **+20.8% relative error increase**, confirming that promotional volatility elevates forecast risk.
3. **Head-to-Head Baseline Dominance:**
   - XGBoost achieved lower WAPE than the Previous-Week Naive baseline in **35 out of 45 stores (77.8% win rate)**.
   - Across the 10 stores where Naive was competitive, all were small, flat-demand Type C or suburban Type B stores with minimal week-to-week sales variance.
4. **Rolling Backtest Consistency:**
   - Across all 4 expanding-window folds (simulating periodic model retraining), XGBoost won **4 out of 4 folds (100% consistency)**.
   - Average error reduction over naive persistence was **24.36%** across all historical backtest folds.

---

## 6. Methods and Libraries Used
- **Pandas 2.3+ & NumPy 2.5+:** Store-level error rollups, rolling temporal windowing, and parity calculations.
- **Plotly Express 6.9+:** Spatial ranking bars, scatter parity plots, and backtest comparison charts.
- **Streamlit:** 4-tab interactive diagnostic interface.
- **Pytest 9.1+:** 6 automated tests validating store error bounds, holiday degradation, win rate thresholds, and backtest consistency.

---

## 7. Why Each Method Was Selected
- **Store-by-Store WAPE:** Global dollar MAE unfairly penalizes large stores (e.g. Store 20 at \$2M/wk) over small stores (e.g. Store 33 at \$250k/wk). WAPE normalizes error relative to store volume, enabling fair cross-store comparison.
- **Scatter Parity Plot:** Plotting XGBoost WAPE vs Naive WAPE with a $y=x$ reference line provides immediate visual proof of outperformance (points below the line indicate model superiority).
- **Expanding-Window Backtest:** Expanding windows preserve the chronological sequence and mimic real-world retraining where historical training data grows over time.

---

## 8. Work Completed
1. **Error Analysis Engine (`src/retailiq/reliability/error_analysis.py`):**
   - Implemented `analyze_store_errors()`: computes store MAE, RMSE, WAPE, max error, and reliability rank.
   - Implemented `analyze_weekly_errors()`: computes weekly error trajectory.
   - Implemented `analyze_holiday_error_impact()`: evaluates error degradation during holiday weeks.
   - Implemented `compute_relative_baseline_errors()`: computes store-level win rates.
2. **Rolling Backtest Engine (`src/retailiq/reliability/backtest.py`):**
   - Implemented `run_rolling_backtest()`: 4-fold expanding window backtest over historical subsets.
3. **Execution Script (`scripts/05_forecast_reliability/run_reliability.py`):**
   - CLI script generating 5 CSV reports and 1 summary JSON in 2.54 seconds.
4. **Interactive Reliability Dashboard (`src/retailiq/dashboard/pages/p05_reliability.py`):**
   - 4-tab Streamlit interface covering Store Error Disparity, Weekly & Holiday Error, Baseline Head-to-Head, and Rolling Backtest Stability.
5. **Automated Test Suite (`tests/test_forecast_reliability.py`):**
   - 6 tests asserting spatial error bounds, holiday error shift, store win rate >= 75%, and 100% backtest fold victory.

---

## 9. Validation and Tests
Executed via Pytest:
- `TestStoreErrorAnalysis`: Verified 45 stores, ranked 1 to 45, best store WAPE < 3%, worst > 8%.
- `TestWeeklyErrorAnalysis`: Verified 22 test weeks, all MAE and WAPE positive.
- `TestHolidayErrorDegradation`: Verified holiday WAPE (4.58%) > non-holiday WAPE (3.79%).
- `TestBaselineHeadToHead`: Verified XGBoost win rate is 77.8% (>= 75% threshold).
- `TestRollingBacktestReports`: Verified 4 folds, 100% won by XGBoost, mean reduction > 15%.
- **Overall Result:** 76 tests passed across project in 6.47s.

---

## 10. Actual Results
Generated pre-computed reports in `reports/forecast_reliability/`:
- `01_store_error_summary.csv`: Complete error profile for all 45 stores.
- `02_weekly_error_summary.csv`: Week-by-week actuals, predictions, and WAPE across 22 test weeks.
- `03_holiday_error_summary.csv`: Holiday vs non-holiday error comparison.
- `04_store_relative_errors.csv`: Store-by-store win rate and error reduction audit.
- `05_rolling_backtest_results.csv`: 4-fold expanding backtest metrics.
- `06_forecast_reliability_summary.json`: High-level reliability summary.

---

## 11. Interpretation of Results
- The champion model exhibits **exceptionally high reliability across the vast majority of the network** (median store WAPE: 3.46%).
- Forecast degradation during holiday weeks (+20.8% error increase) is expected in retail due to promotional volatility, consumer buying surges, and inventory constraints.
- Rolling backtest consistency (100% of folds won) demonstrates that outperformance over the persistence baseline is an enduring characteristic of the model rather than a fluke of split selection.

---

## 12. Limitations
- Store 14 exhibits elevated error (10.59% WAPE) due to localized historical sales swings not fully captured by macro features (`CPI`, `Fuel_Price`).
- The 4-fold backtest uses 8-week evaluation windows; multi-step recursive forecasts beyond 1-week-ahead would compound errors further.

---

## 13. Problems Encountered and Resolutions
- **Issue:** Evaluating error strictly via dollar MAE suggested large supercenters were performing poorly compared to small stores.
- **Resolution:** Introduced volume-weighted percentage error (WAPE) alongside dollar MAE, showing that large stores actually maintain lower relative error (higher predictability) despite larger absolute revenue volume.

---

## 14. Outputs Produced
| Output Artifact | Location | Type |
|---|---|---|
| Store Error Summary Report | `reports/forecast_reliability/01_store_error_summary.csv` | CSV Report |
| Weekly Error Summary Report | `reports/forecast_reliability/02_weekly_error_summary.csv` | CSV Report |
| Holiday Error Summary Report | `reports/forecast_reliability/03_holiday_error_summary.csv` | CSV Report |
| Store Relative Errors Report | `reports/forecast_reliability/04_store_relative_errors.csv` | CSV Report |
| Rolling Backtest Results | `reports/forecast_reliability/05_rolling_backtest_results.csv` | CSV Report |
| Reliability Summary JSON | `reports/forecast_reliability/06_forecast_reliability_summary.json` | JSON Metric Report |
| Error Analysis Module | `src/retailiq/reliability/error_analysis.py` | Python Module |
| Rolling Backtest Module | `src/retailiq/reliability/backtest.py` | Python Module |
| Reliability Script | `scripts/05_forecast_reliability/run_reliability.py` | Python Script |
| Forecast Reliability Page | `src/retailiq/dashboard/pages/p05_reliability.py` | Streamlit Page |
| Reliability Tests | `tests/test_forecast_reliability.py` | Pytest Suite |
| Phase 05 Documentation | `docs/phases/05_forecast_reliability.md` | Research Document |

---

## 15. How This Connects to the Next Phase
With forecasting residuals and reliability profiles established, the project moves to **Phase 06 (Anomaly Detection & Investigation Queue)**:
- Statistical residual Z-score detector ($|Z| > 3.0$).
- Unsupervised Isolation Forest detector on multidimensional sales and residual features.
- Detector agreement analysis (unanimous flags vs single-detector flags).
- Construct a prioritized, ranked investigation queue for business review without making unverified fraud claims.

---

## 16. Git Branch and Commit Reference
- **Branch:** `phase-05-forecast-reliability`
- **Commit:** `feat(phase-05): implement forecast reliability analysis, backtesting, and dashboard`
