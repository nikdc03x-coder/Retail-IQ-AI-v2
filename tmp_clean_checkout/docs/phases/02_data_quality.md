# Phase 02 ??? Data Quality & Readiness Audit

## 1. Phase Title
**Phase 02: Statistical Profiling, Structural Missingness Classification, Returns Accounting, and Observation Continuity Audit**

---

## 2. Problem Addressed
Predictive retail models frequently fail in production due to unexamined data quality issues:
- Assuming missing values are missing completely at random (MCAR) when they are structurally absent (e.g., MarkDown promotions).
- Silently removing negative sales values, which distorts accounting truth and artificially inflates baseline sales estimates.
- Overlooking irregular store-department observations (intermittent time series vs continuous series).
- Ignoring extreme skewness and heavy-tailed kurtosis in retail targets.

Phase 02 provides empirical quality auditing, structural missingness taxonomy, merchandise returns accounting, and observation continuity analysis.

---

## 3. Why the Phase Was Needed
- **Health Verification:** Validate that every column in `analytical_dataset.parquet` conforms to statistical expectations before feature engineering.
- **Missingness Strategy Formulation:** Distinguish between economic indicators (which are 100% complete) and promotional markdowns (which require explicit imputation or absence indicators).
- **Return Accounting:** Isolate the 1,285 negative weekly sales observations (-\$88,161.56 cumulative) to analyze which departments and store types account for merchandise returns.
- **Continuity Categorization:** Determine what proportion of store-department pairs have continuous 143-week coverage versus seasonal or intermittent presence.

---

## 4. Input Data
- **Primary Dataset:** `data/processed/analytical_dataset.parquet` (421,570 observations, 20 columns).
- **Lineage Metadata:** `reports/data_quality/01_ingestion_summary.json`.

---

## 5. Data Understanding
Empirical statistical profiling produced the following verified insights:
1. **Economic Indicators:** Both `CPI` and `Unemployment` contain **0 missing values** across all 421,570 historical observations (mean CPI: 171.20, mean Unemployment: 7.96%).
2. **Promotional MarkDowns:** `MarkDown1` through `MarkDown5` are missing for ~64% of the dataset. Specifically, they were structurally unrecorded by Walmart prior to November 11, 2011. After this date, MarkDowns are consistently recorded.
3. **Merchandise Returns:** 1,285 records (0.30%) contain negative `Weekly_Sales` figures. The cumulative dollar return volume is **-\$88,161.56**. Department 47 experienced the highest return volume (-\$28,241.98 across 58 return weeks).
4. **Target Distribution Shape:** `Weekly_Sales` exhibits strong positive skewness (**3.26**) and heavy kurtosis (**21.49**), driven by high-volume departments (e.g. Dept 92 groceries/dry goods) and massive holiday surges (Thanksgiving, Christmas).
5. **Series Continuity:** Across 45 stores, there are 3,331 active store-department pairs. **2,660 pairs (79.86%)** have full, continuous coverage spanning all 143 historical weeks. The remaining 671 pairs represent seasonal departments (e.g. holiday decor) or newly launched departments.

---

## 6. Methods and Libraries Used
- **Pandas 2.3+ & NumPy 2.5+:** Vectorized statistical aggregation, quantile calculation, skewness/kurtosis profiling, and group-by aggregations.
- **JSON & Pathlib:** Structured reporting and artifact export.
- **Streamlit & Plotly:** Interactive data quality dashboard page with tabs, metric cards, and horizontal return charts.
- **Pytest 9.1+:** Automated regression tests verifying statistical boundaries and reporting contracts.

---

## 7. Why Each Method Was Selected
- **Summary Reports (CSV/JSON) Pattern:** Decoupling quality calculation from dashboard rendering ensures that UI pages load instantaneously without scanning 420K rows on every user interaction.
- **Group-By Return Profiling:** Grouping returns by department reveals concentration (top 5 departments account for >60% of return dollars), proving returns are departmental phenomena rather than random data entry errors.
- **IQR & Non-Parametric Dispersion:** In addition to standard deviation, reporting Q25, Median, Q75, and IQR provides robust characterization of heavy-tailed retail data.

---

## 8. Work Completed
1. **Quality Reporter Engine (`src/retailiq/quality/reporter.py`):**
   - Implemented `generate_column_statistics()`: Computes count, missing %, mean, std, min, Q25, median, Q75, max, IQR, skewness, and kurtosis across 11 continuous features.
   - Implemented `generate_missingness_profile()`: Classifies columns as Complete, Structural (pre-Nov 2011), or Random.
   - Implemented `generate_negative_sales_audit()`: Department and store level aggregation of negative sales.
   - Implemented `generate_store_department_coverage()`: Audits weekly continuity out of 143 weeks.
   - Implemented `run_quality_audit()`: Orchestrates profiling and exports reports.
2. **Quality Script (`scripts/02_data_quality/run_quality_checks.py`):**
   - Standalone CLI execution script that generates 4 CSV reports and 1 JSON summary in 0.68 seconds.
3. **Interactive Quality Dashboard (`src/retailiq/dashboard/pages/p02_data_quality.py`):**
   - 4-tab Streamlit interface covering Missingness Profiling, Feature Distributions, Merchandise Returns Audit, and Store-Dept Continuity.
4. **Automated Test Suite (`tests/test_data_quality.py`):**
   - 8 unit and integration tests asserting statistical bounds, missingness classifications, and return totals.

---

## 9. Validation and Tests
Executed via Pytest:
- `TestStatisticalProfiling`: Verified 11 numerical features profiled, `Weekly_Sales` skewness > 2.0, min sales < 0.
- `TestMissingnessClassification`: Verified CPI/Unemployment 0 missing, MarkDown1 > 50% missing with structural classification and earliest date `2011-11-11`.
- `TestMerchandiseReturnsAudit`: Verified 1,285 return rows, total return value < -\$80,000, 50 affected departments.
- `TestStoreDeptContinuity`: Verified 3,331 pairs total, exactly 2,660 full-coverage pairs, max active weeks 143.
- `TestQualityAuditExecution`: Verified all 5 report artifacts exist and JSON summary fields match expected schema.
- **Overall Result:** 57 tests passed (49 earlier + 8 new) in 5.45s.

---

## 10. Actual Results
Generated report artifacts in `reports/data_quality/`:
- `02_column_statistics.csv`: Complete distribution metrics for all 11 continuous features.
- `03_missingness_profile.csv`: Full classification of missingness for all 20 columns.
- `04_negative_sales_audit.csv`: Return frequency and cumulative amounts for 50 departments.
- `05_store_department_coverage.csv`: Continuity audit for all 3,331 store-department combinations.
- `06_data_quality_summary.json`: High-level quality metrics summary.

---

## 11. Interpretation of Results
- The analytical dataset is in **outstanding structural health** for forecasting.
- The lack of missing values in economic indicators (`CPI`, `Fuel_Price`, `Unemployment`) means these features can be fed directly to machine learning models without imputation during the historical period.
- For `MarkDown1`???`MarkDown5`, two modeling strategies are viable:
  1. Fill missing values with `0.0` (indicating no promotional markdown was active/recorded).
  2. Add binary indicator features `has_markdown` to signal the presence of promotional data.
- Merchandise returns are heavily concentrated in specific departments (e.g. Dept 47 Jewelry/Accessories, Dept 54).

---

## 12. Limitations
- Store-department continuity varies: models predicting individual department-weeks must account for the 671 partial-coverage series.
- MarkDown data begins in week 92 (Nov 2011), meaning models trained strictly on pre-2011 data cannot learn MarkDown response coefficients without specialized handling.

---

## 13. Problems Encountered and Resolutions
- **Issue:** Several departments (e.g. Dept 47, 54) had extreme negative sales weeks that skew simple department averages.
- **Resolution:** Generated dedicated returns audit tables that separate return transactions from positive sales, allowing business analysts to inspect returns in isolation without dropping records from the main analytical table.

---

## 14. Outputs Produced
| Output Artifact | Location | Type |
|---|---|---|
| Column Statistics Report | `reports/data_quality/02_column_statistics.csv` | CSV Report |
| Missingness Profile Report | `reports/data_quality/03_missingness_profile.csv` | CSV Report |
| Negative Sales Audit Report | `reports/data_quality/04_negative_sales_audit.csv` | CSV Report |
| Store-Dept Coverage Report | `reports/data_quality/05_store_department_coverage.csv` | CSV Report |
| Data Quality Summary JSON | `reports/data_quality/06_data_quality_summary.json` | JSON Metric Report |
| Quality Reporter Module | `src/retailiq/quality/reporter.py` | Python Module |
| Quality Audit Script | `scripts/02_data_quality/run_quality_checks.py` | Python Script |
| Data Quality Dashboard Page | `src/retailiq/dashboard/pages/p02_data_quality.py` | Streamlit Page |
| Quality Tests | `tests/test_data_quality.py` | Pytest Suite |
| Phase 02 Documentation | `docs/phases/02_data_quality.md` | Research Document |

---

## 15. How This Connects to the Next Phase
With data quality verified and documented, the project moves to **Phase 03 (Business Analytics & Exploratory Data Analysis)**:
- Pre-compute store sales rankings, department contributions, and store-type benchmarks.
- Analyze holiday surges (Super Bowl, Labor Day, Thanksgiving, Christmas) versus regular weeks.
- Aggregate sales at proper hierarchical levels (department-week vs store-week vs total network).
- Generate interactive Plotly business intelligence dashboards.

---

## 16. Git Branch and Commit Reference
- **Branch:** `phase-02-data-quality`
- **Commit:** `feat(phase-02): implement data quality reporter, profiling reports, and dashboard`
