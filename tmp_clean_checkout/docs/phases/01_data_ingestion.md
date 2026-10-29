# Phase 01 ??? Data Ingestion & Quality Validation

## 1. Phase Title
**Phase 01: Automated Data Ingestion, Schema Contract Enforcement, and Analytical Dataset Assembly**

---

## 2. Problem Addressed
Raw enterprise and retail datasets rarely exist in clean, ready-to-model states. In the Walmart forecasting dataset:
- Information is segregated across three disparate tables (`train.csv`, `features.csv`, `stores.csv`) with varying temporal and spatial granularities.
- `features.csv` contains observations spanning beyond the labeled training horizon into 2013.
- Promotional `MarkDown1`???`MarkDown5` values are missing prior to November 2011.
- Retail sales contain zero and negative figures representing customer merchandise returns and accounting reconciliations.

Without an automated, validated data pipeline, downstream tasks suffer from silent row loss, inconsistent joins, unhandled missing values, or accidental data leakage.

---

## 3. Why the Phase Was Needed
- **Reproducibility:** Establish an automated ingestion pipeline (`scripts/01_data_ingestion/run_ingestion.py`) that reliably builds the analytical dataset from source files in under 1 second.
- **Join Integrity:** Ensure joins preserve 100% of labeled training observations without accidental drops or Cartesian explosion.
- **Contract Enforcement:** Explicitly type each column and validate date sequences against formal schema definitions (`EXPECTED_SCHEMAS`).
- **Data Lineage:** Track exact record transformations and generate an immutable JSON summary report (`reports/data_quality/01_ingestion_summary.json`).

---

## 4. Input Data
Three public Walmart sales forecasting tables placed in `data/raw/`:

| Source File | Byte Size | Row Count | Primary Key / Index | Description |
|---|---|---|---|---|
| `train.csv` | 12,842,546 bytes | 421,570 | `(Store, Dept, Date)` | Historical department-week sales observations |
| `features.csv` | 592,289 bytes | 8,190 | `(Store, Date)` | External macroeconomic and environmental factors |
| `stores.csv` | 532 bytes | 45 | `Store` | Store physical classification and square footage metadata |

---

## 5. Data Understanding
Detailed empirical examination of the raw source records revealed:
- **Temporal Horizon:** `train.csv` spans 143 distinct weekly dates from `2010-02-05` to `2012-10-26`. `features.csv` extends to `2013-07-26` (182 weeks), accommodating the test period.
- **Spatial Coverage:** 45 distinct stores across three store types:
  - **Type A:** 22 stores (large supercenters, mean size ~177,000 sq ft)
  - **Type B:** 17 stores (medium format, mean size ~101,000 sq ft)
  - **Type C:** 6 stores (compact format, mean size ~40,000 sq ft)
- **Department Breadth:** 81 unique department identifiers (ranging from 1 to 99 with non-consecutive numbering).
- **Holiday Agreement:** `IsHoliday` is present in both `train.csv` and `features.csv`. Comparing all 421,570 records revealed a **100% match with 0 discrepancies**.
- **Negative & Zero Sales:** 1,285 observations (0.30%) contain negative sales values (minimum: -\$4,988.94), and 73 observations (0.02%) contain zero sales. These represent real-world customer returns and adjustments and are **strictly preserved**.

---

## 6. Methods and Libraries Used
- **Pandas 2.3+:** Type-safe CSV parsing, vectorized joins, and aggregation profiling.
- **PyArrow 26.0+:** High-performance columnar serialization for Parquet output.
- **Python Standard Library (`json`, `pathlib`):** Deterministic report generation and cross-platform path management.
- **Pytest 9.1+:** Automated regression and integrity testing.

---

## 7. Why Each Method Was Selected
- **Parquet over CSV:** Columnar storage reduces disk footprint from ~14 MB to ~3.4 MB (76% compression) while providing 10x faster read speeds and embedded schema preservation.
- **Explicit Left Joins with Row Validation:** Guaranteeing that the left table (`train.csv`) row count equals the merged analytical table row count prevents silent data deletion.
- **Preservation over Deletion:** Silently filtering negative sales or imputing missing MarkDowns at the ingestion stage would corrupt raw historical records. True data engineering preserves raw fidelity and applies deliberate, documented transformations downstream.

---

## 8. Work Completed
1. **Raw Ingestion Module (`src/retailiq/data/loader.py`):**
   - Built type-safe loaders `load_raw_train()`, `load_raw_features()`, `load_raw_stores()`, and `load_analytical_dataset()`.
   - Explicitly configured dtypes and datetime parsing (`Date` as `datetime64[ns]`).
2. **Data Validator Module (`src/retailiq/data/validator.py`):**
   - Implemented `validate_schema()` checking required columns and data types.
   - Implemented `validate_dates()` verifying chronological boundaries and NaT absence.
   - Implemented `validate_join_integrity()` checking key existence and match rates.
   - Implemented `check_duplicates()` confirming key uniqueness.
   - Implemented `check_sales_anomalies()` documenting non-positive sales counts.
   - Implemented `check_holiday_consistency()` verifying identical holiday flags.
3. **Pipeline Orchestrator (`src/retailiq/data/pipeline.py`):**
   - Automated ingestion pipeline that loads raw files, executes validations, performs validated left joins, appends temporal features (`Year`, `Month`, `WeekOfYear`, `DayOfMonth`), exports Parquet, and writes JSON lineage summary.
4. **Ingestion Script (`scripts/01_data_ingestion/run_ingestion.py`):**
   - CLI execution script with UTF-8 stdout configuration and formatted execution summary.
5. **Interactive Dashboard Explorer (`src/retailiq/dashboard/pages/p01_dataset.py`):**
   - Streamlit interface with KPI metric cards, data preview with store/dept filters, raw source metadata cards, distribution charts (store types, sales histogram), and ingestion lineage metrics.
6. **Automated Test Suite (`tests/test_data_ingestion.py`):**
   - 14 targeted unit and integration tests covering all validation and loading functions.

---

## 9. Validation and Tests
Executed via Pytest:
- `TestRawDataLoading`: 3 tests verifying raw shapes (421,570; 8,190; 45), columns, and types.
- `TestDataValidation`: 8 tests verifying schema contracts, failure detection, date ranges, duplicate absence, join match rate (1.0), holiday consistency (100%), and sales anomaly tracking.
- `TestAnalyticalDataset`: 3 tests verifying Parquet presence, 421,570 row preservation, 20 columns, calendar features, zero missing primary keys, and JSON report fidelity.
- **Overall Result:** 49 tests passed (35 foundation + 14 ingestion) in 7.83s.

---

## 10. Actual Results
- **Analytical Dataset Output:** `data/processed/analytical_dataset.parquet` (3,407,384 bytes).
- **Row Preservation Ratio:** Exactly 100.000000% (421,570 input rows ??? 421,570 output rows).
- **Execution Speed:** Complete pipeline executes in **0.87 seconds** on local disk.
- **Columns Produced (20):**
  `Store`, `Dept`, `Date`, `Year`, `Month`, `WeekOfYear`, `DayOfMonth`, `Weekly_Sales`, `IsHoliday`, `Type`, `Size`, `Temperature`, `Fuel_Price`, `MarkDown1`, `MarkDown2`, `MarkDown3`, `MarkDown4`, `MarkDown5`, `CPI`, `Unemployment`.
- **Summary Report:** `reports/data_quality/01_ingestion_summary.json` saved and validated.

---

## 11. Interpretation of Results
- The Walmart dataset demonstrates high structural integrity: zero duplicate records on `(Store, Dept, Date)` and zero unmapped keys between `train` and `features`/`stores`.
- The holiday flags in `train.csv` and `features.csv` are 100% congruent, eliminating ambiguity regarding promotional timing.
- Missing values in `CPI` and `Unemployment` within `features.csv` occur exclusively in the post-train period (2013). Within the training window (`2010-02-05` to `2012-10-26`), economic indicators are 100% complete.
- MarkDown missingness before Nov 2011 is structural (reflecting historical business data collection onset) rather than corrupted transmission.

---

## 12. Limitations
- **External Test Set:** `test.csv` (competition evaluation set) lacks actual sales labels; evaluation must rely strictly on chronologically held-out historical training data.
- **MarkDown Sparsity:** MarkDown features are unpopulated for the first ~60% of the dataset, requiring careful missingness strategy during feature engineering.
- **Granularity Differences:** Economic features (`CPI`, `Fuel_Price`, `Unemployment`) are store-week level, while sales targets are department-week level.

---

## 13. Problems Encountered and Resolutions
- **Issue:** Windows console `cp1252` encoding threw a `charmap` error when `run_ingestion.py` printed Unicode line-drawing characters (`???`, `???`).
- **Resolution:** Configured `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the start of the script and replaced decorative characters with standard ASCII markers (`-`, `*`).

---

## 14. Outputs Produced
| Output Artifact | Location | Type |
|---|---|---|
| Analytical Parquet Dataset | `data/processed/analytical_dataset.parquet` | Processed Parquet (3.4 MB) |
| Ingestion Summary Report | `reports/data_quality/01_ingestion_summary.json` | JSON Lineage Metric Report |
| Data Loader Module | `src/retailiq/data/loader.py` | Python Module |
| Data Validator Module | `src/retailiq/data/validator.py` | Python Module |
| Pipeline Orchestrator | `src/retailiq/data/pipeline.py` | Python Module |
| Ingestion Execution Script | `scripts/01_data_ingestion/run_ingestion.py` | Python Script |
| Dataset Explorer UI Page | `src/retailiq/dashboard/pages/p01_dataset.py` | Streamlit Dashboard Page |
| Ingestion Tests | `tests/test_data_ingestion.py` | Pytest Suite |
| Phase 01 Documentation | `docs/phases/01_data_ingestion.md` | Research Document |

---

## 15. How This Connects to the Next Phase
Phase 01 provided the single source of truth: `analytical_dataset.parquet`.
**Phase 02 (Data Quality & Readiness)** will:
1. Conduct deep statistical profiling across all 20 features (distributions, skewness, outliers).
2. Quantify missing value patterns (especially MarkDown structural absence vs random missingness).
3. Investigate the 1,285 negative sales observations by store, department, and time.
4. Generate comprehensive data quality reports (`reports/data_quality/`) and build the interactive Data Quality dashboard page.

---

## 16. Git Branch and Commit Reference
- **Branch:** `phase-01-data-ingestion`
- **Commit:** `feat(phase-01): implement automated data ingestion, validation, and analytical dataset`

