# Phase 06 — Anomaly Detection & Investigation Queue

## 1. Phase Title
**Phase 06: Dual-Detector Anomaly Synthesis, Agreement Matrix Cross-Tabulation, and Operational Investigation Queue**

---

## 2. Problem Addressed
In retail networks spanning dozens of locations and thousands of weekly transactions, unusual operational deviations inevitably occur:
- Sudden surges caused by local unmodeled events or promotions.
- Severe dips driven by severe weather closures, competitor openings, or inventory delivery failures.
- Multidimensional anomalies where sales volume appears plausible in isolation but is abnormal when conditioned on historical store size, seasonal momentum, or economic signals.

Without automated triage, store auditors and financial controllers face an impossible manual audit burden. Furthermore, naive anomaly systems frequently mislabel detector flags as confirmed fraud, creating false alarms.

Phase 06 builds a dual-detector system combining **Univariate Residual Z-scores ($|Z| > 3.0$)** and **Multivariate Isolation Forest (contamination = 2%)**, prioritizing discrepancies into a ranked operational investigation queue without asserting unverified claims of fraud or error.

---

## 3. Why the Phase Was Needed
- **Complementary Detection Strengths:** Statistical residual Z-scores isolate severe univariate forecast misses, while Isolation Forest isolates abnormal conjunctions across actual volume, predictions, residuals, and percentage errors.
- **Agreement Cross-Tabulation:** Identify where detectors agree unanimously versus where they diverge, providing operational clarity on confidence levels.
- **Queue Prioritization:** Rank anomalies by a composite Priority Score [0–100] rather than presenting an undifferentiated list of flags.
- **Evidence-Grounded Review:** Formulate human-readable review hypotheses (e.g. magnitude of discrepancy, holiday context) to guide store investigation.

---

## 4. Input Data
- `reports/sales_forecasting/02_test_predictions.csv` (990 held-out store-week observations evaluated on the champion XGBoost model).

---

## 5. Data Understanding
Empirical analysis across the 990 test records revealed:
1. **Queue Size:** Exactly **23 records (2.32%)** were prioritized for operational review by either detector.
2. **Detector Synthesis Breakdown:**
   - **Unanimous Flags (Both Detectors):** 14 records (highest review priority).
   - **Statistical Only (|Z| > 3.0):** 3 records (univariate residual outliers).
   - **Isolation Forest Only:** 6 records (multivariate conjunction outliers).
   - **Normal Operations (Neither Flagged):** 967 records (97.68%).
3. **Detector Agreement Rate:** **99.09% overall agreement** across the 990 test observations.
4. **Top Priority Investigation Record:**
   - **Store 14 on 2012-08-24:**
     - Actual Sales: \$2,014,146.06 vs Forecast: \$1,593,296.63.
     - Residual: +\$420,849.43 (+20.9% miss, $Z = +4.95$).
     - Composite Priority Score: **100.0 / 100**.
     - Review Hypothesis: Massive late-summer demand surge at Store 14 exceeding historical momentum; review local store marketing, inventory clearance, or competitor stockout.

---

## 6. Methods and Libraries Used
- **NumPy 2.5+:** Standard score calculation ($Z = (e - \mu) / \sigma$) and weighted composite priority scoring.
- **Scikit-Learn 1.9+:** `IsolationForest(n_estimators=100, contamination=0.02, random_state=42)` and `StandardScaler`.
- **Pandas 2.3+:** Cross-tabulation agreement analysis and queue sorting.
- **Streamlit & Plotly:** 4-tab interactive investigation UI with Z-score cutoffs and 2D partition space visualization.
- **Pytest 9.1+:** 4 dedicated unit and integration tests asserting statistical bounds, contamination compliance, and queue ranking.

---

## 7. Why Each Method Was Selected
- **Statistical Cutoff (|Z| > 3.0):** Under a normal residual distribution, the probability of $|Z| > 3.0$ is less than 0.27%. Observations exceeding this threshold are indisputably extreme deviations from expected demand.
- **Isolation Forest for Feature Conjunctions:** Linear statistical thresholds evaluate only the 1D residual. Isolation Forest partitions multidimensional space, isolating cases where sales volume and percentage error form an abnormal cluster.
- **Composite Priority Score:** Weighting normalized $|Z|$ (50%) and normalized tree isolation depth (50%) ensures that unanimous detections naturally rise to the top of the queue.

---

## 8. Work Completed
1. **Statistical Detector Module (`src/retailiq/anomalies/statistical.py`):**
   - Implemented `detect_statistical_anomalies()` computing residual mean, standard deviation, Z-scores, and directional labels (`Extreme Under-prediction` vs `Extreme Over-prediction`).
2. **Isolation Forest Module (`src/retailiq/anomalies/isolation_forest.py`):**
   - Implemented `detect_isolation_forest_anomalies()` training an Isolation Forest on scaled feature space and converting decision functions to a normalized $[0, 1]$ anomaly score.
3. **Investigation Queue Orchestrator (`src/retailiq/anomalies/investigation.py`):**
   - Implemented `build_investigation_queue()`: merges detectors, creates cross-tabulation agreement matrix, computes priority scores, and generates automated investigation hypotheses.
4. **Execution Script (`scripts/06_anomaly_detection/run_anomaly_detection.py`):**
   - Executable CLI pipeline running dual detection and queue export in 0.77 seconds.
5. **Interactive Anomaly Dashboard (`src/retailiq/dashboard/pages/p06_anomalies.py`):**
   - 4-tab UI: Prioritized Investigation Queue, Dual-Detector Agreement, Statistical Z-Scores, and Isolation Forest Space.
6. **Automated Test Suite (`tests/test_anomaly_detection.py`):**
   - 4 tests verifying Z-score alignment, contamination boundaries, queue ranking, and report presence.

---

## 9. Validation and Tests
Executed via Pytest:
- `TestStatisticalDetector`: Verified Z-score calculation, $|Z| > 3.0$ threshold alignment, anomaly rate < 3%.
- `TestIsolationForestDetector`: Verified normalized scores in $[0, 1]$, flagged count between 15 and 25 (~2% contamination).
- `TestInvestigationQueue`: Verified queue non-empty, sorted strictly descending by Priority Score, detector agreement rate > 95%, unanimous flags present.
- `TestAnomalyReportsExist`: Verified all 5 CSV reports and 1 JSON summary exist on disk.
- **Overall Result:** 80 tests passed across project in 18.63s.

---

## 10. Actual Results
Generated pre-computed reports in `reports/anomaly_detection/`:
- `01_statistical_scores.csv`: All 990 test records with Z-scores and 3-sigma flags.
- `02_isolation_forest_scores.csv`: All 990 test records with Isolation Forest raw and normalized anomaly scores.
- `03_detector_comparison.csv`: Merged dataset with both detector flags and agreement classifications.
- `04_agreement_summary.csv`: Cross-tabulation counts and percentages.
- `05_investigation_queue.csv`: Prioritized list of 23 flagged records with evidence notes.
- `06_anomaly_detection_summary.json`: High-level anomaly synthesis metrics.

---

## 11. Interpretation of Results
- The dual-detector approach provides **practical operational triage**: instead of forcing managers to inspect 990 records, it concentrates audit resources on the **14 unanimous anomalies** and **9 single-detector discrepancies**.
- Statistical and tree isolation detectors agreed on **99.09% of all test records**, proving high baseline alignment.
- The 6 "Isolation Forest Only" flags represent subtle multivariate shifts (e.g. unexpected volume drop in an otherwise growing store) that did not trigger a hard 3-sigma threshold but represented an isolated structural state.

---

## 12. Limitations
- **Lack of Ground-Truth Anomaly Labels:** In the public Walmart dataset, true operational anomalies (e.g. system outages, stockouts, shrinkage) are unrecorded. Consequently, detection **precision, recall, and ROC-AUC cannot be measured**.
- The 23 flagged records must be treated as **audit targets requiring reconciliation**, not verified defects.

---

## 13. Problems Encountered and Resolutions
- **Issue:** Isolation Forest outputs raw decision scores where negative values indicate anomalies, whereas Z-scores increase in absolute value with anomaly severity.
- **Resolution:** Normalized Isolation Forest decision functions into a $[0, 1]$ scale where $1.0$ represents maximum outlier depth, then formulated a balanced composite Priority Score combining normalized $|Z|$ and normalized tree isolation.

---

## 14. Outputs Produced
| Output Artifact | Location | Type |
|---|---|---|
| Statistical Scores Report | `reports/anomaly_detection/01_statistical_scores.csv` | CSV Report |
| Isolation Forest Scores Report | `reports/anomaly_detection/02_isolation_forest_scores.csv` | CSV Report |
| Detector Comparison Report | `reports/anomaly_detection/03_detector_comparison.csv` | CSV Report |
| Agreement Summary Report | `reports/anomaly_detection/04_agreement_summary.csv` | CSV Report |
| Investigation Queue Report | `reports/anomaly_detection/05_investigation_queue.csv` | CSV Report |
| Anomaly Summary JSON | `reports/anomaly_detection/06_anomaly_detection_summary.json` | JSON Metric Report |
| Statistical Detector Module | `src/retailiq/anomalies/statistical.py` | Python Module |
| Isolation Forest Module | `src/retailiq/anomalies/isolation_forest.py` | Python Module |
| Investigation Queue Module | `src/retailiq/anomalies/investigation.py` | Python Module |
| Anomaly Execution Script | `scripts/06_anomaly_detection/run_anomaly_detection.py` | Python Script |
| Anomaly Investigation Page | `src/retailiq/dashboard/pages/p06_anomalies.py` | Streamlit Page |
| Anomaly Tests | `tests/test_anomaly_detection.py` | Pytest Suite |
| Phase 06 Documentation | `docs/phases/06_anomaly_detection.md` | Research Document |

---

## 15. How This Connects to the Next Phase
With forecasting models, reliability, and anomalies fully established, the project moves to **Phase 07 (Explainable AI via SHAP)**:
- Implement TreeSHAP on the champion XGBoost model.
- Compute global feature importance and summary distributions.
- Compute local prediction waterfall explanations for individual store-week predictions.
- Perform strict prediction reconstruction checks ($f(x) = \phi_0 + \sum \phi_i$) within $\pm 0.01$ numerical tolerance.
- Build interactive SHAP waterfall and force-plot visualizations.

---

## 16. Git Branch and Commit Reference
- **Branch:** `phase-06-anomaly-detection`
- **Commit:** `feat(phase-06): implement dual-detector anomaly detection, agreement analysis, and investigation queue`
