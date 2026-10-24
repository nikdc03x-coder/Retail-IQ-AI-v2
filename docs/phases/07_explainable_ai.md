# Phase 07 — Explainable AI (SHAP Analysis)

## 1. Phase Title
**Phase 07: TreeSHAP Interpretability, Global Feature Attribution Hierarchy, Local Prediction Additive Decomposition, and Efficiency Axiom Audit**

---

## 2. Problem Addressed
In enterprise retail merchandising and operational forecasting, sophisticated gradient-boosted tree models (such as XGBoost) frequently function as impenetrable black boxes to executive leaders, district managers, and store directors:
- **Opacity and Skepticism:** When a model forecasts an unexpected surge or severe slump in weekly store revenue, stakeholders cannot determine whether the forecast is grounded in historical seasonality, macroeconomic inflation, or artifact noise.
- **Unjustified Decisions:** Without attribution, inventory allocations, staffing schedules, and supply chain logistics are committed blindly, creating resistance to automated decision systems.
- **Conflation of Influence vs. Causation:** Naive interpretations mistake statistical predictive influence for direct operational levers (e.g. assuming that historical lag sales physically generate future revenue rather than capturing recurring annual demand patterns).
- **Audit Deficits:** Regulated financial reporting and enterprise governance require verifiable mathematical guarantees that individual prediction explanations strictly reconstruct model outputs without missing residual factors.

Phase 07 implements **TreeSHAP (Tree Shapley Additive Explanations)** on the champion XGBoost regressor, providing mathematically grounded global feature attribution and local waterfall decompositions that satisfy the classical cooperative game theory efficiency axiom:
$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i$$

---

## 3. Why the Phase Was Needed
- **Mathematically Grounded Attribution:** Shapley values are uniquely guaranteed to satisfy four fundamental axioms: *Efficiency*, *Symmetry*, *Dummy Player (Null Effect)*, and *Additivity*.
- **Executive Transparency:** Explaining forecasts in dollar units (e.g., "$+\$228,670$ driven by prior-year holiday baseline") translates complex machine learning trees into standard retail accounting language.
- **Root-Cause Triage:** Enables merchandising and audit teams to inspect anomalous or low-confidence store-week predictions to pinpoint the specific features responsible for model behavior.
- **Governance & Verification:** Validates that feature contributions sum to the model prediction within floating-point tolerance, preventing phantom or unverified claims.

---

## 4. Input Data
- **Model:** Champion XGBoost Regressor (`models/xgboost_model.joblib`) fitted on the training split ($R^2 = 0.9416$, $\text{RMSE} = \$135,160.84$).
- **Held-Out Test Set:** `reports/sales_forecasting/02_test_predictions.csv` (990 store-week observations spanning 45 stores across the 22-week test horizon from 2012-06-01 to 2012-10-26).
- **Engineered Features:** 18 continuous and categorical features covering autoregressive lags, rolling statistics, calendar seasonality, store physical format, and regional economic indices.

---

## 5. Data Understanding & Key Findings
Empirical SHAP analysis across all 990 held-out test predictions revealed:

1. **Baseline Expected Value ($\phi_0$):**
   - The chain-wide expected weekly sales volume across the training distribution is **\$1,059,041.50**. Every local forecast begins at this baseline.
2. **Global Feature Hierarchy:**
   - **#1 Driver — Annual Seasonality (`lag_52`):** Mean absolute SHAP impact of **\$228,670.69**, accounting for **45.01%** of total predictive variation. Store sales are heavily anchored by identical seasonal weeks from the prior year.
   - **#2 Driver — Short-Term Momentum (`rolling_mean_4`):** Mean absolute SHAP impact of **\$140,471.14**, capturing **27.65%** of predictive power. Reflects the trailing 4-week revenue trend.
   - **Combined Core Drivers:** The top two features alone (`lag_52` + `rolling_mean_4`) explain **72.66%** of model behavior.
   - **Medium-Term Trajectory (`rolling_mean_8`):** Ranks #3 at **\$53,905.49** (10.61%).
   - **Prior-Month Volume (`lag_4`):** Ranks #4 at **\$26,446.84** (5.21%).
   - **Immediate Autoregressive (`lag_1`):** Ranks #5 at **\$11,216.56** (2.21%).
   - **Store Capacity (`Size`):** Ranks #6 at **\$10,480.99** (2.06%).
   - **Macroeconomic & Calendar Features (`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `Month`, `WeekOfYear`):** Each contributes $<1.5\%$ individually, acting as fine-grained calibrators rather than primary revenue drivers.
3. **Efficiency Axiom Verification:**
   - Evaluated across all 990 test records:
     - **Maximum Absolute Discrepancy:** **\$2.0000** (attributable to 32-bit floating-point accumulation across 100 gradient-boosted trees on \$1.5M revenue).
     - **Mean Absolute Discrepancy:** **\$0.2815**.
     - **Maximum Relative Error:** **0.000095%** (< 1 part per million).
     - **Audit Status:** **100% PASSED**.

---

## 6. Methods and Libraries Used
- **SHAP (v0.49+):** `shap.TreeExplainer(model)` utilizing the polynomial-time Lundberg & Lee (2018) tree-traversal algorithm to compute exact Shapley values.
- **NumPy (v2.5+):** Additive vector operations, absolute impact summaries, and efficiency axiom tolerance auditing.
- **Pandas (v2.3+):** Attribution matrix synthesis, feature ranking, and observation formatting.
- **Plotly & Streamlit:** 4-tab interactive explainability suite featuring dynamic `go.Waterfall` charts, horizontal bar hierarchies, and non-linear dependence scatter plots.
- **Pytest (v9.1+):** 10 unit and integration tests verifying rank invariants, relative impact mass sums, additive tolerance limits, and report persistence.

---

## 7. Why Each Method Was Selected
| Method / Tool | Primary Rationale | Alternative Considered | Why Rejected |
| :--- | :--- | :--- | :--- |
| **TreeSHAP** | Computes exact Shapley values in $O(TLD^2)$ polynomial time for tree ensembles. | **KernelSHAP** | Exponential time complexity; requires stochastic sampling and creates sampling variance. |
| **TreeSHAP** | Guarantees local accuracy (Efficiency Axiom: contributions sum to prediction). | **Permutation Importance** | Evaluates global marginal loss increase; cannot explain individual store predictions. |
| **Waterfall Decomposition** | Visually bridges baseline average ($\phi_0$) to final forecast in clear additive steps. | **Force Plot** | Dense JS canvas rendering less intuitive for business users than step-by-step waterfalls. |
| **Relative Impact Pct** | Standardizes feature importance to a 100% scale for executive communication. | **Raw Split Count / Gain** | Tree split counts artificially favor high-cardinality continuous variables without dollar units. |

---

## 8. Deliverables & Artifacts Generated
```
reports/explainable_ai/
├── 01_global_feature_importance.csv         # 18 features ranked by Mean |SHAP| ($), %, and definitions
├── 02_local_prediction_attributions.csv     # 990 test records with base value, predictions, diffs, and SHAP columns
├── 03_reconstruction_verification.json      # Mathematical efficiency axiom audit metrics
├── 04_top_drivers_summary.csv               # Frequency distribution of #1 positive and negative drivers
└── 05_explainability_summary.json           # Comprehensive phase execution summary metadata

src/retailiq/explainability/
├── __init__.py                              # Subpackage exports
└── shap_analysis.py                         # TreeSHAP explainer, verification engine, and waterfall generator

scripts/07_explainability/
└── run_shap_analysis.py                     # Headless automation script executing Phase 07

src/retailiq/dashboard/pages/
└── p07_explainability.py                    # 4-tab interactive Streamlit dashboard page

tests/
└── test_explainable_ai.py                   # 10 unit and integration tests (100% passing)
```

---

## 9. Verification & Test Results
- **Unit & Integration Suite (`tests/test_explainable_ai.py`):**
  - `TestGlobalFeatureImportance`: 1 passed.
  - `TestLocalPredictionAttribution`: 1 passed.
  - `TestEfficiencyAxiomReconstruction`: 1 passed.
  - `TestObservationWaterfallExtraction`: 1 passed.
  - `TestExplainabilityReportsExist`: 6 passed.
  - **Result:** **10 passed in 3.74s**.
- **Full Project Regression:** **90 passed in ~22s** across all 8 test modules.

---

## 10. Business Impact & How Stakeholders Use This
1. **District Managers & Store General Managers:**
   - Inspect local waterfall charts prior to peak holiday periods (e.g. Thanksgiving / Christmas) to understand whether forecasted demand is driven by historical momentum or macro trends.
2. **Merchandising & Inventory Planners:**
   - Differentiate baseline capacity requirements (driven by store physical `Size` and `lag_52`) from short-term promotional adjustments (driven by `rolling_mean_4` and `is_holiday_num`).
3. **Enterprise Model Risk Governance & Auditors:**
   - Satisfy transparency and model explainability mandates using verified additive attribution tables where the efficiency error is proven to be under 0.0001%.
