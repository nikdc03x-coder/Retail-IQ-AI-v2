"""
RetailIQ AI V2 — Anomaly Investigation Queue & Agreement Analysis Module

Combines Statistical Residual Z-scores and Multivariate Isolation Forest detectors:
1. Computes cross-detector agreement matrix (Unanimous, Statistical Only, IForest Only).
2. Generates a prioritized, ranked investigation queue with evidence summaries.
3. Formulates operational review hypotheses without asserting unverified fraud or corruption.
"""

from typing import Any
import numpy as np
import pandas as pd


def synthesize_investigation_notes(row: pd.Series) -> str:
    """Generate business-oriented hypothesis explaining why an observation was flagged."""
    agreement = row["Agreement_Classification"]
    z = row["Z_Score"]
    diff = row["Actual_Sales"] - row["XGBoost_Pred"]
    pct = row["XGBoost_Pct_Error"]
    is_hol = row["IsHoliday"]

    hol_context = " (Holiday Week)" if is_hol else ""

    if agreement == "Unanimous Anomaly (Both Detectors)":
        if z > 0:
            return (
                f"High-priority surge{hol_context}: Actual sales exceeded forecast by ${diff:,.0f} "
                f"(+{pct:.1f}%, Z={z:+.1f}). Investigate local demand catalyst, competitor stockout, or unmodeled promotion."
            )
        else:
            return (
                f"High-priority drop{hol_context}: Actual sales fell short of forecast by ${abs(diff):,.0f} "
                f"(-{pct:.1f}%, Z={z:+.1f}). Investigate severe weather disruption, inventory delivery failure, or store event."
            )

    elif agreement == "Statistical Only (|Z| > 3.0)":
        return (
            f"Univariate residual outlier{hol_context}: Residual magnitude |Z|={abs(z):.1f} exceeds standard 3.0 threshold "
            f"(${abs(diff):,.0f} discrepancy). Review store operational log."
        )

    elif agreement == "Isolation Forest Only":
        return (
            f"Multidimensional feature outlier{hol_context}: Abnormal conjunction of sales volume and percentage variance "
            f"detected by tree isolation. Modest Z-score ({z:+.1f}) but unusual structural profile."
        )

    return "Normal operation."


def build_investigation_queue(
    stat_df: pd.DataFrame,
    iforest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Synthesize dual-detector findings into a unified comparison and ranked queue.

    Args:
        stat_df: Output from detect_statistical_anomalies.
        iforest_df: Output from detect_isolation_forest_anomalies.

    Returns:
        Tuple of (merged_comparison_df, agreement_matrix_df, investigation_queue_df, summary_dict).
    """
    merged = pd.merge(
        stat_df,
        iforest_df[["Store", "Date", "IForest_Raw_Score", "IForest_Anomaly_Score", "Is_IForest_Anomaly"]],
        on=["Store", "Date"],
    )

    # Classification logic
    stat_flag = merged["Is_Statistical_Anomaly"]
    ifor_flag = merged["Is_IForest_Anomaly"]

    conditions = [
        (stat_flag & ifor_flag),
        (stat_flag & ~ifor_flag),
        (~stat_flag & ifor_flag),
        (~stat_flag & ~ifor_flag),
    ]
    choices = [
        "Unanimous Anomaly (Both Detectors)",
        "Statistical Only (|Z| > 3.0)",
        "Isolation Forest Only",
        "Normal Operation",
    ]
    merged["Agreement_Classification"] = np.select(conditions, choices, default="Normal Operation")
    merged["Is_Any_Anomaly"] = stat_flag | ifor_flag

    # Agreement cross-tabulation matrix
    ctab = pd.crosstab(
        merged["Is_Statistical_Anomaly"],
        merged["Is_IForest_Anomaly"],
        rownames=["Statistical Anomaly (|Z| > 3)"],
        colnames=["Isolation Forest Anomaly"],
        margins=True,
    ).reset_index()

    # Agreement breakdown table
    agree_summary = (
        merged["Agreement_Classification"]
        .value_counts()
        .reset_index()
    )
    agree_summary.columns = ["Classification", "Record_Count"]
    agree_summary["Percentage"] = (agree_summary["Record_Count"] / len(merged) * 100.0).round(2)

    # ── Ranked Investigation Queue ──
    flagged_df = merged[merged["Is_Any_Anomaly"]].copy()

    # Compute Composite Priority Score [0 - 100]:
    # 50% weight to standardized residual Z-score, 50% to normalized Isolation Forest score
    norm_z = np.clip(np.abs(flagged_df["Z_Score"]) / 5.0, 0.0, 1.0)
    flagged_df["Priority_Score"] = (
        (0.5 * norm_z + 0.5 * flagged_df["IForest_Anomaly_Score"]) * 100.0
    ).round(1)

    flagged_df["Investigation_Evidence"] = flagged_df.apply(synthesize_investigation_notes, axis=1)

    queue_cols = [
        "Store", "Date", "Type", "IsHoliday",
        "Actual_Sales", "XGBoost_Pred", "Residual", "Z_Score", "XGBoost_Pct_Error",
        "Agreement_Classification", "Priority_Score", "Investigation_Evidence",
    ]
    investigation_queue = (
        flagged_df[queue_cols]
        .sort_values(by="Priority_Score", ascending=False)
        .reset_index(drop=True)
    )
    investigation_queue["Queue_Rank"] = range(1, len(investigation_queue) + 1)

    reordered = ["Queue_Rank"] + [c for c in queue_cols if c != "Queue_Rank"]
    investigation_queue = investigation_queue[reordered]

    unanimous_count = int((merged["Agreement_Classification"] == "Unanimous Anomaly (Both Detectors)").sum())
    stat_only_count = int((merged["Agreement_Classification"] == "Statistical Only (|Z| > 3.0)").sum())
    iforest_only_count = int((merged["Agreement_Classification"] == "Isolation Forest Only").sum())

    summary = {
        "total_evaluated": len(merged),
        "total_flagged": len(investigation_queue),
        "unanimous_anomalies": unanimous_count,
        "statistical_only_anomalies": stat_only_count,
        "isolation_forest_only_anomalies": iforest_only_count,
        "detector_agreement_rate_pct": round(
            (unanimous_count + (len(merged) - len(investigation_queue))) / len(merged) * 100.0, 2
        ),
        "top_priority_store": int(investigation_queue.iloc[0]["Store"]) if not investigation_queue.empty else None,
        "top_priority_date": str(pd.to_datetime(investigation_queue.iloc[0]["Date"]).date()) if not investigation_queue.empty else None,
    }

    return merged, agree_summary, investigation_queue, summary
