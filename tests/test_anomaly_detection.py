"""
RetailIQ AI V2 — Phase 06: Anomaly Detection & Investigation Queue Tests

Validates:
- Statistical residual standard score calculation and |Z| > 3.0 thresholding
- Multivariate Isolation Forest unsupervised anomaly detection
- Dual-detector agreement matrix cross-tabulation
- Ranked investigation queue priority scoring and evidence notes
- Report persistence and schema invariants
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.retailiq.anomalies.investigation import build_investigation_queue
from src.retailiq.anomalies.isolation_forest import detect_isolation_forest_anomalies
from src.retailiq.anomalies.statistical import detect_statistical_anomalies
from src.retailiq.config import REPORT_SUBDIRS


@pytest.fixture(scope="module")
def test_preds() -> pd.DataFrame:
    """Load test predictions from Phase 04."""
    fpath = REPORT_SUBDIRS["sales_forecasting"] / "02_test_predictions.csv"
    return pd.read_csv(fpath, parse_dates=["Date"])


class TestStatisticalDetector:
    """Verify statistical residual Z-score calculations."""

    def test_z_score_threshold_alignment(self, test_preds: pd.DataFrame):
        stat_df, meta = detect_statistical_anomalies(test_preds, threshold_z=3.0)

        assert "Z_Score" in stat_df.columns
        assert "Is_Statistical_Anomaly" in stat_df.columns
        assert (stat_df["Is_Statistical_Anomaly"] == (stat_df["Z_Score"].abs() > 3.0)).all()

        # Under a 3-sigma cutoff, anomaly rate should be under 3%
        assert meta["anomaly_rate_pct"] < 3.0
        assert meta["anomalies_flagged"] > 0


class TestIsolationForestDetector:
    """Verify Isolation Forest anomaly scoring."""

    def test_iforest_score_range_and_flags(self, test_preds: pd.DataFrame):
        ifor_df, meta = detect_isolation_forest_anomalies(test_preds, contamination=0.02, random_state=42)

        assert "IForest_Anomaly_Score" in ifor_df.columns
        assert (ifor_df["IForest_Anomaly_Score"] >= 0.0).all()
        assert (ifor_df["IForest_Anomaly_Score"] <= 1.0).all()

        # Flagged count should be close to 2% of 990 = ~20 records
        assert 15 <= meta["anomalies_flagged"] <= 25


class TestInvestigationQueue:
    """Verify prioritized investigation queue synthesis."""

    def test_queue_ranking_and_evidence(self, test_preds: pd.DataFrame):
        stat_df, _ = detect_statistical_anomalies(test_preds)
        ifor_df, _ = detect_isolation_forest_anomalies(test_preds)
        comparison_df, agree_df, queue_df, meta = build_investigation_queue(stat_df, ifor_df)

        assert len(queue_df) > 0, "Investigation queue should not be empty"
        assert list(queue_df["Queue_Rank"]) == list(range(1, len(queue_df) + 1))

        # Priority scores must be sorted descending
        priority_scores = queue_df["Priority_Score"].tolist()
        assert priority_scores == sorted(priority_scores, reverse=True)

        # High overall detector agreement (> 95%)
        assert meta["detector_agreement_rate_pct"] > 95.0

        # Unanimous flags must exist
        assert meta["unanimous_anomalies"] > 0

        # Evidence notes must be non-empty strings
        for note in queue_df["Investigation_Evidence"]:
            assert isinstance(note, str)
            assert len(note) > 10


class TestAnomalyReportsExist:
    """Verify all report files are present on disk."""

    def test_all_reports_present_on_disk(self):
        out_dir = REPORT_SUBDIRS["anomaly_detection"]
        expected_files = [
            "01_statistical_scores.csv",
            "02_isolation_forest_scores.csv",
            "03_detector_comparison.csv",
            "04_agreement_summary.csv",
            "05_investigation_queue.csv",
            "06_anomaly_detection_summary.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            assert fpath.exists(), f"Missing anomaly report {fname}"
            assert fpath.stat().st_size > 0, f"Report {fname} is empty"
