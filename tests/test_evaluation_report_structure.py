# -*- coding: utf-8 -*-
"""Test the structure of the quota‑aware evaluation report.
The report JSON should contain a list of entries, each with the expected keys.
"""

import json
from pathlib import Path

def test_report_structure():
    report_path = Path("quota_aware_evaluation_report.json")
    assert report_path.is_file(), "Report file not found"
    data = json.load(open(report_path, "r", encoding="utf-8"))
    assert isinstance(data, list), "Report root should be a list"
    required_keys = {
        "question_id",
        "prompt",
        "live_attempted",
        "fallback",
        "provider",
        "model",
        "answer",
        "tools_called",
        "tool_details",
        "numeric_correct",
        "quota_used",
    }
    for entry in data:
        missing = required_keys - entry.keys()
        assert not missing, f"Entry missing keys: {missing}"
        # answer should be a dict with at least a 'verified' flag
        assert isinstance(entry.get("answer"), dict), "Answer should be a dict"
        assert "verified" in entry["answer"], "Answer missing 'verified' key"

