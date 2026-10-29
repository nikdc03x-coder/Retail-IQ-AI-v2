# -*- coding: utf-8 -*-
"""Tests for the quota‑aware evaluation script.
These tests verify that the script respects the MAX_LIVE_REQUESTS limit
and that the generated report contains the expected fields.
"""

import json
import os
from pathlib import Path
import importlib.util
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scratch" / "quota_aware_evaluation.py"

def load_module():
    spec = importlib.util.spec_from_file_location("quota_aware_evaluation", str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_max_live_requests_is_respected(tmp_path, monkeypatch):
    """Run the evaluation with MAX_LIVE_REQUESTS=0 and ensure no live calls are made.
    The script writes a JSON report; we check that every entry reports
    ``live_attempted`` as ``False`` and ``fallback`` as ``True``.
    """
    # Set env before loading module
    monkeypatch.setenv("GEMINI_MAX_LIVE", "0")
    module = load_module()
    # Run the main function – it writes ``quota_aware_evaluation_report.json``
    module.main()
    report_path = Path("quota_aware_evaluation_report.json")
    assert report_path.is_file(), "Report file was not created"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        assert entry.get("live_attempted") is False
        assert entry.get("fallback") is True
        # The deterministic fallback should be marked as verified.
        assert entry.get("answer", {}).get("verified") is True
