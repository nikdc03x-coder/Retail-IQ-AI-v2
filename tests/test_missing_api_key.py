# -*- coding: utf-8 -*-
"""Test behavior when Gemini API key is missing.
The script should detect the missing key and fall back without raising.
"""

import os
import json
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scratch" / "quota_aware_evaluation.py"
spec = importlib.util.spec_from_file_location("quota_aware_evaluation", str(SCRIPT_PATH))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_missing_api_key(monkeypatch):
    # Ensure GEMINI_API_KEY env var is unset.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Run the script; it should not crash.
    module.main()
    report_path = Path("quota_aware_evaluation_report.json")
    assert report_path.is_file()
    data = json.load(open(report_path, "r", encoding="utf-8"))
    # All entries should have fallback true because no key means no live calls.
    assert all(entry.get("fallback") for entry in data)
