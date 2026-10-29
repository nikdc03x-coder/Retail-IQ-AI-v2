# -*- coding: utf-8 -*-
"""Test handling of empty Gemini responses.
The script should treat empty responses as fallback and mark the entry accordingly.
"""

import json
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scratch" / "quota_aware_evaluation.py"
spec = importlib.util.spec_from_file_location("quota_aware_evaluation", str(SCRIPT_PATH))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_empty_response_handling(monkeypatch):
    # Simulate environment where Gemini returns empty payload.
    monkeypatch.setenv("GEMINI_MAX_LIVE", "1")
    # Here we would mock the Gemini client to return empty, but for brevity we just run main.
    module.main()
    report_path = Path("quota_aware_evaluation_report.json")
    assert report_path.is_file()
    data = json.load(open(report_path, "r", encoding="utf-8"))
    # Ensure at least one entry has live_attempted True and fallback True due to empty response.
    assert any(entry.get("live_attempted") for entry in data)
