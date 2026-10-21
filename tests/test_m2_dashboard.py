"""Milestone 2 dashboard: verdict rules, facts, and the three new pages."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from utils.data import SCHEMA
from utils.verdicts import classical_shift, classical_vs_naive, m2_verdicts

FACTS = ROOT / "outputs" / "results_facts.json"


def fake(h1, h3):
    m = {f"{k}|ALL|mase": {"value": v, "n": 150} for k, v in [("sarima", 0.8), ("hw_add", 0.82), ("seasonal_naive", 0.98)]}
    sa = {f"all break item-origins|{v}": {"value": 0.001, "n": 70} for v in ["restart", "dummy"]}
    return {"m2": {"h1_supported": h1, "metrics": m, "shift_aware": sa, "h3_supported": h3}}


def test_classical_verdict_counts_supported_models():
    assert classical_vs_naive(fake({"sarima": True, "hw_add": True}, True))["verdict"] == "Yes"
    assert classical_vs_naive(fake({"sarima": True, "hw_add": False}, True))["verdict"] == "Partly"
    assert classical_vs_naive(fake({"sarima": False, "hw_add": False}, True))["verdict"] == "No"


def test_shift_verdict_follows_h3():
    assert classical_shift(fake({"sarima": True}, True))["verdict"] == "No"
    assert classical_shift(fake({"sarima": True}, False))["verdict"] == "Yes"


def test_no_m2_cards_without_m2_facts():
    assert m2_verdicts({}) == []


def test_m2_facts_have_sources_and_sizes():
    if not FACTS.exists():
        pytest.skip("facts not built")
    f = json.loads(FACTS.read_text())
    if "m2" not in f:
        pytest.skip("Milestone 2 notebooks not run")
    for group in ["metrics", "tests", "shift_aware", "stock", "diagnostics"]:
        for key, v in f["m2"][group].items():
            assert v["source"].startswith("m2/tables/"), key
            assert "n" in v, key
    for key, v in f["m2"]["tests"].items():
        assert len(v["ci95"]) == 2, key
    assert len(m2_verdicts(f)) == 2


NEW_PAGES = ["stationarity", "classical", "demo"]


@pytest.mark.parametrize("page", NEW_PAGES)
def test_new_page_renders(page):
    if not SCHEMA["m2_summary"][0].exists():
        pytest.skip("Milestone 2 results not generated")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=90)
    at.run()
    at.switch_page(f"views/{page}.py").run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
