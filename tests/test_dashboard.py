import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from utils.data import SCHEMA, DataError, read_table
from utils.kpis import (abc_xyz_matrix, base_cost_assumptions, cohens_kappa, filter_items, headline,
                        per_item_variant_diff, tradeoff, verdict)


# ---------------------------------------------------------------- loaders

def test_read_table_missing_file_message(tmp_path):
    with pytest.raises(DataError, match="Missing results file"):
        read_table(tmp_path / "nope.csv", ["a"])


def test_read_table_missing_column_message(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({"a": [1]}).to_csv(f, index=False)
    with pytest.raises(DataError, match="missing column"):
        read_table(f, ["a", "b"])


def test_read_table_parses_dates(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({"d": ["2016-01-01"]}).to_csv(f, index=False)
    assert pd.api.types.is_datetime64_any_dtype(read_table(f, ["d"], ["d"])["d"])


@pytest.mark.parametrize("name", list(SCHEMA))
def test_every_real_results_file_matches_schema(name):
    path, required, dates = SCHEMA[name]
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    read_table(path, required, dates)


# ---------------------------------------------------------------- KPI math

def test_verdict():
    assert verdict(0.001, 0.004) == "worse"
    assert verdict(-0.004, -0.001) == "better"
    assert verdict(-0.001, 0.002) == "no detectable difference"
    assert verdict(np.nan, 0.1).startswith("inconclusive")


def test_headline_reports_negative_result_honestly():
    tests = pd.DataFrame([
        {"comparison": "adaptive (break items)", "model": "best", "category": "ALL", "n": 488,
         "median_diff": 0.0033, "ci_low": 0.0015, "ci_high": 0.0053},
        {"comparison": "placebo (no-break items)", "model": "best", "category": "ALL", "n": 261,
         "median_diff": 0.0009, "ci_low": -0.0015, "ci_high": 0.0035}])
    h = headline(tests)
    assert h.startswith("Retraining after a detected break did not help")
    assert "+0.0033" in h and "n = 488" in h


def test_cohens_kappa():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1
    assert np.isclose(cohens_kappa([1, 0, 1, 0], [1, 1, 0, 0]), 0)


def test_per_item_variant_diff_uses_break_origins_only():
    rows = []
    for origin, has_break in [("2016-01-01", True), ("2016-02-01", False)]:
        for variant, mase in [("static", 1.0), ("adaptive", 1.2 if has_break else 1.0), ("placebo", 1.0)]:
            rows.append({"item_id": "A", "origin": origin, "model": "best", "variant": variant,
                         "has_break": has_break, "mase": mase, "is_eval_origin": True})
    d = per_item_variant_diff(pd.DataFrame(rows), "best", "adaptive")
    assert np.isclose(d["diff"].iloc[0], 0.2)                  # only the break origin counts


def test_filters_and_matrix():
    items = pd.DataFrame({"item_id": list("abcd"), "category": ["F", "F", "H", "H"], "in_scope": [True, False, True, False],
                          "abc_class": ["A", "A", "B", "C"], "xyz_class": ["X", "Y", "X", "Z"],
                          "total_sales": [50, 30, 15, 5]})
    assert filter_items(items, categories=["F"], in_scope_only=True)["item_id"].tolist() == ["a"]
    counts, share = abc_xyz_matrix(items)
    assert counts.loc["A", "X"] == 1 and counts.to_numpy().sum() == 4
    assert np.isclose(share.loc["A", "X"], 0.5) and np.isclose(share.to_numpy().sum(), 1)


def test_tradeoff_fill_rate_is_unit_weighted():
    inv = pd.DataFrame({"item_id": ["a", "b"], "policy": "p", "service_level": 0.95, "lead_time": 7,
                        "demand_units": [100, 900], "lost_units": [50, 0], "stockout_days": [3, 0],
                        "avg_inventory_value": [1.0, 2.0], "holding_cost": [1, 1], "stockout_cost": [2, 0],
                        "total_cost": [3, 1], "stockout_risk_flag": [True, False]})
    t = tradeoff(inv, 7).iloc[0]
    assert np.isclose(t["fill_rate"], 0.95) and t["total_cost"] == 4 and t["items_at_risk"] == 1


def test_base_cost_assumptions_recovered():
    inv = pd.DataFrame({"holding_cost": [2.5, 5.0], "holding_value_days": [10, 20],
                        "stockout_cost": [1.0, 3.0], "lost_value": [2, 6]})
    assert base_cost_assumptions(inv) == (0.25, 0.5)


# ---------------------------------------------------------------- app consistency with the notebooks

def test_app_per_item_medians_match_notebook_tests():
    path_err, path_tests = SCHEMA["forecast_error"][0], SCHEMA["adaptive_tests"][0]
    if not (path_err.exists() and path_tests.exists()):
        pytest.skip("results not generated")
    errors = read_table(path_err, [], ["origin"])
    tests = pd.read_csv(path_tests)
    windows = ROOT / "data" / "processed" / "breaks_by_origin.csv"
    placebo = None
    if windows.exists():
        w = pd.read_csv(windows, parse_dates=["origin", "placebo_start"])
        placebo = w[w["placebo_start"].notna()]
    for variant, comp in [("adaptive", "adaptive (break items)"), ("placebo", "placebo (no-break items)")]:
        d = per_item_variant_diff(errors, "best", variant, placebo)
        row = tests[(tests["comparison"] == comp) & (tests["model"] == "best") & (tests["category"] == "ALL")].iloc[0]
        assert len(d) == row["n"]
        assert np.isclose(d["diff"].median(), row["median_diff"], atol=1e-5)   # export rounds MASE to 5 dp


# ---------------------------------------------------------------- every page renders without an exception

PAGES = ["overview", "data_scope", "demand_patterns", "accuracy", "item_explorer", "demand_shifts", "stock_impact",
         "method_limits"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    if not SCHEMA["items"][0].exists():
        pytest.skip("results not generated")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
