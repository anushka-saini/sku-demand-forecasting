import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from utils.data import DataError
from utils.facts import FACTS_PATH, lookup, read_facts, value
from utils.kpis import BAND_FLAG, BIAS_FLAG, build_watchlist
from utils import verdicts

needs_facts = pytest.mark.skipif(not FACTS_PATH.exists(), reason="results_facts.json not built")


# ---------------------------------------------------------------- facts loader

def test_read_facts_missing_file(tmp_path):
    with pytest.raises(DataError, match="build_facts"):
        read_facts(tmp_path / "nope.json")


def test_lookup_names_missing_key():
    with pytest.raises(DataError, match="a / b / zz"):
        lookup({"a": {"b": {"c": 1}}}, "a", "b", "zz")
    assert value({"x": {"value": 3}}, "x") == 3


# ---------------------------------------------------------------- verdict logic

@needs_facts
def test_verdicts_on_real_facts():
    f = read_facts()
    v = {x["question"]: x for x in verdicts.all_verdicts(f)}
    assert len(v) == 4
    for x in v.values():
        assert x["strength"] in {"strong", "moderate", "weak"} and x["caveat"]


@needs_facts
def test_retraining_verdict_flips_if_a_variant_helps():
    f = copy.deepcopy(read_facts())
    assert verdicts.retraining(f)["verdict"] == "No"
    key = next(k for k in f["forecasts"]["adaptation_variants"] if k.startswith("high-confidence"))
    f["forecasts"]["adaptation_variants"][key]["helps"] = True
    assert verdicts.retraining(f)["verdict"] == "Partly"
    key = next(k for k in f["forecasts"]["adaptation_variants"] if k.startswith("all detected"))
    f["forecasts"]["adaptation_variants"][key]["helps"] = True
    assert verdicts.retraining(f)["verdict"] == "Yes"


@needs_facts
def test_shift_verdict_depends_on_power_and_ci():
    f = copy.deepcopy(read_facts())
    f["breaks"]["injection_power_cusum"]["step_20pct"]["value"] = 0.95
    assert verdicts.shifts(f)["verdict"] == "Yes"
    f["breaks"]["adjusted_share_with_real_shift"]["CUSUM"]["ci95"] = [-0.01, 0.2]
    assert verdicts.shifts(f)["verdict"] == "No"


@needs_facts
def test_stock_verdict_counts_significant_windows():
    f = copy.deepcopy(read_facts())
    md = f["inventory"]["multi_window_cost_differences"]
    keys = [k for k in md if k.endswith("|static - naive|0.95")]
    for k in keys:
        md[k]["ci95"] = [-10.0, -1.0]
    r5 = [k for k in f["replication"] if k.split("|")[1].startswith("R5")]
    for k in r5:
        f["replication"][k]["holds"] = True
    assert verdicts.stock(f)["verdict"] == "Yes"
    f["replication"][r5[0]]["holds"] = False            # "Yes" also needs the saving in both other states
    assert verdicts.stock(f)["verdict"] == "Partly"
    for k in keys:
        md[k]["ci95"] = [-10.0, 5.0]
    assert verdicts.stock(f)["verdict"] == "No"


@needs_facts
def test_best_model_is_lowest_median_mase():
    f = copy.deepcopy(read_facts())
    from utils.models import BEST_SHORT
    assert verdicts.best_model(f)["verdict"] == BEST_SHORT
    f["forecasts"]["median_item_mase"]["ets_ALL"]["value"] = 0.1
    f["forecasts"]["pairwise"].setdefault("ets-best|ALL", {"value": -0.1, "p_value": 0.001})
    assert verdicts.best_model(f)["verdict"] == "ETS"


# ---------------------------------------------------------------- watchlist

def _toy():
    cal = pd.DataFrame({"item_id": ["a", "b", "c"], "model": "best", "variant": "static", "method": "conformal",
                        "service_level": 0.95, "fill_rate": [0.90, 0.99, 0.94], "lost_units": [50.0, 1.0, 10.0],
                        "stockout_days": [5, 0, 2], "total_cost": [100.0, 10.0, 40.0], "mean_safety_stock": [3.0, 3.0, 3.0]})
    bias = pd.DataFrame({"item_id": ["a", "b", "c"], "model": "best", "variant": "static", "bias": [-0.2, 0.0, 0.01]})
    cov = pd.DataFrame({"item_id": ["a", "b", "c"], "model": "best", "variant": "static",
                        "coverage90_before": [0.5, 0.9, 0.9], "coverage90_after": [0.7, 0.9, 0.95]})
    items = pd.DataFrame({"item_id": ["a", "b", "c"], "category": "FOODS", "segment": "AY",
                          "break_at_last_origin": [True, False, False], "intermittent": [False, False, True]})
    hc = pd.DataFrame({"item_id": ["a", "b", "c"], "high_confidence": [True, False, np.nan]})
    return cal, bias, cov, items, hc


def test_watchlist_selects_items_below_target_with_reasons():
    wl = build_watchlist(*_toy(), method="conformal", service_level=0.95)
    assert wl["item_id"].tolist() == ["a", "c"]                     # sorted by lost units; b is above target
    a = wl.iloc[0]["reasons"]
    assert "under-forecast" in a and "band covers only 70%" in a and "high-confidence" in a
    assert wl.iloc[1]["reasons"] == "intermittent demand"
    assert BIAS_FLAG == 0.05 and BAND_FLAG == 0.80


def test_watchlist_uses_matching_band_for_method():
    cal, bias, cov, items, hc = _toy()
    cal["method"] = "normal formula"
    wl = build_watchlist(cal, bias, cov, items, hc, method="normal formula", service_level=0.95)
    assert wl.iloc[0]["band_coverage_90"] == 0.5


@needs_facts
def test_watchlist_count_matches_facts():
    from utils.data import SCHEMA, read_table
    tables = {k: read_table(*SCHEMA[k]) for k in ["calibrated_items", "item_bias", "item_coverage", "items", "hc_items"]}
    tables["items"]["break_at_last_origin"] = tables["items"]["break_at_last_origin"].fillna(False).astype(bool)
    tables["items"]["intermittent"] = tables["items"]["intermittent"].fillna(False).astype(bool)
    f = read_facts()
    for method in ["conformal", "normal formula"]:
        for sl in [0.9, 0.95, 0.99]:
            wl = build_watchlist(tables["calibrated_items"], tables["item_bias"], tables["item_coverage"],
                                 tables["items"], tables["hc_items"], method, sl)
            assert len(wl) == value(f, "planner", f"{method}|{sl}")
