"""Which model is best (07i): its tables reproduce 07h's numbers, have no gaps, and the verdict uses plain words."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import EVAL_ORIGINS  # noqa: E402
from model_verdict import LABEL, verdict  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FILES = ["verdict_metrics", "verdict_ranks", "verdict_tests", "verdict_wins", "verdict_stock"]
# built from pieces so this file does not match itself
WORDS = ["M" + "1", "M" + "2", "M" + "3", "mile" + "stone", "mid" + "-sem", "mid" + "sem", "pha" + "se", "syn" + "opsis",
         "rob" + "ust", "lever" + "age", "del" + "ve", "compre" + "hensive", "key " + "takeaways"]
PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in WORDS) + r")\b", re.IGNORECASE)


def load(name):
    path = TABLES / f"{name}.csv"
    if not path.exists():
        pytest.skip("run notebooks/07i_model_verdict.ipynb first")
    return pd.read_csv(path)


def overall():
    m = load("verdict_metrics")
    return m[m["group"] == "ALL"].set_index("model")


def test_medians_reproduce_the_comparison():
    m = overall()
    assert round(m.loc["combination", "rmse"], 3) == 4.392
    assert round(m.loc["sarima", "rmse"], 3) == 4.445
    assert round(m.loc["ets", "mase"], 3) == 0.744          # the Box-Cox damped Holt-Winters
    assert round(m.loc["combination", "mase"], 3) == 0.747
    head = pd.read_csv(TABLES / "ts_headline.csv").set_index("model")
    gap = (m[list(LABEL)] - head.loc[m.index, list(LABEL)]).abs().to_numpy().max()
    assert gap < 1e-9


def test_rank_table_agrees_with_the_medians():
    m, r = overall(), load("verdict_ranks").set_index("model")
    for k in LABEL:
        assert r[f"rank_{k}"].to_dict() == m[k].rank(method="min").astype(int).to_dict()
    assert (r["avg_rank"] - r[[f"rank_{k}" for k in LABEL]].mean(axis=1)).abs().max() < 1e-12
    assert len(r) == 8 and r["n_first"].sum() >= len(LABEL)


@pytest.mark.parametrize("name", FILES)
def test_no_missing_values(name):
    d = load(name)
    assert len(d) and not d.isna().any().any(), d.columns[d.isna().any()].tolist()


def test_every_pair_is_tested_once_per_measure():
    t = load("verdict_tests")
    assert t.groupby(["metric", "reference"]).size().eq(7).all()
    assert set(t["result"]) <= {"better", "tie", "worse"}
    real = (t["p_holm"] < 0.05) & ((t["ci_low"] > 0) | (t["ci_high"] < 0))
    assert (real == (t["result"] != "tie")).all()


def test_wins_add_up_to_the_products():
    w = load("verdict_wins")
    n = overall()["n_products"].iloc[0]
    assert w.groupby("metric")["wins"].sum().eq(n).all()


def test_verdict_text_is_plain_and_complete():
    m, r = overall(), load("verdict_ranks").set_index("model")
    s = load("verdict_stock")
    rec, text = verdict(m[list(LABEL)], r, load("verdict_tests"), load("verdict_wins"),
                        s[s["method"] == "conformal"].set_index("model"), {}, EVAL_ORIGINS, int(m["n_products"].iloc[0]))
    body = " ".join(text)
    assert rec in m.index
    assert not PATTERN.search(body), PATTERN.findall(body)
    assert "For stock planning" in body and "Limits:" in body
    assert "most accurate" not in body
