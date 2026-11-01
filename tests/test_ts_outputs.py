"""The saved time-series results: shapes, origins, no gaps, and choices that follow from validation numbers only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ts_models as tm  # noqa: E402
from config import EVAL_ORIGINS, HORIZON  # noqa: E402

T = ROOT / "outputs" / "tables"
CONTENDERS = ["seasonal_naive", "ets", "sarima", "sarimax", "theta", "prophet", "autots", "combination"]


def need(name):
    if not (T / name).exists():
        pytest.skip(f"{name} not generated")
    return pd.read_parquet(T / name) if name.endswith(".parquet") else pd.read_csv(T / name)


@pytest.mark.parametrize("name", ["ts_sarima_forecasts.parquet", "ts_blocks_forecasts.parquet",
                                  "ts_other_forecasts.parquet", "ts_forecasts.parquet"])
def test_forecast_files_have_the_same_backtest(name):
    fc = need(name)
    assert fc["forecast"].notna().all() and np.isfinite(fc["forecast"]).all() and (fc["forecast"] >= 0).all()
    assert set(pd.to_datetime(fc["origin"])) <= set(tm.ALL_ORIGINS)
    assert set(EVAL_ORIGINS) <= set(pd.to_datetime(fc["origin"]))
    assert fc.groupby(["model", "item_id", "origin"]).size().eq(HORIZON).all()
    days = (pd.to_datetime(fc["date"]) - pd.to_datetime(fc["origin"])).dt.days
    assert days.between(1, HORIZON).all()
    sample = pd.read_csv(T / "classical_sample_items.csv")
    assert fc.groupby("model")["item_id"].nunique().eq(len(sample)).all()
    assert set(fc["item_id"]) == set(sample["item_id"])


def test_headline_has_every_contender_on_150_products():
    h = need("ts_headline.csv").set_index("model")
    assert set(CONTENDERS) <= set(h.index), set(CONTENDERS) - set(h.index)
    assert h["n_products"].eq(150).all()
    assert h[["rmse", "mae", "mape", "smape", "wape", "mase"]].notna().all().all()
    assert round(h.loc["sarima_07a", "rmse"], 3) == 4.445 and round(h.loc["sarima_07a", "mae"], 3) == 3.522


def replay(steps, rule_for):
    """Walk an ablation table in order and redo each keep/drop decision from the validation columns only."""
    best, out = None, []
    groups = steps.groupby("group", sort=False)
    for _, g in groups:
        if best is None:
            best = g.iloc[0]
            out.append(False)
            continue
        cand = g.sort_values(["val_rmse", "val_mae"]).iloc[0]
        rule = rule_for(cand["step"])
        keep = rule(cand, best)
        out += [bool(keep and s == cand["step"]) for s in g["step"]]
        if keep:
            best = cand
    return out


def helps(a, b):
    return a["val_rmse"] < b["val_rmse"] and a["val_mae"] < b["val_mae"]


def not_worse(a, b):
    return a["val_rmse"] <= b["val_rmse"] and a["val_mae"] <= b["val_mae"]


def test_sarima_tweaks_were_chosen_on_validation_only():
    a = need("ts_sarima_ablation.csv")
    s = a[a["family"] == "SARIMA"].copy()
    # candidates tried together in one step share a group: the two transforms, and the three Fourier K
    s["group"] = s["step"].str.replace(r"Box-Cox|log1p", "transform", regex=True).str.replace(r"K=\d", "K", regex=True)
    expected = replay(s, lambda step: not_worse if "Christmas" in step else helps)
    assert s["kept"].astype(bool).tolist() == expected


def test_combination_members_are_the_validation_top_three():
    h = need("ts_headline.csv").set_index("model")
    members = need("ts_combination_members.csv")["member"].tolist()
    single = h.loc[[m for m in ["ets", "sarima", "sarimax", "theta", "prophet", "autots"] if m in h.index]]
    single = single[single["n_products"] == 150]
    assert members == single.sort_values(["val_rmse", "val_mae"]).index[:3].tolist()




def test_runner_up_on_all_products_matches_07h_on_the_sample():
    sc = need("runner_up_726_scores.csv")
    h = need("ts_headline.csv").set_index("model")
    runner = h[h["kind"] == "contender"].sort_values("rmse").index[1]
    key = {"ets": "ets_boxcox"}.get(runner, runner)
    r = sc[sc["model"] == key]
    assert r["item_id"].nunique() == 726 and set(pd.to_datetime(r["origin"])) == set(EVAL_ORIGINS)
    assert r[["mae", "rmse", "mase"]].notna().all().all()
    path = ROOT / "data" / "processed" / "runner_up_forecasts.parquet"
    if not path.exists():
        pytest.skip("09b forecasts not generated")
    ref = need("ts_forecasts.parquet")
    both = pd.read_parquet(path).merge(ref[ref["model"] == runner], on=["item_id", "origin", "date"])
    assert both["item_id"].nunique() == 150
    assert np.allclose(both["forecast_x"], both["forecast_y"], atol=1e-6)


def test_07h_applies_the_blow_up_guard():
    fc = need("ts_forecasts.parquet")
    sys.path.insert(0, str(ROOT / "src"))
    import best_model as bm
    inp = tm.load_inputs()
    single = fc[fc["model"] != "combination"]
    worst = single.groupby(["item_id", "origin", "model"])["forecast"].max().reset_index()
    peak = {i: d["sales"].max() for i, d in inp["daily"].items()}
    worst["ratio"] = worst["forecast"] / worst["item_id"].map(peak).clip(lower=1)
    assert (worst["ratio"] <= bm.CAP).all(), worst.sort_values("ratio").tail(3)


def test_accuracy_page_shows_the_eight_models():
    if not (T / "ts_headline.csv").exists():
        pytest.skip("results not generated")
    sys.path.insert(0, str(ROOT / "dashboard"))
    from utils.models import CONTENDERS as SHOWN, NAMES
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    at.switch_page("views/accuracy.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert "What this tells us" in "\n".join(str(e.value) for e in at.markdown)
    table = at.dataframe[0].value
    assert set(table["Model"].str.replace("  (used for stock)", "", regex=False)) == {NAMES[m] for m in SHOWN}
    for metric in ["MASE", "MAE", "MAPE", "sMAPE", "WAPE"]:
        at.radio(key="acc_metric").set_value(metric).run()
        assert not at.exception, [e.value for e in at.exception]
    groups = at.multiselect(key="acc_extra").options
    at.multiselect(key="acc_extra").set_value(groups).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe[0].value) > len(SHOWN)


def test_only_the_twelve_models_are_reported():
    """No table the dashboard reads still carries a model that was removed from the project."""
    gone = {"hw_mul", "tsb", "ar", "ma", "arimax", "combination_weighted"}
    for f in ["ts_headline.csv", "ts_item_scores.csv", "comparison_metrics_summary.csv", "model_mase_unrounded.csv",
              "classical_metrics_summary.csv", "pbi_fact_forecast_error.csv"]:
        path = T / f
        if not path.exists():
            continue
        left = gone & set(pd.read_csv(path, usecols=["model"])["model"])
        assert not left, (f, left)
    # 07a's raw damped ETS is gone too; 07h keeps the key 'ets' for the Box-Cox Holt-Winters
    for f in ["classical_metrics_summary.csv", "model_mase_unrounded.csv", "pbi_fact_forecast_error.csv"]:
        if (T / f).exists():
            assert "ets" not in set(pd.read_csv(T / f, usecols=["model"])["model"]), f
