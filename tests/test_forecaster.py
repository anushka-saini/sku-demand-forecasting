"""The Forecaster page: its scorecard maths, its loaders, and that its numbers match the leaderboard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
T = ROOT / "outputs" / "tables"
pytestmark = pytest.mark.skipif(not (T / "final_forecaster_summary.csv").exists(), reason="run scripts/export_forecaster.py first")

from utils import final, lab, score  # noqa: E402


# ---------------------------------------------------------------- scorecard maths

def test_errors_known_values():
    s = score.errors([3, 5], [1, 9])           # errors +2 and -4
    assert s["mae"] == 3 and s["bias"] == -1 and np.isclose(s["rmse"], np.sqrt(10))


def test_totals_and_zero_sales():
    f, a, d = score.totals([2, 2], [1, 1])
    assert (f, a) == (4, 2) and d == 1.0
    assert np.isnan(score.totals([1, 1], [0, 0])[2])           # nothing sold: no percent


def test_coverage_counts_days_inside_the_band():
    assert score.coverage([1, 5, 9], [0, 0, 0], [5, 5, 5]).tolist() == [True, True, False]


def test_verdict_and_closer():
    assert score.verdict(1.0, 2.0) == "beat" and score.verdict(2.0, 1.0) == "did not beat" and score.verdict(1.0, 1.0) == "tied"
    assert score.closer(10, 20, 11) == "you" and score.closer(15, 11, 10) == "the model" and score.closer(8, 12, 10) == "tie"


def test_worst_days_picks_the_biggest_misses():
    assert set(score.worst_days([0, 0, 0, 0], [1, 9, 2, 5], k=2)) == {1, 3}


def test_order_advice_is_demand_plus_safety_stock():
    past = [np.array([1.0] * 28), np.array([3.0] * 28)]          # every 7-day shortfall sum is 7 or 21
    a = score.order_advice([2.0] * 28, past, on_hand=5)
    assert a["need"] == 14 and a["safety"] == 21 and a["level"] == 35 and a["order"] == 30
    assert score.order_advice([1.0] * 28, [], 0)["safety"] == 0          # no history: no buffer, no crash


# ---------------------------------------------------------------- loaders

def test_forecast_and_band_loader_never_carries_actuals():
    fw = final.fc_window("FOODS_1_043", "2015-12-06", "xgboost")
    assert len(fw) == 28 and "actual" not in fw.columns and (fw["lo"] <= fw["hi"]).all()


def test_history_stops_at_the_origin():
    h = final.fc_history("FOODS_1_043", "2015-12-06")
    assert len(h) == 90 and h["date"].max() == pd.Timestamp("2015-12-06")


def test_past_errors_come_only_from_earlier_windows():
    assert len(final.fc_past_errors("FOODS_1_043", "2015-12-06", "xgboost")) == 3          # the three validation windows
    assert len(final.fc_past_errors("FOODS_1_043", "2016-01-03", "xgboost")) == 4
    assert final.fc_past_errors("FOODS_1_043", "2015-09-13", "xgboost") == []          # the first window has nothing before it


def test_loaders_do_not_ask_for_actuals_before_reveal(monkeypatch):
    calls = []
    real = final._read
    monkeypatch.setattr(final, "_read", lambda n, c, f: (calls.append((c, f)), real(n, c, f))[1])
    final.fc_index.clear(), final.fc_window.clear(), final.fc_history.clear()
    final.fc_index(), final.fc_window("FOODS_1_085", "2016-01-03", "sarima"), final.fc_history("FOODS_1_085", "2016-01-03")
    assert not any("actual" in c for c, _ in calls)


# ---------------------------------------------------------------- the numbers agree with the leaderboard

def test_summary_reproduces_leaderboard_mae_and_rmse():
    lb = pd.read_csv(T / "final_leaderboard.csv").set_index("model")
    s = final.fc_summary().xs(28, level="h")
    assert len(s) >= 7
    for m, r in s.iterrows():
        assert np.isclose(r["lb_mae"], lb.loc[m, "mae"]) and np.isclose(r["lb_rmse"], lb.loc[m, "rmse"]), m


def test_xgboost_band_coverage_matches_the_garch_table():
    g = pd.read_csv(T / "final_garch_intervals.csv")
    want = float(g.loc[g["interval"].str.startswith("GARCH") & (g["nominal"] == 0.8), "coverage"].iloc[0])
    b = pd.read_parquet(T / "final_forecaster_bands.parquet")
    b = b[b["model"] == "xgboost"]
    a = pd.read_parquet(T / "final_forecasts.parquet", columns=["item_id", "origin", "date", "model", "actual"], filters=[("model", "==", "xgboost")])
    x = b.merge(a, on=["item_id", "origin", "date", "model"])
    assert np.isclose(((x["actual"] >= x["lo"]) & (x["actual"] <= x["hi"])).mean(), want)


def test_pooled_error_by_category_matches_the_leaderboard_pooled_mae():
    d = lab.test_errors()
    lb = pd.read_csv(T / "final_leaderboard.csv").set_index("model")
    for m in ("xgboost", "sarima", "seasonal_naive"):
        assert np.isclose(d.loc[d["model"] == m, "ae"].mean(), lb.loc[m, "pooled_mae"]), m
        assert np.isclose((d.loc[d["model"] == m, "se"].mean()) ** 0.5, lb.loc[m, "pooled_rmse"]), m
    assert set(lab.cut(d, ["xgboost"], "wd")["wd"]) == set(range(7))
