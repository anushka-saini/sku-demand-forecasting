"""The time-series wrappers: origins kept apart, output format, fallbacks, no lookahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ts_models as tm  # noqa: E402
from config import EVAL_ORIGINS, HORIZON, ORIGINS  # noqa: E402
from forecasting import seasonal_naive  # noqa: E402


def toy(n=800, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-09-01", periods=n)
    y = 10 + 3 * (dates.dayofweek >= 5) + rng.poisson(2, n)
    return dates, y.astype(float)


def test_validation_origins_are_the_three_before_scoring():
    assert [o.date().isoformat() for o in tm.VAL_ORIGINS] == ["2015-09-13", "2015-10-11", "2015-11-08"]
    assert not set(tm.VAL_ORIGINS) & set(EVAL_ORIGINS)
    last_val_day = tm.VAL_ORIGINS[-1] + pd.Timedelta(days=HORIZON)
    assert last_val_day < EVAL_ORIGINS[0] + pd.Timedelta(days=1), "validation days must end before the first scored day"
    assert tm.ALL_ORIGINS[2:] == list(ORIGINS)


def test_tuning_view_never_returns_a_scored_origin():
    df = pd.DataFrame({"origin": tm.ALL_ORIGINS * 2, "x": range(18)})
    out = tm.for_tuning(df)
    assert set(out["origin"]) == set(tm.VAL_ORIGINS)
    assert not out["origin"].isin(EVAL_ORIGINS).any()
    with pytest.raises(ValueError):
        tm.for_tuning(df[df["origin"].isin(EVAL_ORIGINS)])
    assert set(tm.for_scoring(df)["origin"]) == set(EVAL_ORIGINS)


@pytest.mark.parametrize("spec", [{"kind": "naive"}, {"kind": "ets"}, {"kind": "theta"},
                                  {"kind": "sarima", "order": (1, 1, 1), "sorder": (1, 0, 1, 7)},
                                  {"kind": "sarima", "order": (1, 0, 0), "how": "log1p", "clip": True,
                                   "exog": ["fourier2", "christmas", "snap"]}])
def test_backtest_shape_and_no_missing(spec):
    dates, y = toy()
    days = dates.append(pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=60))
    flags = pd.DataFrame({"snap": (days.day <= 10).astype(float), "christmas": 0.0, "event": 0.0}, index=days)
    origins = [dates[600], dates[628]]
    df = pd.DataFrame(tm.backtest("X", dates, y, origins, "m", spec, flags))
    assert len(df) == len(origins) * HORIZON
    assert list(df.columns) == ["item_id", "origin", "date", "model", "forecast", "converged", "fallback"]
    assert df["forecast"].notna().all() and (df["forecast"] >= 0).all()
    assert ((df["date"] > df["origin"]) & (df["date"] <= df["origin"] + pd.Timedelta(days=HORIZON))).all()


def test_no_lookahead():
    dates, y = toy()
    o = dates[650]
    spec = {"kind": "sarima", "order": (1, 1, 1), "sorder": (1, 0, 1, 7)}
    a = pd.DataFrame(tm.backtest("X", dates, y, [o], "m", spec))
    y2 = y.copy()
    y2[651:] = np.random.default_rng(1).poisson(50, len(y) - 651)
    b = pd.DataFrame(tm.backtest("X", dates, y2, [o], "m", spec))
    assert np.allclose(a["forecast"], b["forecast"])


def test_failed_fit_falls_back_to_seasonal_naive():
    _, y = toy()
    fc, ok, fb = tm.sarima_fc(y, order=(1, 1, 1), sorder=(1, 0, 1, 7), how="not-a-transform")
    assert fb and not ok
    assert np.array_equal(fc, seasonal_naive(y, HORIZON))
    fc, ok, fb = tm.finish(np.full(HORIZON, np.nan), y, HORIZON, True)
    assert fb and np.array_equal(fc, seasonal_naive(y, HORIZON))


def test_transforms_round_trip_and_clipping():
    _, y = toy()
    for how in [None, "log1p", "boxcox"]:
        z, back = tm.prepare(y, how=how)
        assert np.allclose(back(z), y)
    z, back = tm.prepare(y * 1000, unit=1000)
    assert np.allclose(z, y) and np.allclose(back(z), y * 1000)
    spiky = np.r_[np.full(100, 5.0), 500.0]
    assert tm.clip_high(spiky).max() < 500 and tm.clip_high(spiky)[:100].tolist() == [5.0] * 100


def test_combination_is_the_average():
    base = pd.DataFrame({"item_id": "X", "origin": ORIGINS[0], "date": pd.date_range("2015-11-09", periods=3)})
    fc = pd.concat([base.assign(model="a", forecast=[1.0, 2, 3]), base.assign(model="b", forecast=[3.0, 4, 5])])
    out = tm.combine(fc, ["a", "b"])
    assert out["forecast"].tolist() == [2.0, 3.0, 4.0]
    out = tm.combine(fc, ["a", "b"], weights={"a": 3.0, "b": 1.0})
    assert out["forecast"].tolist() == [1.5, 2.5, 3.5]


def test_fourier_lines_up_across_train_and_future():
    d = pd.date_range("2015-01-01", periods=10)
    full = tm.fourier(d, 2)
    assert np.allclose(full.iloc[5:].to_numpy(), tm.fourier(d[5:], 2).to_numpy())
    assert tm.exog_columns(["fourier2", "snap"]) == ["sin1", "cos1", "sin2", "cos2", "snap"]


def test_best_model_blow_up_falls_back(monkeypatch):
    import best_model as bm
    d = pd.date_range("2014-01-01", periods=800)
    y = np.tile([2.0, 3, 4, 5, 6, 9, 8], 115)[:800]
    monkeypatch.setattr(bm, "_member_fc", lambda *a, **k: (np.full(HORIZON, 1e4), True, False))
    f, ok, fb = bm.member_fc({"kind": "sarima"}, y, d, d[-30], None, None, None, None)
    assert fb and f.max() <= y.max()
    monkeypatch.setattr(bm, "_member_fc", lambda *a, **k: (np.full(HORIZON, 20.0), True, False))
    assert not bm.member_fc({"kind": "sarima"}, y, d, d[-30], None, None, None, None)[2]
