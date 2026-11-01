"""Classical models: metrics, stationarity classifier, classical backtest (no lookahead, fallback)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classical
from forecasting import seasonal_naive
from metrics import all_errors, mae, mape, mape_excluded, rmse, smape, wape
from stationarity import classify, count_outliers, fix_christmas, swing_rule


# ---------------------------------------------------------------- metrics

def test_rmse_and_mae():
    a, f = [1, 2, 3, 4], [1, 2, 3, 8]
    assert mae(a, f) == 1.0
    assert rmse(a, f) == 2.0


def test_mape_skips_zero_days_and_reports_the_share():
    a, f = [0, 10, 20, 0], [5, 12, 18, 1]
    assert np.isclose(mape(a, f), (0.2 + 0.1) / 2)
    assert mape_excluded(a) == 0.5
    assert np.isnan(mape([0, 0], [1, 2]))


def test_smape_skips_days_where_both_are_zero():
    assert np.isclose(smape([0, 10], [0, 10]), 0.0)
    assert np.isclose(smape([0, 2], [2, 2]), 1.0)          # day 1: 2*2/2 = 2; day 2: 0
    assert np.isnan(smape([0, 0], [0, 0]))


def test_wape_weights_by_volume_and_handles_all_zero():
    assert np.isclose(wape([10, 0], [8, 2]), 4 / 10)
    assert np.isnan(wape([0, 0], [1, 1]))


def test_all_errors_mase_uses_given_scale():
    e = all_errors([2, 4], [1, 5], scale=2.0)
    assert e["mae"] == 1.0 and e["mase"] == 0.5
    assert np.isnan(all_errors([2, 4], [1, 5], scale=0)["mase"])


# ---------------------------------------------------------------- stationarity

@pytest.mark.parametrize("p, label", [
    ((0.01, 0.50, 0.01, 0.50), "stationary"),
    ((0.40, 0.01, 0.01, 0.50), "trend-stationary"),
    ((0.40, 0.01, 0.40, 0.01), "difference-stationary"),
    ((0.01, 0.01, 0.01, 0.01), "conflicting"),
    ((0.40, 0.50, 0.40, 0.50), "conflicting"),
])
def test_classify(p, label):
    assert classify(*p) == label


def test_outliers_and_christmas():
    y = np.r_[np.full(99, 10.0), 1000.0]
    assert count_outliers(y) == 1
    idx = pd.date_range("2014-12-11", "2015-01-08")
    s = pd.Series(100.0, index=idx)
    s["2014-12-25"] = 0.0
    assert fix_christmas(s)["2014-12-25"] == 100.0


def test_swing_rule_detects_growing_season():
    dates = pd.date_range("2014-01-04", periods=7 * 60)            # starts on a Saturday
    level = np.repeat(np.linspace(10, 100, 60), 7)
    week = np.tile([1.0, 1.2, 0.8, 1.0, 0.9, 1.3, 0.8], 60)
    assert swing_rule(dates, level * week)[2] == "multiplicative"
    assert swing_rule(dates, level + 10 * (week - 1))[2] == "additive"


# ---------------------------------------------------------------- classical backtest

def _toy(n=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n)
    y = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 1, n)
    return dates, np.maximum(y, 0)


def test_backtest_has_no_lookahead():
    dates, y = _toy()
    origins = [dates[800], dates[828]]
    rows, orders = classical.backtest_series(dates, y, origins, pq=range(1), PQ=range(1))
    noisy = y.copy()
    noisy[dates > origins[0]] = 1e4                                # wreck everything after origin 0
    rows2, orders2 = classical.backtest_series(dates, noisy, origins, pq=range(1), PQ=range(1))
    assert orders == orders2                                       # orders only see data up to origin 0
    a = pd.DataFrame(rows).query("origin == @origins[0]")["forecast"].to_numpy()
    b = pd.DataFrame(rows2).query("origin == @origins[0]")["forecast"].to_numpy()
    assert np.allclose(a, b)


def test_fit_window_is_last_730_days():
    assert len(classical.last_days(np.arange(1000))) == 730
    assert classical.last_days(np.arange(1000))[0] == 270


def test_failed_fit_falls_back_to_seasonal_naive(monkeypatch):
    def broken(*a, **k):
        raise ValueError("boom")
    monkeypatch.setattr(classical, "fit_sarima", broken)
    monkeypatch.setattr(classical, "fit_hw", broken)
    y = np.arange(60, dtype=float)
    for fc, ok, fb in [classical.sarima_forecast(y, (1, 0, 0)), classical.hw_forecast(y)]:
        assert fb and not ok
        assert np.allclose(fc, seasonal_naive(y, 28))


def test_constant_added_only_without_differencing():
    assert classical.trend_for(0, 0) == "c"
    assert classical.trend_for(1, 0) == "n" and classical.trend_for(0, 1) == "n"


def test_holt_winters_survives_zeros():
    dates, y = _toy(200)
    y[::10] = 0
    fc, ok, fb = classical.hw_forecast(y)
    assert not fb and np.all(np.isfinite(fc)) and np.all(fc >= 0)
