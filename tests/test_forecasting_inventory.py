import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from metrics import mase, mase_scale, paired_test
from forecasting import seasonal_naive, tsb_forecast, is_intermittent
from break_detection import cusum_breaks_online
from inventory import (safety_stock_normal, safety_stock_empirical, lead_time_error_sums,
                       forecast_path, simulate_policy, policy_kpis)


# ---------------------------------------------------------------- metrics

def test_mase_seasonal_naive_scale():
    insample = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 3) + np.repeat([0.0, 1.0, 2.0], 7)
    assert np.isclose(mase_scale(insample), 1.0)               # every lag-7 difference is 1
    assert np.isclose(mase([2, 2], [1, 3], insample), 1.0)


def test_mase_perfect_forecast_is_zero():
    rng = np.random.default_rng(0)
    y = rng.poisson(10, 100).astype(float)
    assert mase(y[-7:], y[-7:], y[:-7]) == 0


def test_paired_test_detects_consistent_improvement():
    d = -np.linspace(0.01, 0.2, 40)
    res = paired_test(d)
    assert res['p_value'] < 0.001 and res['rank_biserial'] == -1.0 and res['ci_high'] < 0


# ---------------------------------------------------------------- forecasting

def test_seasonal_naive_repeats_last_week():
    fc = seasonal_naive(np.arange(14), h=10)
    assert fc.tolist() == [7, 8, 9, 10, 11, 12, 13, 7, 8, 9]


def test_tsb_on_regular_pattern():
    y = np.tile([0.0, 4.0], 100)                               # demand every other day, size 4
    fc = tsb_forecast(y, h=5)
    assert np.allclose(fc, fc[0]) and 1.5 < fc[0] < 2.5        # ~ probability 0.5 * size 4
    assert is_intermittent(y) and not is_intermittent(np.ones(50))


def test_online_cusum_finds_two_breaks_without_lookahead():
    rng = np.random.default_rng(5)
    x = rng.normal(100, 5, 300)
    x[120:] += 30
    x[230:] -= 30
    brks = cusum_breaks_online(x, burn_in=8, ref_len=52, k=1.0, h=10)
    assert len(brks) == 2
    assert abs(brks[0][0] - 120) <= 3 and abs(brks[1][0] - 230) <= 3
    assert brks[0][1] >= 120 and brks[1][1] >= 230
    assert cusum_breaks_online(x[:150], burn_in=8, ref_len=52, k=1.0, h=10)[0] == brks[0]


# ---------------------------------------------------------------- inventory

def test_safety_stock_normal_formula():
    assert np.isclose(safety_stock_normal(10, 4, 0.95), 1.6448536 * 10 * 2, atol=1e-4)
    assert safety_stock_normal(10, 4, 0.5) == 0


def test_safety_stock_empirical_quantile():
    assert safety_stock_empirical(np.arange(101), 0.9) == 90
    assert safety_stock_empirical(-np.ones(10), 0.9) == 0      # never negative


def test_lead_time_error_sums_do_not_cross_windows():
    sums = lead_time_error_sums([np.ones(4), 2 * np.ones(3)], lead_time=3)
    assert sums.tolist() == [3, 3, 6]


def test_forecast_path_uses_latest_window_and_repeats_last_week():
    w0, w1 = np.arange(28.0), 100 + np.arange(28.0)
    f, w = forecast_path([w0, w1], t_next=26, n_days=4)
    assert w == 0 and f.tolist() == [26, 27, 21, 22]          # past the window: repeat last week
    f, w = forecast_path([w0, w1], t_next=28, n_days=2)
    assert w == 1 and f.tolist() == [100, 101]


def test_simulation_perfect_forecast_no_stockouts():
    demand = np.full(56, 10.0)
    sim = simulate_policy(demand, [np.full(28, 10.0)] * 2, safety_stocks=[5.0, 5.0], lead_time=3)
    k = policy_kpis(sim, demand, np.full(56, 2.0), holding_rate_annual=0.365, stockout_multiple=1.0)
    assert k['fill_rate'] == 1.0 and k['stockout_days'] == 0
    assert np.isclose(k['holding_cost'], 0.001 * 2.0 * sim['on_hand'].sum())


def test_simulation_underforecast_causes_lost_sales():
    demand = np.full(56, 10.0)
    sim = simulate_policy(demand, [np.full(28, 5.0)] * 2, safety_stocks=[0.0, 0.0], lead_time=3)
    assert sim['lost'].sum() > 0
    assert np.all(sim['on_hand'] >= 0)
