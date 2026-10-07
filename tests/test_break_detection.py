import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from break_detection import (to_weekly, cusum_detect, surrogate_max_stats, pelt_detect,
                             inject_level_shift, sustained_price_changes)
from segmentation import trim_leading_zeros


def test_trim_leading_zeros_keeps_mid_series_zeros():
    df = pd.DataFrame({
        'item_id': ['A'] * 6 + ['B'] * 3,
        'date': list(pd.date_range('2020-01-01', periods=6)) + list(pd.date_range('2020-01-01', periods=3)),
        'sales': [0, 0, 3, 0, 2, 1, 0, 0, 0],
    })
    out = trim_leading_zeros(df)
    assert out['item_id'].unique().tolist() == ['A']          # all-zero item dropped
    assert out['sales'].tolist() == [3, 0, 2, 1]               # mid-series zero kept


def test_to_weekly_drops_partial_weeks():
    dates = pd.date_range('2016-01-01', '2016-01-20')          # Fri .. Wed
    w = to_weekly(pd.DataFrame({'date': dates, 'sales': 1}))
    assert (w.index.dayofweek == 5).all()                      # weeks start Saturday
    assert w.tolist() == [7.0, 7.0]                            # only 2 full weeks


def test_cusum_no_alarm_on_stable_series():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 10, 200)
    res = cusum_detect(x, burn_in=8, ref_len=52, k=1.0, h=15)
    assert res['alarm_idx'] is None


def test_cusum_detects_level_shift_after_it_happens():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 10, 200)
    x[120:] += 40                                              # 4 sd upward shift at t=120
    res = cusum_detect(x, burn_in=8, ref_len=52, k=1.0, h=15)
    assert res['direction'] == 1
    assert res['alarm_idx'] >= 120                             # no lookahead
    assert abs(res['change_idx'] - 120) <= 3


def test_cusum_ignores_burn_in_and_reference_window():
    x = np.full(120, 100.0) + np.tile([1.0, -1.0], 60)
    x[:5] = 0                                                  # ramp-up inside burn-in
    res = cusum_detect(x, burn_in=8, ref_len=52, k=1.0, h=5)
    assert res['alarm_idx'] is None


def test_cusum_too_short_returns_none():
    assert cusum_detect(np.ones(50), burn_in=8, ref_len=52) is None


def test_surrogate_threshold_controls_false_alarms():
    rng = np.random.default_rng(2)
    series = {i: rng.normal(50, 5, 250) for i in range(30)}
    null_a = surrogate_max_stats(series, k=1.0, n_surrogates=20, seed=1)
    null_b = surrogate_max_stats(series, k=1.0, n_surrogates=20, seed=2)
    h = np.quantile(null_a, 0.95)
    assert 0.02 <= (null_b > h).mean() <= 0.08


def test_pelt_finds_injected_shift_and_nothing_on_flat():
    rng = np.random.default_rng(3)
    x = rng.normal(100, 10, 200)
    assert pelt_detect(x, pen=30) == []
    x[130:] -= 40
    cps = pelt_detect(x, pen=30)
    assert len(cps) == 1 and abs(cps[0] - 130) <= 3


def test_inject_level_shift_clips_at_zero():
    x = np.array([5.0, 5.0, 5.0, 5.0])
    out = inject_level_shift(x, t0=2, size_sd=-3, sd_ref=2.0)
    assert out.tolist() == [5.0, 5.0, 0.0, 0.0]
    assert x.tolist() == [5.0] * 4                             # input not modified


def test_sustained_price_change_ignores_one_week_promo():
    idx = pd.date_range('2015-01-03', periods=12, freq='7D')
    p = pd.Series([2.0, 2.0, 1.5, 2.0, 2.0, 2.0, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4], index=idx)
    ev = sustained_price_changes(p, min_change=0.05, hold_weeks=4)
    assert ev['week_start'].tolist() == [idx[6]]               # promo at idx[2] ignored
    assert np.isclose(ev['pct_change'].iloc[0], 0.2)
