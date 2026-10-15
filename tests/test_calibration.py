import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from calibration import (adjusted_share, adjusted_share_ci, deseasonalize, inject, injection_trials,
                         seasonal_null, stl_parts)


def seasonal_series(n=260, level=100.0, amp=30.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return level + amp * np.sin(2 * np.pi * t / 52) + rng.normal(0, noise, n)


def test_stl_needs_two_cycles():
    assert stl_parts(np.ones(8 + 100), burn_in=8) is None
    assert stl_parts(seasonal_series(), burn_in=8) is not None


def test_seasonal_null_keeps_seasonality_and_removes_level_shift():
    y = seasonal_series()
    y[150:] += 60                                              # real break in the source series
    nulls, kind = seasonal_null(y, n_series=10, rng=np.random.default_rng(1))
    assert kind == 'seasonal_residual_bootstrap' and len(nulls) == 10
    s = nulls[0]
    assert len(s) == len(y) and np.isnan(s[:8]).all() and (s[8:] >= 0).all()
    # no level shift: second half is not ~60 above the first half, on average over null series
    gap = np.mean([np.nanmean(n[150:]) - np.nanmean(n[8:150]) for n in nulls])
    assert abs(gap) < 20
    # seasonal pattern survives: correlation with the true sinusoid
    t = np.arange(8, len(y))
    assert np.corrcoef(s[8:], np.sin(2 * np.pi * t / 52))[0, 1] > 0.5


def test_seasonal_null_falls_back_for_short_series():
    nulls, kind = seasonal_null(seasonal_series(n=100), n_series=3, rng=np.random.default_rng(1))
    assert kind == 'reference_bootstrap' and len(nulls[0]) == 100


def test_deseasonalize_reduces_seasonal_variance():
    y = seasonal_series(noise=2.0)
    d = deseasonalize(y)
    assert d[8:].std() < 0.5 * y[8:].std()
    assert deseasonalize(np.ones(50)) is None


def test_inject_step_and_drift_shapes():
    base = np.full(100, 50.0)
    step = inject(base, 40, 0.2, ref_mean=50.0, shape='step')
    drift = inject(base, 40, 0.2, ref_mean=50.0, shape='drift', ramp_weeks=10)
    assert (step[:40] == 50).all() and np.allclose(step[40:], 60)
    assert np.isclose(drift[40], 51) and np.isclose(drift[49], 60) and np.allclose(drift[50:], 60)
    assert (inject(base, 40, -2.0, 50.0) >= 0).all()           # clipped at zero
    assert (base == 50).all()                                  # input untouched


def test_adjusted_share_formula_and_ci():
    assert np.isclose(adjusted_share(0.67, 0.25), 0.56)
    assert adjusted_share(0.3, 0.3) == 0
    res = adjusted_share_ci(np.r_[np.ones(60), np.zeros(40)], np.full(100, 0.2))
    assert np.isclose(res['adjusted'], 0.5) and res['ci_low'] < 0.5 < res['ci_high']


def test_injection_trials_large_step_is_detected_and_control_is_not():
    rng = np.random.default_rng(3)
    hosts = [np.concatenate([np.full(8, np.nan), rng.normal(100, 5, 252)]) for _ in range(20)]
    t = injection_trials(hosts, ref_mean=100.0, h=14.0, pen=30.0, sizes=[0.0, 0.3], rng=np.random.default_rng(4))
    ok = t[~t['excluded']]
    power = ok.groupby(['size', 'shape'])['cusum_detected'].mean()
    assert power.loc[(0.3, 'step')] > 0.9
    assert power.loc[(0.0, 'none')] < 0.2
    assert (ok.loc[ok['cusum_detected'], 'cusum_delay'] >= 0).all()   # never before the injection
