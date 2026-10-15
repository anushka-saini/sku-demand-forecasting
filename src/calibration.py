"""Detector calibration on a seasonality- and autocorrelation-preserving null (docs/PREREGISTRATION.md, Task 1)."""
import numpy as np
import pandas as pd

from break_detection import block_bootstrap, cusum_detect, pelt_detect

STL_PERIOD = 52
MIN_WEEKS_STL = 2 * STL_PERIOD


def stl_parts(values, burn_in=8, period=STL_PERIOD):
    """
    Robust STL of the post-burn-in weekly series. Returns (trend, seasonal, remainder),
    each the length of values[burn_in:], or None if there are fewer than two full cycles.
    """
    from statsmodels.tsa.seasonal import STL
    y = np.asarray(values, dtype=float)[burn_in:]
    if len(y) < 2 * period:
        return None
    res = STL(y, period=period, robust=True).fit()
    return np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid)


def seasonal_null(values, burn_in=8, ref_len=52, n_series=20, block_len=13, rng=None):
    """
    Break-free null series for one item:
        reference-window mean + STL seasonal + moving-block bootstrap of the STL remainder.
    The STL trend is dropped, so there is no level shift or drift by construction, while the
    annual seasonal pattern and the remainder's autocorrelation (within 13-week blocks) are kept.
    Items too short for STL fall back to a 4-week block bootstrap of the reference window.
    Returns (list of arrays aligned with `values`, burn-in filled with NaN; null kind).
    """
    rng = rng or np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    ref_mean = values[burn_in:burn_in + ref_len].mean()
    parts = stl_parts(values, burn_in)
    pad = np.full(burn_in, np.nan)
    out = []
    if parts is None:
        ref = values[burn_in:burn_in + ref_len]
        for _ in range(n_series):
            out.append(np.concatenate([pad, block_bootstrap(ref, len(values) - burn_in, 4, rng)]))
        return out, 'reference_bootstrap'
    _, seasonal, resid = parts
    for _ in range(n_series):
        sim = ref_mean + seasonal + block_bootstrap(resid, len(resid), block_len, rng)
        out.append(np.concatenate([pad, np.maximum(sim, 0.0)]))
    return out, 'seasonal_residual_bootstrap'


def deseasonalize(values, burn_in=8):
    """Series minus its STL seasonal component (offline diagnostic). None if too short."""
    parts = stl_parts(values, burn_in)
    if parts is None:
        return None
    out = np.asarray(values, dtype=float).copy()
    out[burn_in:] = out[burn_in:] - parts[1]
    return out


def cusum_alarm(values, h, k=1.0, burn_in=8, ref_len=52):
    """(alarm_idx or None, direction). None alarm if the series cannot be monitored."""
    res = cusum_detect(values, burn_in=burn_in, ref_len=ref_len, k=k, h=h)
    if res is None or res['alarm_idx'] is None:
        return None, 0
    return res['alarm_idx'], res['direction']


def pelt_changepoints(values, pen, burn_in=8, ref_len=52):
    cps = pelt_detect(values, burn_in=burn_in, ref_len=ref_len, pen=pen)
    return [] if cps is None else cps


def inject(values, t0, size_frac, ref_mean, shape='step', ramp_weeks=26):
    """
    Add a shift of size_frac * ref_mean starting at index t0. 'step' is immediate;
    'drift' ramps linearly to full size over ramp_weeks, then stays flat. Clipped at zero.
    """
    out = np.asarray(values, dtype=float).copy()
    n_after = len(out) - t0
    if shape == 'step':
        profile = np.ones(n_after)
    elif shape == 'drift':
        profile = np.minimum(np.arange(1, n_after + 1) / ramp_weeks, 1.0)
    else:
        raise ValueError(shape)
    out[t0:] = np.maximum(out[t0:] + size_frac * ref_mean * profile, 0.0)
    return out


def adjusted_share(observed, far):
    """Share of items with a real shift after removing chance alarms: (observed - far) / (1 - far)."""
    return (observed - far) / (1 - far) if far < 1 else np.nan


def adjusted_share_ci(item_flags, item_far, n_boot=2000, seed=42):
    """
    Bootstrap over items. item_flags: 0/1 alarm on the real series; item_far: that item's
    false-alarm rate over its null series. Returns dict with observed, far, adjusted, ci_low, ci_high.
    """
    flags, far = np.asarray(item_flags, dtype=float), np.asarray(item_far, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(flags), size=(n_boot, len(flags)))
    obs_b, far_b = flags[idx].mean(axis=1), far[idx].mean(axis=1)
    adj_b = (obs_b - far_b) / (1 - far_b)
    lo, hi = np.quantile(adj_b, [0.025, 0.975])
    return {'n_items': len(flags), 'observed': flags.mean(), 'far': far.mean(),
            'adjusted': adjusted_share(flags.mean(), far.mean()), 'ci_low': lo, 'ci_high': hi}


def injection_trials(null_series, ref_mean, h, pen, sizes, shapes=('step', 'drift'), k=1.0,
                     burn_in=8, ref_len=52, lead_in=10, min_after=39, tol=8, ramp_weeks=26, rng=None):
    """
    Inject known shifts into break-free null hosts. One random t0 and sign per host; hosts
    that already alarm before t0 are skipped (returned with excluded=True). Size 0 is the control.
    """
    rng = rng or np.random.default_rng(42)
    start = burn_in + ref_len
    rows = []
    for host in null_series:
        n = len(host)
        if n < start + lead_in + min_after + 1:
            continue
        t0 = int(rng.integers(start + lead_in, n - min_after + 1))
        sign = int(rng.choice([-1, 1]))
        base_alarm, _ = cusum_alarm(host, h, k, burn_in, ref_len)
        if base_alarm is not None and base_alarm < t0:
            rows.append({'excluded': True})
            continue
        for size in sizes:
            for shape in (shapes if size > 0 else ('none',)):
                x = host if size == 0 else inject(host, t0, sign * size, ref_mean, shape, ramp_weeks)
                alarm, direction = cusum_alarm(x, h, k, burn_in, ref_len)
                hit = alarm is not None and alarm >= t0
                cps = pelt_changepoints(x, pen, burn_in, ref_len)
                hi = t0 + tol if shape != 'drift' else t0 + ramp_weeks + tol
                rows.append({'excluded': False, 'size': size, 'shape': shape, 't0': t0, 'sign': sign,
                             'cusum_detected': hit,
                             'cusum_within_26': hit and alarm - t0 <= 26,
                             'cusum_delay': alarm - t0 if hit else np.nan,
                             'cusum_sign_ok': hit and direction == sign,
                             'pelt_hit': any(t0 - tol <= c <= hi for c in cps)})
    return pd.DataFrame(rows)
