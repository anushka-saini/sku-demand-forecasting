"""High-confidence break definition (docs/PREREGISTRATION.md, Task 2). All indices are weekly."""
import numpy as np

AGREE_WEEKS = 8          # +/- 56 days
EFFECT_MIN = 0.20        # |post - pre| / pre
PERSIST_MIN = 0.50       # (later - pre) / (post - pre)
WINDOW = 26              # weeks in pre / post / later windows
RAMP_WEEKS = 26          # launch ramp: within this many weeks of monitoring start
END_WEEKS = 26           # near the end: within this many weeks of the last week


def assess_break(values, change_idx, pelt_cps):
    """
    Full-series assessment of one CUSUM break (offline: uses data after the break).
      agreement   : a PELT changepoint within +/- 8 weeks of the change week
      effect      : |post - pre| / pre >= 20%   (pre = 26 weeks before the change week,
                                                 post = weeks 1-26 after it)
      persistence : (later - pre) / (post - pre) >= 0.5   (later = weeks 27-52 after)
    A break with fewer than 52 weeks after the change week is 'not assessable' and
    is not high-confidence.
    """
    y = np.asarray(values, dtype=float)
    c = int(change_idx)
    pre = y[max(0, c - WINDOW):c].mean()
    post_w = y[c + 1:c + 1 + WINDOW]
    later_w = y[c + 1 + WINDOW:c + 1 + 2 * WINDOW]
    assessable = len(later_w) == WINDOW
    post = post_w.mean() if len(post_w) else np.nan
    later = later_w.mean() if assessable else np.nan

    agreement = any(abs(cp - c) <= AGREE_WEEKS for cp in pelt_cps)
    rel = (post - pre) / pre if pre > 0 and np.isfinite(post) else np.nan
    effect = bool(np.isfinite(rel) and abs(rel) >= EFFECT_MIN)
    ratio = (later - pre) / (post - pre) if assessable and post != pre else np.nan
    persistence = bool(np.isfinite(ratio) and ratio >= PERSIST_MIN)
    return {'agreement': agreement, 'effect': effect, 'assessable': assessable, 'persistence': persistence,
            'high_confidence': agreement and effect and assessable and persistence,
            'pre_mean': pre, 'post_mean': post, 'later_mean': later,
            'relative_change': rel, 'persistence_ratio': ratio}


def assess_break_at_origin(values, change_idx, pelt_cps):
    """
    No-lookahead version: `values` and `pelt_cps` must come from data up to the origin only.
      agreement   : as above
      effect      : post = up to 26 weeks after the change week, at least 8 available
      persistence : later = the last 8 weeks before the origin; required only when at
                    least 34 weeks have passed since the change week
    """
    y = np.asarray(values, dtype=float)
    c = int(change_idx)
    n_after = len(y) - 1 - c
    pre = y[max(0, c - WINDOW):c].mean()
    post_w = y[c + 1:c + 1 + WINDOW]
    agreement = any(abs(cp - c) <= AGREE_WEEKS for cp in pelt_cps)
    if len(post_w) < 8 or pre <= 0:
        return {'agreement': agreement, 'effect': False, 'persistence': False, 'high_confidence': False,
                'weeks_since_change': n_after}
    post = post_w.mean()
    effect = abs(post - pre) / pre >= EFFECT_MIN
    if n_after >= 34:
        later = y[-8:].mean()
        persistence = post != pre and (later - pre) / (post - pre) >= PERSIST_MIN
    else:
        persistence = True          # too recent to test; not required
    return {'agreement': agreement, 'effect': bool(effect), 'persistence': bool(persistence),
            'high_confidence': bool(agreement and effect and persistence), 'weeks_since_change': n_after}


def break_position(values, change_idx, direction, monitor_start, pre_mean, post_mean):
    """Where the break falls in the series, and whether it looks like a delisting / stockout artifact."""
    y = np.asarray(values, dtype=float)
    c, n = int(change_idx), len(y)
    last8_zero = bool((y[-8:] == 0).all())
    collapse = bool(pre_mean > 0 and np.isfinite(post_mean) and post_mean < 0.10 * pre_mean)
    return {'weeks_since_first_sale': c,
            'position_share': c / (n - 1),
            'in_launch_ramp': c < monitor_start + RAMP_WEEKS,
            'near_series_end': c > n - 1 - END_WEEKS,
            'last_8_weeks_zero': last8_zero,
            'post_below_10pct_of_pre': collapse,
            'possible_delisting_artifact': bool(direction < 0 and (last8_zero or collapse))}
