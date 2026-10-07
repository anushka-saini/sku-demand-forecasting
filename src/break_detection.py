import numpy as np
import pandas as pd


def to_weekly(item_df):
    """
    Aggregate one item's daily sales into full Saturday-Friday weeks (Walmart weeks).
    Partial weeks at either end are dropped. Returns a Series indexed by week start.
    """
    df = item_df[['date', 'sales']].copy()
    offset = (df['date'].dt.dayofweek - 5) % 7          # days since Saturday
    df['week_start'] = df['date'] - pd.to_timedelta(offset, unit='D')
    weekly = df.groupby('week_start')['sales'].agg(['sum', 'count'])
    return weekly.loc[weekly['count'] == 7, 'sum'].astype(float)


def cusum_detect(values, burn_in=8, ref_len=52, k=0.5, h=5.0):
    """
    One-sided-pair (two-sided) tabular CUSUM with a fixed reference window.

    values[:burn_in]                      ignored (launch ramp-up)
    values[burn_in:burn_in+ref_len]       reference window -> baseline mean/std
    values[burn_in+ref_len:]              monitored; z = (x - mu_ref) / sd_ref

    S+_t = max(0, S+_{t-1} + z_t - k),  S-_t = max(0, S-_{t-1} - z_t - k)
    Alarm at the first t where S+ or S- exceeds h. The change point estimate is
    the step after the last time that side's statistic was zero (standard CUSUM
    estimator). Only data up to the alarm is used, so detection is online.

    Returns dict with alarm_idx, change_idx, direction (+1/-1), max_stat,
    mu_ref, sd_ref. alarm_idx/change_idx are None if no alarm; returns None if
    the series is too short or the reference window is constant.
    """
    values = np.asarray(values, dtype=float)
    start = burn_in + ref_len
    if len(values) <= start:
        return None
    ref = values[burn_in:start]
    mu, sd = ref.mean(), ref.std(ddof=1)
    if sd == 0:
        return None

    z = (values[start:] - mu) / sd
    s_pos = s_neg = 0.0
    last_zero_pos = last_zero_neg = -1
    max_stat = 0.0
    for t, zt in enumerate(z):
        s_pos = max(0.0, s_pos + zt - k)
        s_neg = max(0.0, s_neg - zt - k)
        if s_pos == 0.0:
            last_zero_pos = t
        if s_neg == 0.0:
            last_zero_neg = t
        max_stat = max(max_stat, s_pos, s_neg)
        if s_pos > h or s_neg > h:
            direction = 1 if s_pos > h else -1
            last_zero = last_zero_pos if direction == 1 else last_zero_neg
            return {'alarm_idx': start + t, 'change_idx': start + last_zero + 1,
                    'direction': direction, 'max_stat': max_stat, 'mu_ref': mu, 'sd_ref': sd}
    return {'alarm_idx': None, 'change_idx': None, 'direction': 0,
            'max_stat': max_stat, 'mu_ref': mu, 'sd_ref': sd}


def cusum_max_stat_batch(z, k=0.5):
    """
    Max over time of max(S+, S-) for many standardized series at once.
    z: array (n_series, n_time). Used to calibrate h on surrogate series.
    """
    s_pos = np.zeros(z.shape[0])
    s_neg = np.zeros(z.shape[0])
    best = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        s_pos = np.maximum(0.0, s_pos + z[:, t] - k)
        s_neg = np.maximum(0.0, s_neg - z[:, t] - k)
        best = np.maximum(best, np.maximum(s_pos, s_neg))
    return best


def block_bootstrap(ref, n_out, block_len, rng):
    """Moving-block bootstrap: resample contiguous blocks of `ref` to length n_out."""
    n_blocks = int(np.ceil(n_out / block_len))
    starts = rng.integers(0, len(ref) - block_len + 1, size=n_blocks)
    return np.concatenate([ref[s:s + block_len] for s in starts])[:n_out]


def surrogate_max_stats(weekly_by_item, burn_in=8, ref_len=52, k=0.5,
                        n_surrogates=20, block_len=4, seed=42):
    """
    Break-free null distribution of the CUSUM max statistic.

    For each item, build surrogates by block-bootstrapping its own reference
    window to the length of its monitoring period. Surrogates keep the item's
    level, variance and short-range autocorrelation but contain no level shift
    by construction. Each surrogate is standardized with its own resampled
    reference window, mirroring how real series are processed.
    Returns a 1D array of max statistics (n_items * n_surrogates).
    """
    rng = np.random.default_rng(seed)
    out = []
    for values in weekly_by_item.values():
        values = np.asarray(values, dtype=float)
        start = burn_in + ref_len
        if len(values) <= start:
            continue
        ref = values[burn_in:start]
        n_mon = len(values) - start
        sims = np.array([block_bootstrap(ref, ref_len + n_mon, block_len, rng)
                         for _ in range(n_surrogates)])
        sim_ref, sim_mon = sims[:, :ref_len], sims[:, ref_len:]
        sd = sim_ref.std(axis=1, ddof=1)
        ok = sd > 0
        z = (sim_mon[ok] - sim_ref[ok].mean(axis=1, keepdims=True)) / sd[ok, None]
        out.append(cusum_max_stat_batch(z, k=k))
    return np.concatenate(out)


def run_cusum_on_candidates(long_df, candidates, burn_in=8, ref_len=52, k=0.5, h=5.0):
    """Run fixed-reference weekly CUSUM across a list of item_ids."""
    results = []
    grouped = dict(tuple(long_df[long_df['item_id'].isin(candidates)].groupby('item_id')))
    for item_id in candidates:
        weekly = to_weekly(grouped[item_id].sort_values('date'))
        res = cusum_detect(weekly.to_numpy(), burn_in=burn_in, ref_len=ref_len, k=k, h=h)
        row = {'item_id': item_id, 'n_weeks': len(weekly), 'break_detected': False,
               'change_week': pd.NaT, 'alarm_week': pd.NaT, 'direction': 0,
               'detection_delay_weeks': np.nan, 'max_stat': np.nan,
               'mu_ref': np.nan, 'mu_post_alarm': np.nan, 'shift_in_sd': np.nan}
        if res is not None:
            row.update(max_stat=res['max_stat'], mu_ref=res['mu_ref'])
            if res['alarm_idx'] is not None:
                c, a = res['change_idx'], res['alarm_idx']
                post = weekly.iloc[c:a + 1].mean()   # only data up to the alarm
                row.update(break_detected=True,
                           change_week=weekly.index[c], alarm_week=weekly.index[a],
                           direction=res['direction'], detection_delay_weeks=a - c,
                           mu_post_alarm=post,
                           shift_in_sd=(post - res['mu_ref']) / res['sd_ref'])
        results.append(row)
    return pd.DataFrame(results)


def cusum_breaks_online(values, burn_in=8, ref_len=52, k=1.0, h=5.0):
    """
    Repeated CUSUM: after each alarm, restart with a new reference window that
    begins at the estimated change point (no burn-in), so later breaks can be
    found. Pass only data up to the forecast origin and nothing after it is used.
    Returns a list of (change_idx, alarm_idx) in chronological order.
    """
    values = np.asarray(values, dtype=float)
    breaks, offset, b = [], 0, burn_in
    while True:
        res = cusum_detect(values[offset:], burn_in=b, ref_len=ref_len, k=k, h=h)
        if res is None or res['alarm_idx'] is None:
            return breaks
        breaks.append((offset + res['change_idx'], offset + res['alarm_idx']))
        offset, b = offset + res['change_idx'], 0


# ---------------------------------------------------------------- PELT cross-check

def pelt_detect(values, burn_in=8, ref_len=52, pen=20.0, min_size=8):
    """
    Offline PELT (l2 cost = mean shifts) on the post-burn-in series, standardized
    with the same reference-window mean/std as CUSUM so the penalty is unit-free.
    Uses ruptures' C implementation (KernelCPD with a linear kernel), which gives
    the same changepoints as rpt.Pelt(model='l2') but ~4000x faster.
    Returns changepoint indices in the coordinates of `values` (end point excluded),
    or None if the series is too short / constant.
    """
    import ruptures as rpt
    values = np.asarray(values, dtype=float)
    start = burn_in + ref_len
    if len(values) <= start:
        return None
    ref = values[burn_in:start]
    sd = ref.std(ddof=1)
    if sd == 0:
        return None
    z = (values[burn_in:] - ref.mean()) / sd
    bkps = rpt.KernelCPD(kernel='linear', min_size=min_size, jump=1).fit(z).predict(pen=pen)
    return [burn_in + b for b in bkps[:-1]]


def surrogate_series(weekly_by_item, burn_in=8, ref_len=52, n_surrogates=10, block_len=4, seed=42):
    """Break-free surrogates (block-bootstrapped reference window), full post-burn-in length."""
    rng = np.random.default_rng(seed)
    out = []
    for values in weekly_by_item.values():
        values = np.asarray(values, dtype=float)
        if len(values) <= burn_in + ref_len:
            continue
        ref = values[burn_in:burn_in + ref_len]
        for _ in range(n_surrogates):
            sim = block_bootstrap(ref, len(values) - burn_in, block_len, rng)
            out.append(np.concatenate([np.full(burn_in, np.nan), sim]))
    return out


def calibrate_pelt_penalty(surrogates, burn_in=8, ref_len=52, target_fp=0.05,
                           lo=1.0, hi=500.0, iters=14):
    """Bisection for the smallest penalty at which <= target_fp of surrogates get any changepoint."""
    def fp(pen):
        hits = [pelt_detect(s, burn_in, ref_len, pen=pen) for s in surrogates]
        return np.mean([h is not None and len(h) > 0 for h in hits])
    for _ in range(iters):
        mid = np.sqrt(lo * hi)                      # bisect on log scale
        if fp(mid) > target_fp:
            lo = mid
        else:
            hi = mid
    return hi, fp(hi)


# ---------------------------------------------------------------- power check

def inject_level_shift(values, t0, size_sd, sd_ref):
    """Add a permanent shift of size_sd * sd_ref from index t0; clip at zero (no negative sales)."""
    out = np.asarray(values, dtype=float).copy()
    out[t0:] = np.maximum(0.0, out[t0:] + size_sd * sd_ref)
    return out


def power_check(weekly_by_item, sizes_sd, h, k=1.0, burn_in=8, ref_len=52,
                pelt_pen=None, min_after=26, n_reps=3, tol=8, seed=42):
    """
    Inject a known level shift (random sign, random date t0 in the monitoring
    period with >= min_after weeks left) into real series and record whether
    CUSUM alarms at/after t0, the delay, and the sign. Series whose unmodified
    CUSUM already alarms before t0 are skipped: data before t0 is identical, so
    that alarm would happen anyway and the injected shift could never be seen.
    If pelt_pen is given, PELT counts as a hit when a changepoint lies within
    +/- tol weeks of t0.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = burn_in + ref_len
    for key, values in weekly_by_item.items():
        values = np.asarray(values, dtype=float)
        if len(values) < start + min_after + 10:
            continue
        base = cusum_detect(values, burn_in, ref_len, k=k, h=h)
        if base is None:
            continue
        for _ in range(n_reps):
            t0 = int(rng.integers(start + 10, len(values) - min_after + 1))
            if base['alarm_idx'] is not None and base['alarm_idx'] < t0:
                continue
            sign = rng.choice([-1, 1])
            for size in sizes_sd:
                x = inject_level_shift(values, t0, sign * size, base['sd_ref'])
                r = cusum_detect(x, burn_in, ref_len, k=k, h=h)
                hit = r['alarm_idx'] is not None and r['alarm_idx'] >= t0
                row = {'item': key, 'size_sd': size, 'sign': sign, 't0': t0,
                       'shift_pct_of_mean': 100 * size * base['sd_ref'] / base['mu_ref'],
                       'cusum_detected': hit,
                       'cusum_delay': r['alarm_idx'] - t0 if hit else np.nan,
                       'cusum_sign_ok': hit and r['direction'] == sign}
                if pelt_pen is not None:
                    cps = pelt_detect(x, burn_in, ref_len, pen=pelt_pen)
                    row['pelt_detected'] = any(abs(c - t0) <= tol for c in cps)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step vs trend

def step_vs_trend_bic(values, burn_in=8, min_size=8):
    """
    Compare a single mean shift (best split, 3 params: two means + location)
    against a linear trend (2 params) on the post-burn-in series using
    Gaussian BIC. Returns (bic_step, bic_trend); lower is better.
    """
    y = np.asarray(values, dtype=float)[burn_in:]
    n = len(y)
    t = np.arange(n)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    sse_trend = (resid ** 2).sum()

    cs, cs2 = np.cumsum(y), np.cumsum(y ** 2)
    taus = np.arange(min_size, n - min_size + 1)
    left = cs2[taus - 1] - cs[taus - 1] ** 2 / taus
    right = (cs2[-1] - cs2[taus - 1]) - (cs[-1] - cs[taus - 1]) ** 2 / (n - taus)
    sse_step = (left + right).min()

    bic = lambda sse, p: n * np.log(max(sse, 1e-12) / n) + p * np.log(n)
    return bic(sse_step, 3), bic(sse_trend, 2)


# ---------------------------------------------------------------- event alignment

def weekly_ca_prices(prices, calendar, item_ids):
    """Median CA sell price per item per Walmart week, indexed by week start (Saturday)."""
    p = prices[prices['store_id'].str.startswith('CA') & prices['item_id'].isin(item_ids)]
    p = p.groupby(['item_id', 'wm_yr_wk'])['sell_price'].median().reset_index()
    week_start = calendar.groupby('wm_yr_wk')['date'].min()
    p['week_start'] = p['wm_yr_wk'].map(week_start)
    return p.pivot(index='week_start', columns='item_id', values='sell_price').sort_index()


def sustained_price_changes(price_series, min_change=0.05, hold_weeks=4, tol=0.02):
    """
    Weeks where the price changes and differs by >= min_change from the median
    of the previous hold_weeks weeks, then stays within +/- tol of the new price
    for hold_weeks weeks. Comparing to the prior median (not just last week)
    stops the end of a short promotion being counted as a price change.
    Returns a DataFrame of week_start and pct_change.
    """
    s = price_series.dropna()
    vals, idx = s.to_numpy(), s.index
    rows = []
    for i in range(1, len(vals) - hold_weeks + 1):
        if vals[i] == vals[i - 1]:
            continue
        change = vals[i] / np.median(vals[max(0, i - hold_weeks):i]) - 1
        if abs(change) < min_change:
            continue
        held = np.abs(vals[i:i + hold_weeks] / vals[i] - 1) <= tol
        if held.all():
            rows.append({'week_start': idx[i], 'pct_change': change})
    return pd.DataFrame(rows, columns=['week_start', 'pct_change'])


def alignment_vs_chance(break_weeks, event_weeks, window_weeks, candidate_weeks,
                        n_draws=200, seed=42):
    """
    Share of breaks with an event within +/- window_weeks, vs the share expected
    if break dates were placed at random among each item's candidate weeks.
    break_weeks: dict item -> Timestamp; event_weeks: dict item -> DatetimeIndex;
    candidate_weeks: dict item -> DatetimeIndex (that item's monitoring weeks).
    """
    rng = np.random.default_rng(seed)
    win = pd.Timedelta(weeks=window_weeks)
    observed, chance = [], []
    for item, bw in break_weeks.items():
        ev = event_weeks.get(item, pd.DatetimeIndex([]))
        near = lambda w: len(ev) > 0 and np.abs(ev - w).min() <= win
        observed.append(near(bw))
        cand = candidate_weeks[item]
        draws = cand[rng.integers(0, len(cand), size=n_draws)]
        chance.append(np.mean([near(w) for w in draws]))
    return np.array(observed), np.array(chance)


def validate_against_christmas(break_df, long_df):
    """
    Sanity check: verify Dec 25 reads as anomalous (zero-sales) across items/years.
    """
    checks = []
    for item_id in break_df['item_id'].unique():
        item_df = long_df[long_df['item_id'] == item_id]
        christmases = item_df[item_df['date'].dt.strftime('%m-%d') == '12-25']
        for _, row in christmases.iterrows():
            checks.append({
                'item_id': item_id,
                'year': row['date'].year,
                'sales_on_christmas': row['sales'],
                'is_zero': row['sales'] == 0
            })
    return pd.DataFrame(checks)
