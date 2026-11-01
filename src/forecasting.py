import numpy as np
import pandas as pd

HORIZON = 28


# ---------------------------------------------------------------- local (per-item) models

def seasonal_naive(train, h=HORIZON, m=7):
    """Repeat the last observed week."""
    train = np.asarray(train, dtype=float)
    return np.resize(train[-m:], h)


def is_intermittent(series, adi_cut=1.32):
    """Syntetos-Boylan rule: average inter-demand interval > 1.32 -> intermittent."""
    y = np.asarray(series, dtype=float)
    n_nonzero = np.count_nonzero(y)
    return n_nonzero == 0 or len(y) / n_nonzero > adi_cut


def local_forecasts(item_id, dates, sales, origins, starts, h=HORIZON):
    """
    Seasonal naive forecasts for one item across origins and training-window
    variants. starts: dict origin -> {variant: start_date or None}; a variant
    whose start equals the static start (or is None) reuses the static forecast.
    Returns a list of dict rows (item_id, origin, date, model, variant, forecast).
    """
    dates = pd.DatetimeIndex(dates)
    sales = np.asarray(sales, dtype=float)
    rows = []
    for origin in origins:
        n_train = dates.searchsorted(origin, side='right')
        fc_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=h)
        static_start = dates[0]
        fitted = {}
        for variant, start in starts[origin].items():
            start = static_start if start is None or start <= static_start else start
            key = start
            if key not in fitted:
                i0 = dates.searchsorted(start)
                train = sales[i0:n_train]
                fitted[key] = {'seasonal_naive': seasonal_naive(train, h)}
            for model, fc in fitted[key].items():
                rows.extend({'item_id': item_id, 'origin': origin, 'date': d, 'model': model,
                             'variant': variant, 'forecast': f} for d, f in zip(fc_dates, fc))
    return rows


# ---------------------------------------------------------------- training windows

def training_windows(daily_by_item, origins, h_cusum, k=1.0, burn_in=8, ref_len=52,
                     min_window_days=56, seed=42):
    """
    For every (item, origin): run the online CUSUM on full weeks up to the origin
    only, and define the training-window start for each variant.

      static   : all history (start = None)
      adaptive : items with a break before the origin start at the most recent
                 estimated change week (at least min_window_days of data)
      placebo  : items WITHOUT a break get a random truncation whose length is
                 drawn from the adaptive window lengths at the same origin. This
                 separates "a break was detected" from "trained on less, more
                 recent data".
    Returns a DataFrame with one row per (item, origin).
    """
    from break_detection import to_weekly, cusum_breaks_online
    rng = np.random.default_rng(seed)
    rows = []
    for origin in origins:
        for item, d in daily_by_item.items():
            weekly = to_weekly(d[d['date'] <= origin])
            first = d['date'].iloc[0]
            brks = cusum_breaks_online(weekly.to_numpy(), burn_in, ref_len, k, h_cusum)
            row = {'item_id': item, 'origin': origin, 'first_date': first, 'has_break': len(brks) > 0,
                   'n_breaks': len(brks), 'last_change_week': pd.NaT, 'last_alarm_week': pd.NaT,
                   'adaptive_start': pd.NaT, 'placebo_start': pd.NaT}
            if brks:
                c, a = brks[-1]
                start = min(weekly.index[c], origin - pd.Timedelta(days=min_window_days - 1))
                row.update(last_change_week=weekly.index[c], last_alarm_week=weekly.index[a],
                           adaptive_start=start)
            rows.append(row)
    out = pd.DataFrame(rows)
    out['adaptive_window_days'] = (out['origin'] - out['adaptive_start']).dt.days + 1

    for origin, grp in out.groupby('origin'):
        lengths = grp['adaptive_window_days'].dropna().to_numpy()
        idx = grp.index[~grp['has_break']]
        if len(lengths) == 0:
            continue
        draw = rng.choice(lengths, size=len(idx))
        start = origin - pd.to_timedelta(draw - 1, unit='D')
        start = pd.Series(start, index=idx)
        out.loc[idx, 'placebo_start'] = start.where(start > out.loc[idx, 'first_date'])
    return out
