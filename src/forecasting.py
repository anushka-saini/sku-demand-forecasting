import warnings

import numpy as np
import pandas as pd

HORIZON = 28


# ---------------------------------------------------------------- local (per-item) models

def seasonal_naive(train, h=HORIZON, m=7):
    """Repeat the last observed week."""
    train = np.asarray(train, dtype=float)
    return np.resize(train[-m:], h)


def ets_forecast(train, h=HORIZON, m=7):
    """
    Holt-Winters ETS: additive damped trend + additive weekly seasonality,
    parameters estimated by statsmodels. Falls back to seasonal naive if the fit
    fails. Forecasts are clipped at zero.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    train = np.asarray(train, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = ExponentialSmoothing(train, trend='add', damped_trend=True, seasonal='add',
                                       seasonal_periods=m, initialization_method='estimated').fit()
            fc = fit.forecast(h)
        if not np.all(np.isfinite(fc)):
            raise ValueError
    except Exception:
        fc = seasonal_naive(train, h, m)
    return np.maximum(fc, 0.0)


def tsb_forecast(train, h=HORIZON, alphas=(0.05, 0.1, 0.2, 0.3), betas=(0.05, 0.1, 0.2, 0.3)):
    """
    Teunter-Syntetos-Babai (TSB) method for intermittent demand: separately
    smooths demand size (alpha, updated only on non-zero days) and demand
    probability (beta, updated every day). Forecast = probability * size, flat
    over the horizon. Smoothing constants chosen by in-sample one-step MSE.
    """
    y = np.asarray(train, dtype=float)
    nz = np.flatnonzero(y)
    if len(nz) == 0:
        return np.zeros(h)
    best = (np.inf, 0.0)
    for a in alphas:
        for b in betas:
            z, p = y[nz[0]], 1.0 / max(1, np.mean(np.diff(nz))) if len(nz) > 1 else 1.0
            sse = 0.0
            for t in range(nz[0] + 1, len(y)):
                sse += (y[t] - p * z) ** 2
                if y[t] > 0:
                    z += a * (y[t] - z)
                    p += b * (1 - p)
                else:
                    p += b * (0 - p)
            if sse < best[0]:
                best = (sse, p * z)
    return np.full(h, best[1])


def is_intermittent(series, adi_cut=1.32):
    """Syntetos-Boylan rule: average inter-demand interval > 1.32 -> intermittent."""
    y = np.asarray(series, dtype=float)
    n_nonzero = np.count_nonzero(y)
    return n_nonzero == 0 or len(y) / n_nonzero > adi_cut


def local_forecasts(item_id, dates, sales, origins, starts, run_tsb=False, h=HORIZON):
    """
    All local-model forecasts for one item across origins and training-window
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
                fitted[key] = {'seasonal_naive': seasonal_naive(train, h), 'ets': ets_forecast(train, h)}
                if run_tsb:
                    fitted[key]['tsb'] = tsb_forecast(train, h)
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


# ---------------------------------------------------------------- global LightGBM

LGB_FEATURES = ['item_code', 'dept_code', 'dow', 'month', 'snap_CA', 'is_event', 'days_since_launch',
                'sell_price', 'price_rel_365', 'lag_28', 'lag_35', 'lag_42', 'lag_49',
                'rmean_7_l28', 'rmean_28_l28', 'rmean_56_l28', 'rstd_28_l28']


def build_lgb_panel(long_df, calendar, price_wk):
    """
    Daily item panel with features that only use sales lagged >= 28 days, so one
    model forecasts all 28 days ahead with no recursion and no lookahead.
    Calendar (SNAP, events) and sell prices are treated as known in advance,
    as in the M5 competition.
    """
    df = long_df[['item_id', 'date', 'sales']].sort_values(['item_id', 'date']).reset_index(drop=True)
    cal = calendar[['date', 'snap_CA', 'event_name_1']].copy()
    cal['is_event'] = cal['event_name_1'].notna().astype(int)
    df = df.merge(cal[['date', 'snap_CA', 'is_event']], on='date', how='left')

    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['days_since_launch'] = (df['date'] - df.groupby('item_id')['date'].transform('min')).dt.days
    df['item_code'] = df['item_id'].astype('category').cat.codes
    df['dept_code'] = df['item_id'].str.rsplit('_', n=1).str[0].astype('category').cat.codes

    week_start = df['date'] - pd.to_timedelta((df['date'].dt.dayofweek - 5) % 7, unit='D')
    prices_long = price_wk.stack().rename('sell_price').reset_index()
    prices_long.columns = ['week_start', 'item_id', 'sell_price']
    df = df.assign(week_start=week_start).merge(prices_long, on=['week_start', 'item_id'], how='left')
    df['sell_price'] = df.groupby('item_id')['sell_price'].ffill()
    df['price_rel_365'] = df['sell_price'] / df.groupby('item_id')['sell_price'].transform(
        lambda s: s.rolling(365, min_periods=28).mean())

    g = df.groupby('item_id')['sales']
    for lag in [28, 35, 42, 49]:
        df[f'lag_{lag}'] = g.shift(lag)
    shifted = g.shift(28)
    sg = shifted.groupby(df['item_id'])
    for w in [7, 28, 56]:
        df[f'rmean_{w}_l28'] = sg.transform(lambda s: s.rolling(w).mean())
    df['rstd_28_l28'] = sg.transform(lambda s: s.rolling(28).std())
    return df.drop(columns='week_start')


LGB_PARAMS = {'objective': 'tweedie', 'tweedie_variance_power': 1.1, 'learning_rate': 0.05,
              'num_leaves': 63, 'min_data_in_leaf': 100, 'feature_fraction': 0.8,
              'lambda_l2': 1.0, 'verbose': -1, 'seed': 42, 'deterministic': True,
              'force_row_wise': True, 'num_threads': 8}


def lgb_forecast(panel, origin, train_start_by_item=None, num_rounds=300, h=HORIZON):
    """
    Train one global LightGBM on rows dated <= origin and forecast origin+1..origin+h.
    train_start_by_item: optional dict item_id -> earliest training date for that
    item (used for the adaptive / placebo variants). Returns DataFrame
    (item_id, date, forecast).
    """
    import lightgbm as lgb
    train = panel[(panel['date'] <= origin) & panel['rmean_56_l28'].notna()]
    if train_start_by_item:
        starts = train['item_id'].map(train_start_by_item)
        train = train[starts.isna() | (train['date'] >= starts)]
    test = panel[(panel['date'] > origin) & (panel['date'] <= origin + pd.Timedelta(days=h))]
    ds = lgb.Dataset(train[LGB_FEATURES], label=train['sales'],
                     categorical_feature=['item_code', 'dept_code'], free_raw_data=True)
    model = lgb.train(LGB_PARAMS, ds, num_boost_round=num_rounds)
    out = test[['item_id', 'date']].copy()
    out['forecast'] = np.maximum(model.predict(test[LGB_FEATURES]), 0.0)
    return out, model
