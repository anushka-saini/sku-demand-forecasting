"""Classical models for Milestone 2: Holt-Winters, ARIMA and SARIMA, backtested like notebook 02."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting import seasonal_naive

M = 7            # weekly season
FIT_DAYS = 730   # last two years at each origin
H = 28           # forecast horizon
NO_SEASON = (0, 0, 0, 0)


def last_days(y, n=FIT_DAYS):
    """Keep only the most recent n days."""
    return np.asarray(y, dtype=float)[-n:]


def trend_for(d, D):
    """SARIMAX has no constant by default, so an undifferenced model would pull the forecast to 0.
    Add a constant only when nothing is differenced (with differencing it would become a drift)."""
    return "c" if d == 0 and D == 0 else "n"


def fit_sarima(y, order, sorder=NO_SEASON, exog=None):
    """Fit one (S)ARIMA(X) quietly. Returns (fitted result, converged?)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y, exog=exog, order=order, seasonal_order=sorder, trend=trend_for(order[1], sorder[1]))
        res = mod.fit(disp=False)
    return res, bool(res.mle_retvals.get("converged", True))


def choose_order(y, d, D, pq=range(3), PQ=range(2), seasonal=True):
    """Try every order in a small grid and keep the one with the lowest AICc.
    Returns (aicc, order, seasonal order), or None if every fit failed."""
    best = None
    for p in pq:
        for q in pq:
            for P in (PQ if seasonal else [0]):
                for Q in (PQ if seasonal else [0]):
                    so = (P, D, Q, M) if seasonal else NO_SEASON
                    try:
                        res, _ = fit_sarima(y, (p, d, q), so)
                    except Exception:
                        continue
                    if np.isfinite(res.aicc) and (best is None or res.aicc < best[0]):
                        best = (float(res.aicc), (p, d, q), so)
    return best


def sarima_forecast(y, order, sorder=NO_SEASON, h=H, exog=None, exog_future=None):
    """Forecast h days. If the fit breaks, use seasonal naive instead and say so.
    Returns (forecast, converged, fell_back)."""
    try:
        res, ok = fit_sarima(y, order, sorder, exog)
        fc = np.asarray(res.forecast(h, exog=exog_future), dtype=float)
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        return np.maximum(fc, 0.0), ok, False
    except Exception:
        return seasonal_naive(y, h), False, True


def fit_hw(y, seasonal="add"):
    """Holt-Winters with an additive trend and a weekly season. Returns (result, shift used).
    A multiplicative season needs positive data, so zeros get one unit added (removed again after)."""
    y = np.asarray(y, dtype=float)
    shift = 1.0 if seasonal == "mul" and (y <= 0).any() else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(y + shift, trend="add", seasonal=seasonal, seasonal_periods=M,
                                   initialization_method="estimated").fit()
    return res, shift


def hw_forecast(y, seasonal="add", h=H):
    """Holt-Winters forecast with the same fallback rule as SARIMA. Returns (forecast, converged, fell_back)."""
    try:
        res, shift = fit_hw(y, seasonal)
        fc = np.asarray(res.forecast(h), dtype=float) - shift
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite forecast")
        ok = bool(getattr(res.mle_retvals, "success", True))
        return np.maximum(fc, 0.0), ok, False
    except Exception:
        return seasonal_naive(y, h), False, True


def pick_orders(y0, pq=range(3), PQ=range(2)):
    """Orders for one series, chosen once on the data up to origin 0."""
    from stationarity import recommend_D, recommend_d
    d, D = recommend_d(y0), recommend_D(y0, M)
    s = choose_order(y0, d, D, pq, PQ, seasonal=True)
    a = choose_order(y0, d, 0, pq, PQ, seasonal=False)
    return {"d": d, "D": D,
            "sarima_order": s[1] if s else (0, d, 0), "sarima_sorder": s[2] if s else (0, D, 0, M),
            "sarima_aicc": s[0] if s else np.nan, "sarima_grid_failed": s is None,
            "arima_order": a[1] if a else (0, d, 0), "arima_aicc": a[0] if a else np.nan,
            "arima_grid_failed": a is None}


def draw_sample(has_break, categories, per_group=25, seed=42):
    """The Milestone 2 item sample: per category, per_group items with a break and per_group without.
    has_break maps item_id -> True/False (break at any scored origin)."""
    rng = np.random.default_rng(seed)
    picked = []
    for cat in categories:
        for flag in [True, False]:
            pool = sorted(i for i, b in has_break.items() if b == flag and i.startswith(cat))
            for item in rng.choice(pool, per_group, replace=False):
                picked.append({"item_id": item, "category": cat, "break_item": flag})
    return pd.DataFrame(picked).sort_values("item_id").reset_index(drop=True)


def run_item_to_file(item, dates, y, origins, path, **kw):
    """Backtest one item and save it, so a rerun can skip it (used by the joblib workers)."""
    import json
    from pathlib import Path
    path = Path(path)
    if path.exists():
        return item
    rows, orders = backtest_series(dates, y, origins, **kw)
    df = pd.DataFrame(rows).assign(item_id=item)
    df.attrs = {}
    df.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                                      for k, v in orders.items()}))
    return item


def backtest_series(dates, y, origins, h=H, pq=range(3), PQ=range(2), models=("hw_add", "hw_mul", "arima", "sarima"),
                    orders=None):
    """Every classical model at every origin, each fitted only on data up to that origin.
    Orders are picked at origins[0] and then kept fixed. Returns (forecast rows, orders)."""
    dates, y = pd.DatetimeIndex(dates), np.asarray(y, dtype=float)
    if orders is None:
        orders = pick_orders(last_days(y[:dates.searchsorted(origins[0], side="right")]), pq, PQ)
    rows = []
    for o in origins:
        yt = last_days(y[:dates.searchsorted(o, side="right")])
        fc_dates = pd.date_range(o + pd.Timedelta(days=1), periods=h)
        for model in models:
            if model == "hw_add":
                fc, ok, fb = hw_forecast(yt, "add", h)
            elif model == "hw_mul":
                fc, ok, fb = hw_forecast(yt, "mul", h)
            elif model == "arima":
                fc, ok, fb = sarima_forecast(yt, orders["arima_order"], NO_SEASON, h)
            else:
                fc, ok, fb = sarima_forecast(yt, orders["sarima_order"], orders["sarima_sorder"], h)
            for d_, f_ in zip(fc_dates, fc):
                rows.append({"origin": o, "date": d_, "model": model, "forecast": float(f_),
                             "converged": ok, "fallback": fb})
    return rows, orders
