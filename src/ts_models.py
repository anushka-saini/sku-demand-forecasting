"""Time-series contenders and building blocks for 07c to 07h, on the same backtest as 07a.

Every wrapper takes the training sales (already cut at the origin) and returns (forecast, converged, fell_back),
like classical.sarima_forecast. A fit that crashes or gives a non-finite forecast falls back to seasonal naive, and
the fallback is counted. backtest() turns that into rows in the same long format as classical_forecasts.parquet.

Tuning only ever sees the validation origins (VAL_ORIGINS). for_tuning() is the one way the notebooks get scores
to choose from, and it refuses anything that touches a scored origin.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import EVAL_ORIGINS, FIT_DAYS, HORIZON, ORIGINS
from evaluation import METRICS, score
from forecasting import seasonal_naive
from stationarity import outlier_cut

M = 7
H = HORIZON

# ---------------------------------------------------------------- origins

# the warm-up origin plus two earlier ones, each 28 days apart; their windows end on 2015-12-06, before any scored day
VAL_ORIGINS = [ORIGINS[0] - pd.Timedelta(days=H * k) for k in (2, 1, 0)]
ALL_ORIGINS = VAL_ORIGINS[:2] + list(ORIGINS)          # 9 origins: 3 for tuning, 6 scored


def for_tuning(df, col="origin"):
    """Only the rows of the validation origins. Every choice made while tuning goes through this."""
    o = pd.to_datetime(df[col])
    if not o.isin(VAL_ORIGINS).any():
        raise ValueError("no validation origins in this table")
    out = df[o.isin(VAL_ORIGINS)]
    assert not pd.to_datetime(out[col]).isin(EVAL_ORIGINS).any()
    return out


def for_scoring(df, col="origin"):
    """Only the rows of the six scored origins."""
    return df[pd.to_datetime(df[col]).isin(EVAL_ORIGINS)]


# ---------------------------------------------------------------- data changes before a fit

def clip_high(y):
    """Cap training days above Q3 + 3 IQR at that fence (the same fence 03 uses to flag extreme days)."""
    y = np.asarray(y, dtype=float)
    return np.minimum(y, outlier_cut(y))


def transform(y, how=None):
    """Returns (z, back) where back() turns a forecast of z into units. Box-Cox needs positive data, so it works on
    y + 1 with lambda by maximum likelihood on this training window only. The back-transform is the plain inverse,
    so it gives a median forecast, not a mean."""
    y = np.asarray(y, dtype=float)
    if how is None:
        return y, lambda z: z
    if how == "log1p":
        return np.log1p(y), np.expm1
    if how == "boxcox":
        lam = float(stats.boxcox_normmax(y + 1, method="mle"))
        return stats.boxcox(y + 1, lam), lambda z: inv_boxcox(z, lam) - 1
    raise ValueError(how)


def prepare(y, clip=False, how=None, unit=1.0):
    """Clip, rescale (totals go in thousands) and transform. Returns (z, back)."""
    y = np.asarray(y, dtype=float) / unit
    if clip:
        y = clip_high(y)
    z, back = transform(y, how)
    return z, (lambda f: np.asarray(back(np.asarray(f, dtype=float)), dtype=float) * unit)


def finish(fc, y, h, ok):
    """Clip at 0, or fall back to seasonal naive if the forecast is not finite."""
    fc = np.asarray(fc, dtype=float)
    if len(fc) != h or not np.all(np.isfinite(fc)):
        return seasonal_naive(y, h), False, True
    return np.maximum(fc, 0.0), ok, False


# ---------------------------------------------------------------- regressors

def fourier(dates, K, period=365.25):
    """K sine and cosine pairs for the yearly pattern, on a fixed day count so train and future line up."""
    t = (pd.DatetimeIndex(dates) - pd.Timestamp("2011-01-29")).days.to_numpy()
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


def calendar_flags(cal, snap_col="snap_CA"):
    """Daily SNAP (California unless told otherwise), event and Christmas closure flags. Christmas is kept out of the
    event flag."""
    c = cal.set_index(pd.to_datetime(cal["date"]))
    out = pd.DataFrame(index=c.index)
    out["snap"] = c[snap_col].astype(float)
    out["christmas"] = ((c.index.month == 12) & (c.index.day == 25)).astype(float)
    out["event"] = (c["event_name_1"].notna() & (c["event_name_1"] != "Christmas")).astype(float)
    return out


def relative_price(weekly_price, cal):
    """Daily price over the product's own average price of the last 365 days. Prices are set by the retailer and
    known ahead, as in the M5 competition, so the future values can be used."""
    wk = cal[["date", "wm_yr_wk"]].assign(date=lambda d: pd.to_datetime(d["date"]))
    week_start = wk.groupby("wm_yr_wk")["date"].min()
    p = pd.Series(weekly_price.to_numpy(), index=weekly_price.index)
    daily = wk.assign(week_start=wk["wm_yr_wk"].map(week_start)).set_index("date")["week_start"].map(p)
    daily = daily.ffill().bfill()
    return (daily / daily.rolling(365, min_periods=1).mean()).fillna(1.0)


def make_exog(dates, flags, names, rel_price=None):
    """The regressor matrix for these dates. names can hold 'fourierK', 'christmas', 'snap', 'event', 'rel_price'."""
    dates = pd.DatetimeIndex(dates)
    parts = []
    for n in names:
        if n.startswith("fourier"):
            parts.append(fourier(dates, int(n[7:])))
        elif n == "rel_price":
            parts.append(rel_price.reindex(dates).fillna(1.0).rename("rel_price").to_frame())
        else:
            parts.append(flags[[n]].reindex(dates).fillna(0.0))
    return pd.concat(parts, axis=1).to_numpy(dtype=float) if parts else None


# ---------------------------------------------------------------- models

def drop_constant(ex, ex_f=None):
    """Drop regressors that do not vary in the training window (a product whose price never changed): after
    differencing they carry nothing and make the fit fail. Returns (ex, ex_f, kept column positions)."""
    if ex is None:
        return None, None, []
    keep = np.flatnonzero(np.std(ex, axis=0) > 0)
    if len(keep) == 0:
        return None, None, []
    return ex[:, keep], (None if ex_f is None else ex_f[:, keep]), list(keep)


def trend_for(d, D):
    """Same rule as 07a: a constant only when nothing is differenced."""
    return "c" if d == 0 and D == 0 else "n"


def fit_sarimax(z, order, sorder=(0, 0, 0, M), exog=None, maxiter=None):
    """One (S)ARIMA(X) fit. maxiter=None is the statsmodels default (lbfgs, 50 steps), exactly as in 07a."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(z, exog=exog, order=order, seasonal_order=sorder, trend=trend_for(order[1], sorder[1]))
        res = mod.fit(disp=False) if maxiter is None else mod.fit(disp=False, maxiter=maxiter)
    return res, bool(res.mle_retvals.get("converged", True))


FALLBACK_ORDER = ((1, None, 1), (0, None, 1))      # (1, d, 1)(0, D, 1) when the chosen order does not converge


def sarima_fc(y, order, sorder=(0, 0, 0, M), h=H, how=None, clip=False, exog=None, exog_f=None, fix=False, unit=1.0):
    """SARIMA(X) forecast with the optional tweaks. fix=True is the convergence fix: lbfgs with up to 500 steps, and
    if that still does not converge, the simpler fallback order. Returns (forecast, converged, fell_back)."""
    y = np.asarray(y, dtype=float)
    exog, exog_f, _ = drop_constant(exog, exog_f)
    try:
        z, back = prepare(y, clip, how, unit)
        res, ok = fit_sarimax(z, order, sorder, exog, maxiter=500 if fix else None)
        if fix and not ok:
            o2 = (FALLBACK_ORDER[0][0], order[1], FALLBACK_ORDER[0][2])
            s2 = (FALLBACK_ORDER[1][0], sorder[1], FALLBACK_ORDER[1][2], M)
            res2, ok2 = fit_sarimax(z, o2, s2, exog, maxiter=500)
            if ok2:
                res, ok = res2, True
        fc = back(res.forecast(h, exog=exog_f))
    except Exception:
        return seasonal_naive(y, h), False, True
    return finish(fc, y, h, ok)


def ets_fc(y, h=H, how=None, clip=False, unit=1.0):
    """Damped Holt-Winters: additive damped trend and additive weekly season, with an optional transform (how)."""
    y = np.asarray(y, dtype=float)
    try:
        z, back = prepare(y, clip, how, unit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(z, trend="add", damped_trend=True, seasonal="add", seasonal_periods=M,
                                       initialization_method="estimated").fit()
        ok = bool(res.mle_retvals.get("success", True)) if isinstance(res.mle_retvals, dict) else \
            bool(getattr(res.mle_retvals, "success", True))
        fc = back(res.forecast(h))
    except Exception:
        return seasonal_naive(y, h), False, True
    return finish(fc, y, h, ok)


def theta_fc(y, h=H, clip=False, unit=1.0):
    """statsmodels ThetaModel with the weekly period. It tests for seasonality and deseasonalises when the test says
    so, multiplicatively unless the data has zeros (method 'auto'), as the method prescribes."""
    from statsmodels.tsa.forecasting.theta import ThetaModel
    y = np.asarray(y, dtype=float)
    try:
        z, back = prepare(y, clip, None, unit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ThetaModel(z, period=M, deseasonalize=True, method="auto").fit()
            fc = back(np.asarray(res.forecast(h)))
    except Exception:
        return seasonal_naive(y, h), False, True
    return finish(fc, y, h, True)


def prophet_fc(dates, y, h=H, regs=None, regs_f=None, clip=False, yearly="auto"):
    """Prophet with weekly and yearly seasonality and optional extra regressors (DataFrames with one column each)."""
    import logging
    for n in ["cmdstanpy", "prophet"]:
        logging.getLogger(n).disabled = True
    from prophet import Prophet
    y = np.asarray(y, dtype=float)
    try:
        z, back = prepare(y, clip)
        dates = pd.DatetimeIndex(dates)
        tr = pd.DataFrame({"ds": dates, "y": z})
        fut = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=h)})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=yearly, daily_seasonality=False)
        if regs is not None:
            for c in regs.columns:
                m.add_regressor(c)
                tr[c] = regs[c].to_numpy()
                fut[c] = regs_f[c].to_numpy()
        m.fit(tr)
        fc = back(m.predict(fut)["yhat"].to_numpy())
    except Exception:
        return seasonal_naive(y, h), False, True
    return finish(fc, y, h, True)


def arma_grid(y, ps, qs):
    """AICc for every ARMA(p, q) in the grid, d = 0 with a constant, default optimiser.
    Returns a DataFrame with p, q, aicc and converged (NaN aicc if the fit failed)."""
    rows = []
    for p in ps:
        for q in qs:
            try:
                res, ok = fit_sarimax(np.asarray(y, dtype=float), (p, 0, q))
                rows.append({"p": p, "q": q, "aicc": float(res.aicc), "converged": ok})
            except Exception:
                rows.append({"p": p, "q": q, "aicc": np.nan, "converged": False})
    return pd.DataFrame(rows)


def combine(fc, members, weights=None, name="combination"):
    """Average the members' daily forecasts. weights: None (equal), {model: w} or a DataFrame with item_id, model, w."""
    x = fc[fc["model"].isin(members)].copy()
    if weights is None:
        x["w"] = 1.0
    elif isinstance(weights, dict):
        x["w"] = x["model"].map(weights)
    else:
        x = x.merge(weights[["item_id", "model", "w"]], on=["item_id", "model"])
    x["wf"] = x["w"] * x["forecast"]
    g = x.groupby(["item_id", "origin", "date"], as_index=False)[["wf", "w"]].sum()
    g["forecast"] = g["wf"] / g["w"]
    n = x.groupby(["item_id", "origin", "date"])["model"].nunique().to_numpy()
    assert (n == len(members)).all(), "every member needs a forecast for every day"
    return g.drop(columns=["wf", "w"]).assign(model=name, converged=True, fallback=False)


# ---------------------------------------------------------------- backtest

def window(y, dates, origin, n=FIT_DAYS):
    """Training sales and dates up to the origin, the last n days (n=None keeps all history)."""
    k = pd.DatetimeIndex(dates).searchsorted(origin, side="right")
    lo = 0 if n is None else max(0, k - n)
    return np.asarray(y, dtype=float)[lo:k], pd.DatetimeIndex(dates)[lo:k]


def backtest(item, dates, y, origins, model, spec, flags=None, rel_price=None, h=H):
    """One series, one model spec, every origin. spec['kind'] is naive, ets, sarima, theta or prophet; the rest are
    that wrapper's options. Regressors are built from flags and rel_price for spec['exog'] (names for make_exog)."""
    spec = dict(spec)
    kind = spec.pop("kind")
    names = spec.pop("exog", [])
    n_days = spec.pop("fit_days", FIT_DAYS)
    rows = []
    for o in origins:
        yt, dt = window(y, dates, o, n_days)
        fd = pd.date_range(o + pd.Timedelta(days=1), periods=h)
        if kind == "naive":
            fc, ok, fb = seasonal_naive(yt, h), True, False
        elif kind == "ets":
            fc, ok, fb = ets_fc(yt, h, **spec)
        elif kind == "theta":
            fc, ok, fb = theta_fc(yt, h, **spec)
        elif kind == "sarima":
            ex = make_exog(dt, flags, names, rel_price) if names else None
            ex_f = make_exog(fd, flags, names, rel_price) if names else None
            fc, ok, fb = sarima_fc(yt, h=h, exog=ex, exog_f=ex_f, **spec)
        elif kind == "prophet":
            regs = regs_f = None
            if names:
                cols = lambda d: pd.DataFrame(make_exog(d, flags, names, rel_price), columns=exog_columns(names))
                regs, regs_f = cols(dt), cols(fd)
            fc, ok, fb = prophet_fc(dt, yt, h, regs, regs_f, **spec)
        else:
            raise ValueError(kind)
        rows += [{"item_id": item, "origin": o, "date": d, "model": model, "forecast": float(f),
                  "converged": ok, "fallback": fb} for d, f in zip(fd, fc)]
    return rows


def exog_columns(names):
    out = []
    for n in names:
        out += [f"{a}{k}" for k in range(1, int(n[7:]) + 1) for a in ("sin", "cos")] if n.startswith("fourier") else [n]
    return out


def run_to_file(item, dates, y, origins, model, spec, path, flags=None, rel_price=None):
    """backtest() for one item, saved so a rerun skips it (used by the joblib workers)."""
    path = Path(path)
    if path.exists():
        return item
    df = pd.DataFrame(backtest(item, dates, y, origins, model, spec, flags, rel_price))
    df.attrs = {}
    df.to_parquet(path, index=False)
    return item


def run_model(model, spec, series, origins, folder, flags=None, prices=None, n_jobs=10, rerun=False):
    """Every series for one model spec, in parallel, one checkpoint file per series.
    spec: a dict, or a function item -> dict when the settings differ by product (like the SARIMA orders).
    series: {item: DataFrame with date and sales}; prices: {item: daily relative price} or None."""
    from joblib import Parallel, delayed
    spec_for = spec if callable(spec) else (lambda i: spec)
    folder = Path(folder) / model
    folder.mkdir(parents=True, exist_ok=True)
    first = next(iter(series))
    (folder / "spec.json").write_text(json.dumps({"example": first, "spec": spec_for(first),
                                                  "origins": [str(o.date()) for o in origins]}, indent=1, default=str))
    todo = [i for i in series if rerun or not (folder / f"{i}.parquet").exists()]
    Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(run_to_file)(i, series[i]["date"], series[i]["sales"], origins, model, spec_for(i),
                             folder / f"{i}.parquet", flags, None if prices is None else prices.get(i))
        for i in todo)
    return pd.concat([pd.read_parquet(folder / f"{i}.parquet") for i in series], ignore_index=True)


def load_inputs(items=None):
    """What every time-series notebook starts from: sales, the 150 sample, the 07a orders, the calendar flags and
    the daily relative prices (computed once and cached, since reading sell_prices.csv takes a while)."""
    import ast
    from config import INTERIM, PROCESSED, RAW, TABLES
    long_df = pd.concat([pd.read_csv(INTERIM / f"{c}_candidates.csv", parse_dates=["date"])
                         for c in ["foods", "hobbies", "household"]], ignore_index=True)
    long_df = long_df.sort_values(["item_id", "date"]).reset_index(drop=True)
    long_df["category"] = long_df["item_id"].str.split("_").str[0]
    sample = pd.read_csv(TABLES / "classical_sample_items.csv")
    orders = pd.read_csv(TABLES / "classical_orders.csv")
    orders = orders[orders["level"] == "item"].set_index("series")
    for c in ["sarima_order", "sarima_sorder", "arima_order"]:
        orders[c] = orders[c].map(ast.literal_eval)
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    path = PROCESSED / "ts_relative_price.parquet"
    if not path.exists():
        from break_detection import weekly_ca_prices
        prices = pd.read_csv(RAW / "sell_prices.csv", dtype={"store_id": "category"})
        pw = weekly_ca_prices(prices, cal, long_df["item_id"].unique())
        pd.DataFrame({i: relative_price(pw[i], cal) for i in pw.columns}).to_parquet(path)
    rel = pd.read_parquet(path)
    items = list(sample["item_id"]) if items is None else list(items)
    daily = {i: g[["date", "sales"]].reset_index(drop=True) for i, g in long_df.groupby("item_id")}
    return {"long_df": long_df, "sample": sample, "orders": orders, "cal": cal, "flags": calendar_flags(cal),
            "prices": {i: rel[i] for i in rel.columns}, "series": {i: daily[i] for i in items}, "daily": daily,
            "actuals": long_df[["item_id", "date", "sales"]].rename(columns={"sales": "actual"})}


# ---------------------------------------------------------------- scores

def item_scores(fc, actuals, scales):
    """Every metric per item and origin. fc in the long format; actuals has item_id, date, actual."""
    x = fc.merge(actuals, on=["item_id", "date"])
    return score(x, scales)


def per_item(m, origins):
    """Each item's mean score over these origins."""
    x = m[pd.to_datetime(m["origin"]).isin(origins)]
    return x.groupby(["item_id", "model"], as_index=False)[METRICS].mean()


def medians(pi, metrics=("rmse", "mae", "mape", "smape", "wape", "mase")):
    """Median across items, one row per model."""
    return pi.groupby("model")[list(metrics)].median()


def fit_stats(fc):
    """Share of fits (item-origins) that fell back or did not converge, per model."""
    x = fc.drop_duplicates(["item_id", "origin", "model"]).copy()
    x["not_converged"] = ~x["converged"].astype(bool) & ~x["fallback"].astype(bool)     # as in 07a: kept, but flagged
    return x.groupby("model").agg(fits=("fallback", "size"), fallback_rate=("fallback", "mean"),
                                  not_converged_rate=("not_converged", "mean"))
