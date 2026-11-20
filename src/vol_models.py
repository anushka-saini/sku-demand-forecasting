"""GARCH on forecast residuals, VAR on the category totals, and the Diebold-Mariano test.

GARCH models the variance of the errors. It does not change the point forecast, so it cannot lower MAE or RMSE by
itself; what it gives is a spread for each future day, which is used for prediction intervals and safety stock.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

H = 28
Z = {0.8: 1.2816, 0.95: 1.96}


# ---------------------------------------------------------------- Diebold-Mariano

def dm_test(loss_a, loss_b, h=H):
    """Diebold-Mariano test on a series of loss differences d = loss_a - loss_b (one value per day).
    Newey-West variance with h - 1 lags and the Harvey-Leybourne-Newbold small-sample correction, t distribution.
    Negative statistic: A has the lower loss. Returns (statistic, two-sided p, mean difference)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    dbar = d.mean()
    u = d - dbar
    lrv = np.sum(u * u) / T
    for k in range(1, h):
        w = 1 - k / h
        lrv += 2 * w * np.sum(u[k:] * u[:-k]) / T
    if lrv <= 0:
        return np.nan, np.nan, dbar
    dm = dbar / np.sqrt(lrv / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(dm), df=T - 1)
    return float(dm), float(p), float(dbar)


def daily_loss(fc, model, kind):
    """Loss of one model per calendar day, averaged over items. kind: 'sq' or 'abs'."""
    x = fc[fc["model"] == model]
    e = x["actual"] - x["forecast"]
    e = e ** 2 if kind == "sq" else e.abs()
    return e.groupby(x["date"]).mean().sort_index()


def dm_table(fc, pairs):
    """One row per pair (a, b) and loss. fc has model, date, forecast, actual for the test origins."""
    rows = []
    for a, b in pairs:
        for kind in ("abs", "sq"):
            la, lb = daily_loss(fc, a, kind), daily_loss(fc, b, kind)
            s, p, d = dm_test(la, lb)
            rows.append({"model_a": a, "model_b": b, "loss": "absolute error" if kind == "abs" else "squared error",
                         "days": len(la), "mean_loss_a": la.mean(), "mean_loss_b": lb.mean(), "dm_stat": s,
                         "p_value": p, "a_better": bool(d < 0), "significant_5pct": bool(p < 0.05)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- GARCH on residuals

def garch_vol(e, steps=H):
    """GARCH(1,1) with a zero mean on the residual series e. Returns (sigma path for the next `steps` days,
    fitted conditional sigma, ok flag). A failed fit gives the constant sample std."""
    from arch import arch_model
    e = np.asarray(e, dtype=float)
    const = float(np.std(e, ddof=1)) if len(e) > 2 else 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = arch_model(e, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
            f = r.forecast(horizon=steps, reindex=False).variance.to_numpy()[-1]
            if not np.all(np.isfinite(f)) or r.convergence_flag != 0:
                raise ValueError
            return np.sqrt(f), np.asarray(r.conditional_volatility), True
    except Exception:
        return np.full(steps, const), np.full(len(e), const), False


def arch_lm(e, lags=7):
    """Engle's ARCH-LM test on a residual series: p-value (small = the variance changes over time)."""
    from statsmodels.stats.diagnostic import het_arch
    try:
        return float(het_arch(np.asarray(e, dtype=float), nlags=lags)[1])
    except Exception:
        return np.nan


def garch_intervals(res, scale, test_origins, levels=(0.8, 0.95)):
    """Prediction intervals around the best model's forecast, for every item and scored origin.

    res: DataFrame item_id, origin, date, forecast, actual of the mean model, for ALL origins (validation and
    scored). scale: item -> mean sales (to put the residuals on one scale). For an origin o only residuals of
    windows that ended on or before o are used. Two interval types are compared: constant (sample std of those
    residuals) and GARCH. Returns a long frame, one row per item-day."""
    out = []
    for item, g in res.groupby("item_id"):
        g = g.sort_values("date")
        s = scale[item]
        for o in test_origins:
            past = g[g["origin"] < o]
            now = g[g["origin"] == o]
            if len(now) != H or len(past) < 3 * H:
                continue
            e = ((past["actual"] - past["forecast"]) / s).to_numpy()
            sig_g, _, ok = garch_vol(e)
            sig_c = np.full(H, e.std(ddof=1))
            f = now["forecast"].to_numpy()
            a = now["actual"].to_numpy()
            row = pd.DataFrame({"item_id": item, "origin": o, "date": now["date"].to_numpy(), "h": np.arange(1, H + 1),
                                "forecast": f, "actual": a, "sig_garch": sig_g * s, "sig_const": sig_c * s,
                                "garch_ok": ok})
            out.append(row)
    return pd.concat(out, ignore_index=True)


def interval_summary(iv, levels=(0.8, 0.95)):
    """Coverage, mean width and the interval score (lower is better), constant vs GARCH. The lower end is cut at 0
    because sales cannot be negative."""
    rows = []
    for lev in levels:
        z = Z[lev]
        alpha = 1 - lev
        for kind, col in [("constant std", "sig_const"), ("GARCH(1,1)", "sig_garch")]:
            lo = np.maximum(iv["forecast"] - z * iv[col], 0)
            hi = iv["forecast"] + z * iv[col]
            a = iv["actual"]
            cover = ((a >= lo) & (a <= hi)).mean()
            width = (hi - lo).mean()
            score = (hi - lo) + (2 / alpha) * (lo - a).clip(lower=0) + (2 / alpha) * (a - hi).clip(lower=0)
            rows.append({"nominal": lev, "interval": kind, "coverage": cover, "mean_width": width,
                         "interval_score": score.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- VAR on category totals

def adf_d(x, maxd=2):
    """Number of differences until the ADF test (5%) calls the series stationary."""
    from statsmodels.tsa.stattools import adfuller
    d = 0
    x = np.asarray(x, dtype=float)
    while d < maxd and adfuller(x, autolag="AIC", result_object=False)[1] > 0.05:
        x = np.diff(x)
        d += 1
    return d


def var_forecast(tot, exog, origin, fit_days=730, maxlags=21, h=H):
    """VAR on the category totals up to `origin`. tot: DataFrame (date x category). exog: DataFrame (date x
    regressors known ahead: weekday dummies, SNAP, event). Differencing d is the largest the ADF test asks for
    across the series; lags by AIC up to maxlags. Returns (forecast DataFrame, chosen lag, d, ADF table)."""
    from statsmodels.tsa.api import VAR
    y = tot.loc[:origin].iloc[-fit_days:]
    ds = {c: adf_d(y[c].to_numpy()) for c in y.columns}
    d = max(ds.values())
    z = y.diff(d).dropna() if d else y
    ex = exog.reindex(z.index)
    fd = pd.date_range(origin + pd.Timedelta(days=1), periods=h)
    ex_f = exog.reindex(fd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(z, exog=ex).fit(maxlags=maxlags, ic="aic")
        k = max(res.k_ar, 1)
        f = res.forecast(z.to_numpy()[-k:], steps=h, exog_future=ex_f.to_numpy())
    f = pd.DataFrame(f, index=fd, columns=y.columns)
    if d == 1:
        f = f.cumsum() + y.iloc[-1].to_numpy()
    elif d == 2:
        d1 = y.diff().iloc[-1].to_numpy()
        f = (f.cumsum() + d1).cumsum() + y.iloc[-1].to_numpy()
    return f.clip(lower=0), int(res.k_ar), d, ds
