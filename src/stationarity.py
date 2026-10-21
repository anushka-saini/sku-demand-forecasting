"""Stationarity tests, outlier counts and decomposition helpers."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

ALPHA = 0.05


def adf_p(y, reg="c"):
    """ADF p-value. Small p means we reject 'the series has a unit root'."""
    return float(adfuller(np.asarray(y, dtype=float), regression=reg, autolag="AIC")[1])


def kpss_p(y, reg="c"):
    """KPSS p-value. Small p means we reject 'the series is stationary'.
    statsmodels only reports p between 0.01 and 0.10 and warns outside that range, which is fine for a 5% test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kpss(np.asarray(y, dtype=float), regression=reg, nlags="auto")[1])


def classify(adf_c, kpss_c, adf_ct, kpss_ct, alpha=ALPHA):
    """Turn the four p-values into one label (rule fixed in docs/PREREGISTRATION_M2.md)."""
    adf_c_rej, kpss_c_rej = adf_c < alpha, kpss_c < alpha
    adf_ct_rej, kpss_ct_rej = adf_ct < alpha, kpss_ct < alpha
    if adf_c_rej and not kpss_c_rej:
        return "stationary"
    if kpss_c_rej and not kpss_ct_rej and adf_ct_rej:
        return "trend-stationary"
    if kpss_c_rej and not adf_ct_rej:
        return "difference-stationary"
    return "conflicting"


def recommend_d(y):
    """Number of ordinary differences (KPSS-based, as in pmdarima's auto_arima)."""
    from pmdarima.arima import ndiffs
    return int(ndiffs(np.asarray(y, dtype=float), test="kpss", max_d=2))


def recommend_D(y, m=7):
    """Number of seasonal differences at period m (OCSB test)."""
    from pmdarima.arima import nsdiffs
    return int(nsdiffs(np.asarray(y, dtype=float), m=m, test="ocsb", max_D=1))


def check_series(y):
    """All stationarity results for one series."""
    y = np.asarray(y, dtype=float)
    p = {"adf_c": adf_p(y, "c"), "kpss_c": kpss_p(y, "c"), "adf_ct": adf_p(y, "ct"), "kpss_ct": kpss_p(y, "ct")}
    p["label"] = classify(p["adf_c"], p["kpss_c"], p["adf_ct"], p["kpss_ct"])
    p["d"] = recommend_d(y)
    p["D"] = recommend_D(y)
    return p


def outlier_cut(y):
    """Upper fence Q3 + 3 * IQR. Only very large days count; we flag them but never change them."""
    q1, q3 = np.percentile(np.asarray(y, dtype=float), [25, 75])
    return q3 + 3 * (q3 - q1)


def count_outliers(y):
    y = np.asarray(y, dtype=float)
    return int(np.sum(y > outlier_cut(y)))


def swing_rule(dates, y, min_rho=0.5):
    """Does the weekly swing grow with the level? If yes, multiplicative seasonality fits better.
    Returns (Spearman rho, p-value, 'multiplicative' or 'additive')."""
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "y": np.asarray(y, dtype=float)})
    df["week"] = df["date"] - pd.to_timedelta((df["date"].dt.dayofweek - 5) % 7, unit="D")
    wk = df.groupby("week")["y"].agg(["mean", "max", "min", "size"])
    wk = wk[wk["size"] == 7]
    rho, p = stats.spearmanr(wk["mean"], wk["max"] - wk["min"])
    choice = "multiplicative" if rho >= min_rho and p < ALPHA else "additive"
    return float(rho), float(p), choice


def fix_christmas(s):
    """Stores close on 25 December, so the total is almost 0 that day. For decomposition only,
    replace it by the average of the same weekday one week before and after."""
    s = s.astype(float)          # the average can be fractional (and astype copies, so the input is untouched)
    for day in s.index[(s.index.month == 12) & (s.index.day == 25)]:
        near = [day - pd.Timedelta(days=7), day + pd.Timedelta(days=7)]
        near = [d for d in near if d in s.index]
        if near:
            s[day] = s[near].mean()
    return s
