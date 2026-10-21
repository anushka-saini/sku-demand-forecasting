import numpy as np
from scipy import stats


def mase_scale(insample, m=7):
    """In-sample MAE of the seasonal-naive (lag m) forecast. Denominator of MASE."""
    y = np.asarray(insample, dtype=float)
    if len(y) <= m:
        return np.nan
    scale = np.mean(np.abs(y[m:] - y[:-m]))
    return scale if scale > 0 else np.nan


def mase(actual, forecast, insample, m=7):
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006) with a seasonal-naive scale."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return np.mean(np.abs(actual - forecast)) / mase_scale(insample, m)


def rmsse(actual, forecast, insample):
    """Root Mean Squared Scaled Error as in the M5 competition (lag-1 naive scale)."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    y = np.asarray(insample, dtype=float)
    scale = np.mean(np.diff(y) ** 2)
    return np.sqrt(np.mean((actual - forecast) ** 2) / scale) if scale > 0 else np.nan


def wrmsse(rmsse_values, dollar_weights):
    """Weighted RMSSE. Item level only (not the full 12-level M5 hierarchy)."""
    r, w = np.asarray(rmsse_values, dtype=float), np.asarray(dollar_weights, dtype=float)
    ok = ~np.isnan(r) & ~np.isnan(w)
    return np.sum(r[ok] * w[ok]) / np.sum(w[ok])


def paired_test(diff, n_boot=5000, seed=42):
    """
    Wilcoxon signed-rank test on paired differences (e.g. adaptive - static MASE
    per item), with matched-pairs rank-biserial correlation as effect size and a
    bootstrap 95% CI for the median difference. Zero differences are dropped
    (Wilcoxon 'wilcox' method) and counted separately.
    """
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    out = {'n': len(d), 'n_nonzero': len(nz), 'median_diff': np.median(d) if len(d) else np.nan,
           'mean_diff': d.mean() if len(d) else np.nan,
           'share_improved': np.mean(d < 0) if len(d) else np.nan}
    if len(nz) < 10:
        out.update(p_value=np.nan, rank_biserial=np.nan, ci_low=np.nan, ci_high=np.nan)
        return out
    out['p_value'] = stats.wilcoxon(nz).pvalue
    ranks = stats.rankdata(np.abs(nz))
    out['rank_biserial'] = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    out['ci_low'], out['ci_high'] = np.quantile(boots, [0.025, 0.975])
    return out


def standardized_mean_diff(a, b):
    """Cohen's d style standardized difference between two groups (for balance checks)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


# ---------------------------------------------------------------- point-forecast errors

def rmse(actual, forecast):
    """Root mean squared error, in units sold."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mae(actual, forecast):
    """Mean absolute error, in units sold."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(a - f)))


def mape(actual, forecast):
    """Mean absolute percentage error over days with sales > 0 (a zero day would divide by zero)."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    keep = a > 0
    if not keep.any():
        return np.nan
    return float(np.mean(np.abs(a[keep] - f[keep]) / a[keep]))


def mape_excluded(actual):
    """Share of days left out of MAPE because nothing sold."""
    a = np.asarray(actual, dtype=float)
    return float(np.mean(a <= 0)) if len(a) else np.nan


def smape(actual, forecast):
    """Symmetric MAPE, 0 to 2. Days where both actual and forecast are 0 are skipped."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    den = np.abs(a) + np.abs(f)
    keep = den > 0
    if not keep.any():
        return np.nan
    return float(np.mean(2 * np.abs(a[keep] - f[keep]) / den[keep]))


def wape(actual, forecast):
    """Total absolute error divided by total sales, so busy days count more."""
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    total = np.sum(np.abs(a))
    return float(np.sum(np.abs(a - f)) / total) if total > 0 else np.nan


def all_errors(actual, forecast, scale):
    """Every point metric for one forecast window. scale is the MASE denominator."""
    m = mae(actual, forecast)
    return {'rmse': rmse(actual, forecast), 'mae': m, 'mape': mape(actual, forecast),
            'mape_excluded': mape_excluded(actual), 'smape': smape(actual, forecast),
            'wape': wape(actual, forecast), 'mase': m / scale if scale and scale > 0 else np.nan}


def holm_adjust(p_values):
    """
    Holm step-down adjusted p-values (controls the family-wise error rate).
    NaN inputs stay NaN and are not counted in the family size.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    order = idx[np.argsort(p[idx])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out
