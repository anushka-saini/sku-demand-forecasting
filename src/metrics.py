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
