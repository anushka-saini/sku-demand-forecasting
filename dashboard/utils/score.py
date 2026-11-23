"""Small pure functions for the Forecaster page: the scorecard, the verdicts and the order advice. No file reads here,
so they are easy to test."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from inventory import lead_time_error_sums, safety_stock_empirical  # noqa: E402

LEAD, LEVEL = 7, 0.95                # lead time in days and the service target used for the advice


def errors(forecast, actual):
    """MAE, RMSE and bias (forecast minus actual, so plus means too high) over the days given."""
    e = np.asarray(forecast, float) - np.asarray(actual, float)
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}


def totals(forecast, actual):
    """Total forecast, total actual and the gap in percent (nan if nothing sold)."""
    f, a = float(np.sum(forecast)), float(np.sum(actual))
    return f, a, (f - a) / a if a > 0 else np.nan


def coverage(actual, lo, hi):
    """Days inside the band, as a flag per day."""
    a = np.asarray(actual, float)
    return (a >= np.asarray(lo, float)) & (a <= np.asarray(hi, float))


def verdict(mae_model, mae_base):
    """'beat', 'did not beat' or 'tied' on MAE."""
    if np.isclose(mae_model, mae_base):
        return "tied"
    return "beat" if mae_model < mae_base else "did not beat"


def closer(guess, forecast_total, actual_total):
    """Who was nearer the real total."""
    g, m = abs(guess - actual_total), abs(forecast_total - actual_total)
    return "you" if g < m else "the model" if m < g else "tie"


def worst_days(forecast, actual, k=3):
    """Positions of the k days with the biggest miss."""
    e = np.abs(np.asarray(forecast, float) - np.asarray(actual, float))
    return list(np.argsort(-e, kind="stable")[:k])


def order_advice(forecast, past_errors, on_hand=0.0):
    """Demand over the lead time, a safety stock from the item's own past lead-time errors, and what to order.
    past_errors: one array of daily (actual - forecast) per earlier window. A simulation, not a measured result."""
    need = float(np.sum(np.asarray(forecast, float)[:LEAD]))
    sums = lead_time_error_sums(past_errors, LEAD)
    ss = float(safety_stock_empirical(sums, LEVEL)) if len(sums) else 0.0
    level = need + ss
    return {"need": need, "safety": ss, "level": level, "order": max(0.0, level - float(on_hand)), "n": len(sums)}
