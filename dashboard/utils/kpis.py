"""Pure functions for the numbers shown in the app (no Streamlit imports, unit-tested)."""
import numpy as np
import pandas as pd

SMALL_N = 20


def verdict(ci_low, ci_high, lower_is_better=True):
    """Classify a difference (new - old) from its 95% CI."""
    if pd.isna(ci_low) or pd.isna(ci_high):
        return "inconclusive (too few items for a confidence interval)"
    if ci_low > 0:
        return "worse" if lower_is_better else "better"
    if ci_high < 0:
        return "better" if lower_is_better else "worse"
    return "no detectable difference"


def headline(tests, model="lightgbm"):
    """One-sentence finding for adaptive vs static, built from the test table."""
    a = tests[(tests["comparison"].str.startswith("adaptive")) & (tests["model"] == model) & (tests["category"] == "ALL")].iloc[0]
    p = tests[(tests["comparison"].str.startswith("placebo")) & (tests["model"] == model) & (tests["category"] == "ALL")].iloc[0]
    v = verdict(a["ci_low"], a["ci_high"])
    lead = {"worse": "Retraining after a detected break did not help: it made forecasts slightly worse",
            "better": "Retraining after a detected break improved forecasts",
            "no detectable difference": "Retraining after a detected break made no detectable difference to forecasts"}.get(
        v, "The effect of retraining after a detected break is inconclusive")
    return (f"{lead} (median per-item MASE change {a['median_diff']:+.4f}, 95% CI {a['ci_low']:+.4f} to "
            f"{a['ci_high']:+.4f}, n = {int(a['n'])} items), while a random training cut on no-break items changed it by "
            f"{p['median_diff']:+.4f} (95% CI {p['ci_low']:+.4f} to {p['ci_high']:+.4f}, n = {int(p['n'])}).")


def per_item_variant_diff(errors, model, variant):
    """
    Per-item mean of (variant - static) MASE over eval origins, as in 09a_lightgbm:
    adaptive -> origins where a break had been detected; placebo -> origins without a break.
    Placebo rows that equal static (no truncation was possible) are dropped.
    """
    e = errors[errors["is_eval_origin"] & (errors["model"] == model)]
    piv = e.pivot_table(index=["item_id", "origin", "has_break"], columns="variant", values="mase").reset_index()
    piv = piv[piv["has_break"]] if variant == "adaptive" else piv[~piv["has_break"]]
    piv["diff"] = piv[variant] - piv["static"]
    if variant == "placebo":
        piv = piv[piv["diff"] != 0]
    return piv.groupby("item_id", as_index=False)["diff"].mean()


def cohens_kappa(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def break_share_by_category(items, flag="cusum_first_break"):
    """Share of in-scope items with a break, per category, with n."""
    s = items[items["in_scope"]]
    out = s.groupby("category").agg(n_items=("item_id", "size"), n_break=(flag, "sum")).reset_index()
    out["share"] = out["n_break"] / out["n_items"]
    return out


def filter_items(items, categories=None, in_scope_only=False, abc=None, xyz=None):
    out = items
    if categories:
        out = out[out["category"].isin(categories)]
    if in_scope_only:
        out = out[out["in_scope"]]
    if abc:
        out = out[out["abc_class"] == abc]
    if xyz:
        out = out[out["xyz_class"] == xyz]
    return out


def abc_xyz_matrix(items):
    """Counts and unit shares per ABC x XYZ cell (rows A,B,C; columns X,Y,Z)."""
    counts = pd.crosstab(items["abc_class"], items["xyz_class"]).reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    units = items.pivot_table(index="abc_class", columns="xyz_class", values="total_sales", aggfunc="sum")
    units = units.reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
    share = units / units.to_numpy().sum() if units.to_numpy().sum() > 0 else units
    return counts, share


def tradeoff(inventory, lead_time):
    """Aggregate pre-computed item simulations to policy x service level (no policy recomputation)."""
    s = inventory[inventory["lead_time"] == lead_time]
    g = s.groupby(["policy", "service_level"]).agg(
        n_items=("item_id", "nunique"), demand=("demand_units", "sum"), lost=("lost_units", "sum"),
        stockout_days=("stockout_days", "sum"), avg_inventory_value=("avg_inventory_value", "sum"),
        holding_cost=("holding_cost", "sum"), stockout_cost=("stockout_cost", "sum"),
        total_cost=("total_cost", "sum"), items_at_risk=("stockout_risk_flag", "sum")).reset_index()
    g["fill_rate"] = 1 - g["lost"] / g["demand"]
    return g


def pct_change(new, old):
    return 100 * (new / old - 1) if old else np.nan


def base_cost_assumptions(inventory):
    """Recover the assumed cost parameters from the simulation table (cost / cost basis)."""
    s = inventory[(inventory["holding_value_days"] > 0) & (inventory["lost_value"] > 0)]
    h = (s["holding_cost"] / s["holding_value_days"]).median()
    m = (s["stockout_cost"] / s["lost_value"]).median()
    return round(float(h), 4), round(float(m), 4)


def is_small(n):
    return n < SMALL_N


# Planner watchlist reason thresholds (DERIVED display rules, shown on the page)
BIAS_FLAG = 0.05          # |forecast / actual - 1| above this is flagged as under- or over-forecast
BAND_FLAG = 0.80          # 90% band covering fewer than this share of days is flagged as "band misses often"


def build_watchlist(cal_items, item_bias, item_cov, items, hc_items, method, service_level,
                    model="lightgbm", variant="static"):
    """
    Items whose simulated fill rate fell below the target, for one forecast policy and safety-stock method,
    with plain-language reasons. Only filters and joins pre-computed tables.
    """
    w = cal_items[(cal_items["model"] == model) & (cal_items["variant"] == variant) &
                  (cal_items["method"] == method) & (cal_items["service_level"] == service_level)]
    w = w[w["fill_rate"] < service_level].copy()
    b = item_bias[(item_bias["model"] == model) & (item_bias["variant"] == variant)].set_index("item_id")["bias"]
    cov_col = "coverage90_after" if method == "conformal" else "coverage90_before"
    c = item_cov[(item_cov["model"] == model) & (item_cov["variant"] == variant)].set_index("item_id")[cov_col]
    meta = items.set_index("item_id")
    hc = hc_items.set_index("item_id")["high_confidence"].fillna(False).astype(bool)
    w["category"] = w["item_id"].map(meta["category"])
    w["segment"] = w["item_id"].map(meta["segment"])
    w["forecast_bias"] = w["item_id"].map(b)
    w["band_coverage_90"] = w["item_id"].map(c)
    w["shift_detected"] = w["item_id"].map(meta["break_at_last_origin"]).fillna(False).astype(bool)
    w["high_confidence_shift"] = w["item_id"].map(hc).fillna(False).astype(bool)
    w["intermittent"] = w["item_id"].map(meta["intermittent"]).fillna(False).astype(bool)

    def reasons(r):
        out = []
        if r["forecast_bias"] < -BIAS_FLAG:
            out.append(f"under-forecast {r['forecast_bias']:+.1%}")
        elif r["forecast_bias"] > BIAS_FLAG:
            out.append(f"over-forecast {r['forecast_bias']:+.1%}")
        if r["band_coverage_90"] < BAND_FLAG:
            out.append(f"90% band covers only {r['band_coverage_90']:.0%} of days")
        if r["high_confidence_shift"]:
            out.append("high-confidence demand shift")
        elif r["shift_detected"]:
            out.append("demand shift flagged")
        if r["intermittent"]:
            out.append("intermittent demand")
        return "; ".join(out) if out else "no single cause flagged"

    w["reasons"] = w.apply(reasons, axis=1) if len(w) else []
    cols = ["item_id", "category", "segment", "fill_rate", "lost_units", "stockout_days", "total_cost", "mean_safety_stock",
            "forecast_bias", "band_coverage_90", "shift_detected", "high_confidence_shift", "intermittent", "reasons"]
    return w[cols].sort_values("lost_units", ascending=False).reset_index(drop=True)
