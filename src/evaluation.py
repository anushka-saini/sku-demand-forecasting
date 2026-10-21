"""Scoring helpers shared by the classical, machine-learning and comparison notebooks."""
import numpy as np
import pandas as pd

from metrics import all_errors, mase_scale, paired_test, holm_adjust

METRICS = ["rmse", "mae", "mape", "smape", "wape", "mase"]


def mase_scales(daily, origins):
    """MASE denominator per item and origin: in-sample seasonal-naive error on all history up to the origin."""
    rows = [{"item_id": i, "origin": o, "mase_scale": mase_scale(d.loc[d["date"] <= o, "sales"].to_numpy())}
            for i, d in daily.items() for o in origins]
    return pd.DataFrame(rows)


def score(fc, scales):
    """Every metric for each item, origin and model. fc needs item_id, origin, model, forecast, actual."""
    sc = scales.set_index(["item_id", "origin"])["mase_scale"]
    rows = []
    for (item, origin, model), g in fc.groupby(["item_id", "origin", "model"]):
        rows.append({"item_id": item, "origin": origin, "model": model,
                     **all_errors(g["actual"], g["forecast"], sc[(item, origin)])})
    return pd.DataFrame(rows)


def summarise(per_item, models, categories):
    """Median and mean across items of each item's mean score, per category and model."""
    out = []
    for cat in ["ALL"] + categories:
        d = per_item if cat == "ALL" else per_item[per_item["category"] == cat]
        for model in models:
            x = d[d["model"] == model]
            row = {"category": cat, "model": model, "n_items": len(x), "mape_excluded_share": x["mape_excluded"].mean()}
            for k in METRICS:
                row[f"median_{k}"], row[f"mean_{k}"] = x[k].median(), x[k].mean()
            out.append(row)
    return pd.DataFrame(out)


def paired_family(wide, pairs, hypothesis, seed=42):
    """Paired Wilcoxon tests on per-item MASE differences, Holm-corrected within the family."""
    rows = [{"hypothesis": hypothesis, "model_a": a, "model_b": b, **paired_test(wide[a] - wide[b], seed=seed)}
            for a, b in pairs]
    for r, p in zip(rows, holm_adjust([r["p_value"] for r in rows])):
        r["p_holm"] = p
        r["a_better"] = bool(r["median_diff"] < 0 and p < 0.05)
    return rows
