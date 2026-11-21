"""The final comparison: every model scored once on the six test origins, plus the three combinations.

Rules (the same as for the time-series models in 07h):
  * validation origins (ts_models.VAL_ORIGINS) are the only place anything is chosen: top models for the average,
    stacking weights, which classical model feeds the hybrid;
  * the six scored origins (config.EVAL_ORIGINS) are scored once, at the end.
The headline numbers keep the convention of the earlier notebooks: each item's mean over origins, then the median
across the 150 items. Pooled numbers (all item-days together) are reported next to them.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import features as F
import ml_models as M
import ts_models as tm
from config import EVAL_ORIGINS, ORIGINS, TABLES, SEED
from evaluation import mase_scales, score
from metrics import mae, mape, rmse, wape

VAL = tm.VAL_ORIGINS
ALL = tm.ALL_ORIGINS
H = 28
SEEDS = (0, 1, 2)
KEYS = ["item_id", "origin", "date"]

OLD = ["seasonal_naive", "hw_add", "ets_boxcox", "theta", "arma", "arima", "sarima", "sarimax", "prophet", "autots",
       "combination"]
NEW_SINGLE = ["prophet_tuned", "xgboost", "lightgbm", "lstm", "gru", "patchtst"]
NAMES = {"seasonal_naive": "Seasonal naive", "hw_add": "Holt-Winters additive", "ets_boxcox": "Holt-Winters (damped, Box-Cox)",
         "theta": "Theta", "arma": "ARMA(p, q)", "arima": "ARIMA", "sarima": "SARIMA", "sarimax": "SARIMAX",
         "prophet": "Prophet", "autots": "AutoTS", "combination": "Combination (SARIMAX, HW, Theta)",
         "prophet_tuned": "Prophet (tuned, events)", "xgboost": "XGBoost (Optuna)", "lightgbm": "LightGBM (reference)",
         "lstm": "LSTM", "gru": "GRU", "patchtst": "PatchTST", "var": "VAR (top-down to items)",
         "avg_top3": "Average of top 3", "stack": "Stacked (non-negative weights)", "hybrid": "Hybrid (classical + XGBoost residuals)"}
FAMILY = {"seasonal_naive": "Baseline", "hw_add": "Classical", "ets_boxcox": "Classical", "theta": "Classical",
          "arma": "Classical", "arima": "Classical", "sarima": "Classical", "sarimax": "Classical",
          "prophet": "Classical", "autots": "Classical", "combination": "Combination", "prophet_tuned": "Classical",
          "xgboost": "Machine learning", "lightgbm": "Machine learning", "lstm": "Deep learning", "gru": "Deep learning",
          "patchtst": "Deep learning", "var": "Classical", "avg_top3": "Combination", "stack": "Combination",
          "hybrid": "Combination"}
CLASSICAL = [m for m, f in FAMILY.items() if f in ("Classical", "Baseline")]


# ---------------------------------------------------------------- loading

def load_old():
    """Old models at the three validation origins and the six scored ones (07h's saved forecasts, already guarded).
    hw_add was only run at the scored origins, so it has no validation rows and is never picked by validation."""
    t = pd.read_parquet(TABLES / "ts_forecasts.parquet")
    t = t[t["model"].isin(["sarima", "sarimax", "ets", "seasonal_naive", "theta", "prophet", "autots", "combination",
                           "arma", "arima"])]
    t = t.assign(model=t["model"].replace({"ets": "ets_boxcox"}))[KEYS + ["model", "forecast", "actual"]]
    c = pd.read_parquet(TABLES / "comparison_forecasts.parquet")
    hw = c[c["model"] == "hw_add"][KEYS + ["model", "forecast", "actual"]]
    return pd.concat([t, hw], ignore_index=True)


def load_new(daily=None):
    """The new models. Neural nets come back twice: one row per seed, and the mean over the seeds (the one the
    combinations use)."""
    parts, seeds = [], []
    for m in ["prophet_tuned", "xgboost", "lightgbm"]:
        p = M.CACHE / f"{m}_forecasts.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p)[KEYS + ["model", "forecast", "actual"]])
    for m in ["lstm", "gru", "patchtst"]:
        p = M.CACHE / f"{m}_forecasts.parquet"
        if p.exists():
            x = pd.read_parquet(p)
            seeds.append(x[KEYS + ["model", "seed", "forecast", "actual"]])
            parts.append(x.groupby(KEYS + ["model"], as_index=False)[["forecast", "actual"]].mean())
    fc = pd.concat(parts, ignore_index=True)
    if daily is not None:                                   # the same blow-up guard as every other model
        import best_model as bm
        fc, caught = bm.cap_forecasts(fc, daily)
        fc = fc[KEYS + ["model", "forecast", "actual"]]
    else:
        caught = pd.DataFrame()
    return fc, (pd.concat(seeds, ignore_index=True) if seeds else pd.DataFrame()), caught


def tune_seconds():
    """Wall-clock seconds used to tune and fit each new model, read from the files the runs wrote."""
    out = {}
    for m in ["xgboost", "lightgbm"]:
        p = M.CACHE / "times_xgb.csv"
        if p.exists():
            t = pd.read_csv(p, parse_dates=["origin"])
            t = t[(t["model"] == m) & t["origin"].isin(EVAL_ORIGINS)]
            out[m] = float(t["seconds"].sum())
    for m in ["lstm", "gru", "patchtst"]:
        p = M.CACHE / f"{m}_forecasts.parquet"
        if p.exists():
            x = pd.read_parquet(p)
            x = x[x["stage"] == "test"].drop_duplicates(["seed"])
            out[m] = float(x["seconds"].mean())    # one training run per seed
    p = M.CACHE / "prophet_tuned_forecasts.parquet"
    if p.exists():
        x = pd.read_parquet(p)
        out["prophet_tuned"] = (float(x[x["origin"].isin(EVAL_ORIGINS)].drop_duplicates(["item_id", "origin"])["seconds"].sum())
                                if "seconds" in x else np.nan)         # seconds repeat on each of the 28 rows of a fit
    return out


# ---------------------------------------------------------------- scoring

def per_item(fc, scales, origins=EVAL_ORIGINS):
    x = fc[fc["origin"].isin(origins)]
    s = score(x, scales)
    return s.groupby(["item_id", "model"], as_index=False)[["rmse", "mae", "mape", "wape", "smape", "mase"]].mean()


def pooled(fc, origins=EVAL_ORIGINS):
    """All item-days of the given origins together, per model."""
    x = fc[fc["origin"].isin(origins)]
    rows = []
    for m, g in x.groupby("model"):
        rows.append({"model": m, "pooled_mae": mae(g["actual"], g["forecast"]), "pooled_rmse": rmse(g["actual"], g["forecast"]),
                     "pooled_mape_nonzero": mape(g["actual"], g["forecast"]), "pooled_wape": wape(g["actual"], g["forecast"]),
                     "bias": float((g["forecast"] - g["actual"]).mean())})
    return pd.DataFrame(rows).set_index("model")


def val_rank(fc, models):
    """Validation score per model: pooled MAE and RMSE on the three validation origins, and their mean rank."""
    p = pooled(fc[fc["model"].isin(models)], VAL)
    p["rank"] = (p["pooled_mae"].rank() + p["pooled_rmse"].rank()) / 2
    return p.sort_values(["rank", "pooled_rmse"])


# ---------------------------------------------------------------- combinations

def candidates(fc):
    """Single models that have forecasts for all 150 items at every validation and scored origin."""
    n = fc.groupby("model").apply(lambda g: g[["item_id", "origin"]].drop_duplicates().shape[0], include_groups=False)
    full = 150 * len(ALL)
    ok = [m for m in n.index if n[m] == full and m in OLD + NEW_SINGLE and m not in ("combination", "arma", "arima")]
    return sorted(ok)


def wide(fc, models, origins):
    x = fc[fc["model"].isin(models) & fc["origin"].isin(origins)]
    w = x.pivot_table(index=KEYS + ["actual"], columns="model", values="forecast").reset_index()
    return w


def average_top(fc, cands, k=3):
    """Plain average of the k models with the best validation rank. Returns (forecast frame, members)."""
    top = list(val_rank(fc, cands).index[:k])
    w = wide(fc, top, ALL)
    w["forecast"] = w[top].mean(axis=1)
    return w[KEYS + ["actual", "forecast"]].assign(model="avg_top3"), top


def stack(fc, cands):
    """Non-negative least squares on the validation forecasts only; the weights are then fixed for the test."""
    w = wide(fc, cands, VAL)
    coef, _ = nnls(w[cands].to_numpy(), w["actual"].to_numpy())
    weights = pd.Series(coef, index=cands)
    allw = wide(fc, cands, ALL)
    allw["forecast"] = np.clip(allw[cands].to_numpy() @ coef, 0, None)
    return allw[KEYS + ["actual", "forecast"]].assign(model="stack"), weights


def hybrid(fc, rows, base):
    """The classical model `base` plus XGBoost on its residuals. For each scored origin the residual model is fitted
    on the earlier origins only (their forecasts and actuals are all known by then), with the shared features and the
    classical forecast as inputs. Fixed, strongly regularised settings: there are too few origins to tune them."""
    b = fc[fc["model"] == base][KEYS + ["forecast"]].rename(columns={"forecast": "cl"})
    r = rows.merge(b, on=KEYS)
    r["cl_ratio"] = r["cl"] / np.maximum(r["mean28"], 0.1)
    r["res"] = r["y"] - r["cl"]
    cols = F.FEATURES + ["cl", "cl_ratio"]
    params = dict(objective="reg:squarederror", learning_rate=0.05, max_depth=3, min_child_weight=50, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=20.0)
    out = []
    for o in EVAL_ORIGINS:
        tr = r[r["origin"] <= o - pd.Timedelta(days=H)]
        ev = r[r["origin"] == o]
        m = M.make_xgb(params, 150, threads=4)
        m.fit(tr[cols], tr["res"])
        pred = np.clip(ev["cl"].to_numpy() + m.predict(ev[cols]), 0, None)
        out.append(pd.DataFrame({"item_id": ev["item_id"].to_numpy(), "origin": o, "date": ev["date"].to_numpy(),
                                 "actual": ev["y"].to_numpy(), "forecast": pred, "model": "hybrid"}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- VAR on category totals

def category_totals(daily, items):
    cat = lambda i: i.split("_")[0]
    d = pd.concat([daily[i].assign(cat=cat(i)) for i in items])
    return d.pivot_table(index="date", columns="cat", values="sales", aggfunc="sum").sort_index()


def var_exog(cal):
    """Regressors known ahead for the VAR: weekday dummies, SNAP and event flags."""
    c = cal.copy()
    c["date"] = pd.to_datetime(c["date"])
    c = c.set_index("date")
    dow = pd.get_dummies(c.index.dayofweek, prefix="d", drop_first=True, dtype=float)
    dow.index = c.index
    return pd.concat([dow, tm.calendar_flags(cal)[["snap", "event"]]], axis=1)


def time_var(daily, cal, items):
    """Seconds for the six scored VAR fits (the three category series)."""
    import time
    from vol_models import var_forecast
    tot, ex = category_totals(daily, items), var_exog(cal)
    t = time.time()
    for o in EVAL_ORIGINS:
        var_forecast(tot, ex, o)
    return time.time() - t


def var_study(daily, cal, items, origins=ALL, maxlags=21):
    """VAR on the three category totals at each origin. Returns (totals forecasts, per-origin settings, item-level
    forecasts made top-down: the category forecast times the item's share of its category over the last 56 days)."""
    tot = category_totals(daily, items)
    ex = var_exog(cal)
    from vol_models import var_forecast
    rows, sets, item_rows = [], [], []
    for o in origins:
        f, lag, d, ds = var_forecast(tot, ex, o, maxlags=maxlags)
        sets.append({"origin": o, "lag_aic": lag, "differences": d, **{f"adf_d_{k}": v for k, v in ds.items()}})
        for k in f.columns:
            a = tot[k].reindex(f.index)
            rows.append(pd.DataFrame({"series": k, "origin": o, "date": f.index, "forecast": f[k].to_numpy(),
                                      "actual": a.to_numpy(), "model": "var"}))
        for it in items:
            k = it.split("_")[0]
            past = {i: daily[i].loc[daily[i]["date"] <= o, "sales"].iloc[-56:].sum() for i in items if i.split("_")[0] == k}
            share = past[it] / max(sum(past.values()), 1)
            a = daily[it].set_index("date")["sales"].reindex(f.index)
            item_rows.append(pd.DataFrame({"item_id": it, "origin": o, "date": f.index, "forecast": f[k].to_numpy() * share,
                                           "actual": a.to_numpy(), "model": "var"}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sets), pd.concat(item_rows, ignore_index=True)


def totals_of(fc, items_cat):
    """Sum item forecasts to the category totals, per model, origin and date."""
    x = fc.assign(series=fc["item_id"].map(items_cat))
    return x.groupby(["series", "origin", "date", "model"], as_index=False)[["forecast", "actual"]].sum()


# ---------------------------------------------------------------- where models fail

def segment_table(fc, models, cal, daily, breaks):
    """MAE and bias of the given models on groups of item-days (scored origins only)."""
    x = fc[fc["origin"].isin(EVAL_ORIGINS) & fc["model"].isin(models)].copy()
    flags = tm.calendar_flags(cal)
    x["snap"] = flags["snap"].reindex(x["date"]).to_numpy()
    x["event"] = flags["event"].reindex(x["date"]).to_numpy()
    # a zero spell: nothing sold in the 7 days before the origin
    last7 = {}
    for it, d in daily.items():
        for o in EVAL_ORIGINS:
            last7[(it, o)] = float(d.loc[(d["date"] <= o), "sales"].iloc[-7:].sum())
    x["zero_spell"] = [last7[(i, o)] == 0 for i, o in zip(x["item_id"], x["origin"])]
    x["zero_day"] = x["actual"] == 0
    br = breaks.groupby("item_id")["has_break"].max()
    x["break_item"] = x["item_id"].map(br).fillna(False).astype(bool)
    x["weekend"] = pd.DatetimeIndex(x["date"]).dayofweek >= 5
    groups = [("all days", np.ones(len(x), bool)), ("event days", x["event"] == 1), ("no event", x["event"] == 0),
              ("SNAP days", x["snap"] == 1), ("non-SNAP days", x["snap"] == 0), ("weekend", x["weekend"]),
              ("weekday", ~x["weekend"]), ("days with zero sales", x["zero_day"]),
              ("zero week before origin", x["zero_spell"]), ("items with a detected shift", x["break_item"]),
              ("items without a shift", ~x["break_item"])]
    rows = []
    for name, mask in groups:
        g = x[np.asarray(mask)]
        for m, gm in g.groupby("model"):
            rows.append({"segment": name, "model": m, "item_days": len(gm), "mae": mae(gm["actual"], gm["forecast"]),
                         "rmse": rmse(gm["actual"], gm["forecast"]), "bias": float((gm["forecast"] - gm["actual"]).mean()),
                         "mean_actual": float(gm["actual"].mean())})
    return pd.DataFrame(rows)


def horizon_table(fc, models):
    x = fc[fc["origin"].isin(EVAL_ORIGINS) & fc["model"].isin(models)].copy()
    x["h"] = (x["date"] - x["origin"]).dt.days
    x["ae"] = (x["actual"] - x["forecast"]).abs()
    x["se"] = (x["actual"] - x["forecast"]) ** 2
    g = x.groupby(["model", "h"])[["ae", "se"]].mean().reset_index()
    g["mae"], g["rmse"] = g["ae"], np.sqrt(g["se"])
    return g[["model", "h", "mae", "rmse"]]


# ---------------------------------------------------------------- GARCH on the best model's residuals

def garch_study(fc_best, daily, cal, items):
    """Prediction intervals from a GARCH(1,1) on the residuals of the best mean model, against a constant-spread
    interval, plus how the fitted volatility moves with the calendar. fc_best: that model's forecasts at all nine
    origins. Residuals are put on one scale by dividing by the item's mean sales over the year before the first
    validation origin, so only past data is used for the scale."""
    import vol_models as vm
    scale = {i: max(float(daily[i].loc[daily[i]["date"] <= VAL[0], "sales"].iloc[-365:].mean()), 0.1) for i in items}
    iv = vm.garch_intervals(fc_best, scale, EVAL_ORIGINS)
    flags = tm.calendar_flags(cal)
    iv["snap"] = flags["snap"].reindex(iv["date"]).to_numpy()
    iv["event"] = flags["event"].reindex(iv["date"]).to_numpy()
    iv["weekend"] = (pd.DatetimeIndex(iv["date"]).dayofweek >= 5).astype(float)
    iv["abs_err"] = (iv["actual"] - iv["forecast"]).abs()
    iv["scale"] = iv["item_id"].map(scale)
    summ = vm.interval_summary(iv)
    lm = []
    for it, g in fc_best.sort_values("date").groupby("item_id"):
        e = ((g["actual"] - g["forecast"]) / scale[it]).to_numpy()
        lm.append({"item_id": it, "arch_lm_p": vm.arch_lm(e), "n": len(e)})
    lm = pd.DataFrame(lm)
    rows = []
    for name, col, val in [("event day", "event", 1), ("no event", "event", 0), ("SNAP day", "snap", 1), ("non-SNAP day", "snap", 0),
                           ("weekend", "weekend", 1), ("weekday", "weekend", 0)]:
        g = iv[iv[col] == val]
        rows.append({"day_type": name, "item_days": len(g), "mean_sigma_over_scale": float((g["sig_garch"] / g["scale"]).mean()),
                     "mean_abs_error_over_scale": float((g["abs_err"] / g["scale"]).mean())})
    daytype = pd.DataFrame(rows)
    return iv, summ, lm, daytype


# ---------------------------------------------------------------- inventory with the final forecasts

def inventory_rerun(fc, models, long_df, cal, items, sl=0.95):
    """The 11e stock simulation for the given models: normal-formula and calibrated (conformal) safety stock, the same
    lead time and costs. fc needs the warm-up origin (ORIGINS[0]) because the first safety stock uses its errors."""
    from break_detection import weekly_ca_prices
    from config import HOLDING_RATE, LEAD_TIME, PROCESSED, RAW, STOCKOUT_MULTIPLE
    from pipeline import intermittency, price_daily_frame, simulate_inventory, summarise_inventory
    prices = pd.read_csv(RAW / "sell_prices.csv", dtype={"store_id": "category"})
    price_wk = weekly_ca_prices(prices, cal, list(items))
    del prices
    days = pd.date_range(ORIGINS[1] + pd.Timedelta(days=1), periods=H * 6)
    price_daily = price_daily_frame(price_wk, days)
    inter = intermittency(long_df[long_df["item_id"].isin(items)], ORIGINS[0])
    f = fc[fc["model"].isin(models) & fc["origin"].isin(ORIGINS)].assign(variant="static")
    scales = pd.read_csv(PROCESSED / "backtest_errors.csv", parse_dates=["origin"])[["item_id", "origin", "mase_scale"]].drop_duplicates()
    pol = [(m, "static") for m in models]
    normal = simulate_inventory(f, ORIGINS, price_daily, pol, [sl], LEAD_TIME, inter, HOLDING_RATE, STOCKOUT_MULTIPLE)
    g = f.merge(scales, on=["item_id", "origin"])
    g["e"] = g["actual"] - g["forecast"]
    cat_of = {i: i.split("_")[0] for i in items}
    sums = []
    for (item, origin, model), x in g.groupby(["item_id", "origin", "model"]):
        s = np.convolve(x.sort_values("date")["e"].to_numpy(), np.ones(LEAD_TIME), mode="valid") / x["mase_scale"].iloc[0]
        sums.append(pd.DataFrame({"origin": origin, "model": model, "category": cat_of[item], "r7": s}))
    sums = pd.concat(sums, ignore_index=True)
    scale_of = scales.set_index(["item_id", "origin"])["mase_scale"]
    cache = {}

    def conformal_ss(item, model, variant, origin, lead_time, sl_, past_errors):
        key = (cat_of[item], model, origin, sl_)
        if key not in cache:
            pool = sums[(sums["category"] == key[0]) & (sums["model"] == model) & (sums["origin"] < origin)]["r7"].to_numpy()
            n = len(pool)
            cache[key] = np.quantile(pool, min(1.0, np.ceil((n + 1) * sl_) / n), method="higher")
        return max(0.0, scale_of[(item, origin)] * cache[key])

    conf = simulate_inventory(f, ORIGINS, price_daily, pol, [sl], LEAD_TIME, inter, HOLDING_RATE, STOCKOUT_MULTIPLE, ss_fn=conformal_ss)
    items_df = pd.concat([normal.assign(method="normal formula"), conf.assign(method="conformal")], ignore_index=True)
    summ = summarise_inventory(items_df, by=("method", "model", "service_level"))
    return items_df, summ


# ---------------------------------------------------------------- training time of the earlier models

def time_classical(inp, n_items=6, origin=None):
    """Seconds for one fit at one origin, averaged over a few items and scaled to 150 items x 6 scored origins. A short
    timing run, not the original fits (those were not timed). SARIMA orders are the stored ones, so the order search
    is not included. AutoTS, ARIMA and ARMA are not timed."""
    import time
    import best_model as bm
    origin = origin or EVAL_ORIGINS[2]
    its = list(inp["series"])[::25][:n_items]
    ords = inp["orders"]
    specs = {"seasonal_naive": {"kind": "naive"}, "hw_add": {"kind": "ets"}, "ets_boxcox": bm.spec_of("ets"),
             "theta": bm.spec_of("theta"), "sarima": bm.spec_of("sarima"), "sarimax": bm.spec_of("sarimax"),
             "prophet": bm.spec_of("prophet")}
    out = {}
    for m, sp in specs.items():
        t = time.time()
        for it in its:
            s = dict(sp)
            if s["kind"] == "sarima":
                s["order"], s["sorder"] = ords.loc[it, "sarima_order"], ords.loc[it, "sarima_sorder"]
            d = inp["series"][it]
            tm.backtest(it, d["date"], d["sales"].to_numpy(float), [origin], m, s, inp["flags"], inp["prices"][it])
        out[m] = (time.time() - t) / len(its) * 150 * len(EVAL_ORIGINS)
    mem = pd.read_csv(TABLES / "ts_combination_members.csv")["member"].tolist()
    out["combination"] = sum(out[{"ets": "ets_boxcox"}.get(k, k)] for k in mem)
    return out
