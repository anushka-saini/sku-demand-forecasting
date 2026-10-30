"""The best time-series model from 07h, fitted the same way on any products, origins, seasons or states.

The notebooks that used LightGBM before (09, 10, 11a to 11f) use this instead. 07h writes the winner to
outputs/tables/ts_best_model.json: its name and one spec per member (one member for a single model, several for a
combination, whose forecast is the plain average). Each spec is the one tuned on the validation origins in 07d/07f.

Its forecasts are stored under the key 'best', whatever model won. Versions, as for LightGBM before: 'static' is the usual fit window; 'adaptive' trains only from the latest detected
shift (the start dates in breaks_by_origin.csv), and 'placebo' from a random start for products without one. A
version without a start date is the static forecast.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ts_models as tm
from config import FIT_DAYS, HORIZON, PROCESSED, TABLES

BEST_PATH = TABLES / "ts_best_model.json"
M = 7


def load_best(path=BEST_PATH):
    return json.loads(Path(path).read_text())


def spec_of(model):
    """The setting 07d/07f tuned for one contender, without the per-product SARIMA orders (as 07h saves it)."""
    setting = {}
    for f in ["ts_sarima_forecasts.parquet", "ts_other_forecasts.parquet"]:
        s = pd.read_parquet(TABLES / f, columns=["model", "setting"]).drop_duplicates("model")
        setting.update(dict(zip(s["model"], s["setting"])))
    s = json.loads((PROCESSED / "ts_fits" / setting[model] / "spec.json").read_text())["spec"]
    return {k: v for k, v in s.items() if k not in ("order", "sorder")}


def combination_spec(path=TABLES / "ts_combination_members.csv"):
    """The 07h combination, a plain average of its members, in the same form as ts_best_model.json."""
    members = pd.read_csv(path)["member"].tolist()
    return {"model": "combination", "name": "Combination", "members": {m: spec_of(m) for m in members}}


def needs_orders(best):
    return any(s["kind"] == "sarima" for s in best["members"].values())


def search_order(y0, pq=range(3)):
    """SARIMA order for one product, chosen once on its window up to the warm-up origin. Same rule as 07a (d from
    KPSS, lowest AICc), except that the seasonal part is fixed at (1, 0, 1)7, the order 07a chose for 129 of the 150
    sample products, so only p and q are searched (9 fits instead of 36)."""
    from classical import fit_sarima
    from stationarity import recommend_d
    d = recommend_d(y0)
    best = None
    for p in pq:
        for q in pq:
            try:
                res, _ = fit_sarima(y0, (p, d, q), (1, 0, 1, M))
            except Exception:
                continue
            if np.isfinite(res.aicc) and (best is None or res.aicc < best[0]):
                best = (float(res.aicc), (p, d, q))
    return {"sarima_order": best[1] if best else (0, d, 0), "sarima_sorder": (1, 0, 1, M)}


def orders_for(series, warm, folder, known=None, n_jobs=10):
    """Orders for every product at the warm-up origin, cached. known: {item: (order, sorder)} to reuse (07a)."""
    from joblib import Parallel, delayed
    path = Path(folder) / f"orders_{pd.Timestamp(warm).date()}.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    known = known or {}
    todo = [i for i in series if i not in have and i not in known]

    def one(i):
        y, _ = tm.window(series[i]["sales"], series[i]["date"], warm)
        return i, search_order(y)

    for i, o in Parallel(n_jobs=n_jobs, batch_size=4)(delayed(one)(i) for i in todo):
        have[i] = {k: list(v) for k, v in o.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(have))
    out = {i: (tuple(v["sarima_order"]), tuple(v["sarima_sorder"])) for i, v in have.items()}
    out.update({i: (tuple(o), tuple(s)) for i, (o, s) in known.items()})
    return {i: out[i] for i in series}


CAP = 3          # a forecast above 3 times the most the product ever sold in its training window is a blow-up


def member_fc(spec, y, dates, origin, start, order, flags, rel_price, h=HORIZON):
    """One member's forecast at one origin, trained from `start` (or its usual window if start is None).
    A forecast above CAP times the training maximum falls back to seasonal naive, like a failed fit."""
    yt, dt = tm.window(y, dates, origin, spec.get("fit_days", FIT_DAYS))
    if start is not None:
        yt = yt[dt >= start]
    f, ok, fb = _member_fc(spec, y, dates, origin, start, order, flags, rel_price, h)
    if np.max(f) > CAP * max(np.max(yt), 1):
        return tm.seasonal_naive(yt, h), ok, True
    return f, ok, fb


def cap_forecasts(fc, daily, fit_days=None):
    """The same guard as member_fc, on forecasts already saved: per product, origin and model, a forecast above CAP
    times the training maximum is replaced by seasonal naive. fit_days: {model: days} where a model's window is not
    FIT_DAYS (None = all history). Returns the guarded forecasts and the list of what was replaced."""
    fit_days = fit_days or {}
    out, hits = [], []
    for (item, origin, model), g in fc.groupby(["item_id", "origin", "model"], sort=False):
        d = daily[item]
        yt, _ = tm.window(d["sales"].to_numpy(dtype=float), pd.DatetimeIndex(d["date"]), origin,
                          fit_days.get(model, FIT_DAYS))
        g = g.sort_values("date")
        f = g["forecast"].to_numpy()
        if np.max(f) > CAP * max(np.max(yt), 1):
            hits.append({"item_id": item, "origin": origin, "model": model, "forecast_max": float(np.max(f)),
                         "training_max": float(np.max(yt))})
            g = g.assign(forecast=tm.seasonal_naive(yt, len(f)), fallback=True)
        out.append(g)
    return pd.concat(out, ignore_index=True), pd.DataFrame(hits)


def _member_fc(spec, y, dates, origin, start, order, flags, rel_price, h=HORIZON):
    spec = dict(spec)
    kind = spec.pop("kind")
    names = spec.pop("exog", [])
    yt, dt = tm.window(y, dates, origin, spec.pop("fit_days", FIT_DAYS))
    if start is not None:
        keep = dt >= start
        yt, dt = yt[keep], dt[keep]
    fd = pd.date_range(origin + pd.Timedelta(days=1), periods=h)
    if kind == "naive":
        return tm.seasonal_naive(yt, h), True, False
    if kind == "ets":
        return tm.ets_fc(yt, h, **spec)
    if kind == "theta":
        return tm.theta_fc(yt, h, **spec)
    if kind == "sarima":
        ex = tm.make_exog(dt, flags, names, rel_price) if names else None
        ex_f = tm.make_exog(fd, flags, names, rel_price) if names else None
        return tm.sarima_fc(yt, order[0], order[1], h, exog=ex, exog_f=ex_f, **spec)
    if kind == "prophet":
        cols = tm.exog_columns(names)
        regs = pd.DataFrame(tm.make_exog(dt, flags, names, rel_price), columns=cols) if names else None
        regs_f = pd.DataFrame(tm.make_exog(fd, flags, names, rel_price), columns=cols) if names else None
        return tm.prophet_fc(dt, yt, h, regs, regs_f, **spec)
    raise ValueError(kind)


def forecast_item(item, dates, y, origins, starts, best, order=None, flags=None, rel_price=None, h=HORIZON):
    """Every version at every origin for one product. starts: {origin: {variant: start date or None}}."""
    dates, y = pd.DatetimeIndex(dates), np.asarray(y, dtype=float)
    rows = []
    for o in origins:
        fd = pd.date_range(o + pd.Timedelta(days=1), periods=h)
        done = {}
        for variant, start in {"static": None, **starts.get(o, {})}.items():
            key = None if start is None or pd.isna(start) or start <= dates[0] else pd.Timestamp(start)
            if key not in done:
                out = [member_fc(s, y, dates, o, key, order, flags, rel_price, h) for s in best["members"].values()]
                done[key] = (np.mean([f for f, _, _ in out], axis=0), all(ok for _, ok, _ in out),
                             any(fb for _, _, fb in out))
            fc, ok, fb = done[key]
            rows += [{"item_id": item, "origin": o, "date": d, "model": "best", "variant": variant,
                      "forecast": float(f), "converged": ok, "fallback": fb} for d, f in zip(fd, fc)]
    return rows


def _to_file(item, dates, y, origins, starts, best, order, flags, rel_price, path):
    path = Path(path)
    if not path.exists():
        df = pd.DataFrame(forecast_item(item, dates, y, origins, starts, best, order, flags, rel_price))
        df.attrs = {}
        df.to_parquet(path, index=False)
    return item


def run_best(series, origins, starts, folder, best=None, orders=None, flags=None, prices=None, n_jobs=10, rerun=False):
    """The best model on every product, in parallel, one checkpoint per product.
    starts: {item: {origin: {variant: start}}}; orders: {item: (order, sorder)} if a member is a SARIMA."""
    from joblib import Parallel, delayed
    best = best or load_best()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "best.json"
    if not saved.exists() or json.loads(saved.read_text())["members"] != best["members"]:
        rerun = True                     # no record of the model, or another model's forecasts: never reuse them
    if rerun:                            # _to_file skips a product whose checkpoint exists, so clear them first
        for i in series:
            (folder / f"{i}.parquet").unlink(missing_ok=True)
    saved.write_text(json.dumps(best, indent=1))
    todo = [i for i in series if not (folder / f"{i}.parquet").exists()]
    Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_to_file)(i, series[i]["date"], series[i]["sales"], origins, starts.get(i, {}), best,
                          None if orders is None else orders[i], flags, None if prices is None else prices.get(i),
                          folder / f"{i}.parquet")
        for i in todo)
    return pd.concat([pd.read_parquet(folder / f"{i}.parquet") for i in series], ignore_index=True)


def starts_from(windows, variants=("adaptive", "placebo")):
    """{item: {origin: {variant: start}}} from a breaks_by_origin table (or the windows run_backtest makes)."""
    out = {}
    for r in windows.itertuples():
        out.setdefault(r.item_id, {})[r.origin] = {v: getattr(r, f"{v}_start", None) for v in variants
                                                   if hasattr(r, f"{v}_start")}
    return out
