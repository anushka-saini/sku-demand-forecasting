"""Prophet with the M5 events as holidays, SNAP and the relative price as regressors, tuned on the validation origins.

Grid: changepoint_prior_scale x seasonality_mode x seasonality_prior_scale (12 settings). The grid is scored on a
fixed subset of items (every 5th) at the three validation origins; the winner is then fitted for all 150 items at
all nine origins. Same 730-day fit window and blow-up guard as the other time-series models."""
import itertools
import logging
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import ml_models as M
import ts_models as tm
from config import FIT_DAYS, TABLES
from forecasting import seasonal_naive

GRID = list(itertools.product([0.01, 0.05, 0.2], ["additive", "multiplicative"], [1.0, 10.0]))


def holidays_from(cal):
    """Both event columns of the calendar as a Prophet holidays table."""
    c = cal.copy()
    c["date"] = pd.to_datetime(c["date"])
    parts = []
    for k in (1, 2):
        e = c[c[f"event_name_{k}"].notna()]
        parts.append(pd.DataFrame({"holiday": e[f"event_name_{k}"].str.replace(" ", "_"), "ds": e["date"],
                                   "lower_window": 0, "upper_window": 0}))
    return pd.concat(parts, ignore_index=True)


def fit_one(item, dates, y, origin, cfg, hol, flags, rel_price, h=28):
    """Prophet forecast for one item at one origin; a failed fit falls back to seasonal naive."""
    for n in ["cmdstanpy", "prophet"]:
        logging.getLogger(n).disabled = True
    from prophet import Prophet
    t = time.time()
    yt, dt = tm.window(y, dates, origin, FIT_DAYS)
    fd = pd.date_range(origin + pd.Timedelta(days=1), periods=h)
    cps, mode, sps = cfg
    try:
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False, holidays=hol,
                    changepoint_prior_scale=cps, seasonality_mode=mode, seasonality_prior_scale=sps)
        tr = pd.DataFrame({"ds": dt, "y": yt, "snap": flags["snap"].reindex(dt).to_numpy(),
                           "rel_price": rel_price.reindex(dt).fillna(1.0).to_numpy()})
        fu = pd.DataFrame({"ds": fd, "snap": flags["snap"].reindex(fd).to_numpy(),
                           "rel_price": rel_price.reindex(fd).fillna(1.0).to_numpy()})
        m.add_regressor("snap")
        m.add_regressor("rel_price")
        m.fit(tr)
        fc, ok, fb = tm.finish(m.predict(fu)["yhat"].to_numpy(), yt, h, True)
    except Exception:
        fc, ok, fb = seasonal_naive(yt, h), False, True
    return [{"item_id": item, "origin": origin, "date": d, "model": "prophet_tuned", "forecast": float(f),
             "fallback": fb, "seconds": time.time() - t} for d, f in zip(fd, fc)]


def run_set(inp, items, origins, cfg, hol, n_jobs=4):
    res = Parallel(n_jobs=n_jobs)(delayed(fit_one)(i, inp["series"][i]["date"], inp["series"][i]["sales"].to_numpy(float),
                                                    o, cfg, hol, inp["flags"], inp["prices"][i]) for i in items for o in origins)
    fc = pd.DataFrame([r for rows in res for r in rows])
    return fc.merge(inp["actuals"], on=["item_id", "date"])


def run(n_jobs=4):
    inp = tm.load_inputs()
    hol = holidays_from(inp["cal"])
    items = list(inp["series"])
    tp = TABLES / "final_prophet_tuning.csv"
    if tp.exists():
        tune = pd.read_csv(tp)
    else:
        sub = items[::5]
        rows = []
        for cfg in GRID:
            fc = run_set(inp, sub, tm.VAL_ORIGINS, cfg, hol, n_jobs)
            rows.append({"changepoint_prior_scale": cfg[0], "seasonality_mode": cfg[1], "seasonality_prior_scale": cfg[2],
                         "val_score": M.score_pair(fc["actual"], fc["forecast"])})
            print("prophet", cfg, round(rows[-1]["val_score"], 4), flush=True)
        tune = pd.DataFrame(rows)
        tune.to_csv(tp, index=False)
    b = tune.sort_values("val_score").iloc[0]
    cfg = (float(b["changepoint_prior_scale"]), b["seasonality_mode"], float(b["seasonality_prior_scale"]))
    print("prophet best", cfg, flush=True)
    fc = run_set(inp, items, tm.ALL_ORIGINS, cfg, hol, n_jobs)
    fc.to_parquet(M.CACHE / "prophet_tuned_forecasts.parquet")
    M.save_json({"changepoint_prior_scale": cfg[0], "seasonality_mode": cfg[1], "seasonality_prior_scale": cfg[2]},
                M.CACHE / "prophet_config.json")
