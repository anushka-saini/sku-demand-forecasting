"""Runs one of the heavy models and caches its forecasts, so notebook 13 can load them instead of refitting.

    python src/run_models.py xgb | lstm | gru | patchtst | prophet

Every fit is saved as soon as it finishes, so a stopped run picks up where it stopped. The timings go to
data/processed/final/times_<model>.csv (seconds per fit)."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import features as F
import ml_models as M
import nn_models as N
import ts_models as tm
from config import EVAL_ORIGINS, TABLES, SEED
from metrics import mae, rmse

ALL = tm.ALL_ORIGINS
VAL = tm.VAL_ORIGINS
SEEDS = (0, 1, 2)


def run_xgb(threads=5):
    inp = tm.load_inputs()
    rows = M.get_rows(inp)
    pj = M.CACHE / "xgb_params.json"
    import json
    if pj.exists():
        d = json.loads(pj.read_text())
        params, n_est = d["params"], d["n_est"]
    else:
        params, n_est, trials = M.tune_xgb(rows, threads=threads)
        trials.to_csv(TABLES / "final_xgb_trials.csv", index=False)
        M.save_json({"params": params, "n_est": n_est}, pj)
    lp = M.CACHE / "lgbm_params.json"
    if lp.exists():
        d = json.loads(lp.read_text())
        lparams, ln = d["params"], d["n_est"]
    else:
        lparams, ln = M.tune_lgbm(rows, threads=threads)
        M.save_json({"params": lparams, "n_est": ln}, lp)
    times = []
    for name, fn in [("xgboost", M.xgb_fit_fn(params, n_est, threads)), ("lightgbm", M.lgbm_fit_fn(lparams, ln, threads))]:
        fc, secs = M.predict_origins(rows, ALL, fn, name)
        fc.to_parquet(M.CACHE / f"{name}_forecasts.parquet")
        times += [{"model": name, "origin": o, "seconds": s} for o, s in secs.items()]
        print(name, "test MAE", mae(fc[fc.origin.isin(EVAL_ORIGINS)].actual, fc[fc.origin.isin(EVAL_ORIGINS)].forecast))
    pd.DataFrame(times).to_csv(M.CACHE / "times_xgb.csv", index=False)


def nn_items(inp):
    cal_t = F.calendar_table(inp["cal"])
    return {i: N.prep_item(d, cal_t, inp["prices"][i]) for i, d in inp["series"].items()}


def run_nn(kind, threads=3):
    """Tuning and the final runs of one net. Two stages, each a single training run per seed:
    validation (cutoff = first validation origin, forecasts the three validation origins) and
    test (cutoff = first scored origin, forecasts the six scored origins)."""
    inp = tm.load_inputs()
    items = nn_items(inp)
    act = inp["actuals"]
    folder = M.CACHE / "nn"
    folder.mkdir(parents=True, exist_ok=True)
    tp = TABLES / f"final_{kind}_tuning.csv"
    if tp.exists():
        tune = pd.read_csv(tp)
    else:
        res = []
        for k, cfg in enumerate(N.sample_configs(kind)):
            fc, secs, ep = N.fit_predict(items, VAL[0], VAL, kind, cfg, 0, threads)
            m = fc.merge(act, on=["item_id", "date"])
            res.append({**cfg, "val_score": M.score_pair(m["actual"], m["forecast"]), "epochs": ep, "seconds": secs, "config": k})
            print(kind, "config", k, cfg, round(res[-1]["val_score"], 4), f"{secs:.0f}s", flush=True)
        tune = pd.DataFrame(res)
        tune.to_csv(tp, index=False)
    best = tune.sort_values("val_score").iloc[0]
    cfg = {k: (int(best[k]) if k in ("hidden", "layers", "W", "patch") else float(best[k]))
           for k in ("hidden", "layers", "dropout", "W", "lr", "patch") if k in best and not pd.isna(best[k])}
    print(kind, "best", cfg, flush=True)
    for stage, cutoff, origins in [("val", VAL[0], VAL), ("test", EVAL_ORIGINS[0], list(EVAL_ORIGINS))]:
        for s in SEEDS:
            p = folder / f"{kind}_{stage}_{s}.parquet"
            if p.exists():
                continue
            fc, secs, ep = N.fit_predict(items, cutoff, origins, kind, cfg, s, threads)
            fc = fc.merge(act, on=["item_id", "date"]).assign(model=kind, seed=s, seconds=secs, epochs=ep, stage=stage)
            fc.to_parquet(p)
            print(kind, stage, "seed", s, f"{secs:.0f}s", ep, "epochs", flush=True)
    parts = [pd.read_parquet(folder / f"{kind}_{ph}_{s}.parquet") for ph in ("val", "test") for s in SEEDS]
    pd.concat(parts, ignore_index=True).to_parquet(M.CACHE / f"{kind}_forecasts.parquet")
    M.save_json({"config": cfg}, M.CACHE / f"{kind}_config.json")


if __name__ == "__main__":
    what = sys.argv[1]
    t = time.time()
    if what == "xgb":
        run_xgb()
    elif what in ("lstm", "gru", "patchtst"):
        run_nn(what)
    elif what == "prophet":
        import prophet_tuned
        prophet_tuned.run()
    print(what, "done in", round(time.time() - t), "s")
