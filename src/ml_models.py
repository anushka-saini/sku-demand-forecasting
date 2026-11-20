"""XGBoost (tuned with Optuna) and LightGBM (reference) on the shared feature rows.

Protocol, the same as for the time-series models:
  * tuning only sees the three validation origins (ts_models.VAL_ORIGINS), as three expanding-window folds;
  * a fit for origin o trains on rows whose whole target window ends on or before o (row origin <= o - 28 days);
  * the six scored origins are predicted once, after the settings are frozen.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from config import PROCESSED, SEED, TABLES
from ts_models import ALL_ORIGINS, VAL_ORIGINS
from config import EVAL_ORIGINS, ORIGINS

CACHE = PROCESSED / "final"
H = F.H
DAY = pd.Timedelta(days=1)


def get_rows(inp):
    """Feature rows for all items, cached. Weekly origin grid anchored on the first scored origin."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "rows.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = F.build_rows(inp["series"], inp["cal"], inp["prices"], ALL_ORIGINS, anchor=ORIGINS[0])
    df.to_parquet(p)
    return df


def train_part(rows, origin, every=1):
    """Rows usable when forecasting at `origin`: the target window is over by then. every=2 keeps each second week."""
    r = rows[rows["origin"] <= origin - H * DAY]
    if every > 1:
        wk = ((origin - r["origin"]).dt.days // 7)
        r = r[wk % every == 0]
    return r


def at_origin(rows, origin):
    return rows[rows["origin"] == origin]


def score_pair(y, p):
    """The tuning score: mean of RMSE and MAE in units sold (the two ranking metrics)."""
    e = np.asarray(y) - np.asarray(p)
    return float((np.sqrt(np.mean(e ** 2)) + np.mean(np.abs(e))) / 2)


def make_xgb(params, n_est, threads=6, early=None):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=n_est, tree_method="hist", n_jobs=threads, random_state=SEED,
                            early_stopping_rounds=early, verbosity=0, **params)


def fold_data(rows, o, every):
    """Training rows, early-stopping rows and prediction rows for one validation fold at origin o.
    Early stopping uses the 28 days just before o, so nothing inside the scored window helps pick the tree count."""
    stop = at_origin(rows, o - H * DAY)
    tr = train_part(rows, o - H * DAY, every)
    return tr, stop, at_origin(rows, o)


def tune_xgb(rows, n_trials=25, every=2, threads=6, log=print):
    """Optuna search; the score is the mean over the three validation folds. Returns (best params, n_est, trials)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = {o: fold_data(rows, o, every) for o in VAL_ORIGINS}
    best_iters = {}

    def objective(trial):
        obj = trial.suggest_categorical("objective", ["reg:squarederror", "reg:tweedie", "reg:pseudohubererror"])
        p = {"objective": obj,
             "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
             "max_depth": trial.suggest_int("max_depth", 3, 9),
             "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True)}
        if obj == "reg:tweedie":
            p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.8)
        scores, its = [], []
        for o, (tr, stop, ev) in folds.items():
            m = make_xgb(p, 1500, threads, early=40)
            m.fit(tr[F.FEATURES], tr["y"], eval_set=[(stop[F.FEATURES], stop["y"])], verbose=False)
            its.append(m.best_iteration + 1)
            scores.append(score_pair(ev["y"], np.clip(m.predict(ev[F.FEATURES]), 0, None)))
        best_iters[trial.number] = int(np.mean(its))
        trial.set_user_attr("n_est", best_iters[trial.number])
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t = time.time()
    study.optimize(objective, n_trials=n_trials)
    log(f"xgboost tuning: {n_trials} trials in {time.time() - t:.0f}s, best validation score {study.best_value:.4f}")
    trials = study.trials_dataframe()[["number", "value", "state"] + [c for c in study.trials_dataframe().columns
                                                                        if c.startswith("params_") or c == "user_attrs_n_est"]]
    return dict(study.best_params), int(study.best_trial.user_attrs["n_est"]), trials


def tune_lgbm(rows, every=2, threads=6, log=print):
    """LightGBM is only a reference: fixed, sensible settings, and early stopping on the validation folds to pick
    the number of trees."""
    import lightgbm as lgb
    p = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, objective="regression")
    its, scores = [], []
    for o in VAL_ORIGINS:
        tr, stop, ev = fold_data(rows, o, every)
        m = lgb.LGBMRegressor(n_estimators=1500, n_jobs=threads, random_state=SEED, verbose=-1, **p)
        m.fit(tr[F.FEATURES], tr["y"], eval_set=[(stop[F.FEATURES], stop["y"])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        its.append(m.best_iteration_)
        scores.append(score_pair(ev["y"], np.clip(m.predict(ev[F.FEATURES]), 0, None)))
    log(f"lightgbm validation score {np.mean(scores):.4f}, trees {int(np.mean(its))}")
    return p, int(np.mean(its))


def predict_origins(rows, origins, fit_fn, name, every=1):
    """Refit at every origin on its own past rows, predict its 28 days. Returns (forecasts, seconds per origin)."""
    out, secs = [], {}
    for o in origins:
        t = time.time()
        tr = train_part(rows, o, every)
        ev = at_origin(rows, o)
        pred = np.clip(fit_fn(tr, ev), 0, None)
        secs[o] = time.time() - t
        out.append(pd.DataFrame({"item_id": ev["item_id"].to_numpy(), "origin": o, "date": ev["date"].to_numpy(),
                                 "model": name, "forecast": pred, "actual": ev["y"].to_numpy()}))
    return pd.concat(out, ignore_index=True), secs


def xgb_fit_fn(params, n_est, threads=6):
    def fn(tr, ev):
        m = make_xgb(params, n_est, threads)
        m.fit(tr[F.FEATURES], tr["y"])
        return m.predict(ev[F.FEATURES])
    return fn


def lgbm_fit_fn(params, n_est, threads=6):
    import lightgbm as lgb

    def fn(tr, ev):
        m = lgb.LGBMRegressor(n_estimators=n_est, n_jobs=threads, random_state=SEED, verbose=-1, **params)
        m.fit(tr[F.FEATURES], tr["y"])
        return m.predict(ev[F.FEATURES])
    return fn


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
