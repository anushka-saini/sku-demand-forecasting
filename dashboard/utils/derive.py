"""
Numbers the pages compute from the saved outputs when they load: per-item scores over the chosen forecast dates,
error by forecast horizon, and the shift-aware versions on the same items. Nothing here fits a model.
"""
import numpy as np
import pandas as pd
import streamlit as st

from utils.data import get
from utils.models import SAMPLE_DUPLICATES

METRICS = {"MASE": "mase", "MAE": "mae", "RMSE": "rmse", "WAPE": "wape", "sMAPE": "smape", "MAPE": "mape"}
METRIC_HELP = {
    "MASE": "Average error divided by the error of repeating last week, as measured on the item's own history. "
            "Below 1 beats that simple rule. Has no units, so items of any size can be compared.",
    "MAE": "Average size of the miss, in units per day.",
    "RMSE": "Like MAE, but large misses count more. In units per day.",
    "WAPE": "Total miss divided by total sales: the share of volume the forecast got wrong.",
    "sMAPE": "Percentage error that treats over- and under-forecasting alike; 0 is perfect, 2 is the maximum.",
    "MAPE": "Average percentage error. Days with no sales have to be left out, because you cannot divide by zero.",
}


def eval_origins():
    """The scored forecast dates (the warm-up date is not scored)."""
    b = get("breaks_by_origin")
    return sorted(pd.Timestamp(x) for x in b.loc[b["is_eval_origin"], "origin"].unique())


def in_window(df, window, col="origin"):
    if not window:
        return df
    a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return df[(df[col] >= a) & (df[col] <= b)]


TS_KEYS = {"ets": "ets_boxcox"}                                 # 07h's ETS is the Box-Cox one; 07a already has an 'ets'
TS_SKIP = ["seasonal_naive", "sarima", "arima", "sarima_07a"]   # 07h repeats 07a's forecasts for these; 07a's are kept


@st.cache_data(show_spinner=False)
def sample_forecasts():
    """Daily forecasts on the 150-item sample: the 07a models and the time-series models of 07c to 07h, one key each."""
    cols = ["item_id", "category", "origin", "date", "model", "forecast", "actual"]
    ts = get("ts_forecasts")
    ts = ts[~ts["model"].isin(TS_SKIP)].assign(model=lambda d: d["model"].replace(TS_KEYS))
    out = pd.concat([get("classical_forecasts")[cols], ts[cols]], ignore_index=True)
    return out[~out["model"].isin(list(SAMPLE_DUPLICATES))]


def _with_scale(fc):
    scale = get("forecast_error")[["item_id", "origin", "mase_scale"]].drop_duplicates(["item_id", "origin"])
    return fc.merge(scale, on=["item_id", "origin"], how="left")


def _scores(fc, keys):
    """Every metric per group of daily forecasts, as 07a and 07h score them. fc carries the MASE scale."""
    e = fc["actual"] - fc["forecast"]
    ae = e.abs()
    den = fc["actual"].abs() + fc["forecast"].abs()
    x = fc.assign(ae=ae, se=e ** 2, sc=ae / fc["mase_scale"], sm=(2 * ae / den.where(den > 0)),
                  pe=(ae / fc["actual"].where(fc["actual"] > 0)))
    per = x.groupby(keys).agg(mae=("ae", "mean"), mse=("se", "mean"), mase=("sc", "mean"), smape=("sm", "mean"),
                              mape=("pe", "mean"), abs_sum=("ae", "sum"), act_sum=("actual", "sum")).reset_index()
    per["rmse"] = np.sqrt(per["mse"])
    per["wape"] = per["abs_sum"] / per["act_sum"].where(per["act_sum"] > 0)
    return per.drop(columns=["mse", "abs_sum", "act_sum"])


@st.cache_data(show_spinner=False)
def sample_metrics():
    """Every metric per item, scored forecast date and static model on the 150-item sample. 07a saved its scores;
    the 07c to 07h models are scored here from their forecasts, which gives 07h's numbers exactly."""
    evals = set(eval_origins())
    m = get("classical_item_metrics")
    m = m[m["origin"].isin(evals)]
    fc = sample_forecasts()
    fc = fc[fc["origin"].isin(evals) & ~fc["model"].isin(set(m["model"]))]
    ts = _scores(_with_scale(fc), ["item_id", "category", "origin", "model"])
    ts = ts.merge(get("classical_sample")[["item_id", "break_item"]], on="item_id")
    return pd.concat([m, ts], ignore_index=True)


@st.cache_data(show_spinner=False)
def sample_scores(window):
    """
    One row per item and model on the 150-item sample: each metric averaged over the chosen forecast dates.
    Static models carry every metric; shift-aware versions carry MASE (and MAE where it was saved).
    """
    evals = set(eval_origins())
    m = in_window(sample_metrics(), window)
    keys = ["item_id", "category", "break_item"]
    out = [m.groupby(keys + ["model"], as_index=False)[list(METRICS.values())].mean()]
    items = m[keys].drop_duplicates()
    ids = set(items["item_id"])

    e = get("forecast_error")
    e = in_window(e[e["is_eval_origin"] & e["item_id"].isin(ids)], window)
    for model, key in [("ets", "ets_retrained"), ("best", "best_restart"), ("tsb", "tsb_retrained")]:
        d = e[(e["model"] == model) & (e["variant"] == "adaptive")]
        if len(d):
            out.append(d.groupby("item_id", as_index=False)[["mase", "mae"]].mean().merge(items).assign(model=key))

    # SARIMA versions are stored only where a shift was known; elsewhere they are the static SARIMA by definition
    sar = m[m["model"] == "sarima"][["item_id", "origin", "mase"]]
    sa = get("shift_aware_items")
    for variant, key in [("restart", "sarima_restart"), ("dummy", "sarima_dummy")]:
        v = sa[sa["variant"] == variant][["item_id", "origin", "mase_variant"]]
        d = sar.merge(v, on=["item_id", "origin"], how="left")
        d["mase"] = d["mase_variant"].fillna(d["mase"])
        out.append(d.groupby("item_id", as_index=False)["mase"].mean().merge(items).assign(model=key))

    va = in_window(get("variants_items"), window)
    va = va[va["origin"].isin(evals) & va["item_id"].isin(ids)]
    for col, key in [("blend", "best_blend")]:
        out.append(va.groupby("item_id", as_index=False)[col].mean().rename(columns={col: "mase"})
                   .merge(items).assign(model=key))
    out = pd.concat(out, ignore_index=True)
    return out[~out["model"].isin(list(SAMPLE_DUPLICATES))]


@st.cache_data(show_spinner=False)
def horizon_scores(window):
    """Error by week of the 28-day forecast, per item and static model, on the 150-item sample."""
    evals = set(eval_origins())
    fc = sample_forecasts()
    fc = _with_scale(in_window(fc[fc["origin"].isin(evals)], window))
    fc["week"] = ((fc["date"] - fc["origin"]).dt.days - 1) // 7 + 1
    return _scores(fc, ["item_id", "category", "model", "week"])


@st.cache_data(show_spinner=False)
def all_item_scores(window):
    """Mean MASE and MAE per item over the chosen forecast dates, all 726 items, every saved model and version."""
    e = get("forecast_error")
    e = in_window(e[e["is_eval_origin"]], window)
    return e.groupby(["item_id", "model", "variant"], as_index=False)[["mase", "mae"]].mean()


def leaderboard(scores, metric, baseline="seasonal_naive"):
    """Median over items per model, with the share of items where the model beats the baseline."""
    col = METRICS[metric]
    d = scores.dropna(subset=[col])
    med = d.groupby("model")[col].agg(["median", "size"]).rename(columns={"median": "value", "size": "n_items"})
    wide = d.pivot_table(index="item_id", columns="model", values=col)
    if baseline in wide:
        med["beats_baseline"] = [(wide[m] < wide[baseline]).mean() if m != baseline else np.nan for m in med.index]
    return med.reset_index()
