"""Per item and window numbers for the SKU Tracker and Value Added pages. Read from the saved test forecasts, nothing is refitted."""
import pandas as pd
import streamlit as st

from utils.final import FILES, TABLES


@st.cache_data(show_spinner=False)
def item_windows():
    """One row per item, test window and model: errors over the 28 days and the item's average demand in that window.
    bias is forecast minus actual, so plus means too high."""
    d = pd.read_parquet(TABLES / FILES["forecasts"], columns=["item_id", "origin", "model", "forecast", "actual", "split"],
                        filters=[("split", "==", "test")])
    e = d["forecast"] - d["actual"]
    d = d.assign(e=e, ae=e.abs(), se=e ** 2)
    g = d.groupby(["item_id", "origin", "model"]).agg(mae=("ae", "mean"), mse=("se", "mean"), bias=("e", "mean"),
                                                      demand=("actual", "mean")).reset_index()
    g["rmse"] = g["mse"] ** 0.5
    g["cat"] = g["item_id"].str.split("_").str[0]
    return g.drop(columns="mse")


def movers(w, k=15):
    """Items whose MAE changed most from one window to the next, from the table of one model."""
    w = w.sort_values(["item_id", "origin"]).copy()
    w["prev_mae"] = w.groupby("item_id")["mae"].shift()
    w["prev_origin"] = w.groupby("item_id")["origin"].shift()
    w = w.dropna(subset=["prev_mae"])
    w["change"] = w["mae"] - w["prev_mae"]
    return w.reindex(w["change"].abs().sort_values(ascending=False).index).head(k)


def bias_flags(w, band):
    """For each item: in how many windows its bias was outside plus or minus `band` of its average demand.
    Items that sold nothing are left out, since a percent of zero is not defined."""
    avg = w.groupby("item_id")["demand"].mean()
    x = w.merge(avg.rename("avg_demand"), on="item_id")
    x = x[x["avg_demand"] > 0].assign(bias_pct=lambda y: y["bias"] / y["avg_demand"])
    x["out"] = x["bias_pct"].abs() > band
    g = x.groupby("item_id").agg(windows=("out", "size"), outside=("out", "sum"), mean_bias_pct=("bias_pct", "mean"),
                                 avg_demand=("avg_demand", "first")).reset_index()
    g["direction"] = g["mean_bias_pct"].map(lambda v: "over-forecast" if v > 0 else "under-forecast")
    return g
