import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, MAIN, best_info, load, names, need_results
from utils.ui import chart, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
fc = load("forecasts")
info = best_info()
page("Forecast viewer", "Pick an item and models and see their forecasts against what really sold.")

items = sorted(fc["item_id"].unique())
c1, c2 = st.columns([1, 2])
item = c1.selectbox("Item", items, key="fv_item")
allm = st.checkbox("Offer all models, not just the main set", key="fv_all")
have = set(fc["model"].unique())
order = [m for m in lb[lb["level"] == "item"].sort_values("rank")["model"] if m in have and (allm or m in MAIN)]
models = c2.multiselect("Models", order, default=[m for m in [info["model"], "seasonal_naive"] if m in order],
                        format_func=nm.get, key="fv_models")
part = st.radio("Period", ["Test windows", "Validation windows"], horizontal=True, key="fv_part")
d = fc[(fc["item_id"] == item) & (fc["split"] == ("test" if part == "Test windows" else "validation"))]
if d.empty or not models:
    st.info("Pick at least one model.")
    st.stop()
act = d.drop_duplicates("date").sort_values("date")
fig = go.Figure(go.Scatter(x=act["date"], y=act["actual"], name="actual", mode="lines", line=dict(color="#222222", width=2)))
for m in models:
    g = d[d["model"] == m].sort_values("date")
    if len(g):
        fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], name=nm[m], mode="lines", line=dict(color=COL.get(m), width=1.6)))
if part == "Test windows" and st.checkbox("Show the 80% GARCH band around the best model", key="fv_band"):
    iv = load("intervals")
    g = iv[iv["item_id"] == item].sort_values("date")
    lo = np.maximum(g["forecast"] - 1.2816 * g["sig_garch"], 0)
    hi = g["forecast"] + 1.2816 * g["sig_garch"]
    fig.add_trace(go.Scatter(x=pd.concat([g["date"], g["date"][::-1]]), y=pd.concat([hi, lo[::-1]]), fill="toself",
                             fillcolor="rgba(0,114,178,0.15)", line=dict(width=0), name="80% GARCH band", hoverinfo="skip"))
fig.update_yaxes(title="units a day")
chart(style(fig, height=420), f"{item}: forecasts and actual sales",
      "Each forecast covers 28 days; the lines jump where a new window starts.", key="fv_chart")

e = d[d["model"].isin(models)].assign(ae=lambda x: (x["actual"] - x["forecast"]).abs(), se=lambda x: (x["actual"] - x["forecast"]) ** 2,
                                      b=lambda x: x["forecast"] - x["actual"])
tab = e.groupby("model").agg(MAE=("ae", "mean"), RMSE=("se", lambda s: float(np.sqrt(s.mean()))), bias=("b", "mean"))
tab.index = [nm[m] for m in tab.index]
st.dataframe(tab.round(2), width="stretch")
st.caption("Error for this one item over the windows shown (bias = forecast minus actual). One item is a small sample; the "
           "leaderboard has the full picture.")
tells("A single item can look better or worse than the whole set. Use it to see how a model behaves on event days, weekends and "
      "after a quiet spell, then check the leaderboard before drawing a conclusion.")
