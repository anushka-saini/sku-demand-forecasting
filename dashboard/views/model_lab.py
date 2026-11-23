import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, load, names, need_results
from utils.lab import ABOUT, DAYS, cut, test_errors
from utils.ui import chart, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
lbi = lb[lb["level"] == "item"].sort_values("rank")
page("Model Lab", "Pick two or more models and see where each one is strong or weak.")
st.caption("Errors are pooled over every item-day of the six test windows (150 items). The Leaderboard uses the typical error per item, "
           "so its numbers are a little different; the order of the models is the same.")

have = list(lbi["model"])
pick = st.multiselect("Models to compare (two or more)", have, default=[m for m in ["xgboost", "sarima", "seasonal_naive"] if m in have],
                      format_func=nm.get, key="lab_models")
k = st.radio("Error measure", ["MAE", "RMSE"], horizontal=True, key="lab_k").lower()
if len(pick) < 2:
    st.info("Pick at least two models.")
    st.stop()

with st.expander("What each model does", expanded=True):
    for m in pick:
        st.markdown(f"**{nm[m]}**: {ABOUT.get(m, '')}")

d = test_errors()
t1, t2, t3 = st.tabs(["By category", "By weekday", "By forecast day"])
with t1:
    g = cut(d, pick, "cat")
    fig = go.Figure()
    for m in pick:
        x = g[g["model"] == m]
        fig.add_bar(x=[c.title() for c in x["cat"]], y=x[k], name=nm[m], marker_color=COL.get(m))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=f"{k.upper()}, units a day")
    chart(style(fig), f"{k.upper()} by category", "Lower is better. Foods usually sells most, so its errors in units are larger.", key="lab_cat")
with t2:
    g = cut(d, pick, "wd")
    fig = go.Figure()
    for m in pick:
        x = g[g["model"] == m].sort_values("wd")
        fig.add_trace(go.Scatter(x=[DAYS[i] for i in x["wd"]], y=x[k], mode="lines+markers", name=nm[m], line=dict(color=COL.get(m), width=2)))
    fig.update_yaxes(title=f"{k.upper()}, units a day")
    chart(style(fig), f"{k.upper()} by day of the week", "Weekends sell most, so every model misses more in units on those days.", key="lab_wd")
with t3:
    hz = load("horizon")
    fig = go.Figure()
    for m in pick:
        x = hz[hz["model"] == m].sort_values("h")
        fig.add_trace(go.Scatter(x=x["h"], y=x[k], mode="lines", name=nm[m], line=dict(color=COL.get(m), width=2)))
    fig.update_xaxes(title="days ahead")
    fig.update_yaxes(title=f"{k.upper()}, units a day")
    chart(style(fig), f"{k.upper()} by forecast day", "Error at each of the 28 days ahead, over all test windows and items.", key="lab_h")

tot = d[d["model"].isin(pick)].groupby("model")[["ae", "se"]].mean()
tot = tot.assign(rmse=tot["se"] ** 0.5).rename(columns={"ae": "mae"})[["mae", "rmse"]]
best = tot[k].idxmin()
tells(f"Of the models picked, {nm[best]} has the lowest {k.upper()} ({tot.loc[best, k]:.2f} units a day over all item-days). "
      "A model that wins overall can still lose on one category or weekday, which is why this page cuts the error three ways.")
