import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, best_info, load, names, need_results
from utils.track import bias_flags, item_windows, movers
from utils.ui import chart, download, kpis, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
lbi = lb[lb["level"] == "item"].sort_values("rank")
page("SKU Tracker", "How accurate and how biased is the forecast for each item, window by window?")
st.caption("Error and bias per item over the six test windows. Bias is forecast minus actual, so plus means the forecast ran too high. "
           "Use it to find items a forecaster should look at first.")

w_all = item_windows()
models = list(lbi["model"])
c1, c2, c3 = st.columns(3)
m = c1.selectbox("Model", models, index=models.index(best_info()["model"]), format_func=nm.get, key="tr_model")
cats = sorted(w_all["cat"].unique())
cat = c2.selectbox("Category", ["All"] + cats, format_func=lambda c: c if c == "All" else c.title(), key="tr_cat")
wins = sorted(w_all["origin"].unique())
span = lambda o: f"{pd.Timestamp(o) + pd.Timedelta(days=1):%d %b %Y} to {pd.Timestamp(o) + pd.Timedelta(days=28):%d %b %Y}"
win = c3.selectbox("Test window", ["All windows"] + wins, format_func=lambda o: o if isinstance(o, str) else span(o), key="tr_win")

w = w_all[w_all["model"] == m]
if cat != "All":
    w = w[w["cat"] == cat]
ws = w if win == "All windows" else w[w["origin"] == win]
if ws.empty:
    st.info("Nothing to show for this choice.")
    st.stop()

dem = ws["demand"].mean()
kpis([{"label": "Items", "value": f"{ws['item_id'].nunique()}"},
      {"label": "Median MAE", "value": f"{ws['mae'].median():.2f}", "compare": "units a day, per item-window"},
      {"label": "Median RMSE", "value": f"{ws['rmse'].median():.2f}", "compare": "units a day, per item-window"},
      {"label": "Average bias", "value": f"{ws['bias'].mean():+.2f}", "compare": (f"{ws['bias'].mean() / dem:+.1%} of average demand" if dem > 0 else "no demand")}])

t1, t2, t3, t4 = st.tabs(["By item", "By window", "Movers", "Bias monitor"])

with t1:
    g = ws.groupby(["item_id", "cat"]).agg(demand=("demand", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean")).reset_index()
    g["bias_pct"] = g["bias"] / g["demand"].where(g["demand"] > 0)
    fig = go.Figure(go.Scatter(x=g["demand"], y=g["mae"], mode="markers", text=g["item_id"], marker=dict(color=COL.get(m), size=7, opacity=0.75),
                               hovertemplate="%{text}<br>demand %{x:.1f}<br>MAE %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="average demand, units a day")
    fig.update_yaxes(title="MAE, units a day")
    chart(style(fig, legend=False), f"Error against demand, {nm[m]}", "Busier items miss more in units. Look for dots well above the rest.", key="tr_sc")
    out = g.rename(columns={"item_id": "Item", "cat": "Category", "demand": "Average demand", "mae": "MAE", "rmse": "RMSE", "bias": "Bias",
                            "bias_pct": "Bias % of demand"}).sort_values("MAE", ascending=False)
    st.dataframe(out.round(3), hide_index=True, width="stretch")
    download(out.round(4), "sku_tracker_by_item.csv", key="dl_tr_item")

with t2:
    a = w.groupby("origin").agg(items=("item_id", "nunique"), mae=("mae", "median"), rmse=("rmse", "median"), bias=("bias", "mean"),
                                demand=("demand", "mean")).reset_index()
    a["bias_pct"] = a["bias"] / a["demand"]
    a.insert(0, "Window", [span(o) for o in a["origin"]])
    fig = go.Figure()
    for c in cats if cat == "All" else [cat]:
        x = w_all[(w_all["model"] == m) & (w_all["cat"] == c)].groupby("origin")["mae"].median().reset_index()
        fig.add_trace(go.Scatter(x=[span(o) for o in x["origin"]], y=x["mae"], mode="lines+markers", name=c.title()))
    fig.update_yaxes(title="median MAE per item, units a day", rangemode="tozero")
    fig.update_xaxes(title="test window")
    chart(style(fig), f"Median error by window, {nm[m]}", "A line that climbs means that category got harder to forecast.", key="tr_win_chart")
    out = a.drop(columns="origin").rename(columns={"items": "Items", "mae": "Median MAE", "rmse": "Median RMSE", "bias": "Average bias",
                                                   "demand": "Average demand", "bias_pct": "Bias % of demand"})
    st.dataframe(out.round(3), hide_index=True, width="stretch")
    download(out.round(4), "sku_tracker_by_window.csv", key="dl_tr_win")

with t3:
    st.markdown("Items whose error changed most from one test window to the next.")
    way = st.radio("Show", ["Both", "Got worse", "Got better"], horizontal=True, key="tr_way")
    mv = movers(w, k=300)
    if way == "Got worse":
        mv = mv[mv["change"] > 0]
    elif way == "Got better":
        mv = mv[mv["change"] < 0]
    mv = mv.head(15)
    out = pd.DataFrame({"Item": mv["item_id"], "Window before": [span(o) for o in mv["prev_origin"]], "Window after": [span(o) for o in mv["origin"]],
                        "MAE before": mv["prev_mae"], "MAE after": mv["mae"], "Change": mv["change"]})
    st.dataframe(out.round(3), hide_index=True, width="stretch")
    download(out.round(4), "sku_tracker_movers.csv", key="dl_tr_mv")

with t4:
    band = st.slider("Target band: plus or minus this share of average demand", 0.02, 0.40, 0.10, 0.01, format="%.2f", key="tr_band",
                     help="0.10 means plus or minus 10 percent of the item's average daily demand.")
    fl = bias_flags(w, band)
    bad = fl[fl["outside"] > fl["windows"] / 2].sort_values(["outside", "mean_bias_pct"], ascending=[False, False])
    kpis([{"label": "Items outside the band in most windows", "value": f"{len(bad)}", "compare": f"of {len(fl)} items with sales"},
          {"label": "Too high", "value": f"{(bad['mean_bias_pct'] > 0).sum()}", "compare": "over-forecast"},
          {"label": "Too low", "value": f"{(bad['mean_bias_pct'] < 0).sum()}", "compare": "under-forecast"}])
    out = bad.rename(columns={"item_id": "Item", "windows": "Windows", "outside": "Windows outside band", "mean_bias_pct": "Average bias % of demand",
                              "avg_demand": "Average demand", "direction": "Direction"})
    st.dataframe(out.round(3), hide_index=True, width="stretch")
    st.caption("Bias in a window is the average of forecast minus actual over its days. An item is listed if it is outside the band in more than half of its windows.")
    download(out.round(4), "sku_tracker_bias_monitor.csv", key="dl_tr_bias")

tells(f"This page is for finding items to look at, not for ranking models. A model with low average bias can still be far off on single items; "
      f"for {nm[m]} the median item misses by {ws['mae'].median():.2f} units a day.")
