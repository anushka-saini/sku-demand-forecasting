import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, MAIN, best_info, load, names, need_results
from utils.track import item_windows
from utils.ui import chart, download, kpis, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
lbi = lb[lb["level"] == "item"].sort_values("rank")
page("Value Added", "For each item and window, is the model better than simply repeating last week, and by how much?")
st.caption("Forecast value added = the MAE of seasonal naive (last week repeated) minus the model's MAE, for the same item and window. "
           "Above zero means the model helped.")

w_all = item_windows()
models = [m for m in lbi["model"] if m != "seasonal_naive"]
c1, c2 = st.columns(2)
m = c1.selectbox("Model", models, index=models.index(best_info()["model"]), format_func=nm.get, key="va_model")
cats = sorted(w_all["cat"].unique())
cat = c2.selectbox("Category", ["All"] + cats, format_func=lambda c: c if c == "All" else c.title(), key="va_cat")

naive = w_all[w_all["model"] == "seasonal_naive"].set_index(["item_id", "origin"])["mae"]
d = w_all[w_all["model"] == m].set_index(["item_id", "origin"])
d = d.assign(naive=naive, fva=naive - d["mae"]).reset_index()
if cat != "All":
    d = d[d["cat"] == cat]
per_item = d.groupby("item_id")["fva"].mean()
kpis([{"label": "Item-windows with positive value", "value": f"{(d['fva'] > 0).mean():.0%}", "compare": f"of {len(d)}", "tone": "good" if (d["fva"] > 0).mean() > 0.5 else "bad"},
      {"label": "Items with positive average value", "value": f"{(per_item > 0).mean():.0%}", "compare": f"of {len(per_item)} items"},
      {"label": "Average value added", "value": f"{d['fva'].mean():+.2f}", "compare": "units a day of MAE saved"},
      {"label": "Median value added", "value": f"{d['fva'].median():+.2f}", "compare": "units a day of MAE saved"}])

fig = go.Figure(go.Histogram(x=d["fva"], nbinsx=60, marker_color=COL.get(m, "#0E7C86")))
fig.add_vline(x=0, line_dash="dot", line_color="#5B6670")
fig.update_xaxes(title="MAE of seasonal naive minus MAE of the model, units a day")
fig.update_yaxes(title="item-windows")
chart(style(fig, legend=False), f"Value added by {nm[m]} over repeating last week",
      "Bars to the right of the dotted line are item-windows where the model did better.", key="va_hist")

t = per_item.sort_values(ascending=False)
best10 = pd.DataFrame({"Item": t.head(10).index, "Average value added": t.head(10).to_numpy()})
worst10 = pd.DataFrame({"Item": t.tail(10).index[::-1], "Average value added": t.tail(10).to_numpy()[::-1]})
a, b = st.columns(2)
a.markdown("**Best 10 items**")
a.dataframe(best10.round(3), hide_index=True, width="stretch")
b.markdown("**Worst 10 items**")
b.dataframe(worst10.round(3), hide_index=True, width="stretch")
download(pd.concat([best10.assign(group="best 10"), worst10.assign(group="worst 10")]).round(4), "value_added_best_worst.csv", key="dl_va_bw")

st.markdown("#### Which model did best on each item? (descriptive only)")
st.warning("Descriptive only: the best model per item is chosen after seeing the test results, so it is not a forecast result and must not "
           "be used as a model score.")
pool = st.radio("Models considered", ["Main set", "All models"], horizontal=True, key="va_pool")
cand = [x for x in lbi["model"] if pool == "All models" or x in MAIN]
x = w_all[w_all["model"].isin(cand)]
if cat != "All":
    x = x[x["cat"] == cat]
mean_mae = x.groupby(["item_id", "model"])["mae"].mean().reset_index()
win = mean_mae.loc[mean_mae.groupby("item_id")["mae"].idxmin()]
counts = win["model"].map(nm).value_counts().rename_axis("Model").reset_index(name="Items where it had the lowest MAE")
st.dataframe(counts, hide_index=True, width="stretch")
out = win.assign(model=win["model"].map(nm)).rename(columns={"item_id": "Item", "model": "Lowest-MAE model", "mae": "Its MAE"})
with st.expander("Item by item"):
    st.dataframe(out.round(3), hide_index=True, width="stretch")
download(out.round(4), "value_added_best_model_per_item_descriptive.csv", key="dl_va_best")

tells(f"{nm[m]} added value in {(d['fva'] > 0).mean():.0%} of item-windows. The table of best models per item is a description of this one test, "
      "not a way to choose a model for next time.")
