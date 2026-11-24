import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, load, names, need_results
from utils.lab import ABOUT, DAYS, cut, errors
from utils.ui import chart, download, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
lbi = lb[lb["level"] == "item"].sort_values("rank")
page("Model Lab", "Pick two or more models and see where each one is strong or weak.")
st.caption("Errors on the six test windows (150 items). The Leaderboard uses the typical error per item, so its numbers differ a little "
           "from the pooled ones here; the order of the models is the same.")

have = list(lbi["model"])
c1, c2, c3 = st.columns([3, 1, 1])
pick = c1.multiselect("Models to compare (two or more)", have, default=[m for m in ["xgboost", "sarima", "seasonal_naive"] if m in have],
                      format_func=nm.get, key="lab_models")
k = c2.radio("Error measure", ["MAE", "RMSE"], key="lab_k").lower()
lvl = c3.radio("Aggregation", ["Item", "Category", "Total"], key="lab_lvl",
               help="Item scores every item-day. Category and Total add the item forecasts and the actuals first, then score the sums.").lower()
if len(pick) < 2:
    st.info("Pick at least two models.")
    st.stop()
unit = {"item": "units a day per item", "category": "units a day per category", "total": "units a day, all items added"}[lvl]
if lvl != "item":
    st.caption("The category and total forecasts are the sums of the item forecasts, so they always agree with the item-level results.")

with st.expander("What each model does", expanded=True):
    for m in pick:
        st.markdown(f"**{nm[m]}**: {ABOUT.get(m, '')}")

g = errors(tuple(pick), lvl)
t1, t2, t3 = st.tabs(["By category", "By weekday", "By forecast day"])
for tab, by, title, key in [(t1, "grp", "category", "lab_cat"), (t2, "wd", "day of the week", "lab_wd"), (t3, "h", "forecast day", "lab_h")]:
    with tab:
        c = cut(g, by)
        lab = lambda col: [v.title() if by == "grp" else DAYS[v] if by == "wd" else v for v in col]
        fig = go.Figure()
        for m in pick:
            x = c[c["model"] == m].sort_values(by)
            if by == "grp":
                fig.add_bar(x=lab(x[by]), y=x[k], name=nm[m], marker_color=COL.get(m))
            else:
                fig.add_trace(go.Scatter(x=lab(x[by]), y=x[k], mode="lines+markers" if by == "wd" else "lines", name=nm[m],
                                         line=dict(color=COL.get(m), width=2)))
        if by == "grp":
            fig.update_layout(barmode="group")
        fig.update_xaxes(title="days ahead" if by == "h" else None)
        fig.update_yaxes(title=f"{k.upper()}, {unit}")
        chart(style(fig), f"{k.upper()} by {title}, {lvl} level", "Lower is better.", key=key)
        out = c.assign(model=c["model"].map(nm))
        out[by] = lab(out[by])
        download(out.round(4), f"model_lab_{lvl}_{by}.csv", key=f"dl_{key}")

tot = g.groupby("model")[["ae", "se"]].mean()
tot["rmse"] = tot["se"] ** 0.5
col = "ae" if k == "mae" else "rmse"
best = tot[col].idxmin()
tells(f"Of the models picked, {nm[best]} has the lowest {k.upper()} at the {lvl} level ({tot.loc[best, col]:.2f} {unit}). "
      "A model that wins overall can still lose on one category or weekday, which is why this page cuts the error three ways.")
