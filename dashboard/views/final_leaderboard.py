import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, MAIN, best_info, load, names, need_results, text
from utils.ui import chart, download, kpis, page, style, tells, verdict_box

need_results()
lb = load("leaderboard")
nm = names(lb)
info = best_info()
page("Final leaderboard", "Which model forecasts best on the six untouched test windows, and does a combination beat the best single model?")

lbi = lb[lb["level"] == "item"].sort_values("rank")
best = lbi.iloc[0]
naive = lbi[lbi["model"] == "seasonal_naive"].iloc[0]
single = lbi[lbi["family"] != "Combination"].iloc[0]
split = load("split")
st.caption(f"150 items, 28 days ahead, 6 test windows ({split.iloc[-1]['origins']}). Tuning used the three validation windows only; "
           f"the test windows were scored once. Headline: each item's mean error, then the median over items.")

kpis([
    {"label": "Best model", "value": best["name"], "compare": f"rank 1 of {len(lbi)} by MAE and RMSE"},
    {"label": "MAE (units a day)", "value": f"{best['mae']:.2f}", "compare": f"{best['mae'] / naive['mae'] - 1:+.1%} vs seasonal naive",
     "tone": "good" if best["mae"] < naive["mae"] else "bad", "help": "Median over items of the mean absolute error."},
    {"label": "RMSE (units a day)", "value": f"{best['rmse']:.2f}", "compare": f"{best['rmse'] / naive['rmse'] - 1:+.1%} vs seasonal naive",
     "tone": "good" if best["rmse"] < naive["rmse"] else "bad"},
    {"label": "Best single model", "value": single["name"], "compare": f"MAE {single['mae']:.2f}, RMSE {single['rmse']:.2f}"},
])

c1, c2 = st.columns(2)
shown = c1.radio("Models shown", [f"Main set ({len(MAIN)})", f"All {len(lbi)}"], horizontal=True, key="lb_fam",
                 help="The main set is a baseline, the classical models that matter, the two tree models, one neural net and the "
                      "best combination. All models adds the rest.")
metric = c2.radio("Error measure", ["MAE", "RMSE"], horizontal=True, key="lb_metric",
                  help="MAE and RMSE decide the ranking. Lower is better.")
d = lbi if shown.startswith("All") else lbi[lbi["model"].isin(MAIN)]
k = metric.lower()
d = d.sort_values(k)
fig = go.Figure(go.Bar(y=[nm[m] for m in d["model"]], x=d[k], orientation="h", marker_color=[COL.get(m, "#888") for m in d["model"]],
                       error_x=dict(type="data", array=d[f"{k}_std"].fillna(0), thickness=1.2), text=[f"{v:.2f}" for v in d[k]],
                       textposition="outside", hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title=f"median {metric} per item, units a day (lower is better)")
chart(style(fig, height=max(300, 30 * len(d) + 80), legend=False), f"{metric} of every model on the test windows",
      "Whiskers on the neural nets are the standard deviation over three seeds.", key="lb_bar")

tbl = lbi if shown.startswith("All") else lbi[lbi["model"].isin(MAIN)]
mine = tbl[["rank", "name", "family", "mae", "mae_std", "rmse", "rmse_std", "mape_nonzero", "wape", "pooled_mae", "pooled_rmse",
            "fit_seconds", "fit_seconds_kind"]].copy()
mine.columns = ["Rank", "Model", "Group", "MAE", "MAE sd (seeds)", "RMSE", "RMSE sd (seeds)", "MAPE (non-zero days)", "WAPE",
                "Pooled MAE", "Pooled RMSE", "Fit seconds (6 windows)", "Time is"]
st.dataframe(mine.round(3), hide_index=True, width="stretch")
st.caption("MAPE leaves out days with no sales (about 5% of item-days), so it is not comparable across items with many zeros; "
           "WAPE is total absolute error over total sales and needs no such cut. Rank is the mean of the MAE rank and the RMSE rank. "
           "Fit seconds for the earlier models are an estimate from a short timing run; AutoTS, ARIMA and ARMA were not timed.")
download(lb, "final_leaderboard.csv")

rows = []
for _, r in lbi[lbi["family"] == "Combination"].iterrows():
    both = r["mae"] < single["mae"] and r["rmse"] < single["rmse"]
    one = r["mae"] < single["mae"] or r["rmse"] < single["rmse"]
    rows.append(f"{r['name']}: MAE {r['mae']:.2f}, RMSE {r['rmse']:.2f}; "
                + ("beats" if both else "beats on one measure only" if one else "does not beat")
                + f" the best single model ({single['name']}: {single['mae']:.2f} / {single['rmse']:.2f}).")
verdict_box(rows, head="Does a combination win?")

t1, t2, t3, t4, t5 = st.tabs(["Error by horizon", "Ensemble weights", "Is the gain real?", "Where models fail", "What it means"])
with t1:
    hz = load("horizon")
    pick = st.multiselect("Models", list(lbi["model"]), default=[m for m in MAIN if m in ("xgboost", "combination", "sarima", "seasonal_naive")],
                          format_func=nm.get, key="lb_hz")
    hk = st.radio("Measure", ["MAE", "RMSE"], horizontal=True, key="lb_hzm").lower()
    f2 = go.Figure()
    for m in pick:
        g = hz[hz["model"] == m]
        f2.add_trace(go.Scatter(x=g["h"], y=g[hk], mode="lines", name=nm[m], line=dict(color=COL.get(m), width=2)))
    f2.update_xaxes(title="days ahead")
    f2.update_yaxes(title=f"{hk.upper()}, units a day")
    chart(style(f2), "Error by forecast day", "Error at each of the 28 days ahead, over all test windows and items.", key="lb_h")
with t2:
    w = load("weights")
    w = w[w["weight"] > 1e-6].sort_values("weight")
    f3 = go.Figure(go.Bar(y=w["name"], x=w["weight"], orientation="h", marker_color=[COL.get(m, "#888") for m in w.iloc[:, 0]],
                          text=[f"{v:.2f}" for v in w["weight"]], textposition="outside"))
    f3.update_xaxes(title="weight in the stacked forecast")
    chart(style(f3, legend=False), "Stacking weights (non-negative, fitted on validation windows only)",
          f"Weights sum to {w['weight'].sum():.2f}. The simple average uses: " + ", ".join(nm.get(m, m) for m in info["avg_top3_members"]) + ".",
          key="lb_w")
with t3:
    dm = load("dm")
    show = dm.copy()
    show["model_a"], show["model_b"] = show["model_a"].map(nm), show["model_b"].map(nm)
    st.dataframe(show[["model_a", "model_b", "loss", "mean_loss_a", "mean_loss_b", "dm_stat", "p_value", "significant_5pct",
                       "share_items_a_better", "wilcoxon_p"]].round(4), hide_index=True, width="stretch")
    sig = dm[dm["significant_5pct"] & dm["a_better"]]
    st.caption("Diebold-Mariano on per-day loss differences over the 168 test days (Newey-West, 27 lags, small-sample correction), plus "
               f"a paired Wilcoxon test per item. {len(sig)} of {len(dm)} tests show the first model significantly better at 5%.")
with t4:
    seg = load("segments")
    ms = [m for m in [best["model"], "seasonal_naive", "sarima"] if m in set(seg["model"])]
    s = seg[seg["model"].isin(ms)].pivot(index="segment", columns="model", values="mae")
    s.columns = [nm[c] for c in s.columns]
    s["Item-days"] = seg[seg["model"] == best["model"]].set_index("segment")["item_days"]
    st.dataframe(s.round(3), width="stretch")
    st.caption("MAE on groups of item-days (test windows). Event and SNAP days come from the M5 calendar; 'zero week before origin' "
               "means nothing sold in the 7 days before the forecast was made.")
with t5:
    tx = text()
    st.markdown("**Which model wins, and why**")
    st.markdown(tx["winner"])
    st.markdown(tx["combos"])
    st.markdown(tx["dm"])
    st.markdown(tx["why"])
    st.markdown(tx["nn"])
    st.markdown("**Where the best model fails**")
    for t in tx["failure"]:
        st.markdown("- " + t)
    st.markdown("**Limitations**")
    for t in tx["limitations"]:
        st.markdown("- " + t)
    st.markdown("**Challenges**")
    for t in tx["challenges"]:
        st.markdown("- " + t)
    st.markdown("**Future scope**")
    for t in tx["future"]:
        st.markdown("- " + t)
tells(f"{best['name']} has the best combined rank on MAE and RMSE. Its MAE is {abs(best['mae'] / naive['mae'] - 1):.0%} "
      f"{'below' if best['mae'] < naive['mae'] else 'above'} seasonal naive and its RMSE {abs(best['rmse'] / naive['rmse'] - 1):.0%} "
      f"{'below' if best['rmse'] < naive['rmse'] else 'above'}.")
