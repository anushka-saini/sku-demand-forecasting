import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, MAIN, best_info, load, names, need_results
from utils.ui import chart, download, kpis, page, style, tells, verdict_box

need_results()
lb = load("leaderboard")
nm = names(lb)
info = best_info()
page("Leaderboard", "Which model forecasts best on the six untouched test windows, and does a combination beat the best single model?")

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

with st.expander("Stacking weights (how the stacked forecast mixes models)"):
    w = load("weights")
    w = w[w["weight"] > 1e-6].sort_values("weight")
    f3 = go.Figure(go.Bar(y=w["name"], x=w["weight"], orientation="h", marker_color=[COL.get(m, "#888") for m in w.iloc[:, 0]],
                          text=[f"{v:.2f}" for v in w["weight"]], textposition="outside"))
    f3.update_xaxes(title="weight in the stacked forecast")
    chart(style(f3, legend=False), "Stacking weights (non-negative, fitted on validation windows only)",
          f"Weights sum to {w['weight'].sum():.2f}. The simple average uses: " + ", ".join(nm.get(m, m) for m in info["avg_top3_members"]) + ".",
          key="lb_w")
st.caption("Error by forecast day is on the Model Lab page. Significance tests and where the models fail are on the Diagnostics page.")
tells(f"{best['name']} has the best combined rank on MAE and RMSE. Its MAE is {abs(best['mae'] / naive['mae'] - 1):.0%} "
      f"{'below' if best['mae'] < naive['mae'] else 'above'} seasonal naive and its RMSE {abs(best['rmse'] / naive['rmse'] - 1):.0%} "
      f"{'below' if best['rmse'] < naive['rmse'] else 'above'}.")
