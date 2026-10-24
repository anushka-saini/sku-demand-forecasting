import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.models import COLORS, ORDER, name
from utils.ui import ACCENT, DARK, SECOND, chart, filters, header, insight, n_note, style

overview = get("comparison_overview")
im = get("classical_item_metrics")
summ = get("classical_summary")
agg = get("aggregate_metrics")
pw = get("pairwise")
mase = get("mase_unrounded")
bias = get("bias")
cats, _, _ = filters()

header("ML and comparison",
       "Every model side by side, scored the same way on the same products and dates: the classical models, the "
       "machine-learning model LightGBM, and seasonal naive as the baseline.")

# the key figure: per-product MASE of every model on the 150-product sample
ev = im[im["origin"] > im["origin"].min()]
wide = ev.groupby(["item_id", "model"])["mase"].mean().unstack()[ORDER]
fig = go.Figure()
for m in ORDER:
    fig.add_box(y=wide[m], name=name(m), marker_color=COLORS[m], boxpoints=False,
                hovertemplate=name(m) + ": %{y:.3f}<extra></extra>")
fig.add_hline(y=1, line_dash="dot", line_color=DARK)
style(fig, height=420, legend=False).update_yaxes(title="MASE per product (lower is better)", range=[0.3, 1.5])
best = overview.set_index("model")["median_mase"].idxmin()
chart(fig, f"Every model beats seasonal naive by a wide margin; {best} is lowest, but only just",
      "Each box is the spread of 150 products' MASE, averaged over the six scored origins. Below 1 beats repeating "
      "last week in-sample. Boxes are drawn without outliers.")

st.markdown("#### One table for everything")
shown = overview.copy()
for col in ["median_mase_726", "beats_naive_share", "beats_lightgbm_share"]:          # blank, not "None", where a model has no score
    shown[col] = shown[col].map(lambda v: "" if v != v else (f"{v:.3f}" if col == "median_mase_726" else f"{v:.0%}"))
st.dataframe(shown, hide_index=True, width="stretch",
             column_config={"model": "Model", "n_items": "Products",
                            "median_mase": st.column_config.NumberColumn("MASE", format="%.3f"),
                            "median_rmse": st.column_config.NumberColumn("RMSE (units)", format="%.2f"),
                            "median_mae": st.column_config.NumberColumn("MAE (units)", format="%.2f"),
                            "median_smape": st.column_config.NumberColumn("sMAPE", format="%.3f"),
                            "median_wape": st.column_config.NumberColumn("WAPE", format="%.3f"),
                            "median_mase_726": "MASE, all 726",
                            "beats_naive_share": "beats naive on",
                            "beats_lightgbm_share": "beats LightGBM on"})
st.caption("Medians over products of each product's mean over the six scored origins, on the 150-product sample. "
           "Seasonal naive, ETS and LightGBM also have a score on all 726 products. 'Beats' = lower MASE for that product.")
n_note(int(overview["n_items"].iloc[0]), "products")

st.markdown("#### Where the simple models win")
winner = wide.idxmin(axis=1).map(name).value_counts()
fig = go.Figure(go.Bar(x=winner.index, y=winner.to_numpy(), marker_color=ACCENT,
                       hovertemplate="%{x}: lowest MASE for %{y} products<extra></extra>"))
style(fig, height=300, legend=False).update_yaxes(title="products where it is the best of the seven")
chart(fig, f"LightGBM is the best model for only {winner.get('LightGBM', 0)} of {len(wide)} products",
      "For each product, the model with the lowest MASE. The classical models together win more often than LightGBM.")
cat_best = summ[summ["category"] != "ALL"].loc[lambda x: x.groupby("category")["median_mase"].idxmin()]
tot_best = agg.loc[agg.groupby("series")["mase"].idxmin()]
c1, c2 = st.columns(2)
c1.markdown("Lowest median MASE by category:\n" + "\n".join(f"- {r.category}: {name(r.model)} ({r.median_mase:.3f})"
                                                           for r in cat_best.itertuples()))
c2.markdown("Lowest MASE on each total:\n" + "\n".join(f"- {r.series}: {name(r.model)} ({r.mase:.3f}, {r.how})"
                                                      for r in tot_best.itertuples()))

st.markdown("#### On all 726 products: LightGBM against ETS and seasonal naive")
MODEL_NAMES = {"lightgbm": "LightGBM", "ets": "ETS", "seasonal_naive": "Seasonal naive"}
cat_rows = ["ALL"] + (cats if cats else ["FOODS", "HOBBIES", "HOUSEHOLD"])
p = pw[pw["category"].isin(cat_rows)].copy()
p["row"] = p["model_a"].map(MODEL_NAMES) + " − " + p["model_b"].map(MODEL_NAMES) + " · " + p["category"] + " (n=" + p["n"].astype(str) + ")"
fig = go.Figure(go.Scatter(
    x=p["median_diff"], y=p["row"], mode="markers", marker=dict(color=ACCENT, size=10),
    error_x=dict(type="data", symmetric=False, array=p["ci_high"] - p["median_diff"], arrayminus=p["median_diff"] - p["ci_low"],
                 color=ACCENT, thickness=1.5),
    customdata=p[["p_value", "share_improved"]],
    hovertemplate="%{y}<br>median difference %{x:+.4f}<br>p = %{customdata[0]:.2e}<br>first model better on "
                  "%{customdata[1]:.1%} of products<extra></extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=80 + 32 * len(p), legend=False).update_xaxes(title="median per-product MASE difference (negative = first model better)")
fig.update_yaxes(autorange="reversed")
le = p[(p["model_a"] == "lightgbm") & (p["model_b"] == "ets") & (p["category"] == "ALL")].iloc[0]
chart(fig, f"LightGBM beats ETS by a small margin ({le['median_diff']:+.4f}) and both beat the baseline by a lot",
      "Median per-product difference in MASE between two models, with a 95% bootstrap interval (Wilcoxon signed-rank test).")

st.markdown("#### Do the models guess too high or too low?")
b = bias[(bias["category"] == "ALL") & (bias["variant"] == "static")].copy()
b["label"] = b["model"].map(MODEL_NAMES)
fig = go.Figure()
fig.add_bar(x=b["label"], y=100 * b["pooled_bias"], name="total", marker_color=ACCENT,
            hovertemplate="%{x}: %{y:+.1f}% in total<extra></extra>")
fig.add_scatter(x=b["label"], y=100 * b["median_item_bias"], mode="markers", name="median product (95% interval)",
                marker=dict(color=SECOND, symbol="diamond", size=11),
                error_y=dict(type="data", symmetric=False, array=100 * (b["ci_high"] - b["median_item_bias"]),
                             arrayminus=100 * (b["median_item_bias"] - b["ci_low"]), color=SECOND),
                hovertemplate="%{x}: median product %{y:+.1f}%<extra></extra>")
fig.add_hline(y=0, line_color=DARK, line_width=1)
style(fig, height=320).update_yaxes(title="forecast against actual (%)", ticksuffix="%")
lb = b[b["model"] == "lightgbm"].iloc[0]
chart(fig, f"LightGBM forecasts too low: {lb['pooled_bias']:+.1%} in total, and for {lb['share_items_under']:.0%} of products",
      "Bars: total forecast against total sales over the scored days, all 726 products. Diamonds: the median product's bias.")

tsb = mase[(mase["model"] == "tsb") & (mase["category"] == "ALL")].iloc[0]
st.caption(f"TSB, built for intermittent demand, only runs on the {int(tsb['n_items'])} intermittent products "
           f"(median MASE {tsb['median_item_mase']:.3f}), too few for this comparison.")

ov = overview.set_index("model")
gap = ov.loc["SARIMA", "median_mase"] - ov.loc["LightGBM", "median_mase"]
eb = b[b["model"] == "ets"].iloc[0]
insight(f"LightGBM has the lowest error, but its lead over SARIMA is only {gap:.3f} MASE, and the classical models "
        f"together win on more products than LightGBM does. ETS is also closer in total ({eb['pooled_bias']:+.1%} against "
        f"{lb['pooled_bias']:+.1%}). For a planner, a well-set-up classical model gives nearly all of the accuracy "
        f"and is easier to explain.")
