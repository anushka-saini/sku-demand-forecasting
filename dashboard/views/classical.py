import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.m2 import COLORS, ORDER, name
from utils.ui import chart, header, n_note, style

header("Classical models",
       "Milestone 2, notebooks 13-15. Holt-Winters, ARIMA and SARIMA on the same backtest as Milestone 1 (7 origins, "
       "28-day forecasts, origins 1-6 scored), on a 150-item sample, compared with seasonal naive, damped Holt-Winters "
       "(the Milestone 1 ETS) and LightGBM.")

summ = get("m2_summary")
tests = get("m2_tests")
agg = get("m2_aggregate")
fc = get("m2_forecasts")
im = get("m2_item_metrics")
orders = get("m2_orders")
diag = get("m2_diagnostics")
diag_items = get("m2_diag_items")
params = get("m2_params")
stats = get("m2_run_stats")
shift = get("m2_shift")
h4 = get("m2_h4")

cat = st.radio("Category", ["ALL", "FOODS", "HOBBIES", "HOUSEHOLD"], horizontal=True, key="m2_cat")
s = summ[summ["category"] == cat].set_index("model").loc[ORDER].reset_index()
best = s.loc[s["median_mase"].idxmin(), "model"]
st.markdown(f"#### {name(best)} has the lowest median MASE ({s['median_mase'].min():.3f}) in {cat.lower() if cat != 'ALL' else 'all categories'}")
table = s.assign(model=s["model"].map(name))[["model", "n_items", "median_rmse", "median_mae", "median_mape",
                                              "median_smape", "median_wape", "median_mase", "mean_mase"]]
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"median_rmse": st.column_config.NumberColumn("RMSE (units)", format="%.2f"),
                            "median_mae": st.column_config.NumberColumn("MAE (units)", format="%.2f"),
                            "median_mape": st.column_config.NumberColumn("MAPE", format="%.3f"),
                            "median_smape": st.column_config.NumberColumn("sMAPE", format="%.3f"),
                            "median_wape": st.column_config.NumberColumn("WAPE", format="%.3f"),
                            "median_mase": st.column_config.NumberColumn("MASE", format="%.3f"),
                            "mean_mase": st.column_config.NumberColumn("MASE (mean)", format="%.3f")})
st.caption(f"Median across items of each item's mean over origins 1-6. MAPE leaves out the "
           f"{s['mape_excluded_share'].iloc[0]:.1%} of days with no sales. Lower is better for every column.")
n_note(int(s["n_items"].iloc[0]))

st.markdown("#### Pre-registered tests (paired Wilcoxon on per-item MASE, Holm-corrected)")
t = tests.assign(model_a=tests["model_a"].map(name), model_b=tests["model_b"].map(name))
st.dataframe(t[["hypothesis", "model_a", "model_b", "n", "median_diff", "ci_low", "ci_high", "p_holm", "a_better"]],
             hide_index=True, width="stretch",
             column_config={"median_diff": st.column_config.NumberColumn("median MASE difference (A − B)", format="%+.4f"),
                            "ci_low": st.column_config.NumberColumn("95% CI low", format="%+.4f"),
                            "ci_high": st.column_config.NumberColumn("95% CI high", format="%+.4f"),
                            "p_holm": st.column_config.NumberColumn("Holm p", format="%.4f"),
                            "a_better": st.column_config.CheckboxColumn("A significantly better")})

st.markdown("#### Totals (category and California)")
st.caption("Classical models are fitted on each total. Seasonal naive, ETS and LightGBM are the sum of their item "
           "forecasts, because they were never fitted on totals.")
piv = agg.pivot(index="model", columns="series", values="mase").reindex(ORDER)
piv.index = piv.index.map(name)
st.dataframe(piv, width="stretch", column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in piv.columns})

st.markdown("#### Holt-Winters additive vs multiplicative (H4, descriptive)")
groups = ["zero-heavy", "not zero-heavy", "all"]
st.caption("Items: median MASE; zero-heavy = share of zero days before origin 0 above the sample median. "
           "Multiplicative on items uses sales + 1.")
st.dataframe(h4[h4["group"].isin(groups)][["group", "n_items", "zero_share_median", "median_mase_hw_add",
                                         "median_mase_hw_mul", "share_items_add_better"]], hide_index=True, width="stretch")
st.caption("Totals: MASE of each model fitted on the total; competitive = multiplicative within 5% of additive.")
st.dataframe(h4[~h4["group"].isin(groups)][["group", "median_mase_hw_add", "median_mase_hw_mul", "mul_vs_add",
                                          "mul_within_5pct"]].rename(columns={"group": "total", "median_mase_hw_add": "MASE additive",
                                                                              "median_mase_hw_mul": "MASE multiplicative"}),
             hide_index=True, width="stretch")

st.markdown("---")
st.markdown("### One item, all models")
items = sorted(fc["item_id"].unique())
c1, c2 = st.columns([2, 1])
item = c1.selectbox("Item", items, key="m2_item")
origins = sorted(fc["origin"].unique())[1:]
origin = c2.selectbox("Forecast origin", origins, index=len(origins) - 1, format_func=lambda o: str(o.date()), key="m2_origin")
show = st.multiselect("Models", ORDER, default=["seasonal_naive", "hw_add", "sarima", "lightgbm"], format_func=name, key="m2_models")
g = fc[(fc["item_id"] == item) & (fc["origin"] == origin)]
fig = go.Figure()
a = g[g["model"] == "sarima"].sort_values("date")
fig.add_scatter(x=a["date"], y=a["actual"], mode="lines+markers", name="actual", line=dict(color="#7F7F7F", width=2.5))
for m in show:
    x = g[g["model"] == m].sort_values("date")
    fig.add_scatter(x=x["date"], y=x["forecast"], mode="lines", name=name(m),
                    line=dict(color=COLORS[m], dash="dot" if m == "seasonal_naive" else "solid"))
err = im[(im["item_id"] == item) & (im["origin"] == origin)].set_index("model")
closest = err.loc[[m for m in show if m in err.index], "mae"].idxmin() if show else None
chart(style(fig), f"{item}, origin {origin.date()}: {name(closest)} is closest (MAE {err.loc[closest, 'mae']:.2f})" if closest
      else f"{item}, origin {origin.date()}", "Black is what sold; each line is one model's 28-day forecast made at the origin.")
e = im[(im["item_id"] == item) & im["origin"].isin(origins)]
e = e.groupby("model")[["rmse", "mae", "mape", "mase"]].mean().reindex(ORDER).dropna(how="all")
e.index = e.index.map(name)
st.caption("This item's errors, averaged over origins 1-6:")
st.dataframe(e, width="stretch", column_config={c: st.column_config.NumberColumn(c.upper(), format="%.3f") for c in e.columns})

o = orders[orders["series"] == item].iloc[0]
st.markdown(f"**Chosen orders (AICc at origin 0):** SARIMA {o['sarima_order']}{o['sarima_sorder']}, "
            f"ARIMA {o['arima_order']}, d = {o['d']}, D = {o['D']}")
p = params[params["series"] == item].set_index("model").dropna(axis=1, how="all").drop(columns="level")
st.dataframe(p, width="stretch")

st.markdown("---")
st.markdown("### Diagnostics (residuals at origin 0)")
dd = diag[diag["level"] == "item"].assign(model=lambda x: x["model"].map(name))
st.dataframe(dd[["model", "n", "ljung_box_pass", "normality_pass", "roots_ok"]], hide_index=True, width="stretch",
             column_config={"ljung_box_pass": st.column_config.NumberColumn("Ljung-Box pass (lags 7, 14)", format="percent"),
                            "normality_pass": st.column_config.NumberColumn("Jarque-Bera pass", format="percent"),
                            "roots_ok": st.column_config.NumberColumn("stationary and invertible", format="percent")})
st.caption("Ljung-Box pass = no autocorrelation left in the residuals. Normality matters for intervals and the textbook "
           "safety-stock formula, not for the point forecasts.")
di = diag_items[diag_items["series"] == item].assign(model=lambda x: x["model"].map(name))
st.dataframe(di[["model", "lb_p7", "lb_p14", "jb_p"]], hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in ["lb_p7", "lb_p14", "jb_p"]})
rs = stats.assign(model=stats["model"].map(name))
st.caption("Fallbacks to seasonal naive and convergence warnings, per fit:")
st.dataframe(rs[["model", "level", "fits", "fallback_rate", "not_converged_rate"]], hide_index=True, width="stretch")

st.markdown("---")
st.markdown("### Does a shift-aware SARIMA help? (H3)")
fig = go.Figure()
lab = shift["variant"] + ": " + shift["group"]
fig.add_scatter(x=shift["median_diff"], y=lab, mode="markers", marker=dict(color="#D55E00", size=10),
                error_x=dict(type="data", symmetric=False, array=shift["ci_high"] - shift["median_diff"],
                             arrayminus=shift["median_diff"] - shift["ci_low"], color="#999999"),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>", showlegend=False)
fig.add_vline(x=0, line_color="#666666")
main = shift[shift["group"] == "all break item-origins"]
helps = main["helps"].astype(str).str.lower().eq("true").any()
chart(style(fig, height=330, legend=False),
      "A shift-aware SARIMA variant meets the 'helps' rule" if helps else "No shift-aware SARIMA variant meets the 'helps' rule",
      "Each dot is the median per-item change in MASE versus static SARIMA, with a 95% bootstrap CI. Negative = better. "
      "Restart = fit only after the detected break; dummy = a level-shift regressor; placebo = a random cut on items without a break.")
