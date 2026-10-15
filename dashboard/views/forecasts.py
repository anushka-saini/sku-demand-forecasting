import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup
from utils.ui import ACCENT, DARK, NEUTRAL, SECOND, VARIANT_STYLE, chart, empty_figure, filters, header, n_note, style, vline

f = facts()
items = get("items")
errors = get("forecast_error")
mase = get("mase_unrounded")
pw = get("pairwise")
bias = get("bias")
cov = get("coverage")
cats, segs, _ = filters()
MODEL_NAMES = {"lightgbm": "LightGBM", "ets": "ETS", "tsb": "TSB", "seasonal_naive": "Seasonal naive"}

header("Forecast comparison",
       "Four models were backtested at 6 forecast dates, 28 days apart (2015-12-07 to 2016-05-22), using only data "
       "available at each date. Error is MASE: lower is better, and below 1 beats repeating last week's sales.")

# 1. Model table
m = mase[mase["model"].isin(["seasonal_naive", "ets", "lightgbm"])]
cat_rows = ["ALL"] + (cats if cats else sorted(m["category"].unique()))
tbl = m[m["category"].isin(cat_rows)].pivot_table(index="category", columns="model", values="median_item_mase").reindex(cat_rows)
n = m[m["category"].isin(cat_rows)].pivot_table(index="category", columns="model", values="n_items").reindex(cat_rows)["lightgbm"]
tbl = tbl[["seasonal_naive", "ets", "lightgbm"]].rename(columns=MODEL_NAMES)
tbl.insert(0, "Items (n)", n.astype(int))
st.markdown(f"#### LightGBM has the lowest median error in every category")
st.caption("How to read this: median over items of each item's error averaged across the 6 forecast windows. "
           "The ALL row ignores the category filter.")
st.dataframe(tbl, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                  for c in ["Seasonal naive", "ETS", "LightGBM"]})
tsb = mase[(mase["model"] == "tsb") & (mase["category"] == "ALL")].iloc[0]
st.caption(f"TSB (for intermittent items only): median {tsb['median_item_mase']:.3f}, n = {int(tsb['n_items'])} items "
           "(too few for a fair comparison, so it is left out of the table).")

# 2. Paired tests with CIs and win rates
p = pw[pw["category"].isin(cat_rows)].copy()
p["pair"] = p["model_a"].map(MODEL_NAMES) + " − " + p["model_b"].map(MODEL_NAMES)
p["row"] = p["pair"] + " · " + p["category"] + " (n=" + p["n"].astype(str) + ")"
fig = go.Figure(go.Scatter(
    x=p["median_diff"], y=p["row"], mode="markers", marker=dict(color=ACCENT, size=10),
    error_x=dict(type="data", symmetric=False, array=p["ci_high"] - p["median_diff"], arrayminus=p["median_diff"] - p["ci_low"],
                 color=ACCENT, thickness=1.5),
    customdata=p[["ci_low", "ci_high", "p_value", "share_improved"]],
    hovertemplate="%{y}<br>median difference %{x:+.4f}<br>95% CI %{customdata[0]:+.4f} to %{customdata[1]:+.4f}<br>"
                  "p = %{customdata[2]:.2e}<br>first model better on %{customdata[3]:.1%} of items<extra></extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=80 + 32 * len(p), legend=False).update_xaxes(title="median per-item MASE difference (negative = first model better)")
fig.update_yaxes(autorange="reversed")
le = lookup(f, "forecasts", "pairwise", "lightgbm-ets|ALL")
chart(fig, f"LightGBM beats ETS by a small margin ({le['value']:+.4f}, p = {le['p_value']:.1e}) and both beat the baseline by a lot",
      "Each dot is the median per-item difference in error between two models, with its 95% bootstrap confidence interval "
      "(Wilcoxon signed-rank test). Hover for the share of items where the first model wins.")
win = p[["pair", "category", "n", "share_improved", "p_value"]].rename(columns={"share_improved": "first model better on", "pair": "comparison"})
win["first model better on"] = 100 * win["first model better on"]
st.dataframe(win, hide_index=True, width="stretch", column_config={
    "first model better on": st.column_config.NumberColumn("Items where the first model wins", format="%.1f%%"),
    "p_value": st.column_config.NumberColumn("p-value", format="%.2e")})
small = p[p["n"] < 20]
if len(small):
    st.warning("Fewer than 20 items: " + ", ".join(small["row"]))

# 3. Calibration panel
st.divider()
st.markdown("## Calibration: are the forecasts and their bands honest?")
b = bias[(bias["category"] == "ALL")].copy()
b["label"] = b["model"].map(MODEL_NAMES) + " " + b["variant"]
fig = go.Figure()
fig.add_trace(go.Bar(x=b["label"], y=100 * b["pooled_bias"], name="total (pooled)", marker_color=ACCENT,
                     hovertemplate="%{x}: %{y:+.1f}% total<extra></extra>"))
fig.add_trace(go.Scatter(x=b["label"], y=100 * b["median_item_bias"], mode="markers", name="median item (95% CI)",
                         marker=dict(color=SECOND, symbol="diamond", size=11),
                         error_y=dict(type="data", symmetric=False, array=100 * (b["ci_high"] - b["median_item_bias"]),
                                      arrayminus=100 * (b["median_item_bias"] - b["ci_low"]), color=SECOND),
                         customdata=b[["n_items"]], hovertemplate="%{x}: median item %{y:+.1f}%, n = %{customdata[0]}<extra></extra>"))
fig.add_hline(y=0, line_color=DARK, line_width=1)
style(fig, height=340).update_yaxes(title="forecast vs actual (%)", ticksuffix="%")
lb = b[(b["model"] == "lightgbm") & (b["variant"] == "static")].iloc[0]
chart(fig, f"LightGBM under-forecasts: {lb['pooled_bias']:+.1%} in total, {lb['median_item_bias']:+.1%} for the median item",
      "Bars: total forecast vs total actual over the evaluation days. Diamonds: the median item's bias with a bootstrap 95% CI. "
      "Negative means too low, which leads to stockouts unless safety stock compensates.")

c = cov[(cov["origins"] == "origins 2-6")].copy()
c["label"] = c["model"].map(MODEL_NAMES) + " " + c["variant"] + " · " + c["level"].astype(str) + "%"
fig = go.Figure()
fig.add_trace(go.Bar(x=c["label"], y=100 * c["coverage_before"], name="before (each item's own error quantiles)",
                     marker=dict(color=NEUTRAL, pattern_shape="/"), hovertemplate="%{x}: %{y:.1f}% covered<extra>before</extra>"))
fig.add_trace(go.Bar(x=c["label"], y=100 * c["coverage_after"], name="after (split conformal)", marker_color=ACCENT,
                     hovertemplate="%{x}: %{y:.1f}% covered<extra>after</extra>"))
for lvl in sorted(c["level"].unique()):
    fig.add_hline(y=lvl, line_dash="dot", line_color=DARK, annotation_text=f"{lvl}% target", annotation_position="right")
fig.update_layout(barmode="group")
style(fig, height=360).update_yaxes(title="share of days inside the band", ticksuffix="%", range=[60, 100])
cs = c[(c["model"] == "lightgbm") & (c["variant"] == "static") & (c["level"] == 90)].iloc[0]
chart(fig, f"Calibrated bands hit their targets: LightGBM's 90% band covers {cs['coverage_after']:.1%} of days, up from {cs['coverage_before']:.1%}",
      f"Share of evaluation days (forecast dates 2-6, {int(cs['n_days']):,} item-days per model) on which actual sales fell inside the band. "
      "The dotted lines are the targets.")

# 4. Item explorer
st.divider()
st.markdown("### Item explorer")
fd_all = get("forecast_daily")
weekly = get("sales_weekly")
bo = get("breaks_by_origin")
scope_items = items[items["in_scope"]]
if cats:
    scope_items = scope_items[scope_items["category"].isin(cats)]
if segs:
    scope_items = scope_items[scope_items["segment"].isin(segs)]
scope_ids = sorted(scope_items["item_id"])
if not scope_ids:
    st.info("No modelled items match the sidebar filters.")
    st.stop()
default = st.session_state.get("explorer_item", scope_ids[0])
c1, c2 = st.columns([2, 1])
item = c1.selectbox("Item", scope_ids, index=scope_ids.index(default) if default in scope_ids else 0)
st.session_state["explorer_item"] = item
emodel = c2.radio("Model", ["lightgbm", "ets"], format_func=MODEL_NAMES.get, horizontal=True, key="explorer_model")
band_for = st.radio("Show 90% band for", ["static", "adaptive"], horizontal=True, key="explorer_band")
info = items.set_index("item_id").loc[item]
ibo = bo[(bo["item_id"] == item) & bo["is_eval_origin"]].sort_values("origin")
fd = fd_all[(fd_all["item_id"] == item) & (fd_all["model"] == emodel)]

st.markdown(f"**{item}** · {info['category']} · segment {info['segment']} · mean {info['mean_sales']:.1f} units/day "
            f"· CV {info['cv']:.2f}{' · intermittent' if info['intermittent'] else ''}")
if not ibo["has_break"].any():
    st.info("No break was detected for this item before any forecast date, so its static and adaptive forecasts are "
            "identical by design.")
else:
    last = ibo[ibo["has_break"]].iloc[-1]
    st.markdown(f"Break detected before {ibo['has_break'].sum()} of {len(ibo)} forecast dates. The most recent "
                f"estimated change was the week of **{last['last_change_week'].date()}** (alarm raised "
                f"{last['last_alarm_week'].date()}), so the adaptive model trains on the last "
                f"{int(last['adaptive_window_days'])} days.")

if fd.empty:
    st.plotly_chart(empty_figure("No forecasts stored for this item and model."), width="stretch")
else:
    fig = go.Figure()
    act = fd[fd["variant"] == "static"].sort_values("date")
    for variant in ["static", "adaptive"]:
        v = fd[fd["variant"] == variant].sort_values("date")
        s = VARIANT_STYLE[variant]
        if variant == band_for:
            fig.add_trace(go.Scatter(x=pd.concat([v["date"], v["date"][::-1]]), y=pd.concat([v["hi_90"], v["lo_90"][::-1]]),
                                     fill="toself", fillcolor=s["color"], opacity=0.15, line=dict(width=0),
                                     name=f"{variant} 90% band", hoverinfo="skip", showlegend=True))
        fig.add_trace(go.Scatter(x=v["date"], y=v["forecast"], mode="lines", name=f"{variant} forecast",
                                 line=dict(color=s["color"], dash=s["dash"], width=2),
                                 customdata=v[["lo_90", "hi_90", "horizon"]],
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} (band %{customdata[0]:.1f}-%{customdata[1]:.1f}, "
                                               "day %{customdata[2]} of 28)<extra>" + variant + "</extra>"))
    fig.add_trace(go.Scatter(x=act["date"], y=act["actual"], mode="lines+markers", name="actual",
                             line=dict(color=DARK, width=1), marker=dict(size=4),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:.0f} units sold<extra>actual</extra>"))
    for o in sorted(fd["origin"].unique()):
        fig.add_vline(x=o, line_width=0.8, line_dash="dot", line_color=NEUTRAL)
    style(fig, height=420).update_yaxes(title="units per day")
    item_err = errors[(errors["item_id"] == item) & (errors["model"] == emodel) & errors["is_eval_origin"]]
    me = item_err.groupby("variant")["mase"].mean()
    chart(fig, f"{item}: static MASE {me.get('static', np.nan):.3f} vs adaptive {me.get('adaptive', np.nan):.3f}",
          "Black: actual daily sales. Blue solid: static forecast. Orange dashed: adaptive forecast. Shaded: the "
          "selected variant's 90% band from the item's own past errors (the uncalibrated band, which covers fewer days than 90%; see the calibration panel above). Dotted verticals mark the forecast dates, each "
          "followed by a 28-day window.")

hist = weekly[weekly["item_id"] == item].sort_values("week_start")
fig = go.Figure(go.Scatter(x=hist["week_start"], y=hist["units"], mode="lines", line=dict(color=ACCENT, width=1.2),
                           name="weekly units", hovertemplate="week of %{x|%Y-%m-%d}: %{y:.0f} units<extra></extra>"))
if not fd.empty:
    fig.add_vrect(x0=fd["date"].min(), x1=fd["date"].max(), fillcolor=NEUTRAL, opacity=0.15, line_width=0,
                  annotation_text="evaluation window", annotation_position="top left")
if info["cusum_first_break"]:
    vline(fig, info["first_break_week"], " first break", color=SECOND, dash="dash")
if pd.notna(info["last_break_week"]) and info["last_break_week"] != info["first_break_week"]:
    vline(fig, info["last_break_week"], " latest break", color=SECOND, dash="dot")
style(fig, height=300, legend=False).update_yaxes(title="units per week")
chart(fig, "Full sales history with detected breaks",
      "Weekly units sold since launch. Orange dashed: first detected break. Orange dotted: the most recent break "
      "before the last forecast date (if different). Grey: the evaluation window shown above.")
