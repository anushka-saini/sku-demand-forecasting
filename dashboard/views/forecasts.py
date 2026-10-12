import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.kpis import is_small, per_item_variant_diff, verdict
from utils.ui import ACCENT, DARK, NEUTRAL, SECOND, VARIANT_STYLE, chart, empty_figure, header, n_note, style, vline

items = get("items")
mase = get("model_mase")
tests = get("adaptive_tests")
errors = get("forecast_error")
sb = get("selection_bias")
MODEL_NAMES = {"lightgbm": "LightGBM", "ets": "ETS", "tsb": "TSB", "seasonal_naive": "Seasonal naive"}

header("Forecast comparison",
       "Static models train on all history. Adaptive models retrain from the most recent detected break. Errors are "
       "measured with MASE over 6 rolling 28-day windows (2015-12-07 to 2016-05-22). Lower is better, and below 1 "
       "beats simply repeating last week.")

# 1. Model comparison (static)
m = mase.melt(id_vars="category", var_name="model", value_name="median_mase")
n_cat = items[items["in_scope"]].groupby("category").size()
n_cat["ALL"] = n_cat.sum()
fig = go.Figure()
for model, color, pattern in [("seasonal_naive", NEUTRAL, "."), ("ets", SECOND, "/"), ("lightgbm", ACCENT, "")]:
    d = m[m["model"] == model]
    fig.add_trace(go.Bar(x=d["category"], y=d["median_mase"], name=MODEL_NAMES[model],
                         marker=dict(color=color, pattern_shape=pattern), text=d["median_mase"].round(3),
                         textposition="outside", customdata=d["category"].map(n_cat),
                         hovertemplate="%{x}: median MASE %{y:.3f}<br>n = %{customdata} items<extra>" + MODEL_NAMES[model] + "</extra>"))
fig.add_hline(y=1, line_dash="dot", line_color=DARK, annotation_text="1 = in-sample seasonal naive", annotation_position="top right")
fig.update_layout(barmode="group")
style(fig).update_yaxes(title="median item MASE (static)", range=[0, 1.15])
chart(fig, f"LightGBM is the most accurate static model in every category "
           f"(overall {mase.set_index('category').loc['ALL', 'lightgbm']:.3f} vs "
           f"{mase.set_index('category').loc['ALL', 'seasonal_naive']:.3f} for seasonal naive)",
      "Median over items of each item's MASE averaged across the 6 evaluation windows. Hover for the number of "
      "items in each category.")
st.caption("n per category: " + ", ".join(f"{k} {v}" for k, v in n_cat.items()))

# 2. Forest plot of adaptive vs placebo effects
st.divider()
f = tests[tests["model"].isin(["lightgbm", "ets"])].copy()
f["row"] = f["model"].map(MODEL_NAMES) + " · " + f["category"] + " (n=" + f["n"].astype(str) + ")"
fig = go.Figure()
for comp, color, symbol in [("adaptive (break items)", SECOND, "diamond"), ("placebo (no-break items)", NEUTRAL, "square")]:
    d = f[f["comparison"] == comp]
    fig.add_trace(go.Scatter(
        x=d["median_diff"], y=d["row"], mode="markers", name=comp, marker=dict(color=color, symbol=symbol, size=10),
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["median_diff"],
                     arrayminus=d["median_diff"] - d["ci_low"], color=color, thickness=1.5),
        customdata=d[["ci_low", "ci_high", "p_value", "share_improved", "n"]],
        hovertemplate="%{y}<br>median change %{x:+.4f}<br>95% CI %{customdata[0]:+.4f} to %{customdata[1]:+.4f}"
                      "<br>p = %{customdata[2]:.4f}<br>items improved %{customdata[3]:.1%}<extra>" + comp + "</extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=520).update_xaxes(title="median per-item MASE change vs static (negative = better)")
fig.update_yaxes(autorange="reversed")
a_all = tests[(tests["comparison"].str.startswith("adaptive")) & (tests["model"] == "lightgbm") & (tests["category"] == "ALL")].iloc[0]
adapt_rows = f[f["comparison"].str.startswith("adaptive") & f["ci_high"].notna()]
n_helped = int((adapt_rows["ci_high"] < 0).sum())
title = (f"Retraining after a break did not significantly help in any of the {len(adapt_rows)} model-category groups"
         if n_helped == 0 else f"Retraining significantly helped in {n_helped} of {len(adapt_rows)} model-category groups")
chart(fig, f"{title}. Overall for LightGBM: {verdict(a_all['ci_low'], a_all['ci_high'])}",
      "Each dot is a median per-item change with its 95% bootstrap confidence interval. Whiskers entirely right "
      "of 0 mean adaptation hurt. Whiskers crossing 0 mean no detectable difference. Orange diamonds are items "
      "with a break; grey squares are the placebo control on items without one.")
small = f[f["n"] < 20]
if len(small):
    st.warning("Small subgroups (n < 20): " + ", ".join(small["row"]) + ". Treat these with caution.")

# 3. Distribution of per-item differences
st.divider()
model = st.radio("Model for the distribution", ["lightgbm", "ets"], format_func=MODEL_NAMES.get, horizontal=True)
ad = per_item_variant_diff(errors, model, "adaptive")
pl = per_item_variant_diff(errors, model, "placebo")
lo, hi = np.percentile(np.r_[ad["diff"], pl["diff"]], [1, 99])          # display range only; stats use all items
bins = dict(start=lo, end=hi, size=(hi - lo) / 60)
fig = go.Figure()
fig.add_trace(go.Histogram(x=ad["diff"].clip(lo, hi), xbins=bins, name=f"adaptive, break items (n={len(ad)})",
                           marker=dict(color=SECOND), opacity=0.75, histnorm="percent",
                           hovertemplate="change %{x:+.3f}: %{y:.1f}% of items<extra>adaptive</extra>"))
fig.add_trace(go.Histogram(x=pl["diff"].clip(lo, hi), xbins=bins, name=f"placebo, no-break items (n={len(pl)})",
                           marker=dict(color=NEUTRAL, pattern_shape="/"), opacity=0.6, histnorm="percent",
                           hovertemplate="change %{x:+.3f}: %{y:.1f}% of items<extra>placebo</extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
fig.update_layout(barmode="overlay")
style(fig).update_xaxes(title="per-item MASE change vs static (negative = better)")
fig.update_yaxes(title="% of items")
chart(fig, f"Both distributions are centred almost exactly on zero. Median {ad['diff'].median():+.4f} (adaptive) vs "
           f"{pl['diff'].median():+.4f} (placebo)",
      "Each item's average MASE change across the evaluation windows. The display is trimmed to the 1st-99th "
      "percentile (the tails are piled into the end bins); the medians use all items.")

# 4. Selection bias
st.divider()
st.markdown("#### Are break items simply harder to forecast? Checking that the comparison is fair")
st.caption("How to read this: if items with breaks differed a lot from those without (in volume, variability or "
           "history), comparing their errors directly would be unfair. A standardized difference (SMD) above 0.1 "
           "signals imbalance. The regression then compares their static error after adjusting for those differences.")
bal = sb[sb["kind"] == "balance"][["variable", "break_items", "no_break_items", "smd"]].copy()
bal["variable"] = bal["variable"].map({"mean_sales": "Mean units/day", "cv": "Variability (CV)",
                                       "zero_share": "Share of zero days", "n_days": "Days of history"}).fillna(bal["variable"])
c1, c2 = st.columns([1.15, 1])
c1.dataframe(bal, hide_index=True, width="stretch", column_config={
    "variable": "Before first forecast date",
    "break_items": st.column_config.NumberColumn(f"Break (n={int(sb['n_break'].iloc[0])})", format="%.2f"),
    "no_break_items": st.column_config.NumberColumn(f"No break (n={int(sb['n_no_break'].iloc[0])})", format="%.2f"),
    "smd": st.column_config.NumberColumn("SMD", format="%.2f")})
reg = sb[sb["kind"] == "regression"]
fig = go.Figure(go.Scatter(
    x=reg["pct_effect"], y=reg["variable"], mode="markers", marker=dict(color=ACCENT, size=10),
    error_x=dict(type="data", symmetric=False, array=reg["ci_high_pct"] - reg["pct_effect"],
                 arrayminus=reg["pct_effect"] - reg["ci_low_pct"], color=ACCENT),
    customdata=reg[["ci_low_pct", "ci_high_pct", "p_value"]],
    hovertemplate="%{y}: %{x:+.1f}%<br>95% CI %{customdata[0]:+.1f}% to %{customdata[1]:+.1f}%<br>p = %{customdata[2]:.3f}<extra></extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=260, legend=False).update_xaxes(title="static MASE of break items vs no-break items (%)")
with c2:
    adj = reg[reg["variable"] == "lightgbm adjusted"].iloc[0]
    crosses = adj["ci_low_pct"] < 0 < adj["ci_high_pct"]
    chart(fig, (f"Not detectably harder: {adj['pct_effect']:+.1f}% after adjustment (95% CI includes 0)" if crosses
                else f"Break items are {adj['pct_effect']:+.1f}% harder to forecast after adjustment"),
          "Raw = simple comparison. Adjusted = controlling for volume, variability, zero share, history length "
          "and category.")
imb = bal[bal["smd"].abs() > 0.1]
if len(imb):
    st.caption("Imbalanced characteristics (|SMD| > 0.1): " + ", ".join(f"{r.variable} ({r.smd:+.2f})" for r in imb.itertuples())
               + ". A longer history likely gives more time for a break to be detected. This is why the adjusted "
               "estimate is the one to read.")

# 5. Item explorer
st.divider()
st.markdown("### Item explorer")
fd_all = get("forecast_daily")
weekly = get("sales_weekly")
bo = get("breaks_by_origin")
scope_ids = sorted(items.loc[items["in_scope"], "item_id"])
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
          "selected variant's empirical 90% band, built from the item's errors at earlier forecast dates. Dotted verticals mark the forecast dates, each "
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
