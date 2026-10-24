import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup
from utils.ui import ACCENT, DARK, NEUTRAL, SECOND, VARIANT_STYLE, chart, empty_figure, filters, header, insight, style, vline

items = get("items")
errors = get("forecast_error")
cats, segs, _ = filters()
MODEL_NAMES = {"lightgbm": "LightGBM", "ets": "ETS", "tsb": "TSB", "seasonal_naive": "Seasonal naive"}

header("Forecast explorer",
       "Pick any of the 726 products to see its LightGBM or ETS forecasts at the six scored dates, next to what "
       "actually sold and the product's detected breaks.")

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

f = facts()
trunc = lookup(f, "forecasts", "adaptation_variants", "all detected breaks|truncate")
band = lookup(f, "forecasts", "band_coverage", "lightgbm_static_90_origins2-6")
insight(f"For most products the static and retrained forecasts sit close together: retraining after a shift changed "
        f"the median error by only {trunc['value']:+.4f} MASE, and made it worse. The shaded band is the simple one, "
        f"which held the actual sales on {band['coverage_before']:.1%} of days instead of 90%; the Uncertainty and "
        f"stock page shows the calibrated version.")
