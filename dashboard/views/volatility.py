import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.final import best_info, load, names, need_results, text
from utils.ui import chart, kpis, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
info = best_info()
page("Volatility (GARCH)", "How much do the errors of the best model swing from day to day, and does that help with prediction intervals?")
st.markdown(f"A GARCH(1,1) was fitted to the forecast errors of **{nm[info['model']]}**. GARCH models the *size* of the errors. It does "
            "not change the forecast, so MAE and RMSE are exactly the same with or without it; what it gives is a spread for each future day.")

g = load("garch")
lm = load("archlm")
dt = load("daytype")
iv = load("intervals")
n95 = g[g["nominal"] == 0.95].set_index("interval")
n80 = g[g["nominal"] == 0.8].set_index("interval")
gb, cs = "GARCH(1,1)", "constant std"
kpis([
    {"label": "Items with changing variance", "value": f"{(lm['arch_lm_p'] < 0.05).mean():.0%}",
     "compare": "ARCH-LM test rejects constant variance at 5%", "help": "Engle's ARCH-LM test on each item's forecast errors."},
    {"label": "80% band coverage, GARCH", "value": f"{n80.loc[gb, 'coverage']:.1%}", "compare": f"constant spread: {n80.loc[cs, 'coverage']:.1%}"},
    {"label": "95% band coverage, GARCH", "value": f"{n95.loc[gb, 'coverage']:.1%}", "compare": f"constant spread: {n95.loc[cs, 'coverage']:.1%}"},
    {"label": "Interval score, 80% (lower is better)", "value": f"{n80.loc[gb, 'interval_score']:.2f}",
     "compare": f"constant spread: {n80.loc[cs, 'interval_score']:.2f}",
     "tone": "good" if n80.loc[gb, "interval_score"] < n80.loc[cs, "interval_score"] else "bad"},
])
fig = go.Figure()
fig.add_bar(x=dt["day_type"], y=dt["mean_sigma_over_scale"], name="GARCH spread (sigma)", marker_color="#0072B2")
fig.add_bar(x=dt["day_type"], y=dt["mean_abs_error_over_scale"], name="actual absolute error", marker_color="#D55E00")
fig.update_layout(barmode="group")
fig.update_yaxes(title="relative to the item's usual sales")
chart(style(fig), "Fitted spread and real error by type of day",
      "Where the orange bar is clearly above the blue one, the spread under-states the error on those days.", key="vol_dt")

st.markdown("**Interval table** (scored windows, all items)")
st.dataframe(g.round(3), hide_index=True, width="stretch")
item = st.selectbox("Item", sorted(iv["item_id"].unique()), key="vol_item")
x = iv[iv["item_id"] == item]
origin = st.select_slider("Forecast date", sorted(x["origin"].unique()), format_func=lambda d: f"{pd.Timestamp(d):%d %b %Y}", key="vol_o")
x = x[x["origin"] == origin].sort_values("date")
z = 1.2816
f2 = go.Figure()
f2.add_trace(go.Scatter(x=pd.concat([x["date"], x["date"][::-1]]),
                        y=pd.concat([x["forecast"] + z * x["sig_garch"], np.maximum(x["forecast"] - z * x["sig_garch"], 0)[::-1]]),
                        fill="toself", fillcolor="rgba(0,114,178,0.18)", line=dict(width=0), name="80% band, GARCH"))
f2.add_trace(go.Scatter(x=x["date"], y=x["forecast"] + z * x["sig_const"], mode="lines",
                        line=dict(color="#8C8C8C", dash="dash", width=1), name="80% band, constant"))
f2.add_trace(go.Scatter(x=x["date"], y=np.maximum(x["forecast"] - z * x["sig_const"], 0), mode="lines",
                        line=dict(color="#8C8C8C", dash="dash", width=1), showlegend=False))
f2.add_trace(go.Scatter(x=x["date"], y=x["forecast"], name="forecast", line=dict(color="#0072B2", width=2)))
f2.add_trace(go.Scatter(x=x["date"], y=x["actual"], name="actual", line=dict(color="#222222", width=2)))
f2.update_yaxes(title="units a day")
chart(style(f2, height=400), f"{item}: forecast band for one 28-day window",
      "The GARCH band follows the recent size of the errors; the constant band does not.", key="vol_item_chart")
st.markdown(text()["garch"])
tells("Volatility helps decide how much buffer stock to keep and when a forecast is less certain, but it cannot make the point "
      "forecast more accurate.")
