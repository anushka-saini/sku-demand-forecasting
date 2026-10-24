import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts
from utils.kpis import base_cost_assumptions, pct_change, tradeoff
from utils.ui import (ACCENT, DARK, NEUTRAL, POLICY_STYLE, SECOND, assumed, assumed_chart, chart, filters, header,
                      insight, n_note, style)

f = facts()
kpi = get("kpi_all")
inv = get("inventory_policy")
sens = get("cost_sensitivity")
cal = get("calibrated_inv")
wsum = get("window_summary")
wdiff = get("window_diffs")
wmeta = get("windows")
cov = get("coverage")
nv = get("newsvendor")
be = get("breakeven")
cats, _, sl = filters()
H_BASE, M_BASE = base_cost_assumptions(inv)
ORDER = list(POLICY_STYLE)

header("Uncertainty and stock",
       "A forecast only matters if it changes how much stock a store holds, so we put an honest error band around "
       "each forecast and then simulate restocking every product day by day over 168 days.")

# the key figure: how often the actual sales fall inside the band, before and after calibration
MODEL_NAMES = {"lightgbm": "LightGBM", "seasonal_naive": "Seasonal naive"}
c = cov[(cov["origins"] == "origins 2-6")].copy()
c["label"] = c["model"].map(MODEL_NAMES) + " " + c["variant"] + " · " + c["level"].astype(str) + "%"
fig = go.Figure()
fig.add_trace(go.Bar(x=c["label"], y=100 * c["coverage_before"], name="before (each product's own past errors)",
                     marker=dict(color=NEUTRAL, pattern_shape="/"), hovertemplate="%{x}: %{y:.1f}% covered<extra>before</extra>"))
fig.add_trace(go.Bar(x=c["label"], y=100 * c["coverage_after"], name="after (conformal, pooled by category)", marker_color=ACCENT,
                     hovertemplate="%{x}: %{y:.1f}% covered<extra>after</extra>"))
for lvl in sorted(c["level"].unique()):
    fig.add_hline(y=lvl, line_dash="dot", line_color=DARK, annotation_text=f"{lvl}% target", annotation_position="right")
fig.update_layout(barmode="group")
style(fig, height=360).update_yaxes(title="share of days inside the band", ticksuffix="%", range=[60, 100])
fig.update_xaxes(title="forecast and band level")
band = c[(c["model"] == "lightgbm") & (c["variant"] == "static") & (c["level"] == 90)].iloc[0]
chart(fig, f"Calibrated bands hit their targets: LightGBM's 90% band covers {band['coverage_after']:.1%} of days, up from {band['coverage_before']:.1%}",
      f"Share of scored days (origins 2-6, {int(band['n_days']):,} product-days per model) on which actual sales fell "
      "inside the band. The dotted lines are the targets.")

st.divider()
st.markdown("## The stock simulation")
st.markdown("Each day, if stock on hand plus stock on order falls to the reorder point, we order up to it plus a week "
            "of forecast demand; unmet demand is lost. Every selector below only filters simulations that were run "
            "in advance.")
assumed("Lead time, holding cost and stockout cost are assumptions, not observed data. Charts that depend on them "
        "have a dashed orange frame.")

c1, c2, c3 = st.columns(3)
lt = c1.radio("Lead time, days (assumed)", sorted(kpi["lead_time"].unique()), index=1, horizontal=True)
h = c2.select_slider("Holding cost, % of price per year (assumed)", sorted(sens["holding_rate"].unique()),
                     value=H_BASE, format_func=lambda x: f"{x:.0%}")
m = c3.select_slider("Cost of a lost sale, × price (assumed)", sorted(sens["stockout_multiple"].unique()),
                     value=M_BASE, format_func=lambda x: f"{x:g}×")
st.caption(f"Target service level {sl:.0%} (sidebar). Category filter applies to fill rate, stockouts and inventory; "
           "costs cover all items.")

k = kpi[(kpi["lead_time"] == lt)]
if cats:
    k = k[k["category"].isin(cats)]
t = tradeoff(k.assign(stockout_risk_flag=False), lt)
cur = t[t["service_level"] == sl].set_index("policy")
scen = sens[(sens["lead_time"] == lt) & (sens["holding_rate"] == h) & (sens["stockout_multiple"] == m)]
cost = scen.set_index(["policy", "service_level"])["total_cost"]

st.markdown(f"#### At a {sl:.0%} target and a {lt}-day lead time")
cols = st.columns(3)
for col, p in zip(cols, ORDER):
    r = cur.loc[p]
    with col:
        st.markdown(f"**{POLICY_STYLE[p]['label']}**")
        st.metric("Fill rate (measured in simulation)", f"{r['fill_rate']:.2%}")
        st.metric("Stockout days", f"{int(r['stockout_days']):,}")
        st.metric("Average inventory value", f"${r['avg_inventory_value']:,.0f}")
        delta = None if p == "lightgbm_static" else f"{pct_change(cost[(p, sl)], cost[('lightgbm_static', sl)]):+.1f}% vs LightGBM static"
        st.metric("Total cost (ASSUMED costs, all items)", f"${cost[(p, sl)]:,.0f}", delta=delta, delta_color="inverse")
n_note(int(cur["n_items"].iloc[0]))

fig = go.Figure()
for p in ORDER:
    d = scen[scen["policy"] == p].sort_values("service_level")
    s = POLICY_STYLE[p]
    fig.add_trace(go.Scatter(x=d["service_level"], y=d["total_cost"], mode="lines+markers", name=s["label"],
                             line=dict(color=s["color"], dash=s["dash"]), marker=dict(symbol=s["symbol"], size=9),
                             hovertemplate="%{x:.0%} target: $%{y:,.0f}<extra>" + s["label"] + "</extra>"))
fig.add_vline(x=sl, line_dash="dot", line_color=DARK)
style(fig).update_xaxes(tickvals=[0.90, 0.95, 0.99], tickformat=".0%", title="target service level")
fig.update_yaxes(tickprefix="$", tickformat=",.0f", title="total cost over 168 days")
best = scen[scen["policy"] == "lightgbm_static"].sort_values("total_cost").iloc[0]
chart(assumed_chart(fig), f"Under these assumed costs the cheapest of the three targets is {best['service_level']:.0%}",
      f"Total holding + lost-sale cost for all items at holding {h:.0%}/yr and lost-sale cost {m:g}× price. Move the sliders "
      "to test other assumptions.")

# Calibrated safety stock
st.divider()
st.markdown("## Calibrated safety stock (7-day lead time)")
c = cal[cal["service_level"] == sl].copy()
c["label"] = c["policy"].map(lambda p: POLICY_STYLE[p]["label"])
show = c[["method", "label", "fill_rate", "cycle_service_level", "avg_inventory_value", "total_cost", "n_items"]].copy()
show[["fill_rate", "cycle_service_level"]] *= 100
st.dataframe(show.sort_values(["label", "method"]), hide_index=True, width="stretch", column_config={
    "method": "Safety-stock method", "label": "Forecast",
    "fill_rate": st.column_config.NumberColumn("Fill rate", format="%.2f%%"),
    "cycle_service_level": st.column_config.NumberColumn("Achieved cycle service level", format="%.1f%%"),
    "avg_inventory_value": st.column_config.NumberColumn("Avg. inventory", format="$%.0f"),
    "total_cost": st.column_config.NumberColumn("Total cost (ASSUMED base costs)", format="$%.0f"),
    "n_items": st.column_config.NumberColumn("Items", format="%d")})
cs = c[c["policy"] == "lightgbm_static"].set_index("method")
st.caption(f"How to read this: the 'normal formula' assumes forecast errors are unbiased and independent; 'conformal' "
           f"sizes safety stock from the actual spread of past errors. For LightGBM static at {sl:.0%}, the achieved cycle "
           f"service level rises from {cs.loc['normal formula', 'cycle_service_level']:.1%} to {cs.loc['conformal', 'cycle_service_level']:.1%} "
           f"(target {sl:.0%}). Costs at base assumptions ({H_BASE:.0%}/yr, {M_BASE:g}× price).")

# Heavy tail
st.divider()
st.markdown("## Where does the adaptive cost gap come from?")
pi = k[k["service_level"] == sl].pivot(index="item_id", columns="policy", values="total_cost")
d = (pi["lightgbm_adaptive"] - pi["lightgbm_static"]).dropna()
absd = np.sort(np.abs(d.to_numpy()))[::-1]
cum = np.cumsum(absd) / absd.sum() if absd.sum() > 0 else np.zeros_like(absd)
fig = go.Figure(go.Scatter(x=np.arange(1, len(cum) + 1), y=cum, mode="lines", line=dict(color=ACCENT),
                           hovertemplate="largest %{x} items: %{y:.1%} of the absolute gap<extra></extra>"))
if len(cum) >= 10:
    fig.add_vline(x=10, line_dash="dot", line_color=DARK)
    fig.add_annotation(x=10, y=cum[9], text=f" top 10 = {cum[9]:.0%}", showarrow=False, xanchor="left")
style(fig, legend=False).update_xaxes(title="items, largest absolute cost difference first", range=[0, len(cum) + 1])
fig.update_yaxes(tickformat=".0%", title="cumulative share of absolute gap", range=[0, 1.02])
chart(assumed_chart(fig), f"The gap is spread out: the 10 largest items account for {cum[9]:.0%} of it" if len(cum) >= 10 else "Too few items",
      f"Per-item |adaptive − static| total cost (base assumed costs, {lt}-day lead time, {sl:.0%} target), sorted from largest. "
      f"n = {len(d)} items; items without a detected shift have a difference of 0.")
wt = wdiff[(wdiff["comparison"] == "adaptive - static") & (wdiff["service_level"] == sl)]
if len(wt):
    st.caption("Top-10 share of the absolute gap in each 168-day window (7-day lead time): "
               + ", ".join(f"{r.window} {r.top10_share_of_abs_gap:.0%}" for r in wt.itertuples()))

# Windows
st.divider()
st.markdown("## Does it hold in other seasons?")
wd = wdiff[wdiff["service_level"] == sl].merge(wmeta[["window", "seasonal_coverage"]], on="window")
wd["label"] = wd["window"] + " · " + wd["seasonal_coverage"] + " · " + wd["comparison"]
fig = go.Figure(go.Scatter(
    x=wd["total_diff_pct"], y=wd["label"], mode="markers",
    marker=dict(color=[SECOND if c == "adaptive - static" else ACCENT for c in wd["comparison"]], size=10),
    error_x=dict(type="data", symmetric=False,
                 array=100 * (wd["ci_high"] - wd["total_diff"]) / (wd["total_diff"] / (wd["total_diff_pct"] / 100)),
                 arrayminus=100 * (wd["total_diff"] - wd["ci_low"]) / (wd["total_diff"] / (wd["total_diff_pct"] / 100)),
                 color=NEUTRAL),
    customdata=wd[["total_diff", "ci_low", "ci_high", "n_items"]],
    hovertemplate="%{y}<br>%{x:+.1f}% ($%{customdata[0]:+,.0f})<br>95% CI $%{customdata[1]:+,.0f} to $%{customdata[2]:+,.0f}"
                  "<br>n = %{customdata[3]} items<extra></extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=330, legend=False).update_xaxes(title="total cost difference (%)", ticksuffix="%")
fig.update_yaxes(autorange="reversed")
chart(assumed_chart(fig), "Across three 168-day windows, adaptive costs more than static and static costs less than the baseline",
      f"Total cost difference at a {sl:.0%} target, 7-day lead time, base assumed costs, with a bootstrap 95% CI over items. "
      "Orange: adaptive − static. Blue: static − seasonal naive.")

# Newsvendor
st.markdown("#### Textbook optimum vs simulation")
nvs = nv[(nv["policy"] == "lightgbm_static") & (nv["holding_rate"] == h) & (nv["stockout_multiple"] == m)]
st.dataframe(nvs[["window", "critical_ratio", "nearest_grid_level_to_cr", "simulated_cheapest_level"]], hide_index=True,
             width="stretch", column_config={
                 "critical_ratio": st.column_config.NumberColumn("Newsvendor critical ratio", format="%.4f"),
                 "nearest_grid_level_to_cr": st.column_config.NumberColumn("Nearest tested target", format="%.3f"),
                 "simulated_cheapest_level": st.column_config.NumberColumn("Cheapest target in simulation", format="%.3f")})
bes = be[be["policy"] == "lightgbm_static"]
st.caption(f"How to read this: the critical ratio m / (m + h × 7/365) is the textbook best service level for these "
           f"assumed costs. The simulation was tested from 50% to 99.5% and its cheapest target is higher, because the "
           f"achieved service level runs below the target. A 99% target beats 95% whenever the lost-sale cost multiple "
           f"exceeds {bes['breakeven_stockout_to_holding_ratio'].min():.2f}–{bes['breakeven_stockout_to_holding_ratio'].max():.2f} "
           f"× the annual holding rate (your setting: {m / h:.1f}).")

st.divider()
c1, c2 = st.columns(2)
c1.download_button("Download item results (this lead time and target)",
                   k[k["service_level"] == sl].to_csv(index=False).encode(), file_name=f"inventory_items_L{lt}_SL{sl}.csv",
                   mime="text/csv")
c2.download_button("Download cost scenarios", sens.to_csv(index=False).encode(), file_name="inventory_cost_scenarios.csv",
                   mime="text/csv")

cl = cal[(cal["service_level"] == 0.95) & (cal["policy"] == "lightgbm_static")].set_index("method")
saving = 1 - cl.loc["conformal", "total_cost"] / cl.loc["normal formula", "total_cost"]
insight(f"The textbook safety stock promises more than it delivers, because forecast errors are neither normal nor "
        f"independent. Sizing it from the real errors instead brings the achieved service close to the target and, "
        f"at our assumed costs, cuts total cost by {saving:.0%} at a 95% target. How the safety stock is sized "
        f"matters more than which good forecast feeds it.")
