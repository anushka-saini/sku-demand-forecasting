import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.kpis import BAND_FLAG, BIAS_FLAG, base_cost_assumptions, build_watchlist
from utils.models import POLICY_COLORS, POLICY_NAMES
from utils.ui import DARK, NEUTRAL, chart, download, filters, filters_used, kpis, page, simulated, style, tells

flt = filters()
kpi = get("kpi_all")
sens = get("cost_sensitivity")
inv = get("inventory_policy")
cal = get("calibrated_inv")
cal_items = get("calibrated_items")
wdiff = get("window_diffs")
items = get("items")
H_BASE, M_BASE = base_cost_assumptions(inv)
POL = ["lightgbm_static", "lightgbm_adaptive", "seasonal_naive_static"]

page("Stock impact", "What do the forecasts mean for stock held, lost sales and cost?")
filters_used(["category"])
simulated("These results come from replaying the real sales of the test period against a reorder rule: each day, if "
          "stock on hand plus stock on order falls to the reorder point, order enough to cover the lead time, the "
          "safety stock and one more week. Unserved demand is lost. Lead time and costs are assumptions.")

c1, c2, c3, c4 = st.columns(4)
lt = c1.radio("Lead time (days)", sorted(kpi["lead_time"].unique()), index=1, horizontal=True, key="si_lead",
              help="Days between placing an order and receiving it. Assumed.")
sl = c2.radio("Target service level", sorted(kpi["service_level"].unique()), index=1, horizontal=True, key="si_sl",
              format_func=lambda x: f"{x:.0%}", help="The chance of not running out during one order cycle that the safety stock aims for.")
h = c3.select_slider("Holding cost, % of price a year", sorted(sens["holding_rate"].unique()), value=H_BASE, key="si_h",
                     format_func=lambda x: f"{x:.0%}", help="Cost of keeping one unit in stock for a year. Assumed.")
m = c4.select_slider("Cost of a lost sale, x price", sorted(sens["stockout_multiple"].unique()), value=M_BASE, key="si_m",
                     format_func=lambda x: f"{x:g}x", help="Profit and goodwill lost when a customer finds the shelf empty. Assumed.")

k = kpi[(kpi["lead_time"] == lt) & (kpi["service_level"] == sl)]
if flt["categories"]:
    k = k[k["category"].isin(flt["categories"])]
if k.empty:
    st.info("No items match the category filter.")
    st.stop()
g = k.groupby("policy").agg(demand=("demand_units", "sum"), lost=("lost_units", "sum"), stock=("avg_inventory_value", "sum"),
                            ss=("mean_safety_stock", "mean"), stockout_days=("stockout_days", "sum"), n=("item_id", "nunique"))
g["fill_rate"] = 1 - g["lost"] / g["demand"]
scen = sens[(sens["lead_time"] == lt) & (sens["holding_rate"] == h) & (sens["stockout_multiple"] == m)]
cost = scen.set_index(["policy", "service_level"])["total_cost"]
c_lgb, c_naive = cost[("lightgbm_static", sl)], cost[("seasonal_naive_static", sl)]
kpis([
    {"label": "Total cost, LightGBM forecast", "value": f"${c_lgb:,.0f}", "compare": f"{c_lgb / c_naive - 1:+.1%} vs the baseline forecast",
     "tone": "good" if c_lgb < c_naive else "bad", "help": "Holding plus lost-sale cost over the test period, all items, at the costs you chose."},
    {"label": "Fill rate", "value": f"{g.loc['lightgbm_static', 'fill_rate']:.1%}",
     "compare": f"vs {g.loc['seasonal_naive_static', 'fill_rate']:.1%} with the baseline", "help": "Share of demand served from stock."},
    {"label": "Average stock held", "value": f"${g.loc['lightgbm_static', 'stock']:,.0f}",
     "compare": f"{g.loc['lightgbm_static', 'stock'] / g.loc['seasonal_naive_static', 'stock'] - 1:+.0%} vs the baseline",
     "help": "Average value of stock on hand, summed over the items in view."},
    {"label": "Safety stock per item", "value": f"{g.loc['lightgbm_static', 'ss']:,.1f} units",
     "compare": f"vs {g.loc['seasonal_naive_static', 'ss']:,.1f} with the baseline",
     "help": "Average extra units kept to cover forecast error, from the textbook formula."},
    {"label": "Days out of stock", "value": f"{int(g.loc['lightgbm_static', 'stockout_days']):,}",
     "compare": f"vs {int(g.loc['seasonal_naive_static', 'stockout_days']):,} with the baseline", "help": "Item-days with an empty shelf."},
])

fig = go.Figure()
for p in POL:
    d = scen[scen["policy"] == p].sort_values("service_level")
    fig.add_bar(x=[f"{x:.0%}" for x in d["service_level"]], y=d["total_cost"], name=POLICY_NAMES[p], marker_color=POLICY_COLORS[p],
                hovertemplate="%{x} target: $%{y:,.0f}<extra>" + POLICY_NAMES[p] + "</extra>")
fig.update_layout(barmode="group")
style(fig, height=340).update_yaxes(title="total cost over the test period ($)", tickprefix="$", tickformat=",")
fig.update_xaxes(title="target service level", type="category")
cheap = scen.loc[scen["total_cost"].idxmin()]
chart(fig, f"Cheapest here: {POLICY_NAMES[cheap['policy']]} at a {cheap['service_level']:.0%} target (${cheap['total_cost']:,.0f})",
      f"At a {lt}-day lead time, holding cost {h:.0%} a year and a lost sale costing {m:g}x its price. Change the settings above "
      f"to test other assumptions; the costs cover all items, whatever the category filter.")

# textbook vs calibrated safety stock (saved only for a 7-day lead time and the base costs)
c = cal[cal["service_level"] == sl].copy()
cs = c[c["policy"] == "lightgbm_static"].set_index("method")
fig = go.Figure()
for method, lab, color in [("normal formula", "textbook formula", NEUTRAL), ("conformal", "sized from past errors", "#1F6F8B")]:
    d = c[c["method"] == method].set_index("policy").reindex(POL)
    fig.add_bar(x=[POLICY_NAMES[p] for p in POL], y=d["total_cost"], name=lab, marker_color=color,
                customdata=np.c_[d["fill_rate"], d["cycle_service_level"]],
                hovertemplate="%{x}: $%{y:,.0f}<br>fill rate %{customdata[0]:.1%}, cycles without a stockout %{customdata[1]:.1%}"
                              "<extra>" + lab + "</extra>")
fig.update_layout(barmode="group")
style(fig, height=340).update_yaxes(title="total cost ($)", tickprefix="$", tickformat=",")
save = cs.loc["conformal", "total_cost"] / cs.loc["normal formula", "total_cost"] - 1
diag = get("residual_checks")
normal_pass = diag.loc[diag["level"] == "item", "normality_pass"].max()
chart(fig, f"Sizing safety stock from past errors changes LightGBM's cost by {save:+.0%} at a {sl:.0%} target",
      f"The textbook formula assumes forecast errors are normal; at most {normal_pass:.1%} of items pass a normality test. "
      f"With it, only "
      f"{cs.loc['normal formula', 'cycle_service_level']:.1%} of order cycles avoided a stockout against a {sl:.0%} target; "
      f"sized from past errors, {cs.loc['conformal', 'cycle_service_level']:.1%}. Saved for a 7-day lead time and the base "
      f"costs only ({H_BASE:.0%} a year, {M_BASE:g}x price).")

with st.expander("Does this hold in other half-years?"):
    wd = wdiff[wdiff["service_level"] == sl].copy()
    wd["label"] = wd["window"] + ": " + wd["comparison"].map({"static - naive": "LightGBM vs baseline",
                                                               "adaptive - static": "retrained vs LightGBM"})
    fig = go.Figure(go.Scatter(x=wd["total_diff_pct"], y=wd["label"], mode="markers",
                               marker=dict(size=10, color=["#E69F00" if c == "static - naive" else "#A06A00" for c in wd["comparison"]]),
                               customdata=wd[["ci_low", "ci_high", "n_items"]],
                               hovertemplate="%{y}: %{x:+.1f}%<br>range $%{customdata[0]:+,.0f} to $%{customdata[1]:+,.0f}, "
                                             "%{customdata[2]} items<extra></extra>"))
    fig.add_vline(x=0, line_color=DARK)
    style(fig, height=300, legend=False).update_xaxes(title="difference in total cost (%)", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    sn = wd[wd["comparison"] == "static - naive"]
    chart(fig, f"LightGBM is cheaper than the baseline in {int((sn['ci_high'] < 0).sum())} of {len(sn)} half-years at a {sl:.0%} target",
          "W1 is the test period above; W2 and W3 are two earlier half-years run the same way. 7-day lead time, base costs.")

st.markdown("<div class='chart-title'>Items to check</div>", unsafe_allow_html=True)
method = st.radio("Safety stock", ["conformal", "normal formula"], horizontal=True, key="si_method",
                  format_func={"conformal": "sized from past errors", "normal formula": "textbook formula"}.get)
wl = build_watchlist(cal_items, get("item_bias"), get("item_coverage"), items, get("hc_items"), method, sl)
if flt["categories"]:
    wl = wl[wl["category"].isin(flt["categories"])]
st.caption(f"Items whose simulated fill rate fell below the {sl:.0%} target with LightGBM forecasts (7-day lead time), "
           f"most lost units first. Reasons are simple flags: forecast more than {BIAS_FLAG:.0%} off in total, an interval "
           f"that held the sales on fewer than {BAND_FLAG:.0%} of days, a detected shift, or sales on few days.")
if wl.empty:
    st.info("No items fell below the target with these settings.")
else:
    show = wl[["item_id", "reasons", "fill_rate", "lost_units", "stockout_days", "mean_safety_stock"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"item_id": "Item", "reasons": st.column_config.TextColumn("Why it may be short", width="medium"),
                                "fill_rate": st.column_config.NumberColumn("Fill rate", format="percent"),
                                "lost_units": st.column_config.NumberColumn("Lost units", format="%d"),
                                "stockout_days": st.column_config.NumberColumn("Days out of stock", format="%d"),
                                "mean_safety_stock": st.column_config.NumberColumn("Safety stock", format="%.1f")})
    st.caption("The CSV has every column, including category, forecast bias and interval coverage.")
    download(wl, f"items_to_check_{int(sl * 100)}.csv", key="dl_watch")

adapt = cost[("lightgbm_adaptive", sl)] / c_lgb - 1
tells(f"With these settings the LightGBM forecast costs {c_lgb / c_naive - 1:+.1%} against the baseline forecast, and the "
      f"version retrained after shifts costs {adapt:+.1%} against it. Changing how safety stock is sized moves cost by {save:+.0%}"
      f" at the {sl:.0%} target" + (", so it is the bigger lever. " if abs(save) > abs(c_lgb / c_naive - 1) else ". ")
      + f"{len(wl):,} items still fall below the target with the "
      f"{'past-error' if method == 'conformal' else 'textbook'} safety stock. All of this is simulated with assumed costs.")
