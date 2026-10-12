import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.data import get
from utils.kpis import base_cost_assumptions, pct_change, tradeoff
from utils.ui import DARK, POLICY_STYLE, assumed, chart, derived, header, n_note, style

inv = get("inventory_policy")
inv_tests = get("inventory_tests")
sens = get("cost_sensitivity")
H_BASE, M_BASE = base_cost_assumptions(inv)

header("Inventory decision",
       "Each item's stock was simulated day by day over the 168 evaluation days with a reorder-point policy: when "
       "stock on hand plus on order falls to the reorder point, order up to it plus a week of forecast demand; "
       "unmet demand is lost. Selectors below only filter pre-computed simulations.")
assumed(f"Lead time, holding cost ({H_BASE:.0%} of unit price per year) and stockout cost ({M_BASE:.2g} × unit "
        f"price per lost unit) are assumptions, not observed data. Ordering cost is ignored.")

c1, c2 = st.columns(2)
sl = c1.radio("Target service level", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{x:.0%}", horizontal=True)
lt = c2.radio("Lead time (days, assumed)", sorted(inv["lead_time"].unique()), index=1, horizontal=True)

t = tradeoff(inv, lt)
cur = t[t["service_level"] == sl].set_index("policy")
order = [p for p in POLICY_STYLE if p in cur.index]

st.markdown(f"#### At {sl:.0%} target service and a {lt}-day lead time")
cols = st.columns(len(order))
for col, p in zip(cols, order):
    r = cur.loc[p]
    with col:
        st.markdown(f"**{POLICY_STYLE[p]['label']}**")
        st.metric("Fill rate", f"{r['fill_rate']:.2%}")
        st.metric("Stockout days (all items)", f"{int(r['stockout_days']):,}")
        st.metric("Average inventory value", f"${r['avg_inventory_value']:,.0f}")
        delta = None if p == "lightgbm_static" else f"{pct_change(r['total_cost'], cur.loc['lightgbm_static', 'total_cost']):+.1f}% vs LightGBM static"
        st.metric("Total cost (assumed costs)", f"${r['total_cost']:,.0f}", delta=delta, delta_color="inverse")
n_note(int(cur["n_items"].iloc[0]))

test = inv_tests[(inv_tests["lead_time"] == lt) & (inv_tests["service_level"] == sl)]
if len(test):
    r = test.iloc[0]
    st.caption(f"Item by item, on the {int(r['n_items'])} items with a break: the median adaptive − static cost "
               f"difference is {r['median_cost_diff']:+.2f} dollars (95% CI {r['ci_low']:+.2f} to {r['ci_high']:+.2f}, "
               f"p = {r['p_value']:.3f}). Adaptive was cheaper for {r['share_items_cheaper']:.0%} of them. The total "
               f"gap is driven by a few items.")

# Trade-off curves
fig = make_subplots(rows=1, cols=3, subplot_titles=("Holding cost", "Stockout cost", "Total cost"), shared_yaxes=False)
for p in order:
    d = t[t["policy"] == p].sort_values("service_level")
    s = POLICY_STYLE[p]
    for i, metric in enumerate(["holding_cost", "stockout_cost", "total_cost"], start=1):
        fig.add_trace(go.Scatter(x=d["service_level"], y=d[metric], mode="lines+markers", name=s["label"],
                                 legendgroup=p, showlegend=(i == 1),
                                 line=dict(color=s["color"], dash=s["dash"]), marker=dict(symbol=s["symbol"], size=9),
                                 hovertemplate="%{x:.0%} service: $%{y:,.0f}<extra>" + s["label"] + "</extra>"), row=1, col=i)
for i in range(1, 4):
    fig.add_vline(x=sl, line_dash="dot", line_color=DARK, row=1, col=i)
fig.update_xaxes(tickvals=[0.90, 0.95, 0.99], tickformat=".0%")
fig.update_yaxes(tickprefix="$", tickformat=",.0f")
style(fig, height=400)
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0), margin=dict(t=40, b=40))
best = t[t["policy"] == "lightgbm_static"].sort_values("total_cost").iloc[0]
chart(fig, f"Stockouts cost far more than holding stock, so the highest service level tested "
           f"({best['service_level']:.0%}) is the cheapest for LightGBM static",
      f"Summed over all items for the {lt}-day lead time and base cost assumptions. Raising the service level adds "
      "a little holding cost (left) and removes much more stockout cost (middle). The dotted line marks your selection.")

# Inventory vs fill rate
fig = go.Figure()
for p in order:
    d = t[t["policy"] == p].sort_values("service_level")
    s = POLICY_STYLE[p]
    fig.add_trace(go.Scatter(x=d["avg_inventory_value"], y=d["fill_rate"], mode="lines+markers+text", name=s["label"],
                             text=[f"{x:.0%}" for x in d["service_level"]], textposition="bottom right",
                             line=dict(color=s["color"], dash=s["dash"]), marker=dict(symbol=s["symbol"], size=10),
                             hovertemplate="inventory $%{x:,.0f}<br>fill rate %{y:.2%}<extra>" + s["label"] + "</extra>"))
style(fig).update_xaxes(title="average inventory value held, all items ($)", tickprefix="$", tickformat=",.0f")
fig.update_yaxes(title="fill rate", tickformat=".1%")
chart(fig, "LightGBM static gets a higher fill rate from less inventory than seasonal naive. Adaptive sits slightly below static",
      "Each point is one target service level (labelled). Up and to the left is better: more demand served with "
      "less stock held.")

# Sensitivity
s = sens[(sens["lead_time"] == lt) & (sens["service_level"] == sl)]
piv = s.pivot_table(index=["holding_rate", "stockout_multiple"], columns="policy", values="total_cost").reset_index()
piv["chg"] = 100 * (piv["lightgbm_adaptive"] / piv["lightgbm_static"] - 1)
z = piv.pivot(index="stockout_multiple", columns="holding_rate", values="chg")
fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=[f"{h:.0%}/yr" for h in z.columns], y=[f"{m:g}× price" for m in z.index],
                           text=[[f"{v:+.1f}%" for v in row] for row in z.to_numpy()], texttemplate="%{text}",
                           colorscale=[[0, "rgba(230,159,0,0.15)"], [1, "rgba(230,159,0,0.85)"]], showscale=False,
                           hovertemplate="holding %{x}, stockout %{y}<br>adaptive vs static total cost %{text}<extra></extra>"))
style(fig, height=300, legend=False).update_xaxes(title="holding cost rate (assumed)")
fig.update_yaxes(title="stockout cost (assumed)")
direction = "more" if (z.to_numpy() > 0).all() else ("less" if (z.to_numpy() < 0).all() else "mixed")
chart(fig, f"Across every assumed cost combination, adaptive costs {direction} than static" if direction != "mixed"
      else "The adaptive vs static cost comparison depends on the assumed costs",
      "Each cell is the % change in total cost (all items) from switching LightGBM static to adaptive, for one "
      "combination of assumed holding and stockout costs. Positive means adaptive is more expensive.")

# Item table
st.markdown("#### Item reorder points and stockout risk")
derived("Stockout-risk flag = the item's simulated fill rate fell below the target service level.")
c1, c2, c3 = st.columns(3)
pol = c1.selectbox("Policy", order, format_func=lambda p: POLICY_STYLE[p]["label"])
cats = c2.multiselect("Category", sorted(inv["category"].unique()), default=sorted(inv["category"].unique()))
only_risk = c3.checkbox("Only items at risk")
tab = inv[(inv["lead_time"] == lt) & (inv["service_level"] == sl) & (inv["policy"] == pol) & inv["category"].isin(cats)]
if only_risk:
    tab = tab[tab["stockout_risk_flag"]]
n_risk = int(inv[(inv["lead_time"] == lt) & (inv["service_level"] == sl) & (inv["policy"] == pol)
                 & inv["category"].isin(cats)]["stockout_risk_flag"].sum())
st.caption(f"{n_risk} items flagged at risk. How to read this: click a header to sort. Reorder point is the "
           f"level on the last simulated day; safety stock is the average over the 6 windows (units).")
show = tab[["item_id", "category", "stockout_risk_flag", "fill_rate", "stockout_days", "lost_units", "mean_safety_stock",
            "final_rop", "total_cost", "break_item"]].sort_values(["stockout_risk_flag", "lost_units"], ascending=False).copy()
show["stockout_risk_flag"] = show["stockout_risk_flag"].map({True: "AT RISK", False: "ok"})
if show.empty:
    st.info("No items match these filters.")
else:
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "item_id": "Item", "category": "Category", "stockout_risk_flag": "Risk (derived)",
        "fill_rate": st.column_config.NumberColumn("Fill rate", format="%.3f"),
        "stockout_days": st.column_config.NumberColumn("Stockout days", format="%d"),
        "lost_units": st.column_config.NumberColumn("Lost units", format="%.0f"),
        "mean_safety_stock": st.column_config.NumberColumn("Safety stock (avg)", format="%.1f"),
        "final_rop": st.column_config.NumberColumn("Reorder point", format="%.1f"),
        "total_cost": st.column_config.NumberColumn("Total cost $ (assumed)", format="%.2f"),
        "break_item": st.column_config.CheckboxColumn("Break item")})
n_note(len(show))
