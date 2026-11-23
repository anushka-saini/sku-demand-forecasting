import plotly.graph_objects as go
import streamlit as st

from utils.final import best_info, load, names, need_results, text
from utils.ui import chart, download, kpis, page, simulated, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
info = best_info()
st_ = load("stock")
cmp_ = load("stock_cmp")
page("Inventory", "What does the best model do to stock held, lost sales and cost, against the baselines?")
st.caption("Reorder points and safety stock come from each model's forecast, with the safety stock sized from the model's own past errors. All costs are assumed.")
simulated("The test windows' real sales are replayed against a reorder rule (7-day lead time, 95% service level). Costs are "
          "assumptions: holding 25% of the price a year, and a lost sale costs half the price.")
method = st.radio("Safety stock rule", ["conformal", "normal formula"], horizontal=True, key="fs_m",
                  format_func=lambda s: "Calibrated from past errors" if s == "conformal" else "Textbook formula",
                  help="Calibrated: the safety stock is a quantile of the model's own past lead-time errors. "
                       "Textbook: z times the standard deviation of the errors.")
s = st_[st_["method"] == method].set_index("model")
b = info["model"]
if b not in s.index:
    st.info("The best model has no stock run (it has no forecast at the warm-up window).")
    st.stop()
ref = "combination" if "combination" in s.index and b != "combination" else "seasonal_naive"
nv = "seasonal_naive"
kpis([
    {"label": f"Total cost, {nm[b]}", "value": f"${s.loc[b, 'total_cost']:,.0f}",
     "compare": f"{s.loc[b, 'total_cost'] / s.loc[ref, 'total_cost'] - 1:+.1%} vs {nm[ref]}",
     "tone": "good" if s.loc[b, "total_cost"] < s.loc[ref, "total_cost"] else "bad"},
    {"label": "Fill rate", "value": f"{s.loc[b, 'fill_rate']:.1%}", "compare": f"{nm[ref]}: {s.loc[ref, 'fill_rate']:.1%}"},
    {"label": "Average stock value", "value": f"${s.loc[b, 'avg_inventory_value']:,.0f}",
     "compare": f"{s.loc[b, 'avg_inventory_value'] / s.loc[ref, 'avg_inventory_value'] - 1:+.1%} vs {nm[ref]}"},
    {"label": "Cost vs seasonal naive", "value": f"{s.loc[b, 'total_cost'] / s.loc[nv, 'total_cost'] - 1:+.1%}",
     "compare": f"${s.loc[b, 'total_cost'] - s.loc[nv, 'total_cost']:+,.0f}"},
])
order = list(s.index)
fig = go.Figure()
fig.add_bar(x=[nm[m] for m in order], y=s["holding_cost"], name="holding cost", marker_color="#0072B2")
fig.add_bar(x=[nm[m] for m in order], y=s["stockout_cost"], name="lost-sale cost", marker_color="#D55E00")
fig.update_layout(barmode="stack")
fig.update_yaxes(title="cost over 168 days ($, assumed costs)")
chart(style(fig), "Total cost split into holding and lost sales",
      "Both parts come from the same replay, so they add up to the total in the first card.", key="fs_cost")
tab = s[["fill_rate", "cycle_service_level", "avg_inventory_value", "holding_cost", "stockout_cost", "total_cost"]].copy()
tab.index = [nm[m] for m in tab.index]
st.dataframe(tab.round(3), width="stretch")
c = cmp_[cmp_["method"] == method]
c = c.assign(versus=c["versus"].map(nm), model=c["model"].map(nm))
st.markdown("**Paired comparison** (sum over items of the cost difference, bootstrap 95% interval; negative means the best model is cheaper)")
st.dataframe(c[["model", "versus", "total_cost_diff", "total_cost_diff_pct", "ci_low", "ci_high", "stock_value_diff_pct"]].round(4),
             hide_index=True, width="stretch")

# reorder point = forecast demand over the lead time + safety stock, averaged over the items
it = load("stock_items")
it = it[it["method"] == method].copy()
it["lt_demand"] = it["demand_units"] / 168 * it["lead_time"]          # demand is summed over the 168 test days
it["rop"] = it["lt_demand"] + it["mean_safety_stock"]
ss_ = it.groupby("model")[["lt_demand", "mean_safety_stock", "rop"]].mean()
ss_ = ss_.loc[[m for m in order if m in ss_.index]]
ss_.index = [nm[m] for m in ss_.index]
ss_.columns = ["Demand over the lead time", "Safety stock", "Reorder point"]
st.markdown("**Safety stock and reorder point per item** (units, average over the 150 items; lead time 7 days, 95% target)")
st.dataframe(ss_.round(1), width="stretch")
one = st.selectbox("One item", sorted(it["item_id"].unique()), key="fs_item")
x = it[(it["item_id"] == one) & (it["model"] == b)].iloc[0]
kpis([{"label": f"Demand over {int(x['lead_time'])} days", "value": f"{x['lt_demand']:.0f}", "compare": nm[b]},
      {"label": "Safety stock", "value": f"{x['mean_safety_stock']:.0f}"},
      {"label": "Reorder point", "value": f"{x['rop']:.0f}", "compare": "demand plus safety stock"},
      {"label": "Fill rate", "value": f"{x['fill_rate']:.1%}", "compare": "over the 168 test days"}])
download(st_, "final_stock_summary.csv")
st.markdown(text()["stock"])
tells("Lower forecast error only turns into lower cost if the safety stock follows it. If the interval in the table includes zero, "
      "the cost difference is within the noise of this replay.")
