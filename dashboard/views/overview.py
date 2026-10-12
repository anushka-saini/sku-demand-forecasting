import streamlit as st

from utils.data import get
from utils.kpis import headline, pct_change, verdict

items = get("items")
tests = get("adaptive_tests")
summary = get("policy_summary")
inv_tests = get("inventory_tests")
mase = get("model_mase")

st.title("Adaptive SKU-Level Demand Forecasting & Inventory Decision Intelligence")
st.markdown(
    "<p class='intro'>Walmart's M5 data covers daily unit sales for every item in California stores from 2011 to "
    "2016. This project asks three questions. Do individual items' demand patterns show lasting shifts (structural "
    "breaks) that can be detected statistically? Does ignoring those shifts make forecasts worse? And does "
    "retraining the forecast after a shift lead to better stocking decisions than a model that never adapts?</p>",
    unsafe_allow_html=True)

st.markdown(f"<div class='headline'>{headline(tests, 'lightgbm')}</div>", unsafe_allow_html=True)

scope = items[items["in_scope"]]
n_scope = len(scope)
n_break = int(scope["break_at_last_origin"].sum())
lgb = tests[(tests["comparison"].str.startswith("adaptive")) & (tests["model"] == "lightgbm") & (tests["category"] == "ALL")].iloc[0]
s95 = summary[summary["service_level"] == 0.95].set_index("policy")
cost_chg = pct_change(s95.loc["lightgbm_adaptive", "total_cost"], s95.loc["lightgbm_static", "total_cost"])
it95 = inv_tests[(inv_tests["lead_time"] == 7) & (inv_tests["service_level"] == 0.95)].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Items analysed", f"{n_scope:,}")
    st.caption(f"High-volume, predictable (AX/AY) items out of {len(items):,} California items.")
with c2:
    st.metric("Items with a break", f"{n_break / n_scope:.1%}", help=f"{n_break} of {n_scope} items")
    st.caption("Share of items whose demand shifted by the last forecast date, using only data available at that "
               "time. About half of these alarms reflect slow drift rather than sudden steps (see Break detection).")
with c3:
    st.metric("MASE change (adaptive)", f"{lgb['median_diff']:+.4f}",
              help=f"95% CI {lgb['ci_low']:+.4f} to {lgb['ci_high']:+.4f}, n = {int(lgb['n'])} items with a break")
    st.caption(f"Positive means retraining made LightGBM forecasts worse. Verdict: "
               f"{verdict(lgb['ci_low'], lgb['ci_high'])}, but tiny next to a typical MASE of "
               f"{mase.set_index('category').loc['ALL', 'lightgbm']:.3f}.")
with c4:
    st.metric("Cost change at 95% service", f"{cost_chg:+.1f}%",
              help="Adaptive vs static LightGBM, all 726 items, lead time 7 days, base cost assumptions")
    st.caption(f"Adaptive cost more in total, but most of the gap comes from a few items. Item by item the "
               f"difference is not significant (p = {it95['p_value']:.2f}). Costs use assumed parameters.")

st.markdown("#### What the evidence says")
naive_vs_lgb = mase.set_index("category").loc["ALL"]
s_naive = s95.loc["seasonal_naive_static"]
s_static = s95.loc["lightgbm_static"]
st.markdown(
    f"- **A good static model matters more than adapting.** LightGBM's median MASE is "
    f"{naive_vs_lgb['lightgbm']:.3f} against {naive_vs_lgb['seasonal_naive']:.3f} for the seasonal-naive baseline. "
    f"At a 95% service level its reorder points hold "
    f"{abs(pct_change(s_static['avg_inventory_value'], s_naive['avg_inventory_value'])):.1f}% less inventory at "
    f"{abs(pct_change(s_static['total_cost'], s_naive['total_cost'])):.1f}% lower total cost.\n"
    f"- **Breaks are common but often gradual.** Retraining on the post-break window gave no benefit. A random "
    f"training cut (placebo) on items without a break also had no effect.\n"
    f"- **Cost figures depend on assumed parameters.** Lead time, holding cost and stockout cost are assumptions, "
    f"tested in a sensitivity analysis on the Inventory page.")
st.caption("Use the pages on the left in order: data and method, segmentation, break detection, forecasts, "
           "inventory, then assumptions and limitations.")
