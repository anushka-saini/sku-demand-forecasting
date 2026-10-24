import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup, value
from utils.models import COLORS, ORDER, SHORT, name
from utils.ui import chart, kpis, page, style, tells

f = facts()
mase = get("mase_unrounded")
overview = get("comparison_overview")
cal = get("calibrated_inv")
variants = get("variants")
adaptive = get("adaptive_tests")
shift_sarima = get("shift_aware_tests")

page("Overview", "Can better forecasts, or reacting to shifts in demand, cut the cost of keeping these items in stock?")

n_items = int(value(f, "scope", "ax_ay_items"))
ca_items = int(value(f, "scope", "ca_items"))
units_share = value(f, "scope", "ax_ay_share_of_units")
all_rows = mase[(mase["category"] == "ALL") & mase["model"].isin(["seasonal_naive", "ets", "lightgbm"])].set_index("model")
naive = all_rows.loc["seasonal_naive", "median_item_mase"]
best = all_rows["median_item_mase"].idxmin()
best_val = all_rows.loc[best, "median_item_mase"]
gain = 1 - best_val / naive
adj = lookup(f, "breaks", "adjusted_share_with_real_shift", "CUSUM")
stock = lookup(f, "inventory", "static_vs_naive_base_95")
c95 = cal[(cal["policy"] == "lightgbm_static") & (cal["service_level"] == 0.95)].set_index("method")
fr_text, fr_cal = c95.loc["normal formula", "fill_rate"], c95.loc["conformal", "fill_rate"]

# every version that reacts to a shift, and whether any of them met the "helps" rule
lgb_all = variants[variants["group"] == "all detected breaks"]
reacting = len(lgb_all) + len(adaptive[(adaptive["comparison"].str.startswith("adaptive")) & (adaptive["category"] == "ALL")
                                      & adaptive["model"].isin(["ets", "tsb"])]) \
    + len(shift_sarima[shift_sarima["group"] == "all break item-origins"])
any_better = bool(lgb_all["helps"].astype(str).str.lower().eq("true").any()) or bool(
    shift_sarima.loc[shift_sarima["group"] == "all break item-origins", "helps"].astype(str).str.lower().eq("true").any())

kpis([
    {"label": "Items analysed", "value": f"{n_items:,}", "compare": f"of {ca_items:,} items; {units_share:.0%} of units sold",
     "help": "The high-volume, fairly steady items in California's four stores."},
    {"label": f"{name(best)} forecast error", "value": f"{best_val:.3f}", "compare": f"{gain:.0%} lower than the baseline ({naive:.3f})",
     "tone": "good", "help": "MASE: average error divided by the error of repeating last week. Lower is better; below 1 beats that rule."},
    {"label": "Items with a lasting shift", "value": f"{adj['value']:.0%}",
     "compare": f"range {adj['ci95'][0]:.0%} to {adj['ci95'][1]:.0%}, chance alarms removed",
     "help": "Share of items whose normal sales level changed for good at some point."},
    {"label": "Stock cost, better forecast", "value": f"{stock['value']:+.1%}",
     "compare": f"vs the baseline forecast; {stock['inventory_change']:+.0%} stock held", "tone": "good",
     "help": "Simulated over 168 days at a 95% target and a 7-day lead time, with assumed costs."},
    {"label": "Fill rate, safety stock from past errors", "value": f"{fr_cal:.1%}",
     "compare": f"vs {fr_text:.1%} with the textbook formula", "tone": "good",
     "help": "Share of demand served from stock in the simulation, 95% target."},
])

st.markdown("#### The answer")
s1 = (f"Lasting shifts are common: after removing chance alarms, about {adj['value']:.0%} of the {n_items:,} items "
      f"changed their normal sales level at some point.")
s2 = (f"{name(best)} forecasts best, with a typical error {gain:.0%} below the simple baseline, "
      + (f"and reacting to a detected shift made none of the {reacting} versions we tried more accurate."
         if not any_better else "and at least one version that reacts to shifts improved accuracy."))
s3 = (f"In the stock simulation the better forecast holds {abs(stock['inventory_change']):.0%} less stock at "
      f"{abs(stock['value']):.0%} lower cost, and sizing safety stock from past errors lifts the fill rate from "
      f"{fr_text:.1%} to {fr_cal:.1%}.")
st.markdown(f"{s1} {s2} {s3}")
todo = (f"**What to do with this:** plan with {name(best)} forecasts as they are"
        + (", do not retrain them after a detected shift," if not any_better else ",")
        + " and size safety stock from past forecast errors rather than the textbook formula.")
st.markdown(todo)

# small model comparison: the 150-item sample, where every model was run
ov = overview.set_index("model")
key_by_name = {name(k): k for k in ORDER}
rows = [(key_by_name.get(m, m), ov.loc[m, "median_mase"]) for m in ov.index]
rows.sort(key=lambda r: r[1], reverse=True)
fig = go.Figure(go.Bar(x=[v for _, v in rows], y=[SHORT.get(k, k) for k, _ in rows], orientation="h",
                       marker_color=[COLORS.get(k, "#888") for k, _ in rows],
                       text=[f"{v:.3f}" for _, v in rows], textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}: MASE %{x:.3f}<extra></extra>"))
style(fig, height=300, legend=False).update_xaxes(title="typical forecast error (MASE, lower is better)",
                                                   range=[0, max(v for _, v in rows) * 1.15])
best_s = min(rows, key=lambda r: r[1])
second = sorted(rows, key=lambda r: r[1])[1]
chart(fig, f"{SHORT.get(best_s[0], best_s[0])} is lowest, but the good models sit close together",
      f"On the {int(ov['n_items'].iloc[0])} items where every model ran, {SHORT.get(best_s[0], best_s[0])} "
      f"({best_s[1]:.3f}) is ahead of {SHORT.get(second[0], second[0])} ({second[1]:.3f}) by {second[1] - best_s[1]:.3f}; "
      f"the baseline is at {ov.loc[name('seasonal_naive'), 'median_mase']:.3f}.")

cal_save = lookup(f, "inventory", "conformal_vs_normal_cost_change", "0.95")["value"]
bigger = "how safety stock is sized" if abs(cal_save) > abs(stock["value"]) else "the better forecast"
tells(f"{name(best)} cuts the typical error by {gain:.0%}, "
      + ("while reacting to shifts did not help. " if not any_better else "and some shift-aware versions helped. ")
      + f"In the simulation the larger saving comes from {bigger} ({cal_save:+.0%} cost from safety stock sized on past "
      f"errors, {stock['value']:+.0%} from the better forecast). The costs behind these numbers are assumptions; the "
      f"Stock impact page lets you change them.")
