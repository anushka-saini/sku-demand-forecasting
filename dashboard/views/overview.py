import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup, value
from utils.models import BEST, BLOCKS, COLORS, CONTENDERS, PICKED, SHORT, name
from utils.ui import chart, download, kpis, page, style, tells

f = facts()
mase = get("mase_unrounded")
cal = get("calibrated_inv")
variants = get("variants")
adaptive = get("adaptive_tests")
shift_sarima = get("shift_aware_tests")

page("Overview", "Can better forecasts, or reacting to shifts in demand, cut the cost of keeping these items in stock?")

n_items = int(value(f, "scope", "ax_ay_items"))
ca_items = int(value(f, "scope", "ca_items"))
units_share = value(f, "scope", "ax_ay_share_of_units")
all_rows = mase[(mase["category"] == "ALL") & mase["model"].isin(["seasonal_naive", "ets", BEST])].set_index("model")
naive = all_rows.loc["seasonal_naive", "median_item_mase"]
best = all_rows["median_item_mase"].idxmin()
best_val = all_rows.loc[best, "median_item_mase"]
gain = 1 - best_val / naive
adj = lookup(f, "breaks", "adjusted_share_with_real_shift", "CUSUM")
stock = lookup(f, "inventory", "static_vs_naive_base_95")
c95 = cal[(cal["policy"] == f"{BEST}_static") & (cal["service_level"] == 0.95)].set_index("method")
fr_text, fr_cal = c95.loc["normal formula", "fill_rate"], c95.loc["conformal", "fill_rate"]

# every version that reacts to a shift, and whether any of them met the "helps" rule
best_all = variants[variants["group"] == "all detected breaks"]
reacting = len(best_all) + len(adaptive[(adaptive["comparison"].str.startswith("adaptive")) & (adaptive["category"] == "ALL")
                                      & adaptive["model"].isin(["ets", "tsb"])]) \
    + len(shift_sarima[shift_sarima["group"] == "all break item-origins"])
any_better = bool(best_all["helps"].astype(str).str.lower().eq("true").any()) or bool(
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
     "compare": f"vs the baseline forecast; {stock['inventory_change']:+.0%} stock held",
     "tone": "good" if stock["value"] < 0 else "bad",
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
c95 = cal[cal["service_level"] == 0.95].set_index(["method", "policy"])["total_cost"]
conf_vs = c95[("conformal", f"{BEST}_static")] / c95[("conformal", "seasonal_naive_static")] - 1
more = lambda x, lo, hi: f"{abs(x):.0%} {lo if x < 0 else hi}"
s3 = (f"In the stock simulation with the textbook safety stock, the better forecast holds "
      f"{more(stock['inventory_change'], 'less', 'more')} stock at {more(stock['value'], 'lower', 'higher')} cost than the "
      f"baseline forecast; with safety stock sized from past errors it costs {more(conf_vs, 'less', 'more')}. Sizing "
      f"safety stock from past errors also lifts the fill rate from {fr_text:.1%} to {fr_cal:.1%}.")
st.markdown(f"{s1} {s2} {s3}")
todo = (f"**What to do with this:** plan with {name(best)} forecasts as they are"
        + (", do not retrain them after a detected shift," if not any_better else ",")
        + " and size safety stock from past forecast errors rather than the textbook formula.")
st.markdown(todo)

# the eight time-series models of 07h on the 150-item sample, on RMSE, the measure the winner was picked on, with
# ARIMA beside them for reference
head = get("ts_headline").assign(model=lambda d: d["model"].replace({"ets": "ets_boxcox"})).set_index("model")
head = head.drop(index="sarima_07a", errors="ignore")                   # the same forecasts as the SARIMA contender
ov = head.loc[[m for m in CONTENDERS if m in head.index]].sort_values("rmse")
naive_rmse = ov.loc["seasonal_naive", "rmse"]
REF = [m for m in ["arima"] if m in head.index]
shown = head.loc[list(ov.index) + REF].sort_values("rmse", ascending=False)
tag = {PICKED: "  (used for stock)", **{m: "  (reference, not one of the eight)" for m in REF}}
fig = go.Figure(go.Bar(x=shown["rmse"], y=[SHORT[m] + tag.get(m, "") for m in shown.index],
                       orientation="h", marker=dict(color=[COLORS[m] for m in shown.index],
                                                    opacity=[0.5 if m in REF else 1 for m in shown.index]),
                       customdata=np.c_[shown["mae"], shown["mase"], shown["rmse"] / naive_rmse - 1],
                       text=[f"{v:.3f}" + ("" if m == "seasonal_naive" else f"  ({v / naive_rmse - 1:+.0%})")
                             for m, v in shown["rmse"].items()],
                       textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}<br>RMSE %{x:.3f} (%{customdata[2]:+.1%} vs baseline)<br>MAE %{customdata[0]:.3f}"
                                     "<br>MASE %{customdata[1]:.3f}<extra></extra>"))
fig.add_vline(x=naive_rmse, line_dash="dot", line_color="#8C8C8C")
style(fig, height=110 + 34 * len(shown), legend=False).update_xaxes(
    title="typical forecast error (RMSE in units per day, lower is better)", range=[0, naive_rmse * 1.22])
fitted = ov.drop("seasonal_naive")
top, runner = fitted.index[0], fitted.index[1]
close = fitted[fitted["rmse"] <= fitted["rmse"].iloc[0] * 1.05]
chart(fig, f"Of the eight time-series models, {SHORT[top]} has the lowest typical RMSE, and every fitted model beats the baseline"
      if bool((fitted["rmse"] < naive_rmse).all()) else
      f"Of the eight time-series models, {SHORT[top]} has the lowest typical RMSE",
      f"Typical RMSE over the {int(ov['n_products'].iloc[0])} sample products and the six scored forecast dates (Dec 2015 "
      f"to Apr 2016); the dotted line is the baseline, repeating last week ({naive_rmse:.3f}). {SHORT[top]} "
      f"({fitted['rmse'].iloc[0]:.3f}) is {fitted['rmse'].iloc[1] - fitted['rmse'].iloc[0]:.3f} ahead of {SHORT[runner]}, and "
      f"{len(close)} of the {len(fitted)} fitted models are within 5% of it ({', '.join(SHORT[m] for m in close.index)}), so "
      f"the good ones sit close together. "
      + "".join(f"The pale bar is {name(m)}, shown for reference: {head.loc[m, 'rmse']:.3f}, "
                f"{1 - head.loc[m, 'rmse'] / naive_rmse:.0%} less error than the baseline but "
                f"{head.loc[m, 'rmse'] / fitted['rmse'].iloc[0] - 1:.0%} more than {SHORT[top]}. " for m in REF)
      + "RMSE is the measure the model for the stock work was picked on, before the run. The Forecast accuracy page has "
        "every measure, by date range, item group and forecast horizon.")

# every time-series model, the building blocks of 07c included
KIND = {"contender": "One of the eight", "building block": "Building block (07c)",
        "combination, second version": "Second version of the combination"}
blocks = head[head["kind"] == "building block"].sort_values("rmse")
table = head.sort_values("rmse").assign(
    Model=lambda d: [name(m) + ("  (used for stock)" if m == PICKED else "") for m in d.index],
    Type=lambda d: ["Baseline" if m == "seasonal_naive" else KIND.get(k, k) for m, k in d["kind"].items()],
    vs_baseline=lambda d: d["rmse"] / naive_rmse - 1)
table = table[["Model", "Type", "rmse", "mae", "mape", "mase", "vs_baseline"]].rename(
    columns={"rmse": "RMSE", "mae": "MAE", "mape": "MAPE", "mase": "MASE", "vs_baseline": "RMSE vs baseline"})
st.markdown("<div class='chart-title'>Every time-series model, building blocks included</div>", unsafe_allow_html=True)
st.dataframe(table, hide_index=True, width="stretch",
             column_config={**{c: st.column_config.NumberColumn(format="%.3f") for c in ["RMSE", "MAE", "MASE"]},
                            "MAPE": st.column_config.NumberColumn(format="percent"),
                            "RMSE vs baseline": st.column_config.NumberColumn(format="percent")})
st.markdown(f"<div class='takeaway'>Same {int(ov['n_products'].iloc[0])} products and forecast dates as the chart, lower "
            f"is better for every measure. The building blocks ({', '.join(SHORT[m] for m in BLOCKS if m in blocks.index)}) "
            f"are the parts "
            f"SARIMA is assembled from, fitted one at a time. The best of them, {name(blocks.index[0])} "
            f"({blocks['rmse'].iloc[0]:.3f}), is {blocks['rmse'].iloc[0] / fitted['rmse'].iloc[0] - 1:.0%} behind "
            f"{SHORT[top]}; {name(blocks.index[-1])} is the weakest ({blocks['rmse'].iloc[-1]:.3f}). Adding the weekly term "
            f"turns ARIMA ({head.loc['arima', 'rmse']:.3f}) into SARIMA ({head.loc['sarima', 'rmse']:.3f}).</div>",
            unsafe_allow_html=True)
download(table, "time_series_models.csv", key="dl_ov_models")

cal_save = lookup(f, "inventory", "conformal_vs_normal_cost_change", "0.95")["value"]
bigger = "how safety stock is sized" if abs(cal_save) > abs(stock["value"]) else "the better forecast"
tells(f"{name(best)} cuts the typical error by {gain:.0%}, "
      + ("while reacting to shifts did not help. " if not any_better else "and some shift-aware versions helped. ")
      + f"In the simulation, sizing safety stock from past errors changes total cost by {cal_save:+.0%}, against "
      f"{stock['value']:+.0%} from the better forecast with the textbook safety stock, so {bigger} matters more. The "
      f"costs behind these numbers are assumptions; the Stock impact page lets you change them.")
