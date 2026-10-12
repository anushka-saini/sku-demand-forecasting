import pandas as pd
import streamlit as st

from utils.data import get
from utils.kpis import base_cost_assumptions
from utils.ui import header

inv = get("inventory_policy")
sens = get("cost_sensitivity")
power = get("power_check")
mase = get("model_mase").set_index("category")
items = get("items")
H_BASE, M_BASE = base_cost_assumptions(inv)

header("Assumptions & limitations", "What this analysis can and cannot tell you.")

st.markdown("#### Scope")
st.markdown(
    f"- **California only**: the 4 CA stores, summed into one series per item. Store-level stockouts and "
    f"substitution between items are hidden.\n"
    f"- **AX/AY items only**: {int(items['in_scope'].sum())} of {len(items):,} items, the high-volume and reasonably "
    f"predictable ones. Results may not hold for slow or erratic items.\n"
    f"- **ABC uses units sold, not revenue.**\n"
    f"- **One adaptation mechanism**: retrain on the data after the most recent detected break. Other approaches "
    f"(level-shift terms, down-weighting old data) were not tested.\n"
    f"- **M5 sales end on 2016-05-22**, so there is **no COVID-era data** and nothing here relates to COVID.\n"
    f"- **Short evaluation**: 6 windows × 28 days = 168 days (Dec 2015 to May 2016), including Christmas.\n"
    f"- **Prices and calendar events are treated as known in advance** (the M5 convention).")

st.markdown("#### Assumed parameters (not observed data)")
st.dataframe(pd.DataFrame([
    {"Parameter": "Lead time (days)", "Base value": "7",
     "Tested": ", ".join(str(x) for x in sorted(inv["lead_time"].unique()))},
    {"Parameter": "Holding cost (share of unit price per year)", "Base value": f"{H_BASE:.0%}",
     "Tested": ", ".join(f"{x:.0%}" for x in sorted(sens["holding_rate"].unique()))},
    {"Parameter": "Stockout cost per lost unit (× unit price)", "Base value": f"{M_BASE:g}",
     "Tested": ", ".join(f"{x:g}" for x in sorted(sens["stockout_multiple"].unique()))},
    {"Parameter": "Ordering cost", "Base value": "0 (ignored)", "Tested": "not tested"},
    {"Parameter": "Unit value", "Base value": "CA median sell price that week", "Tested": "not tested"},
    {"Parameter": "Unmet demand", "Base value": "lost (no backorders)", "Tested": "not tested"},
    {"Parameter": "Review period / order cycle", "Base value": "daily review, order up to 7 days extra", "Tested": "not tested"},
]), hide_index=True, width="stretch")

fd = get("forecast_daily")
lgb = fd[(fd["model"] == "lightgbm") & (fd["variant"] == "static")]
coverage = ((lgb["actual"] >= lgb["lo_90"]) & (lgb["actual"] <= lgb["hi_90"])).mean()
bias = lgb["forecast"].sum() / lgb["actual"].sum() - 1
ctrl = power[power["size_sd"] == 0].iloc[0]
one = power[power["size_sd"] == 1].iloc[0]
st.markdown("#### What did not work")
st.markdown(
    f"1. **The first break detector flagged 100% of items.** It used whole-series averages, started on day 1 and "
    f"used a textbook threshold meant for independent data. It was fixed with weekly data, a launch burn-in, a "
    f"baseline year and a threshold calibrated on simulated break-free series.\n"
    f"2. **CUSUM cannot tell sudden steps from slow drift.** With no shift injected, it still alarms in "
    f"{ctrl['cusum_detection_rate']:.0%} of series. A 1σ shift (≈{one['shift_pct_of_mean_median']:.0f}% of mean "
    f"level) is detected only {one['cusum_detection_rate']:.0%} of the time, with a median delay of "
    f"{one['cusum_median_delay_wks']:.0f} weeks.\n"
    f"3. **A step-versus-trend test was not informative.** It preferred 'step' even for items with no break.\n"
    f"4. **Retraining after a break did not improve forecasts or inventory cost.** A placebo with random training "
    f"cuts showed no effect either.\n"
    f"5. **TSB did not beat LightGBM on intermittent items.**\n"
    f"6. **Uncertainty is understated.** LightGBM's 90% bands cover {coverage:.1%} of actual days, and simulated "
    f"fill rates fall below the service targets. Its forecasts are {abs(bias):.1%} {'low' if bias < 0 else 'high'} "
    f"in total, which the normal safety-stock formula does not correct.")

st.markdown("#### What would change the conclusion")
st.markdown(
    "- **A detector that separates steps from drift**, or breaks with a known cause (e.g. a recorded price or "
    "packaging change). Adaptation may help when the shift is abrupt and large.\n"
    "- **A different adaptation**: adding a level-shift term or down-weighting old data instead of discarding it, or "
    "retraining only after large shifts.\n"
    "- **Store-level or slower-moving items**, where shifts are relatively larger and forecasts are less able to "
    "self-correct.\n"
    "- **Real cost data**: if holding costs were far higher than assumed (e.g. perishables), the cheapest service "
    "level could fall below 99%. The adaptive vs static ranking held in every scenario tested.\n"
    "- **A longer evaluation period** with more forecast dates.")
st.caption(f"Benchmark for context: median MASE of LightGBM {mase.loc['ALL', 'lightgbm']:.3f}, seasonal naive "
           f"{mase.loc['ALL', 'seasonal_naive']:.3f}.")
