import pandas as pd
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup, value
from utils.kpis import base_cost_assumptions
from utils.ui import GLOSSARY, header

f = facts()
inv = get("inventory_policy")
sens = get("cost_sensitivity")
items = get("items")
H_BASE, M_BASE = base_cost_assumptions(inv)
w = f["scope"]["windows"]

header("Assumptions, limitations and how to reproduce", "What this analysis can and cannot tell you.")

st.markdown("#### Scope")
st.markdown(
    f"- **California, replicated in Texas and Wisconsin.** Sales are summed across each state's stores into one series "
    f"per item, so store-level stockouts and substitution between items are hidden.\n"
    f"- **High-volume items only:** {value(f, 'scope', 'ax_ay_items')} of {value(f, 'scope', 'ca_items'):,} California items "
    f"(AX/AY), carrying {value(f, 'scope', 'ax_ay_share_of_units'):.0%} of units. Results may not hold for slow or erratic items.\n"
    f"- **ABC uses units sold, not revenue.** The item list was chosen using the full history (a mild lookahead in item "
    f"selection only; no forecast or stock decision uses future data).\n"
    f"- **Evaluation:** forecasts {w['W1']['first_eval_day']} to {w['W1']['last_eval_day']}; inventory also on two earlier "
    f"168-day windows ({w['W2']['seasonal_coverage']}; {w['W3']['seasonal_coverage']}).\n"
    f"- **M5 ends on {w['W1']['last_eval_day']}**, so there is no COVID-era data.\n"
    f"- **Prices and calendar events are treated as known in advance** (the M5 convention).")

st.markdown("#### Assumed parameters (not observed data)")
st.dataframe(pd.DataFrame([
    {"Parameter": "Lead time (days)", "Base value": "7", "Tested": ", ".join(str(x) for x in sorted(inv["lead_time"].unique()))},
    {"Parameter": "Holding cost (share of unit price per year)", "Base value": f"{H_BASE:.0%}",
     "Tested": ", ".join(f"{x:.0%}" for x in sorted(sens["holding_rate"].unique()))},
    {"Parameter": "Cost of a lost sale (× unit price)", "Base value": f"{M_BASE:g}",
     "Tested": ", ".join(f"{x:g}" for x in sorted(sens["stockout_multiple"].unique()))},
    {"Parameter": "Ordering cost", "Base value": "0 (ignored)", "Tested": "not tested"},
    {"Parameter": "Unit value", "Base value": "median sell price that week", "Tested": "not tested"},
    {"Parameter": "Unmet demand", "Base value": "lost (no backorders)", "Tested": "not tested"},
    {"Parameter": "Review period", "Base value": "daily review, order up to 7 extra days", "Tested": "not tested"},
]), hide_index=True, width="stretch")

far = lookup(f, "breaks", "seasonal_null_false_alarm_rate", "CUSUM")["value"]
old_far = lookup(f, "breaks", "old_null_false_alarm_rate", "CUSUM")["value"]
real = lookup(f, "breaks", "real_series_alarm_after_random_date_no_injection")["value"]
s20 = lookup(f, "breaks", "injection_power_cusum", "step_20pct")
cov = lookup(f, "forecasts", "band_coverage", "lightgbm_static_90_origins2-6")
bias = value(f, "forecasts", "bias", "lightgbm_static_ALL")
st.markdown("#### What did not work")
st.markdown(
    "1. **The first break detector flagged 100% of items.** It used whole-series averages, started on day 1 and used a "
    "textbook threshold meant for independent data. It was fixed with weekly data, a launch burn-in, a baseline year "
    "and a threshold calibrated on simulated break-free series.\n"
    f"2. **The first false-alarm estimate was optimistic.** {old_far:.1%} held only on the simple null the threshold was "
    f"set on; on a seasonality-preserving null it is {far:.1%}. An earlier figure ({real:.0%} alarms on real series with "
    "nothing injected) was wrongly described as a false-alarm rate.\n"
    f"3. **The detector is coarse.** A sudden 20% shift is caught {s20['value']:.0%} of the time (median delay "
    f"{s20['median_delay_weeks']:.0f} weeks), and it cannot reliably separate steps from slow drift.\n"
    "4. **Retraining after a shift did not improve forecasts or stock cost**, with four different methods, on "
    "high-confidence shifts only, and in two other states.\n"
    "5. **TSB did not beat LightGBM on intermittent items.**\n"
    f"6. **The original uncertainty bands were too narrow** ({cov['coverage_before']:.1%} of days inside a 90% band) and "
    f"the textbook safety stock under-delivered. Calibration fixed the bands ({cov['value']:.1%}) but forecasts are still "
    f"{abs(bias):.1%} {'low' if bias < 0 else 'high'} in total.")

st.markdown("#### What would change the conclusion")
st.markdown(
    "- **A detector that separates steps from drift**, or shifts with a known cause (a recorded price or packaging change).\n"
    "- **Different adaptation**, such as retraining only after large, confirmed shifts, or a different model family.\n"
    "- **Store-level or slower-moving items**, where shifts are relatively larger.\n"
    "- **Real cost data**: much higher holding costs (perishables) could make a target below 99% cheapest.\n"
    "- **A longer evaluation** with more forecast dates.")

st.markdown("#### How to reproduce")
st.code("pip install -r requirements.txt\n"
        "# put the M5 files in data/raw/ (see docs/DATA.md)\n"
        "make all          # runs every notebook, rebuilds outputs/results_facts.json, runs the tests\n"
        "streamlit run dashboard/app.py", language="bash")
st.markdown("Analyses after the first version were pre-registered (`docs/PREREGISTRATION.md`); every run, including a "
            "failed one, is in `docs/RUN_LOG.md`; full results are in `docs/RESULTS.md`. Every number on these pages is "
            "read from `outputs/results_facts.json` or `outputs/tables/`.")

st.markdown("#### Glossary")
st.dataframe(pd.DataFrame(list(GLOSSARY.items()), columns=["Term", "Meaning"]), hide_index=True, width="stretch",
             column_config={"Meaning": st.column_config.TextColumn(width="large")})
