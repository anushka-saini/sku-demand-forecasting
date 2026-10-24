import streamlit as st

from utils.facts import facts, lookup, value
from utils.verdicts import POWER_FOR_YES, all_verdicts, classical_verdicts
from utils.data import get
from utils.ui import insight, verdict_card

f = facts()
verdicts = all_verdicts(f)

st.title("Do demand shifts matter for forecasting and stock?")
st.markdown(
    "<p class='intro'>Walmart's M5 data records daily unit sales for every item in its California stores from 2011 "
    "to 2016. This project asks whether individual items' demand shifts in lasting ways, whether retraining the "
    "forecast after a shift helps, and what that means for how much stock to hold. Every answer below is computed "
    "from the results files, not typed in.</p>", unsafe_allow_html=True)

retrain = verdicts[2]
if retrain["verdict"] == "No":
    st.markdown("<div class='headline'><b>Retraining after a detected shift does not help.</b> That negative result "
                "is the main finding: the gains come from a better static forecast and better-calibrated safety "
                "stock, not from reacting to shifts.</div>", unsafe_allow_html=True)

cols = st.columns(4)
for col, v in zip(cols, verdicts):
    with col:
        verdict_card(v)
        st.page_link(v["page"], label="See the evidence")

extra = classical_verdicts(f)
if extra:
    st.markdown("#### Classical time-series models")
    for col, v in zip(st.columns(4), extra):
        with col:
            verdict_card(v)
            st.page_link(v["page"], label="See the evidence")

st.markdown("#### Every model, side by side")
ov = get("comparison_overview")
st.dataframe(ov[["model", "median_mase", "median_rmse", "median_mae", "median_smape", "median_mase_726", "beats_naive_share"]],
             hide_index=True, width="stretch",
             column_config={"model": "Model",
                            "median_mase": st.column_config.NumberColumn("MASE (150 products)", format="%.3f"),
                            "median_rmse": st.column_config.NumberColumn("RMSE (units)", format="%.2f"),
                            "median_mae": st.column_config.NumberColumn("MAE (units)", format="%.2f"),
                            "median_smape": st.column_config.NumberColumn("sMAPE", format="%.3f"),
                            "median_mase_726": st.column_config.NumberColumn("MASE (all 726)", format="%.3f"),
                            "beats_naive_share": st.column_config.NumberColumn("beats seasonal naive on", format="percent")})
st.caption("Medians over products, six scored forecast dates, 28 days ahead. Lower is better; MASE below 1 beats "
           "repeating last week. Only seasonal naive, ETS and LightGBM were run on all 726 products.")
st.page_link("views/comparison.py", label="See the full comparison")

st.markdown("#### Key numbers")
adj = lookup(f, "breaks", "adjusted_share_with_real_shift", "CUSUM")
lgb, naive = value(f, "forecasts", "median_item_mase", "lightgbm_ALL"), value(f, "forecasts", "median_item_mase", "seasonal_naive_ALL")
trunc = lookup(f, "forecasts", "adaptation_variants", "all detected breaks|truncate")
cal = lookup(f, "inventory", "conformal_vs_normal_cost_change", "0.95")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Items with a real demand shift", f"{adj['value']:.1%}",
              help=f"95% CI {adj['ci95'][0]:.1%} to {adj['ci95'][1]:.1%}; n = {adj['n']} items")
    st.caption(f"What this means: {adj['value']:.0%} of high-volume items changed their typical sales level at some "
               f"point, after discounting alarms that would happen by chance.")
with c2:
    st.metric("LightGBM forecast error (MASE)", f"{lgb:.3f}", delta=f"{lgb - naive:+.3f} vs baseline {naive:.3f}",
              delta_color="inverse", help="Median MASE across items (lower is better; below 1 beats repeating last week in-sample).")
    st.caption(f"What this means: the machine-learning model's typical error is {1 - lgb / naive:.0%} lower than "
               f"repeating last week's sales.")
with c3:
    st.metric("Retraining effect (MASE)", f"{trunc['value']:+.4f}",
              help=f"Median per-item change, 95% CI {trunc['ci95'][0]:+.4f} to {trunc['ci95'][1]:+.4f}; n = {trunc['n']} items")
    st.caption("What this means: restarting the model from the shift made forecasts "
               + ("slightly worse." if trunc["ci95"][0] > 0 else "slightly better." if trunc["ci95"][1] < 0 else "neither better nor worse."))
with c4:
    st.metric("Cost, calibrated safety stock", f"{cal['value']:+.1%}",
              help=f"Total cost at a 95% target, lead time 7 days, assumed costs; n = {cal['n']} items")
    st.caption("What this means: sizing safety stock from realistic error bands, rather than a textbook formula, "
               "cut simulated cost. Costs are assumptions.")

with st.expander("How the verdicts are decided"):
    st.markdown(
        f"- **Shifts:** *Yes* if the chance-adjusted share's 95% CI is above 0 **and** a sudden 20% shift is detected "
        f"at least {POWER_FOR_YES:.0%} of the time; *Partly* if only the first holds. *Strong* if it also replicates "
        f"in Texas and Wisconsin.\n"
        "- **Best forecast:** the model with the lowest median error. *Strong* if it beats every other model "
        "significantly in all three states, otherwise *moderate*.\n"
        "- **Retraining:** *No* if none of the four methods meets the pre-registered 'helps' rule, on all breaks or on "
        "high-confidence breaks. *Strong* if truncation also fails to help in Texas and Wisconsin.\n"
        "- **Stock cost:** *Yes* if the better forecast is cheaper with a 95% CI below 0 in at least 2 of 3 time windows "
        "and cheaper in both other states. *Strong* if all 3 windows qualify.\n"
        "- **Classical vs baseline (H1):** *Yes* if both SARIMA and Holt-Winters additive beat seasonal "
        "naive (Holm p < 0.05, median difference below 0); *Partly* if one does.\n"
        "- **Shift-aware SARIMA (H3):** *No* if neither restart nor the level-shift dummy meets the "
        "pre-registered 'helps' rule (Holm p < 0.05 and at least 1% of the static SARIMA error).")

ovi = ov.set_index("model")
insight(f"Demand shifts are real and common, but reacting to them does not improve forecasts or stock costs, for "
        f"LightGBM or for the classical models. What helps is a good static forecast, where LightGBM "
        f"({ovi.loc['LightGBM', 'median_mase']:.3f}) is only just ahead of SARIMA ({ovi.loc['SARIMA', 'median_mase']:.3f}), "
        f"and safety stock sized from the real spread of forecast errors.")
