import pandas as pd
import streamlit as st

from utils.data import get
from utils.derive import eval_origins
from utils.facts import facts, lookup, value
from utils.kpis import base_cost_assumptions
from utils.models import name
from utils.ui import download, kpis, page, tells
from utils.verdicts import POWER_FOR_YES

f = facts()
bo = get("breaks_by_origin")
inv = get("inventory_policy")
rep = get("rep_conclusions")
diag = get("residual_checks")
H_BASE, M_BASE = base_cost_assumptions(inv)
w = f["scope"]["windows"]["W1"]
H = int(get("forecast_daily")["horizon"].max())
LEAD = int(get("calibrated_items")["lead_time"].iloc[0])

page("Method and limits", "How were the forecasts tested, and what do the results not prove?")

origins = sorted(pd.Timestamp(x) for x in bo["origin"].unique())
evals = eval_origins()
gap = (evals[1] - evals[0]).days
kpis([
    {"label": "Forecast dates tested", "value": f"{len(evals)}", "compare": f"every {gap} days, plus one warm-up date"},
    {"label": "Days ahead per forecast", "value": f"{H}", "compare": f"each forecast covers the next {H} days"},
    {"label": "Days scored", "value": f"{(pd.Timestamp(w['last_eval_day']) - pd.Timestamp(w['first_eval_day'])).days + 1}",
     "compare": f"{pd.Timestamp(w['first_eval_day']):%d %b %Y} to {pd.Timestamp(w['last_eval_day']):%d %b %Y}"},
    {"label": "Checks that hold in two other states", "value": f"{int(rep['holds'].sum())} of {len(rep)}",
     "compare": ", ".join(sorted(rep["state"].unique()))},
])

st.markdown("#### How the test worked")
st.markdown(f"We stood at {len(origins)} past dates, {gap} days apart: a warm-up on {origins[0]:%d %b %Y} and {len(evals)} scored "
            f"dates from {evals[0]:%d %b %Y} to {evals[-1]:%d %b %Y}. At each one every method was fitted on the sales up to that "
            f"date only and forecast the next {H} days. Within each comparison, every method is scored on the same items and days; "
            f"the warm-up's errors only seed the uncertainty bands and safety stock.")

st.markdown("#### How we kept the future out")
st.markdown("- Each model is fitted only on data before its forecast date, and anything chosen about a model (its settings, its "
            "ARIMA orders) is chosen on that training data.\n"
            "- The shift detector's alarm threshold was set on data before the first forecast date, and shifts are only used if "
            "they were detected by the forecast date.\n"
            "- Model settings (transforms, regressors, which models to combine) were chosen on three validation dates before "
            "the scored ones (Sep to Dec 2015), never on the scored dates.\n"
            "- Two things are known in advance, as in the original competition: prices and the calendar. The item list was chosen "
            "on the full history, which is a mild look-ahead in which items were studied, though not in any forecast.")

st.markdown("#### The main assumptions")
st.markdown(f"Sales of the four California stores are added into one series per item, so store-level stockouts are hidden. "
            f"The stock simulation assumes a {LEAD}-day lead time, a holding cost of {H_BASE:.0%} of the price a year and a lost sale "
            f"costing {M_BASE:g} times the price, with no ordering cost; the Stock impact page lets you change these.")

st.markdown("#### What the results do not prove")
s20 = lookup(f, "breaks", "injection_power_cusum", "step_20pct")["value"]
share = value(f, "scope", "ax_ay_share_of_items")
st.markdown(f"- They cover the high-volume, steady items only ({share:.0%} of the range); slow or erratic items may behave differently.\n"
            f"- The detector catches a sudden 20% shift only {s20:.0%} of the time, so \"reacting does not help\" is about the "
            f"shifts it can see.\n"
            f"- Costs are assumptions. Fill rate, stockouts and stock held come out of the simulation; money depends on the costs.\n"
            f"- Forecasts are scored on one half-year, {pd.Timestamp(w['first_eval_day']):%b %Y} to "
            f"{pd.Timestamp(w['last_eval_day']):%b %Y}; the stock results were also checked on two earlier half-years.\n"
            f"- Every forecast here comes from a time-series model; machine-learning models were left out of the project.")

with st.expander("The same checks in Texas and Wisconsin"):
    t = rep[["state", "conclusion", "holds", "evidence"]].copy()
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"state": "State", "conclusion": "Check", "holds": st.column_config.CheckboxColumn("Holds"),
                                "evidence": st.column_config.TextColumn("Evidence", width="large")})
    download(t, "other_states.csv", key="dl_rep")

with st.expander("Do the classical models' assumptions hold?"):
    d = diag[diag["level"] == "item"].assign(model=lambda x: x["model"].map(name))
    st.dataframe(d[["model", "n", "ljung_box_pass", "normality_pass", "roots_ok"]], hide_index=True, width="stretch",
                 column_config={"model": "Method", "n": st.column_config.NumberColumn("Items", format="%d"),
                                "ljung_box_pass": st.column_config.NumberColumn("No pattern left in the errors", format="percent"),
                                "normality_pass": st.column_config.NumberColumn("Errors look normal", format="percent"),
                                "roots_ok": st.column_config.NumberColumn("Model is stable", format="percent")})
    best = d.loc[d["ljung_box_pass"].idxmax()]
    st.caption(f"{best['model']} leaves no pattern in its errors for {best['ljung_box_pass']:.0%} of items. Errors almost never "
               f"look normal (at most {d['normality_pass'].max():.1%} of items), which is why safety stock is also sized from the "
               f"errors themselves.")

with st.expander("How the Overview's answers are decided"):
    st.markdown(f"- Shifts: yes if the share with a real shift is above zero after removing chance alarms and a sudden 20% shift "
                f"is caught at least {POWER_FOR_YES:.0%} of the time; partly if only the first holds.\n"
                "- Best forecast: the method with the lowest typical error on all items.\n"
                "- Retraining: no if none of the ways of reacting improves error by at least 1% of the static error with a "
                "corrected p-value below 0.05.\n"
                "- Stock cost: yes if the better forecast is cheaper, with the whole 95% range below zero, in at least two of "
                "three half-years and in both other states.")

with st.expander("Where to read more"):
    st.markdown("`README.md` for how to run everything; `docs/MODEL_CHOICES.md` for why each method is there; "
                "`docs/RESULTS.md` for every result with its numbers; `docs/RUN_LOG.md` for every run; `docs/GLOSSARY.md` "
                "for the terms used; `docs/DASHBOARD_MAP.md` for where each old dashboard page went.")

tells(f"Every method saw only the past, on the same {len(evals)} dates and the same "
      f"items, and {int(rep['holds'].sum())} of {len(rep)} checks hold in two other states. The weak points are the coarse "
      f"shift detector, the assumed costs and the narrow scope of high-volume items.")
