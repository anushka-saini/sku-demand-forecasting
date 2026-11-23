import pandas as pd
import streamlit as st

from utils.final import TABLES, best_info, fc_index, load, names, need_results, text
from utils.ui import kpis, page, tells

need_results()
split = load("split")
lb = load("leaderboard")
nm = names(lb)
tx = text()
info = best_info()
idx = fc_index()
n_items, n_win = idx["item_id"].nunique(), idx["origin"].nunique()
page("Method and Limits", "How were the models tested, and what do the results not prove?")
st.caption("The split, the tuning, the checks against peeking at the future, and the limits of the study.")

val = split[split["part"].str.startswith("validation")]
kpis([
    {"label": "Items in the comparison", "value": f"{n_items}", "compare": "a random sample of the study set, drawn with seed 42"},
    {"label": "Validation windows", "value": f"{len(val)}", "compare": "used to tune and to choose combinations"},
    {"label": "Test windows", "value": f"{n_win}", "compare": "scored once, 28 days each"},
    {"label": "Models compared", "value": f"{lb[lb['level'] == 'item']['model'].nunique()}", "compare": "classical, machine learning, deep learning, combinations"},
])

st.markdown("#### The split")
sp = split.copy()
sp.columns = ["Part", "Forecast dates", "First day", "Last day"]
st.dataframe(sp, hide_index=True, width="stretch")
st.markdown("Each model sees only the sales up to a forecast date and forecasts the next 28 days. The test windows were scored once, after all choices were made.")

st.markdown("#### Tuning")
tune = {"XGBoost": "final_xgb_trials.csv", "Prophet": "final_prophet_tuning.csv", "LSTM": "final_lstm_tuning.csv",
        "GRU": "final_gru_tuning.csv", "PatchTST": "final_patchtst_tuning.csv"}
rows = [{"Model": k, "Settings tried": len(pd.read_csv(TABLES / f))} for k, f in tune.items() if (TABLES / f).exists()]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown("Settings were chosen on the three validation windows only, by the score on those windows. The stacking weights and the average "
            "of the top three were also fitted there.")

st.markdown("#### How we kept the future out")
st.markdown("- Every feature is read from sales up to the forecast date. A test scrambles all sales after a forecast date and checks that no feature changes.\n"
            "- Prices and the calendar are known ahead of time, as in the original competition, so they are allowed.\n"
            "- Tuning and combination weights use the validation windows, never the test windows.\n"
            "- The Forecaster page loads the real sales for a window only after you press Reveal.\n"
            "- One caution: the 726 study items were picked from the whole sales history, including the test months. That does not change any "
            "forecast, but the choice of items is not strictly out of sample.")

st.markdown("#### Limitations")
for t in tx["limitations"]:
    st.markdown("- " + t)
st.markdown("#### Challenges")
for t in tx["challenges"]:
    st.markdown("- " + t)
st.markdown("#### Future scope")
for t in tx["future"]:
    st.markdown("- " + t)
tells(f"The test is clean on the forecasts, the gaps between the top models are small, and the stock numbers use assumed costs. "
      f"Treat {nm[info['model']]} as the best choice on this data, not as proven for every item or store.")
