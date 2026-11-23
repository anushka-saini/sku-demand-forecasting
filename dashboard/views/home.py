import streamlit as st

from utils.facts import facts, lookup, value
from utils.final import best_info, load, names, need_results, text
from utils.ui import kpis, page, tells, verdict_box

need_results()
lb = load("leaderboard")
lbi = lb[lb["level"] == "item"].sort_values("rank")
nm = names(lb)
b = lbi.iloc[0]
n = lbi[lbi["model"] == "seasonal_naive"].iloc[0]
tx = text()
F = facts()

page("Overview", "Which forecasting model should drive stock planning for high-volume Walmart items, and how sure are we?")
st.caption("A shop has to guess how much of each product it will sell, so it keeps enough on the shelf without paying for stock "
           "that sits there. This dashboard shows the test of 21 forecasting methods on daily sales, and what the best one means for stock.")

kpis([
    {"label": "Best model", "value": b["name"].split(" (")[0], "compare": f"first of {len(lbi)} models on MAE and RMSE"},
    {"label": "MAE (units a day)", "value": f"{b['mae']:.2f}", "compare": f"{abs(b['mae'] / n['mae'] - 1):.0%} lower than repeating last week ({n['mae']:.2f})", "tone": "good",
     "help": "Mean absolute error: the average miss per day. Typical value over items."},
    {"label": "RMSE (units a day)", "value": f"{b['rmse']:.2f}", "compare": f"{abs(b['rmse'] / n['rmse'] - 1):.0%} lower than repeating last week ({n['rmse']:.2f})", "tone": "good",
     "help": "Root mean squared error: like MAE, but big misses count more."},
    {"label": "Improvement over seasonal naive", "value": f"{abs(b['mae'] / n['mae'] - 1):.0%}", "compare": "on MAE; the baseline repeats last week", "tone": "good"},
])

st.markdown("#### The problem")
st.markdown("Sometimes a product's sales change for good: it gets more popular, a competitor turns up, the price moves. "
            "Forecasts that ignore this can leave shelves empty or full. We use Walmart's public M5 data: daily sales of every "
            "product in its California stores from 2011 to 2016.")
st.markdown("#### The three questions")
adj = lookup(F, "breaks", "adjusted_share_with_real_shift", "CUSUM")
n_all = int(value(F, "scope", "ax_ay_items"))
qs = [("1. Do lasting changes in demand really happen, and can we spot them?",
       f"Yes. After removing chance alarms, about {adj['value']:.0%} of the {n_all} high-volume items changed their normal sales level at some point."),
      ("2. Which forecast is best, and does updating it after a change help?", tx["winner"]),
      ("3. What does a better forecast mean for how much stock to keep?", tx["stock"].split(" With the textbook")[0])]
for q, a in qs:
    with st.expander(q, expanded=False):
        st.markdown(a)

verdict_box(["Pick the model on accuracy and speed together: " + tx["winner"].split(". ")[0] + ".",
             tx["combos"].split(". ")[0] + "."], head="Short answer")

st.markdown("#### How to use this dashboard")
st.markdown("- **Data**: what is in the sample and how the sales are split into training, validation and test.\n"
            "- **Leaderboard**: every model on the same test, with a table you can download.\n"
            "- **Model Lab**: pick two or more models and compare them by category, weekday and forecast day.\n"
            "- **Forecaster**: pick an item and a test window, see the forecast, make your own guess, then reveal what sold.\n"
            "- **Inventory**: what the forecasts mean for stock, cost and fill rate (with assumed costs).\n"
            "- **Diagnostics**: where models fail, which gaps are real, and the GARCH and VAR results.\n"
            "- **Method and Limits**: how the test was run and what it does not prove.")
tells(f"{b['name']} is the leader on both error measures. Judge it on all three pages that follow, "
      "not on one item: the Forecaster shows single cases, the Leaderboard shows the whole test.")
