import html

import streamlit as st

from utils.facts import facts, lookup, value
from utils.final import fc_index, load, need_results, text
from utils.ui import kpis, tells

need_results()
lb = load("leaderboard")
lbi = lb[lb["level"] == "item"].sort_values("rank")
b = lbi.iloc[0]
n = lbi[lbi["model"] == "seasonal_naive"].iloc[0]
tx = text()
F = facts()
idx = fc_index()
n_items, n_win = idx["item_id"].nunique(), idx["origin"].nunique()

# ---------------------------------------------------------------- hero
st.markdown(f"<div class='hero'><div class='tag'>Walmart M5 · {n_items} high-volume items · 28-day forecasts</div>"
            "<h1>Do Sudden Changes in Demand Matter? Forecasting and Stock Planning with Walmart Data</h1>"
            f"<div class='sub'>We tested {len(lbi)} forecasting methods on daily sales and asked what the best one means for how much "
            "stock to keep. Try it on a real item, or read the results.</div></div>", unsafe_allow_html=True)
b1, b2, _ = st.columns([1, 1, 3])
if b1.button("Try the Forecaster", type="primary", width="stretch", key="go_fc"):
    st.switch_page("views/forecaster.py")
if b2.button("See the results", width="stretch", key="go_lb"):
    st.switch_page("views/final_leaderboard.py")

# ---------------------------------------------------------------- headline numbers
st.markdown("#### The headline")
kpis([
    {"label": "Best model", "value": b["name"].split(" (")[0], "compare": f"first of {len(lbi)} on MAE and RMSE"},
    {"label": "MAE (units a day)", "value": f"{b['mae']:.2f}", "compare": f"seasonal naive: {n['mae']:.2f}", "tone": "good",
     "help": "Mean absolute error: the average miss per day. Typical value over items."},
    {"label": "RMSE (units a day)", "value": f"{b['rmse']:.2f}", "compare": f"seasonal naive: {n['rmse']:.2f}", "tone": "good",
     "help": "Like MAE, but big misses count more."},
    {"label": "Better than seasonal naive", "value": f"{abs(b['mae'] / n['mae'] - 1):.0%}", "compare": "lower MAE than repeating last week", "tone": "good"},
    {"label": "Items", "value": f"{n_items}", "compare": "in the final comparison"},
    {"label": "Test windows", "value": f"{n_win}", "compare": "28 days each, scored once"},
])

# ---------------------------------------------------------------- what this does
st.markdown("#### What this does")
what = [("Compare models", "21 methods on one test: accuracy, errors by category, weekday and forecast day.", "views/final_leaderboard.py", "Open the Leaderboard"),
        ("Forecast an item", "Pick an item, see the forecast, guess, then check it against what really sold.", "views/forecaster.py", "Open the Forecaster"),
        ("Plan stock", "Reorder points, safety stock, cost and fill rate, with assumed costs.", "views/final_stock.py", "Open Inventory"),
        ("See where forecasts fail", "Spikes, zero days and event days, and which gaps between models are real.", "views/diagnostics.py", "Open Diagnostics")]
st.markdown("<div class='cards'>" + "".join(f"<div class='card'><div class='t'>{html.escape(t)}</div><div class='d'>{html.escape(d)}</div></div>"
                                              for t, d, _, _ in what) + "</div>", unsafe_allow_html=True)
cols = st.columns(len(what))
for c, (_, _, page, label) in zip(cols, what):
    c.page_link(page, label=label)

# ---------------------------------------------------------------- how it works
st.markdown("#### How it works")
steps = [("Data", "Daily sales of 726 high-volume items; 150 in the final comparison."), ("Features", "Rolling averages, same-weekday history, price and calendar."),
         ("Models", "Classical, machine learning, deep learning and combinations."), ("Test once", "Tuned on three windows, scored once on six later ones."),
         ("Stock decision", "Forecast errors set the safety stock and reorder point.")]
st.markdown("<div class='steps'>" + "".join(f"<div class='step'><div class='n'>STEP {i}</div><div class='t'>{html.escape(t)}</div>"
                                             f"<div class='d'>{html.escape(d)}</div></div>" for i, (t, d) in enumerate(steps, 1)) + "</div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- the question
st.markdown("#### The question we asked")
adj = lookup(F, "breaks", "adjusted_share_with_real_shift", "CUSUM")
n_all = int(value(F, "scope", "ax_ay_items"))
qs = [("Do lasting changes in demand happen, and can we spot them?",
       f"Yes: about {adj['value']:.0%} of the {n_all} high-volume items changed their normal sales level, after removing chance alarms."),
      ("Which forecast is best?", tx["winner"].split(". ")[0] + "."),
      ("What does a better forecast mean for stock?", tx["stock"].split(". ")[0] + ".")]
st.markdown("".join(f"<div class='qa'><div class='q'>{html.escape(q)}</div><div class='a'>{html.escape(a)}</div></div>" for q, a in qs),
            unsafe_allow_html=True)
tells(f"{b['name']} leads on both error measures. One item can mislead, so the Forecaster shows single cases and the Leaderboard shows the whole test.")
