import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup, value
from utils.final import COL, MAIN, best_info, load, names, need_results, text
from utils.ui import chart, kpis, page, style, tells, verdict_box

need_results()
lb = load("leaderboard")
lbi = lb[lb["level"] == "item"].sort_values("rank")
nm = names(lb)
info = best_info()
tx = text()
dm = load("dm")
stock = load("stock").set_index(["method", "model"])
b = lbi.iloc[0]
n = lbi[lbi["model"] == "seasonal_naive"].iloc[0]
prev = lbi[lbi["model"] == "combination"].iloc[0]

page("Summary", "Which forecasting model should drive stock planning for these items, and how sure are we?")
st.caption("150 high-volume California items, daily sales, 28 days ahead. Models were tuned on three validation windows and scored once on "
           "six later test windows. Numbers below are typical (median) errors per item, in units a day.")

d_naive = dm[(dm["model_b"] == "seasonal_naive") & (dm["loss"] == "absolute error")].iloc[0]
d_prev = dm[(dm["model_b"] == "combination") & (dm["loss"] == "absolute error")]
kpis([
    {"label": "Best model", "value": b["name"].split(" (")[0], "compare": f"first of {len(lbi)} models tested on MAE and RMSE"},
    {"label": "Typical error (MAE)", "value": f"{b['mae']:.2f}", "compare": f"{abs(b['mae'] / n['mae'] - 1):.0%} lower than repeating last week ({n['mae']:.2f})", "tone": "good"},
    {"label": "Typical error (RMSE)", "value": f"{b['rmse']:.2f}", "compare": f"{abs(b['rmse'] / n['rmse'] - 1):.0%} lower than repeating last week ({n['rmse']:.2f})", "tone": "good"},
    {"label": "Beats the simple baseline?", "value": "Yes" if d_naive["significant_5pct"] else "Not clearly",
     "compare": "Diebold-Mariano p " + ("< 0.001" if d_naive["p_value"] < 0.001 else f"= {d_naive['p_value']:.3f}")},
    {"label": "Beats the earlier best (combination)?", "value": "Yes" if (len(d_prev) and d_prev["significant_5pct"].any()) else "Not significantly",
     "compare": f"MAE {b['mae']:.2f} vs {prev['mae']:.2f}; p = {d_prev['p_value'].iloc[0]:.2f}" if len(d_prev) else ""},
])

d = lbi[lbi["model"].isin(MAIN)].sort_values("mae")
fig = go.Figure(go.Bar(y=[nm[m].split(" (")[0] for m in d["model"]], x=d["mae"], orientation="h",
                       marker_color=[COL.get(m, "#888") for m in d["model"]], text=[f"{v:.2f}" for v in d["mae"]], textposition="outside",
                       error_x=dict(type="data", array=d["mae_std"].fillna(0), thickness=1.2)))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="typical MAE per item, units a day (lower is better)", range=[0, d["mae"].max() * 1.15])
chart(style(fig, height=380, legend=False), "Eight models, one test: the tree models lead, the classical models are close behind",
      "Main set only; the Final leaderboard page has all models, MAPE, WAPE and fit times.", key="sum_bar")

st.markdown("#### The three findings")
verdict_box([tx["winner"], tx["combos"].split(". ")[0] + ".", tx["dm"].split("; against SARIMA")[0] + "."], head="What the test shows")

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Where it goes wrong")
    for t in tx["failure"][2:5]:
        st.markdown("- " + t)
with c2:
    st.markdown("#### What it means for stock")
    st.markdown(tx["stock"].split(" With the textbook")[0])
    st.markdown("With the textbook safety-stock formula the same model costs more, not less: " + tx["stock"].split("With the textbook safety stock formula, ")[1][:260].split(" The 95%")[0].rstrip(".") + ".")
    st.markdown(tx["garch"].split(". As intervals")[0] + ".")


st.markdown("#### Questions and answers")
F = facts()
adj = lookup(F, "breaks", "adjusted_share_with_real_shift", "CUSUM")
n_all = int(value(F, "scope", "ax_ay_items"))
al = get("event_alignment")
p4 = al[(al["event"] == "sustained price change") & (al["window_wks"] == 4)].iloc[0]
hol = al[al["event"] != "sustained price change"].iloc[0]
va = get("variants")
va = va[va["group"] == "all detected breaks"]
sh = get("shift_aware_tests")
helps = bool(va["helps"].astype(str).str.lower().eq("true").any()) or bool(
    sh.loc[sh["group"] == "all break item-origins", "helps"].astype(str).str.lower().eq("true").any())
n_versions = len(va) + len(sh[sh["group"] == "all break item-origins"])
cal = get("calibrated_inv")
c95 = cal[cal["service_level"] == 0.95].set_index(["method", "policy"])
fr = cal[(cal["service_level"] == 0.95) & (cal["policy"] == "best_static")].set_index("method")["fill_rate"]
conf_vs = c95.loc[("conformal", "best_static"), "total_cost"] / c95.loc[("conformal", "seasonal_naive_static"), "total_cost"] - 1
qas = [
    ("Which model forecasts best?", f"{b['name']}: MAE {b['mae']:.2f}, RMSE {b['rmse']:.2f} units a day, {abs(b['mae'] / n['mae'] - 1):.0%} below repeating last week. "
     "It is not significantly better than the best classical models, so the choice among the top few is about speed and simplicity as much as accuracy."),
    ("Does a combination of models win?", tx["combos"].split(". ")[0] + "."),
    ("Do demand shifts happen?", f"Yes. After removing chance alarms, about {adj['value']:.0%} of the {n_all} items changed their normal sales level at some point "
     "(earlier study, time-series models)."),
    ("Can price changes or holidays explain the shifts?", f"Mostly not. {p4['aligned_pct']:.1f}% of shifts sit within four weeks of a lasting price change, against {p4['chance_pct']:.1f}% by chance; "
     f"holidays line up no more than chance ({hol['aligned_pct']:.1f}% against {hol['chance_pct']:.1f}%)."),
    ("Should we retrain a model after a shift is detected?", f"{'At least one version helped.' if helps else 'No.'} "
     f"{'' if helps else f'None of the {n_versions} shift-aware versions we tried was more accurate than leaving the model alone.'} (earlier study)"),
    ("Does a smaller error cut cost?", f"Only with the right safety stock. Sized from the model's own past errors, the earlier best model costs {abs(conf_vs):.0%} "
     f"{'less' if conf_vs < 0 else 'more'} than the baseline and the fill rate is {fr['conformal']:.1%} against {fr['normal formula']:.1%} with the textbook formula. "
     "For XGBoost the saving is within the noise of the replay (see Stock impact, best model)."),
    ("Does GARCH or VAR help?", "No. GARCH does not change the forecast and did not give better prediction intervals; VAR on the category totals was not better than the item models."),
]
for q, a in qas:
    with st.expander(q):
        st.markdown(a)

st.markdown("#### How far to trust this")
for t in tx["limitations"][:4]:
    st.markdown("- " + t)
st.markdown("Pages: **Final leaderboard** (all models and tests), **Forecast viewer** (pick an item), **Volatility (GARCH)**, "
            "**Stock impact, best model**. The *Earlier study* section holds the data exploration, shift detection and the first stock work.")
tells(f"{b['name']} is the model to use, mainly because it is accurate and quick to refit ({b['fit_seconds']:.0f} seconds for six windows), "
      "not because it is clearly better than the best classical models: that gap is within the noise of six test windows.")
