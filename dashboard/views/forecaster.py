import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, FC_MODELS, best_info, fc_actuals, fc_history, fc_index, fc_summary, fc_window, forecaster_ready, \
    load, names, need_results
from utils.ui import chart, kpis, page, style, tells, verdict_box

need_results()
if not forecaster_ready():
    st.info("The Forecaster reads a few small files made by scripts/export_forecaster.py. Run it once, then reload this page.")
    st.stop()

lb = load("leaderboard")
nm = names(lb)
idx = fc_index()
summ = fc_summary()
cats = sorted({i.split("_")[0] for i in idx["item_id"]})
wins = sorted(idx["origin"].unique())
ss = st.session_state

page("Forecaster", "Pick an item and a test window, see what the model expects, make your own guess, then reveal what really sold.")
st.info("**Test windows only; no future information used.** Every window here is one of the six test windows (Dec 2015 to May 2016). "
        "The forecast was made from sales up to the day before it starts, and the real sales for the window are not loaded "
        "into this page until you press Reveal.")


def pick():
    """Random item and window. The same seed gives the same pick, so a demo can be repeated."""
    seed = ss.get("fc_seed")
    r = idx.iloc[np.random.default_rng(None if seed is None else int(seed)).integers(len(idx))]
    ss.update(fc_cat=r["item_id"].split("_")[0], fc_item=r["item_id"], fc_win=r["origin"])


def span(o):
    o = pd.Timestamp(o)
    return f"{o + pd.Timedelta(days=1):%d %b %Y} to {o + pd.Timedelta(days=28):%d %b %Y}"


# ---------------------------------------------------------------- controls
dflt = best_info()["model"] if best_info()["model"] in FC_MODELS else FC_MODELS[0]
c1, c2, c3 = st.columns([2, 1, 1])
m = c1.selectbox("Model", FC_MODELS, index=FC_MODELS.index(dflt), format_func=nm.get, key="fc_model",
                 help="The main models. GRU is left out here because its leaderboard row is an average over three training seeds.")
c2.number_input("Seed (optional)", min_value=0, step=1, value=None, format="%d", key="fc_seed",
                help="Leave empty for a different pick each time. The same number gives the same pick.")
c3.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
c3.button("Pick a random item and window", on_click=pick, width="stretch")

d1, d2, d3 = st.columns(3)
if ss.get("fc_cat") not in cats:
    ss["fc_cat"] = cats[0]
cat = d1.selectbox("Category", cats, key="fc_cat")
opts = sorted(i for i in idx["item_id"].unique() if i.startswith(cat + "_"))
if ss.get("fc_item") not in opts:
    ss["fc_item"] = opts[0]
item = d2.selectbox("Item", opts, key="fc_item")
if ss.get("fc_win") not in wins:
    ss["fc_win"] = wins[0]
origin = d3.selectbox("Test window", wins, format_func=span, key="fc_win")

# a new item or window starts fresh: no old guess, nothing revealed
sel = (item, str(origin))
if ss.get("fc_sel") != sel:
    ss["fc_sel"], ss["fc_guess"] = sel, None
revealed = ss.get("fc_rev") == sel

# ---------------------------------------------------------------- before the reveal
hist = fc_history(item, origin)
fw = fc_window(item, origin, m)
start, end = fw["date"].min(), fw["date"].max()
st.markdown(f"**{item}**, forecasting **{start:%d %b %Y} to {end:%d %b %Y}** (28 days). Model: **{nm[m]}**.")

fig = go.Figure()
fig.add_vrect(x0=start, x1=end, fillcolor="rgba(128,128,128,0.08)", line_width=0)
fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], name="sales so far", mode="lines", line=dict(color="#222222", width=1.8)))
fig.add_trace(go.Scatter(x=pd.concat([fw["date"], fw["date"][::-1]]), y=pd.concat([fw["hi"], fw["lo"][::-1]]), fill="toself",
                         fillcolor="rgba(0,114,178,0.16)", line=dict(width=0), name="80% band", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fw["date"], y=fw["forecast"], name=f"{nm[m]} forecast", mode="lines",
                         line=dict(color=COL.get(m, "#0072B2"), width=2.4)))
if revealed:
    a = fc_actuals(item, origin, [m, "seasonal_naive", "sarima"])
    act = a[a["model"] == m].sort_values("date")
    fig.add_trace(go.Scatter(x=act["date"], y=act["actual"], name="what really sold", mode="lines+markers",
                             line=dict(color="#B0463C", width=2), marker=dict(size=5)))
fig.update_yaxes(title="units a day", rangemode="tozero")
chart(style(fig, height=430), f"{nm[m]}: last 90 days of sales and the next 28 days",
      "The shaded area is the 28 days being forecast. The blue band should catch about 8 days in 10." if not revealed else
      "Red is what really sold over the 28 days.", key="fc_chart")

# ---------------------------------------------------------------- the guess and the reveal
g1, g2 = st.columns([2, 1])
guess = g1.number_input("Make your own guess: total units for these 28 days", min_value=0.0, step=1.0, value=None, format="%.0f",
                        key="fc_guess", help="Look at the history, then type your guess for the 28-day total.")
g2.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
if g2.button("Reveal actuals", type="primary", width="stretch"):
    ss["fc_rev"] = sel
    st.rerun()

verdict_box(["A forecast is not exactly right on any day, so it is judged three ways, and none of them alone says right or wrong.",
             "(a) The error in units: how far the forecast was from what sold. "
             "(b) The band: whether the real sales stayed inside the 80% band. "
             "(c) The baseline: whether it did better than simply repeating last week (seasonal naive)."],
            head="What does a correct forecast mean?")

if not revealed:
    st.caption("Press Reveal actuals to see how the model did on this item and window.")
else:
    # ------------------------------------------------------------ after the reveal
    def score(model):
        g = a[a["model"] == model].sort_values("date")
        e = g["forecast"].to_numpy() - g["actual"].to_numpy()
        return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}

    s = score(m)
    ftot, atot = float(act["forecast"].sum()), float(act["actual"].sum())
    diff = (ftot - atot) / atot if atot > 0 else np.nan
    inside = (act["actual"].to_numpy() >= fw["lo"].to_numpy()) & (act["actual"].to_numpy() <= fw["hi"].to_numpy())
    cover = float(inside.mean())
    kpis([
        {"label": "MAE (units a day)", "value": f"{s['MAE']:.2f}", "help": "Average size of the daily miss."},
        {"label": "RMSE (units a day)", "value": f"{s['RMSE']:.2f}", "help": "Like MAE but big misses count more."},
        {"label": "Bias (units a day)", "value": f"{s['bias']:+.2f}", "compare": "over-forecast" if s["bias"] > 0 else "under-forecast",
         "help": "Average of forecast minus actual. Positive means the model forecast too much."},
        {"label": "28-day total", "value": f"{ftot:.0f} vs {atot:.0f}",
         "compare": "forecast vs actual" + ("" if np.isnan(diff) else f", {diff:+.1%}")},
        {"label": "Days inside the 80% band", "value": f"{int(inside.sum())} of {len(inside)}", "compare": f"{cover:.0%} coverage"},
    ])

    # against the two baselines on the same item and window
    rows, notes = [], []
    for base, label in [("seasonal_naive", "seasonal naive"), ("sarima", "SARIMA")]:
        if m == base:
            continue
        b = score(base)
        won = s["MAE"] < b["MAE"]
        notes.append(f"{'Beat' if won else 'Did not beat'} {label}" + (" (equal error)" if s["MAE"] == b["MAE"] else "")
                     + f": MAE {s['MAE']:.2f} vs {b['MAE']:.2f}, RMSE {s['RMSE']:.2f} vs {b['RMSE']:.2f}.")
        rows.append({"Compared with": nm[base], "Their MAE": round(b["MAE"], 2), "Their RMSE": round(b["RMSE"], 2),
                     f"{nm[m]} MAE": round(s["MAE"], 2), f"{nm[m]} RMSE": round(s["RMSE"], 2),
                     "Result": ("Beat " if won else "Did not beat ") + label})
    if rows:
        st.markdown("**Against the baselines, same item and window** (judged on MAE)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        notes.append("This model is the seasonal naive baseline, so there is nothing to compare it with here.")

    # you against the model
    if guess is None:
        mine = "You did not enter a guess."
    else:
        dm_, dg = abs(ftot - atot), abs(float(guess) - atot)
        who = "You were closer." if dg < dm_ else ("The model was closer." if dm_ < dg else "You tied with the model.")
        mine = f"Your guess {guess:.0f}, the model {ftot:.0f}, actual {atot:.0f}. {who}"
    st.markdown(f"**You vs the model:** {mine}")

    # day by day, worst three highlighted
    e = act["forecast"].to_numpy() - act["actual"].to_numpy()
    tab = pd.DataFrame({"Date": act["date"].dt.strftime("%a %d %b").to_numpy(), "Forecast": act["forecast"].round(1).to_numpy(),
                        "Actual": act["actual"].astype(int).to_numpy(), "Error (forecast - actual)": np.round(e, 1),
                        "Inside the band": np.where(inside, "yes", "no")})
    worst = set(np.argsort(-np.abs(e), kind="stable")[:3])
    st.markdown("**Day by day** (the three biggest misses are shaded)")
    st.dataframe(tab.style.apply(lambda r: ["background-color: rgba(176,70,60,0.28)" if r.name in worst else "" for _ in r], axis=1),
                 hide_index=True, width="stretch")

    verdict_box([f"Error: on average {s['MAE']:.2f} units a day off (RMSE {s['RMSE']:.2f}); the forecast ran "
                 f"{'over' if s['bias'] > 0 else 'under'} by {abs(s['bias']):.2f} a day, and the 28-day total was "
                 + ("not given as a percent, because nothing sold." if np.isnan(diff) else f"{abs(diff):.0%} {'above' if diff > 0 else 'below'} what sold."),
                 f"Band: {int(inside.sum())} of {len(inside)} days ({cover:.0%}) stayed inside the 80% band. A well-set 80% band "
                 f"catches about 8 days in 10 over many windows, so one window can be higher or lower.",
                 "Baseline: " + " ".join(notes)], head="Verdict for this item and window")

# ---------------------------------------------------------------- the whole test set
st.divider()
st.subheader("Whole test set")
r = summ.loc[m]
st.caption(f"All 150 items and 6 test windows ({int(r['item_windows'])} item-windows), so one unlucky item does not mislead. "
           "Each item-window counts once.")
cards = [{"label": "Beat seasonal naive", "value": "baseline" if m == "seasonal_naive" else f"{r['beat_seasonal_naive']:.0%}",
          "compare": "of item-windows, on MAE", "tone": None if m == "seasonal_naive" else "good" if r["beat_seasonal_naive"] > 0.5 else "bad"},
         {"label": "Median MAE", "value": f"{r['median_mae']:.2f}", "compare": "units a day, per item-window"},
         {"label": "Median RMSE", "value": f"{r['median_rmse']:.2f}", "compare": "units a day, per item-window"},
         {"label": "Band coverage", "value": f"{r['coverage']:.0%}", "compare": "of days inside the 80% band"}]
if m != "sarima":
    cards.insert(1, {"label": "Beat SARIMA", "value": f"{r['beat_sarima']:.0%}", "compare": "of item-windows, on MAE"})
kpis(cards)
st.caption(f"The leaderboard takes each item's mean over its six windows and then the median over items: MAE {r['lb_mae']:.2f}, "
           f"RMSE {r['lb_rmse']:.2f} for {nm[m]}. That is why it differs a little from the item-window medians above. "
           "The band for XGBoost is the GARCH band from the Volatility page; for the other models it is the 80th percentile of that "
           "item's own errors in the validation windows.")

beat = "is the baseline the others are measured against" if m == "seasonal_naive" else     f"did better than seasonal naive in {r['beat_seasonal_naive']:.0%} of item-windows"
tells(f"One item and window is a small sample. Over all {int(r['item_windows'])} item-windows {nm[m]} {beat}, with a median daily miss of "
      f"{r['median_mae']:.2f} units, and its 80% band held {r['coverage']:.0%} of days. Judge a model on that, not on a single pick.")
