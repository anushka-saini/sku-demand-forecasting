import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, FC_MODELS, best_info, fc_actuals, fc_flags, fc_history, fc_index, fc_past_errors, fc_summary, \
    fc_window, forecaster_ready, load, names, need_results
from utils.score import LEAD, LEVEL, closer, coverage, errors, order_advice, totals, verdict, worst_days
from utils.ui import chart, ink, kpis, page, style, tells, verdict_box

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
st.info("**Test windows only; no future information used.** The forecast comes from sales up to the day before it starts. "
        "The real sales for the window are not loaded into this page until you press Reveal.")


def pick():
    """Random item and window. The same seed gives the same pick, so a demo can be repeated."""
    seed = ss.get("fc_seed")
    r = idx.iloc[np.random.default_rng(None if seed is None else int(seed)).integers(len(idx))]
    ss.update(fc_cat=r["item_id"].split("_")[0], fc_item=r["item_id"], fc_win=r["origin"])


def span(o):
    o = pd.Timestamp(o)
    return f"{o + pd.Timedelta(days=1):%d %b %Y} to {o + pd.Timedelta(days=28):%d %b %Y}"


# ---------------------------------------------------------------- left panel: the choices
left, right = st.columns([1, 3], gap="large")
with left:
    st.markdown("#### Choose")
    if ss.get("fc_cat") not in cats:
        ss["fc_cat"] = cats[0]
    cat = st.selectbox("Category", cats, format_func=str.title, key="fc_cat")
    opts = sorted(i for i in idx["item_id"].unique() if i.startswith(cat + "_"))
    if ss.get("fc_item") not in opts:
        ss["fc_item"] = opts[0]
    item = st.selectbox("Item (type to search)", opts, key="fc_item")
    if ss.get("fc_win") not in wins:
        ss["fc_win"] = wins[0]
    origin = st.selectbox("Test window", wins, format_func=span, key="fc_win")
    dflt = best_info()["model"] if best_info()["model"] in FC_MODELS else FC_MODELS[0]
    m = st.selectbox("Model", FC_MODELS, index=FC_MODELS.index(dflt), format_func=nm.get, key="fc_model",
                     help="The main models. GRU is left out because its leaderboard row is an average over three training seeds.")
    h = st.radio("Days to forecast", [7, 14, 28], index=2, horizontal=True, key="fc_h")
    st.markdown("---")
    st.number_input("Seed for surprise me (optional)", min_value=0, step=1, value=None, format="%d", key="fc_seed",
                    help="Leave empty for a different pick each time. The same number gives the same pick.")
    st.button("Surprise me", on_click=pick, width="stretch", help="A random item and window")

# a new item or window starts fresh: no old guess, nothing revealed
sel = (item, str(origin))
if ss.get("fc_sel") != sel:
    ss["fc_sel"], ss["fc_guess"] = sel, None
revealed = ss.get("fc_rev") == sel

hist = fc_history(item, origin)
fw = fc_window(item, origin, m).head(h)
start, end = fw["date"].min(), fw["date"].max()

with right:
    # ------------------------------------------------------------ step 1: the forecast
    st.markdown(f"### Step 1: Forecast")
    st.markdown(f"**{item}**, {h} days from **{start:%d %b %Y} to {end:%d %b %Y}**, model **{nm[m]}**.")

    # your own adjustment: scale the whole forecast, then optionally edit single days
    with st.expander("Adjust the forecast (optional)"):
        mult = st.slider("Scale the model forecast by", 0.5, 1.5, 1.0, 0.05, key="fc_mult",
                         help="1.00 keeps the model forecast. 1.10 adds ten percent to every day.")
        ed = pd.DataFrame({"Date": fw["date"].dt.strftime("%a %d %b"), "Model forecast": fw["forecast"].round(2).to_numpy(),
                           "Your forecast": (fw["forecast"] * mult).round(2).to_numpy()})
        ed = st.data_editor(ed, hide_index=True, width="stretch", disabled=["Date", "Model forecast"],
                            key=f"fc_edit_{item}_{origin:%Y%m%d}_{m}_{h}_{mult}")
        mine = np.clip(ed["Your forecast"].astype(float).to_numpy(), 0, None)
    edited = int((np.abs(mine - (fw["forecast"] * mult).round(2).to_numpy()) > 1e-9).sum())
    changed = mult != 1.0 or edited > 0
    if not changed:
        mine = fw["forecast"].to_numpy()          # untouched: use the model's own numbers, not the rounded ones in the table
    fig = go.Figure()
    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(128,128,128,0.08)", line_width=0)
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], name="sales so far", mode="lines", line=dict(color=ink(), width=1.8)))
    fig.add_trace(go.Scatter(x=pd.concat([fw["date"], fw["date"][::-1]]), y=pd.concat([fw["hi"], fw["lo"][::-1]]), fill="toself",
                             fillcolor="rgba(0,114,178,0.16)", line=dict(width=0), name="80% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fw["date"], y=fw["forecast"], name=f"{nm[m]} forecast", mode="lines",
                             line=dict(color=COL.get(m, "#0072B2"), width=2.4)))
    if changed:
        fig.add_trace(go.Scatter(x=fw["date"], y=mine, name="your adjusted forecast", mode="lines", line=dict(color="#E0A040", width=2.2, dash="dash")))
    if revealed:
        a = fc_actuals(item, origin, [m, "seasonal_naive", "sarima"])
        a = a[a["date"] <= end]
        act = a[a["model"] == m].sort_values("date")
        fig.add_trace(go.Scatter(x=act["date"], y=act["actual"], name="what really sold", mode="lines+markers",
                                 line=dict(color="#B0463C", width=2), marker=dict(size=5)))
    fig.update_yaxes(title="units a day", rangemode="tozero")
    fig.update_xaxes(title="date")
    chart(style(fig, height=430), f"{nm[m]}: last 90 days of sales and the next {h} days",
          "The blue band should catch about 8 days in 10." if not revealed else "Red is what really sold.", key="fc_chart")

    g1, g2 = st.columns([2, 1])
    guess = g1.number_input(f"Your own guess: total units over these {h} days (optional)", min_value=0.0, step=1.0, value=None,
                            format="%.0f", key="fc_guess")
    g2.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    if g2.button("Reveal actuals", type="primary", width="stretch"):
        ss["fc_rev"] = sel
        st.rerun()

    verdict_box(["A forecast is not exactly right on any day, so it is judged three ways, and none of them alone says right or wrong.",
                 "(a) The error in units: how far the forecast was from what sold. "
                 "(b) The band: whether the real sales stayed inside the 80% band. "
                 "(c) The baseline: whether it did better than simply repeating last week (seasonal naive)."],
                head="What does a correct forecast mean?")

    # ------------------------------------------------------------ step 2: the reveal
    st.markdown("### Step 2: Reveal")
    table = fw.rename(columns={"date": "Date", "forecast": "Forecast", "lo": "Band low", "hi": "Band high"})
    table[["Forecast", "Band low", "Band high"]] = table[["Forecast", "Band low", "Band high"]].round(2)
    if not revealed:
        st.caption("Press Reveal actuals to see how the model did on this item and window.")
    else:
        fa = act["forecast"].to_numpy()
        aa = act["actual"].to_numpy()
        s = errors(fa, aa)
        ftot, atot, diff = totals(fa, aa)
        ins = coverage(aa, fw["lo"].to_numpy(), fw["hi"].to_numpy())
        kpis([
            {"label": "MAE (units a day)", "value": f"{s['mae']:.2f}", "help": "Average size of the daily miss."},
            {"label": "RMSE (units a day)", "value": f"{s['rmse']:.2f}", "help": "Like MAE but big misses count more."},
            {"label": "Bias (units a day)", "value": f"{s['bias']:+.2f}", "compare": "over-forecast" if s["bias"] > 0 else "under-forecast",
             "help": "Average of forecast minus actual. Positive means the model forecast too much."},
            {"label": f"{h}-day total", "value": f"{ftot:.0f} vs {atot:.0f}",
             "compare": "forecast vs actual" + ("" if np.isnan(diff) else f", {diff:+.1%}")},
            {"label": "Days inside the 80% band", "value": f"{int(ins.sum())} of {len(ins)}", "compare": f"{ins.mean():.0%} coverage"},
        ])

        rows, notes = [], []
        for base, label in [("seasonal_naive", "seasonal naive"), ("sarima", "SARIMA")]:
            if m == base:
                continue
            x = a[a["model"] == base].sort_values("date")
            b = errors(x["forecast"].to_numpy(), x["actual"].to_numpy())
            v = verdict(s["mae"], b["mae"])
            notes.append(f"{v.capitalize()} {label}: MAE {s['mae']:.2f} vs {b['mae']:.2f}, RMSE {s['rmse']:.2f} vs {b['rmse']:.2f}.")
            rows.append({"Compared with": nm[base], "Their MAE": round(b["mae"], 2), "Their RMSE": round(b["rmse"], 2),
                         f"{nm[m]} MAE": round(s["mae"], 2), f"{nm[m]} RMSE": round(s["rmse"], 2), "Verdict": f"{v.capitalize()} {label}"})
        if rows:
            st.markdown("**Against the baselines, same item and window** (judged on MAE)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            notes.append("This model is the seasonal naive baseline, so there is nothing to compare it with here.")

        if guess is None:
            st.markdown("**You vs the model:** you did not enter a guess.")
        else:
            who = closer(float(guess), ftot, atot)
            st.markdown(f"**You vs the model:** your guess {guess:.0f}, the model {ftot:.0f}, actual {atot:.0f}. "
                        + {"you": "You were closer.", "the model": "The model was closer.", "tie": "A tie."}[who])

        # your adjusted forecast against the model, on the same days
        ym = errors(mine, aa)
        if changed:
            res = "better" if ym["mae"] < s["mae"] else "worse" if ym["mae"] > s["mae"] else "the same"
            st.markdown(f"**Your adjusted forecast vs the model:** MAE {ym['mae']:.2f} against {s['mae']:.2f} for the model, so yours was **{res}**.")
        else:
            res = "kept the model forecast"
            st.markdown("**Your adjusted forecast vs the model:** you kept the model forecast.")

        # one line in the session log for each different set of choices
        gk = None if guess is None else float(guess)
        key = (item, str(origin), m, h, round(float(mult), 2), tuple(np.round(mine, 2)), gk)
        done = ss.setdefault("fc_logged", set())
        if key not in done:
            done.add(key)
            ss.setdefault("fc_log", []).append({
                "Item": item, "Window": span(origin), "Model": nm[m], "Days": h, "Scale": round(float(mult), 2), "Days edited": edited,
                "Model MAE": round(s["mae"], 2), "Your MAE": round(ym["mae"], 2), "Your forecast was": res,
                "Guess": gk, "Model total": round(ftot), "Actual total": round(atot),
                "Closer": "no guess" if gk is None else closer(gk, ftot, atot)})

        # the three worst days, and what kind of day they were
        fl = fc_flags(item, origin).set_index("date")
        worst = worst_days(fa, aa)
        tags = []
        for i, d in enumerate(act["date"]):
            t = []
            if i in worst:
                if fl["event"].get(d, 0) == 1:
                    t.append("event")
                if fl["snap"].get(d, 0) == 1:
                    t.append("SNAP day")
                if aa[i] == 0:
                    t.append("zero sales")
            tags.append(", ".join(t))
        table = pd.DataFrame({"Date": act["date"].dt.strftime("%a %d %b").to_numpy(), "Forecast": np.round(fa, 1), "Actual": aa.astype(int),
                              "Error (forecast - actual)": np.round(fa - aa, 1), "Inside the band": np.where(ins, "yes", "no"),
                              "Worst-day note": tags})
        st.markdown("**Day by day** (the three biggest misses are shaded; the note says if that day was an event, a SNAP day or had no sales)")
        st.dataframe(table.style.apply(lambda r: ["background-color: rgba(176,70,60,0.28)" if r.name in worst else "" for _ in r], axis=1),
                     hide_index=True, width="stretch")
        verdict_box([f"Error: on average {s['mae']:.2f} units a day off (RMSE {s['rmse']:.2f}); the forecast ran "
                     f"{'over' if s['bias'] > 0 else 'under'} by {abs(s['bias']):.2f} a day, and the {h}-day total was "
                     + ("not given as a percent, because nothing sold." if np.isnan(diff)
                        else f"{abs(diff):.0%} {'above' if diff > 0 else 'below'} what sold."),
                     f"Band: {int(ins.sum())} of {len(ins)} days ({ins.mean():.0%}) stayed inside the 80% band. A well-set 80% band "
                     f"catches about 8 days in 10 over many windows, so one window can be higher or lower.",
                     "Baseline: " + " ".join(notes)], head="Verdict for this item and window")

    # ------------------------------------------------------------ step 3: what to order
    st.markdown("### Step 3: Order advice")
    st.caption(f"A simulation with assumed numbers: a {LEAD}-day lead time and a {LEVEL:.0%} target. Costs are the assumptions on the "
               "Inventory page, not Walmart's figures. It uses only this item's earlier windows, never the one being forecast.")
    adv = order_advice(fw["forecast"].to_numpy(), fc_past_errors(item, origin, m))
    o1, o2 = st.columns([1, 3])
    hand = o1.number_input("Stock on hand (units, assumed)", min_value=0.0, step=1.0, value=0.0, key="fc_hand")
    with o2:
        kpis([{"label": f"Forecast demand, next {LEAD} days", "value": f"{adv['need']:.0f}"},
              {"label": "Safety stock", "value": f"{adv['safety']:.0f}", "help": f"The {LEVEL:.0%} quantile of this item's past {LEAD}-day "
                                                                                 f"shortfalls ({adv['n']} cases)."},
              {"label": "Order-up-to level", "value": f"{adv['level']:.0f}", "help": "Forecast demand over the lead time plus safety stock."},
              {"label": "Suggested order", "value": f"{max(0.0, adv['level'] - hand):.0f}", "compare": "level minus stock on hand"}])

    # ------------------------------------------------------------ download and the session log
    if changed:
        table["Your forecast"] = np.round(mine, 2)
    if revealed:
        csv = table.drop(columns=["Worst-day note"]).to_csv(index=False)
    else:
        csv = table.to_csv(index=False)
    st.download_button("Download this forecast as CSV", csv.encode("utf-8"), file_name=f"forecast_{item}_{origin:%Y%m%d}_{m}.csv",
                       mime="text/csv", key="fc_dl")
    log = ss.get("fc_log", [])
    with st.expander(f"Session log ({len(log)} reveals)"):
        if not log:
            st.caption("Each reveal adds a line here: what you changed and how it did against the model.")
        else:
            ld = pd.DataFrame(log)
            st.dataframe(ld, hide_index=True, width="stretch")
            st.download_button("Download the log as CSV", ld.to_csv(index=False).encode("utf-8"), file_name="forecaster_session_log.csv",
                               mime="text/csv", key="fc_log_dl")
            if st.button("Clear the log", key="fc_log_clear"):
                ss["fc_log"] = []                  # the set of logged choices stays, so the current reveal is not added again
                st.rerun()

# ---------------------------------------------------------------- track record over the whole test set
st.divider()
st.subheader("Track record")
r = summ.loc[(m, h)]
st.caption(f"All 150 items and 6 test windows ({int(r['item_windows'])} item-windows), first {h} days of each, so one lucky or unlucky "
           "item does not mislead. Each item-window counts once.")
cards = [{"label": "Beat seasonal naive", "value": "baseline" if m == "seasonal_naive" else f"{r['beat_seasonal_naive']:.0%}",
          "compare": "of item-windows, on MAE", "tone": None if m == "seasonal_naive" else "good" if r["beat_seasonal_naive"] > 0.5 else "bad"},
         {"label": "Median MAE", "value": f"{r['median_mae']:.2f}", "compare": "units a day, per item-window"},
         {"label": "Median RMSE", "value": f"{r['median_rmse']:.2f}", "compare": "units a day, per item-window"},
         {"label": "Band coverage", "value": f"{r['coverage']:.0%}", "compare": "of days inside the 80% band"}]
if m != "sarima":
    cards.insert(1, {"label": "Beat SARIMA", "value": f"{r['beat_sarima']:.0%}", "compare": "of item-windows, on MAE"})
kpis(cards)
if h == 28:
    st.caption(f"The Leaderboard takes each item's mean over its six windows and then the median over items: MAE {r['lb_mae']:.2f}, "
               f"RMSE {r['lb_rmse']:.2f} for {nm[m]}. That is why it differs a little from the item-window medians above.")
st.caption("The XGBoost band is the GARCH band from the Diagnostics page; for the other models it is the 80th percentile of that "
           "item's own errors in the validation windows.")
beat = "is the baseline the others are measured against" if m == "seasonal_naive" else \
    f"did better than seasonal naive in {r['beat_seasonal_naive']:.0%} of item-windows"
tells(f"One item and window is a small sample. Over all {int(r['item_windows'])} item-windows {nm[m]} {beat}, with a median daily miss of "
      f"{r['median_mae']:.2f} units, and its 80% band held {r['coverage']:.0%} of days. Judge a model on that, not on a single pick.")
