import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup
from utils.models import BEST, BEST_SHORT, COLORS
from utils.ui import ACCENT, DARK, NEUTRAL, apply_item_filters, chart, download, filters, filters_used, kpis, page, style, tells

f = facts()
flt = filters()
items = get("items")
hc = get("hc_breaks_items")
inj = get("injection")
variants = get("variants")
adaptive = get("adaptive_tests")
sarima = get("shift_aware_tests")
null = get("null_rates")
agree = get("agreement").iloc[0]
align = get("event_alignment")

page("Demand shifts", "How often does an item's demand change for good, and does reacting to it make forecasts better?")
filters_used(["category", "segment"])

scope = apply_item_filters(items, {**flt, "shift": "All items"})
if scope.empty:
    st.info("No items match the category and item-group filters.")
    st.stop()
flagged = scope[scope["cusum_first_break"]]
b = hc[hc["cusum_break"] & hc["item_id"].isin(set(scope["item_id"]))].copy()
b["delay_wks"] = (b["alarm_week"] - b["change_week"]).dt.days / 7
adj = lookup(f, "breaks", "adjusted_share_with_real_shift", "CUSUM")
far = lookup(f, "breaks", "seasonal_null_false_alarm_rate", "CUSUM")["value"]
kpis([
    {"label": "Items flagged with a shift", "value": f"{len(flagged) / len(scope):.0%}", "compare": f"{len(flagged):,} of {len(scope):,} items",
     "help": "The detector compares each week with the item's first year and raises an alarm when sales stay away from it."},
    {"label": "With a real shift", "value": f"{adj['value']:.0%}", "compare": f"all items, after removing {far:.0%} chance alarms",
     "help": "On made-up series with no shift the detector still alarms on some, so we subtract that rate."},
    {"label": "Clear, lasting shifts", "value": f"{int(b['high_confidence'].sum()):,}",
     "compare": f"{b['high_confidence'].mean():.0%} of the flagged shifts",
     "help": "Confirmed by a second method, at least 20% in size, and still half there a year later."},
    {"label": "Time to raise the alarm", "value": f"{b['delay_wks'].median():.0f} weeks", "compare": "typical, after the change",
     "help": "Weeks between the estimated change and the week the detector raised its alarm."},
    {"label": "Shifts that are drops", "value": f"{(b['direction'] < 0).mean():.0%}", "compare": "the rest are rises"},
])

c1, c2 = st.columns(2)
with c1:
    yr = flagged["first_break_week"].dt.year.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=yr.index.astype(str), y=yr.to_numpy(), marker_color=ACCENT,
                           hovertemplate="%{x}: %{y} items<extra></extra>"))
    style(fig, height=300, legend=False).update_yaxes(title="items with their first shift that year")
    fig.update_xaxes(title="year of the first shift")
    early = yr[yr.index <= 2013].sum() / yr.sum()
    chart(fig, f"{early:.0%} of first shifts fall in 2012-2013",
          "Watching only starts a year after an item launches, so the earliest years collect the first shifts of many items.")
with c2:
    rc = (100 * b["relative_change"]).clip(-100, 150)
    fig = go.Figure(go.Histogram(x=rc, nbinsx=40, marker_color=ACCENT, hovertemplate="%{x:.0f}%: %{y} shifts<extra></extra>"))
    fig.add_vline(x=0, line_color=NEUTRAL)
    style(fig, height=300, legend=False).update_xaxes(title="change in weekly sales level (%), shown from -100% to +150%")
    fig.update_yaxes(title="number of shifts")
    chart(fig, f"A typical shift moves the sales level by {b['relative_change'].abs().median():.0%}",
          f"Measured as the 26 weeks after the change against the 26 weeks before, over {len(b):,} flagged shifts.")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Histogram(x=b["delay_wks"], nbinsx=40, marker_color=ACCENT, hovertemplate="%{x:.0f} weeks: %{y} shifts<extra></extra>"))
    style(fig, height=300, legend=False).update_xaxes(title="weeks from the change to the alarm")
    fig.update_yaxes(title="number of shifts")
    chart(fig, f"The alarm comes a typical {b['delay_wks'].median():.0f} weeks after the change",
          f"{(b['delay_wks'] <= 13).mean():.0%} of alarms come within 13 weeks; the slowest takes {b['delay_wks'].max():.0f} weeks.")
with c2:
    ctrl = inj[inj["shape"] == "none"]["cusum_power"].iloc[0]
    fig = go.Figure()
    for shape, color, dash, lab in [("step", ACCENT, "solid", "sudden step"), ("drift", "#C47F1A", "dash", "slow drift over 26 weeks")]:
        d = inj[inj["shape"] == shape].sort_values("size")
        fig.add_scatter(x=100 * d["size"], y=d["cusum_power"], mode="lines+markers", name=lab, line=dict(color=color, dash=dash),
                        hovertemplate="%{x:.0f}% shift: caught %{y:.0%}<extra>" + lab + "</extra>")
    fig.add_hline(y=ctrl, line_dash="dot", line_color=NEUTRAL, annotation_text=f"no shift: {ctrl:.0%}", annotation_position="top left")
    style(fig, height=300).update_yaxes(tickformat=".0%", range=[0, 1], title="share of planted shifts caught")
    fig.update_xaxes(title="size of the planted shift (% of normal sales)")
    s20 = inj[(inj["shape"] == "step") & (np.isclose(inj["size"], 0.20))]["cusum_power"].iloc[0]
    chart(fig, f"Small shifts go unseen: a sudden 20% shift is caught {s20:.0%} of the time",
          "We planted shifts of known size in series with none and counted how often the detector found them.")

# did reacting to a shift help?
rows = []
for v, lab in [("truncate", f"{BEST_SHORT}, retrained after a shift"), ("blend", f"{BEST_SHORT}, blend of static and retrained")]:
    r = variants[(variants["group"] == "all detected breaks") & (variants["variant"] == v)].iloc[0]
    rows.append({"version": lab, "model": BEST, "items": int(r["n"]), "change": r["median_diff"], "low": r["ci_low"],
                 "high": r["ci_high"], "better_for": r["share_improved"], "meets_rule": str(r["helps"]).lower() == "true"})
for m, lab in [("ets", "ETS, retrained after a shift"), ("tsb", "TSB, retrained after a shift")]:
    r = adaptive[(adaptive["model"] == m) & adaptive["comparison"].str.startswith("adaptive") & (adaptive["category"] == "ALL")].iloc[0]
    rows.append({"version": lab, "model": m, "items": int(r["n"]), "change": r["median_diff"], "low": r["ci_low"], "high": r["ci_high"],
                 "better_for": r["share_improved"], "meets_rule": False})
for v, lab in [("restart", "SARIMA, retrained after a shift"), ("dummy", "SARIMA with a shift marker")]:
    for grp, suffix in [("all break item-origins", ""), ("only where the variant differs", " (only where it differs)")]:
        r = sarima[(sarima["group"] == grp) & (sarima["variant"] == v)].iloc[0]
        rows.append({"version": lab + suffix, "model": "sarima", "items": int(r["n"]), "change": r["median_diff"], "low": r["ci_low"],
                     "high": r["ci_high"], "better_for": r["share_improved"], "meets_rule": str(r["helps"]).lower() == "true"})
pl = adaptive[(adaptive["model"] == BEST) & adaptive["comparison"].str.startswith("placebo") & (adaptive["category"] == "ALL")].iloc[0]
rows.append({"version": f"{BEST_SHORT}, random short window (placebo)", "model": BEST, "items": int(pl["n"]), "change": pl["median_diff"],
             "low": pl["ci_low"], "high": pl["ci_high"], "better_for": pl["share_improved"], "meets_rule": False})
rt = pd.DataFrame(rows)

mu = get("mase_unrounded")
PRACTICAL = round(0.01 * mu[(mu["model"] == BEST) & (mu["category"] == "ALL")]["median_item_mase"].iloc[0], 4)    # 1% of its error
fig = go.Figure()
fig.add_vrect(x0=-PRACTICAL, x1=PRACTICAL, fillcolor=NEUTRAL, opacity=0.10, line_width=0)
fig.add_scatter(x=rt["change"], y=rt["version"], mode="markers", marker=dict(color=[COLORS[m] for m in rt["model"]], size=11),
                error_x=dict(type="data", symmetric=False, array=rt["high"] - rt["change"], arrayminus=rt["change"] - rt["low"],
                             color="rgba(128,128,128,0.7)"),
                customdata=rt[["low", "high", "items", "better_for"]],
                hovertemplate="%{y}<br>change %{x:+.4f} (range %{customdata[0]:+.4f} to %{customdata[1]:+.4f})"
                              "<br>%{customdata[2]} items, better for %{customdata[3]:.0%}<extra></extra>", showlegend=False)
fig.add_vline(x=0, line_color=DARK)
style(fig, height=120 + 34 * len(rt), legend=False).update_xaxes(title="change in error (MASE) against the static model; below 0 = better")
fig.update_yaxes(autorange="reversed")
main = rt[~rt["version"].str.contains("placebo|only where", regex=True)]        # each version counted once
n_better = int((main["high"] < 0).sum())
n_worse = int((main["low"] > 0).sum())
n_tested = len(main)
n_rule = int(main["meets_rule"].sum())
chart(fig, f"Reacting to a shift did not pay off: {n_rule} of {n_tested} versions meet the bar for a useful gain",
      f"{n_worse} {'version is' if n_worse == 1 else 'versions are'} clearly worse than doing nothing and {n_better} clearly better, and the better one is too small "
      f"to matter (the grey band is ±{PRACTICAL}, the smallest gain we decided in advance would count). Lines show a 95% range."
      if n_better else
      f"{n_worse} {'version is' if n_worse == 1 else 'versions are'} clearly worse than doing nothing and none is clearly better. The grey band is ±{PRACTICAL}, the "
      f"smallest gain we decided in advance would count. Lines show a 95% range.")
show = rt.rename(columns={"version": "Version", "items": "Items", "change": "Change in MASE", "low": "95% range from",
                          "high": "95% range to", "better_for": "Better for", "meets_rule": "Useful gain"}).drop(columns="model")
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"Change in MASE": st.column_config.NumberColumn(format="%+.4f"),
                            "95% range from": st.column_config.NumberColumn(format="%+.4f"),
                            "95% range to": st.column_config.NumberColumn(format="%+.4f"),
                            "Better for": st.column_config.NumberColumn("Better for (share of items)", format="percent"),
                            "Useful gain": st.column_config.CheckboxColumn()})
download(show, "reacting_to_shifts.csv", key="dl_react")

with st.expander("How far the detector can be trusted"):
    sn = null[null["group"] == "ALL (seasonal null)"].set_index("method")
    p4 = align[(align["event"] == "sustained price change") & (align["window_wks"] == 4)].iloc[0]
    ev0 = align[(align["event"] == "calendar event") & (align["window_wks"] == 0)].iloc[0]
    st.markdown(f"On made-up series that keep each item's yearly pattern but contain no shift, the detector alarms on "
                f"{sn.loc['CUSUM', 'far']:.1%} of them. A second method agrees with it on {agree['raw_agreement']:.0%} of items. "
                f"{p4['aligned_pct']:.1f}% of shifts sit within four weeks of a lasting price change, against "
                f"{p4['chance_pct']:.1f}% by chance; holidays line up no more than chance ({ev0['aligned_pct']:.1f}% against "
                f"{ev0['chance_pct']:.1f}%). So only a small share of shifts can be tied to a price change in this data.")

tells(f"Shifts are common: {len(flagged) / len(scope):.0%} of items in view are flagged and about {adj['value']:.0%} of all items "
      f"have a real one. But the detector is slow (a typical {b['delay_wks'].median():.0f} weeks) and misses small changes "
      f"(a 20% step is caught {s20:.0%} of the time). "
      + (f"None of the {n_tested} ways of reacting met the bar for a useful gain and {n_worse} made forecasts clearly worse, "
         f"so retraining after an alarm is not worth it with these forecasts." if n_rule == 0 else
         f"{n_rule} of the {n_tested} ways of reacting met the bar for a useful gain."))
