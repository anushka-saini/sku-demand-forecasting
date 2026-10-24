import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup
from utils.kpis import break_share_by_category, filter_items
from utils.ui import ACCENT, DARK, NEUTRAL, SECOND, chart, filters, header, insight, n_note, style

f = facts()
items = get("items")
align = get("event_alignment")
calib = get("calibration").set_index("method")
agree = get("agreement").iloc[0]
null = get("null_rates")
inj = get("injection")
fun = get("hc_funnel").rename(columns={"Unnamed: 0": "group"})
cats, segs, _ = filters()

header("Breaks",
       "A weekly CUSUM compares each item's sales with its own first year (after an 8-week launch period); PELT, a "
       "second method, cross-checks it. This page shows how often shifts are flagged, and how far the detector can be trusted.")

# ---------------------------------------------------------------- rates by category and direction
scope = filter_items(items, categories=cats, in_scope_only=True)
if segs:
    scope = scope[scope["segment"].isin(segs)]
if scope.empty:
    st.info("No items match the sidebar filters.")
    st.stop()
share = break_share_by_category(scope)
dirs = scope[scope["cusum_first_break"]].groupby(["category", "first_break_direction"]).size().unstack(fill_value=0)
fig = go.Figure()
for d, name, color, pattern in [(-1, "demand fell", ACCENT, ""), (1, "demand rose", SECOND, "/")]:
    counts = dirs[d] if d in dirs else pd.Series(0, index=share["category"])
    y = counts.reindex(share["category"]).fillna(0).to_numpy() / share["n_items"].to_numpy()
    fig.add_trace(go.Bar(x=share["category"], y=y, name=name, marker=dict(color=color, pattern_shape=pattern),
                         customdata=counts.reindex(share["category"]).fillna(0).to_numpy(),
                         hovertemplate="%{x}: %{customdata} items (%{y:.1%})<extra>" + name + "</extra>"))
fig.update_layout(barmode="stack")
for _, r in share.iterrows():
    fig.add_annotation(x=r["category"], y=r["share"], text=f"{r['share']:.1%} of n={r['n_items']}", showarrow=False, yshift=12)
style(fig).update_yaxes(tickformat=".0%", title="share of items", range=[0, 1])
down = (scope.loc[scope["cusum_first_break"], "first_break_direction"] < 0).mean()
chart(fig, f"{share['n_break'].sum() / share['n_items'].sum():.0%} of items are flagged with a shift; {down:.0%} of those are drops",
      "Bar height is the share of modelled items with a CUSUM break over the full series. Solid = demand fell, hatched = "
      "demand rose. These are raw flags; the validation panel below discounts chance alarms.")
for _, r in share.iterrows():
    if r["n_items"] < 20:
        st.warning(f"{r['category']}: only n = {r['n_items']} items after filtering.")

b = scope[scope["cusum_first_break"]].copy()
if len(b):
    b["quarter"] = b["first_break_week"].dt.to_period("Q").dt.start_time
    tl = b.groupby(["quarter", "category"]).size().reset_index(name="items")
    fig = px.bar(tl, x="quarter", y="items", color="category", pattern_shape="category",
                 color_discrete_sequence=[ACCENT, SECOND, NEUTRAL])
    fig.update_traces(hovertemplate="quarter starting %{x|%b %Y}: %{y} items<extra>%{fullData.name}</extra>")
    style(fig).update_yaxes(title="items with first break in quarter")
    chart(fig, "Most first breaks fall in 2012-2013, the earliest years in which monitoring was possible",
          "Each bar counts items whose first estimated break falls in that quarter. Monitoring starts 60 weeks after an item's launch.")
    n_note(len(b), "items with a break")

# ---------------------------------------------------------------- validation panel
st.divider()
st.markdown("## Detector validation")
st.caption("How far can the flags above be trusted? Every rate here says exactly how it was measured.")

sn = null[null["group"] == "ALL (seasonal null)"].set_index("method")
real_ctrl = lookup(f, "breaks", "real_series_alarm_after_random_date_no_injection")
far_rows = [
    ("CUSUM", "Simple null", calib.loc["CUSUM", "heldout_false_alarm_rate"], int(calib.loc["CUSUM", "n_heldout_surrogates"]),
     "4-week block bootstrap of each item's first year. The threshold was set on this same kind of series, so ~5% is true by construction."),
    ("CUSUM", "Seasonal null", sn.loc["CUSUM", "far"], int(sn.loc["CUSUM", "n_null_series"]),
     "Each item's yearly seasonal pattern + 13-week block bootstrap of its remainder; no shift. The realistic false-alarm rate."),
    ("PELT", "Simple null", calib.loc["PELT", "heldout_false_alarm_rate"], int(calib.loc["PELT", "n_heldout_surrogates"]),
     "As above, for PELT's penalty."),
    ("PELT", "Seasonal null", sn.loc["PELT", "far"], int(sn.loc["PELT", "n_null_series"]), "As above, for PELT."),
    ("CUSUM", "Real series, nothing injected", real_ctrl["value"], int(real_ctrl["n"]),
     "Alarm after a random date on real sales. NOT a false-alarm rate: real series can genuinely drift."),
]
far = pd.DataFrame(far_rows, columns=["Method", "Measured on", "Alarm rate", "Series (n)", "How it was measured"])
far["Alarm rate"] = 100 * far["Alarm rate"]
st.dataframe(far, hide_index=True, width="stretch",
             column_config={"Alarm rate": st.column_config.NumberColumn(format="%.1f%%"),
                            "Series (n)": st.column_config.NumberColumn(format="%d"),
                            "How it was measured": st.column_config.TextColumn(width="large")})

c1, c2 = st.columns(2)
for col, m in [(c1, "CUSUM"), (c2, "PELT")]:
    col.metric(f"{m}: items with a real shift (chance-adjusted)", f"{sn.loc[m, 'adjusted']:.1%}",
               help=f"(flagged − false-alarm rate) / (1 − false-alarm rate); bootstrap over items")
    col.caption(f"95% CI {sn.loc[m, 'ci_low']:.1%} to {sn.loc[m, 'ci_high']:.1%}; flagged {sn.loc[m, 'observed']:.1%}; "
                f"n = {int(sn.loc[m, 'n_items'])} items with at least two years of data")

ctrl = inj[inj["shape"] == "none"].iloc[0]
fig = go.Figure()
for shape, color, dash, symbol in [("step", ACCENT, "solid", "circle"), ("drift", SECOND, "dash", "diamond")]:
    d = inj[inj["shape"] == shape].sort_values("size")
    fig.add_trace(go.Scatter(x=100 * d["size"], y=d["cusum_power"], mode="lines+markers", name=f"sudden {shape}" if shape == "step" else "slow drift (26 weeks)",
                             line=dict(color=color, dash=dash), marker=dict(symbol=symbol, size=9),
                             customdata=d[["n_trials", "cusum_power_26wk", "cusum_median_delay_wks"]],
                             hovertemplate="%{x:.0f}% shift: detected %{y:.1%}<br>within 26 weeks %{customdata[1]:.1%}<br>"
                                           "median delay %{customdata[2]:.0f} weeks<br>n = %{customdata[0]}<extra></extra>"))
fig.add_hline(y=ctrl["cusum_power"], line_dash="dot", line_color=NEUTRAL,
              annotation_text=f"no shift: {ctrl['cusum_power']:.1%}", annotation_position="bottom right")
style(fig).update_xaxes(title="injected shift (% of the item's normal level)")
fig.update_yaxes(tickformat=".0%", range=[0, 1], title="share of trials detected")
s20 = inj[(inj["shape"] == "step") & (inj["size"] == 0.20)].iloc[0]
chart(fig, f"Small shifts are invisible: a sudden 20% shift is caught {s20['cusum_power']:.0%} of the time, "
           f"with a median delay of {s20['cusum_median_delay_wks']:.0f} weeks",
      f"Known shifts were added to break-free (seasonal-null) series ({int(ctrl['n_trials']):,} trials per point). "
      "Blue: sudden step. Orange dashed: slow drift that reaches full size after 26 weeks. Dotted: alarm rate with nothing injected.")

c1, c2, c3 = st.columns(3)
c1.metric("CUSUM and PELT flag the same items", f"{agree['raw_agreement']:.1%}", help=f"Cohen's kappa {agree['cohens_kappa']:.2f}")
c2.metric("PELT changepoint within 8 weeks of CUSUM's", f"{agree['within_8_wks']:.1%}", help="share of CUSUM breaks")
fa = fun.set_index(fun.columns[0]).loc["ALL"]
c3.metric("High-confidence breaks", f"{int(fa['4_high_confidence'])} of {int(fa['cusum_breaks'])}",
          help="PELT agrees within 8 weeks, level changes at least 20%, and at least half the change persists a year later")
n_note(int(agree["n_items"]))

steps = ["cusum_breaks", "1_pelt_agrees", "2_and_effect_20pct", "3_and_assessable", "4_high_confidence"]
labels = ["CUSUM breaks", "+ PELT agrees", "+ shift ≥ 20%", "+ a year of data after", "+ shift persists"]
fig = go.Figure(go.Bar(x=[int(fa[s]) for s in steps], y=labels, orientation="h", marker_color=ACCENT,
                       text=[int(fa[s]) for s in steps], textposition="outside",
                       hovertemplate="%{y}: %{x} breaks<extra></extra>"))
style(fig, height=300, legend=False).update_yaxes(autorange="reversed")
chart(fig, f"Only {fa['hc_share_of_breaks']:.0%} of flagged breaks pass every quality check",
      "Each bar keeps the breaks that also pass the next check, in order. The final bar is the high-confidence set used on the retraining page.")

al = align.copy()
al["label"] = al["event"] + al["window_wks"].map(lambda w: " (same week)" if w == 0 else f" ±{w} wk")
fig = go.Figure()
fig.add_trace(go.Bar(y=al["label"], x=al["aligned_pct"], orientation="h", name="observed", marker_color=ACCENT,
                     customdata=al[["p_value", "n_breaks"]],
                     hovertemplate="%{y}<br>observed %{x:.1f}%<br>p = %{customdata[0]:.3f}, n = %{customdata[1]} breaks<extra></extra>"))
fig.add_trace(go.Bar(y=al["label"], x=al["chance_pct"], orientation="h", name="expected by chance",
                     marker=dict(color=NEUTRAL, pattern_shape="/"), hovertemplate="%{y}<br>chance %{x:.1f}%<extra></extra>"))
for _, r in al.iterrows():
    fig.add_annotation(y=r["label"], x=max(r["aligned_pct"], r["chance_pct"]), text=f"p = {r['p_value']:.3f}",
                       showarrow=False, xanchor="left", xshift=6, font=dict(size=11))
fig.update_layout(barmode="group")
style(fig, height=400).update_xaxes(title="% of breaks with the event nearby")
p4 = al[(al["event"] == "sustained price change") & (al["window_wks"] == 4)].iloc[0]
chart(fig, f"Price changes explain few shifts: {p4['aligned_pct']:.1f}% of breaks sit near one vs {p4['chance_pct']:.1f}% by chance; holidays no more than chance",
      "Blue: share of breaks with the event within the window. Grey hatched: the share expected if break dates were random. "
      "SNAP days fall on the 1st-10th of every month in every year, so they cannot cause a lasting shift.")
n_note(int(al["n_breaks"].iloc[0]), "breaks")

# ---------------------------------------------------------------- shift-aware classical models
st.divider()
st.markdown("## Does a shift-aware SARIMA forecast better?")
shift = get("shift_aware_tests")
fig = go.Figure()
lab = shift["variant"] + ": " + shift["group"]
fig.add_scatter(x=shift["median_diff"], y=lab, mode="markers", marker=dict(color="#D55E00", size=10),
                error_x=dict(type="data", symmetric=False, array=shift["ci_high"] - shift["median_diff"],
                             arrayminus=shift["median_diff"] - shift["ci_low"], color="#999999"),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>", showlegend=False)
fig.add_vline(x=0, line_color="#666666")
style(fig, height=330, legend=False).update_xaxes(title="median change in MASE against static SARIMA (negative = better)")
main = shift[shift["group"] == "all break item-origins"]
helps = main["helps"].astype(str).str.lower().eq("true").any()
chart(fig, "A shift-aware SARIMA variant meets the 'helps' rule" if helps else "No shift-aware SARIMA variant meets the 'helps' rule",
      "Each dot is the median per-product change in MASE against static SARIMA, with a 95% bootstrap interval. "
      "Restart = fit only after the detected break; dummy = a level-shift input; placebo = a random cut on products "
      "without a break.")
n_note(int(main["n"].iloc[0]), "products with a break in the 150-product sample")

share_all = scope["cusum_first_break"].mean()      # follows the sidebar filters
hc_share = fa["hc_share_of_breaks"]
insight(f"Shifts are common: {share_all:.0%} of the products in view are flagged, and after removing chance alarms "
        f"{sn.loc['CUSUM', 'adjusted']:.0%} still have a real one. But the detector only sees large shifts, only "
        f"{hc_share:.0%} of flags pass every quality check, and reacting to a shift does not make SARIMA (or ETS, or "
        f"LightGBM) forecast better.")
