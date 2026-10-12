import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.kpis import break_share_by_category
from utils.ui import ACCENT, NEUTRAL, SECOND, chart, header, n_note, style

items = get("items")
power = get("power_check")
align = get("event_alignment")
calib = get("calibration").set_index("method")
agree = get("agreement").iloc[0]

header("Break detection",
       "A weekly CUSUM compares each item's sales against its own first year (after an 8-week launch period). The "
       "threshold was calibrated on simulated break-free series. PELT, a second method, is used as a cross-check.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("CUSUM false-alarm rate", f"{calib.loc['CUSUM', 'heldout_false_alarm_rate']:.1%}",
          help=f"On {int(calib.loc['CUSUM', 'n_heldout_surrogates']):,} held-out break-free surrogate series; "
               f"threshold h = {calib.loc['CUSUM', 'value']:.2f}")
c2.metric("PELT false-alarm rate", f"{calib.loc['PELT', 'heldout_false_alarm_rate']:.1%}",
          help=f"On {int(calib.loc['PELT', 'n_heldout_surrogates']):,} held-out surrogates; penalty {calib.loc['PELT', 'value']:.1f}")
c3.metric("CUSUM vs PELT agreement", f"{agree['raw_agreement']:.1%}", help=f"Cohen's kappa {agree['cohens_kappa']:.2f}")
c4.metric("PELT changepoint within ±8 weeks", f"{agree['within_8_wks']:.1%}", help="Share of CUSUM breaks")
st.caption(f"Kappa {agree['cohens_kappa']:.2f}: the methods mostly flag the same items, but "
           f"{1 - agree['within_8_wks']:.0%} of CUSUM breaks have no PELT changepoint within 8 weeks, so the dates "
           f"often differ. n = {int(agree['n_items'])} items.")

# Share by category, split by direction
scope = items[items["in_scope"]]
share = break_share_by_category(items)
dirs = scope[scope["cusum_first_break"]].groupby(["category", "first_break_direction"]).size().unstack(fill_value=0)
fig = go.Figure()
for d, name, color, pattern in [(-1, "demand fell", ACCENT, ""), (1, "demand rose", SECOND, "/")]:
    y = dirs.get(d, 0) / share.set_index("category")["n_items"]
    fig.add_trace(go.Bar(x=share["category"], y=y.reindex(share["category"]).to_numpy(), name=name,
                         marker=dict(color=color, pattern_shape=pattern),
                         customdata=dirs.get(d, 0).reindex(share["category"]).to_numpy(),
                         hovertemplate="%{x}: %{customdata} items (%{y:.1%})<extra>" + name + "</extra>"))
fig.update_layout(barmode="stack")
for _, r in share.iterrows():
    fig.add_annotation(x=r["category"], y=r["share"], text=f"{r['share']:.1%} of n={r['n_items']}", showarrow=False, yshift=12)
style(fig).update_yaxes(tickformat=".0%", title="share of items", range=[0, 1])
chart(fig, "About two-thirds of items in every category show a break, and most breaks are drops in demand",
      "Bar height is the share of modelled items with a CUSUM break over the full series. Solid = demand fell, "
      "hatched = demand rose. Labels give the share and number of items.")

# Timeline
b = scope[scope["cusum_first_break"]].copy()
b["quarter"] = b["first_break_week"].dt.to_period("Q").dt.start_time
tl = b.groupby(["quarter", "category"]).size().reset_index(name="items")
fig = px.bar(tl, x="quarter", y="items", color="category", pattern_shape="category",
             color_discrete_sequence=[ACCENT, SECOND, NEUTRAL])
fig.update_traces(hovertemplate="quarter starting %{x|%b %Y}: %{y} items<extra>%{fullData.name}</extra>")
style(fig).update_yaxes(title="items with first break in quarter")
chart(fig, "Most first breaks fall in 2012-2013, the earliest years in which monitoring was possible",
      "Each bar counts items whose first estimated break falls in that quarter. Monitoring can only start 60 "
      "weeks after an item's launch.")
n_note(len(b), "items with a break")

# Power check
pc = power.copy()
fig = go.Figure()
fig.add_trace(go.Scatter(x=pc["size_sd"], y=pc["cusum_detection_rate"], mode="lines+markers", name="CUSUM detected",
                         line=dict(color=ACCENT), marker=dict(symbol="circle", size=9),
                         customdata=pc[["shift_pct_of_mean_median", "cusum_median_delay_wks", "n_trials"]],
                         hovertemplate="shift %{x}σ (≈%{customdata[0]:.0f}% of mean)<br>detected %{y:.1%}<br>"
                                       "median delay %{customdata[1]:.0f} wks<br>n = %{customdata[2]}<extra></extra>"))
fig.add_trace(go.Scatter(x=pc["size_sd"], y=pc["pelt_hit_rate_8wk"], mode="lines+markers", name="PELT within ±8 wks",
                         line=dict(color=SECOND, dash="dash"), marker=dict(symbol="diamond", size=9),
                         hovertemplate="shift %{x}σ<br>PELT hit %{y:.1%}<extra></extra>"))
control = pc[pc["size_sd"] == 0].iloc[0]
fig.add_hline(y=control["cusum_detection_rate"], line_dash="dot", line_color=NEUTRAL,
              annotation_text=f"no shift injected: CUSUM still alarms {control['cusum_detection_rate']:.0%}",
              annotation_position="bottom right")
style(fig).update_xaxes(title="injected shift size (standard deviations of the baseline year)")
fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title="share of trials")
chart(fig, f"CUSUM reliably catches only large shifts. Even with no shift injected, it alarms "
           f"{control['cusum_detection_rate']:.0%} of the time because the series drift on their own",
      f"Known level shifts were added to real series at a random date ({int(control['n_trials']):,} trials per "
      f"size) and the methods were re-run. Anything above the dotted line is what the injected shift adds. "
      f"1σ ≈ {pc.loc[pc['size_sd'] == 1, 'shift_pct_of_mean_median'].iloc[0]:.0f}% of an item's mean level.")

# Event alignment
al = align.copy()
al["label"] = al["event"] + al["window_wks"].map(lambda w: " (same week)" if w == 0 else f" ±{w} wk")
fig = go.Figure()
fig.add_trace(go.Bar(y=al["label"], x=al["aligned_pct"], orientation="h", name="observed", marker_color=ACCENT,
                     customdata=al[["p_value", "n_breaks"]],
                     hovertemplate="%{y}<br>observed %{x:.1f}%<br>p = %{customdata[0]:.3f}, n = %{customdata[1]} breaks<extra></extra>"))
fig.add_trace(go.Bar(y=al["label"], x=al["chance_pct"], orientation="h", name="expected by chance",
                     marker=dict(color=NEUTRAL, pattern_shape="/"),
                     hovertemplate="%{y}<br>chance %{x:.1f}%<extra></extra>"))
for _, r in al.iterrows():
    fig.add_annotation(y=r["label"], x=max(r["aligned_pct"], r["chance_pct"]), text=f"p = {r['p_value']:.3f}",
                       showarrow=False, xanchor="left", xshift=6, font=dict(size=11))
fig.update_layout(barmode="group")
style(fig, height=420).update_xaxes(title="% of breaks with the event nearby")
price4 = al[(al["event"] == "sustained price change") & (al["window_wks"] == 4)].iloc[0]
chart(fig, f"Only {price4['aligned_pct']:.1f}% of breaks sit near a sustained price change, against "
           f"{price4['chance_pct']:.1f}% by chance. Most breaks have no identifiable cause in this data",
      "Blue: share of breaks with the event within the window. Grey hatched: the share expected if break dates were "
      "random. Holidays recur every year, so their chance rate is high. SNAP days fall on the 1st-10th of every "
      "month in every year, so SNAP cannot cause a lasting shift and is not shown. Co-occurrence is not causation.")
n_note(int(al["n_breaks"].iloc[0]), "breaks")
