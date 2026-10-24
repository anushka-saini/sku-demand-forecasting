import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import get
from utils.models import TOTALS
from utils.ui import ACCENT, NEUTRAL, chart, filters, filters_used, kpis, page, style, tells

CAT_COLORS = {"FOODS": "#1F6F8B", "HOBBIES": "#C47F1A", "HOUSEHOLD": "#5E8C61"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

flt = filters()
day = get("weekday_index").set_index("weekday").reindex(DAYS)
month = get("month_index").set_index("month")
cal = get("calendar_effects").set_index("category")
rho = get("price_correlation")
items = get("items")
decomp = get("demo_decomposition")
stat = get("stationarity_summary")

page("Demand patterns", "How do sales move through the week and the year, and what do events and price changes do to them?")
filters_used(["category"])
cats = [c for c in ["FOODS", "HOBBIES", "HOUSEHOLD"] if c in (flt["categories"] or ["FOODS", "HOBBIES", "HOUSEHOLD"])]
if not cats:
    st.info("No category selected. Pick at least one in the sidebar.")
    st.stop()

weekend = day.loc[["Saturday", "Sunday"], cats]
peak_cat, peak_day = weekend.stack().idxmax()[::-1]
peak = weekend.max().max() - 100
low = 100 - day[cats].min().min()
swing = (month[cats].max() - month[cats].min()).max()
r = rho.merge(items[["item_id", "category"]], on="item_id")
r = r[r["category"].isin(cats)]["spearman_units_vs_price"]
kpis([
    {"label": "Busiest day vs an average day", "value": f"{peak:+.0f}%", "compare": f"{peak_day} in {peak_cat.title()}",
     "help": "Average daily sales on that weekday divided by the average day, minus 100%."},
    {"label": "Quietest day vs an average day", "value": f"{-low:+.0f}%", "compare": "midweek"},
    {"label": "Largest gap between months", "value": f"{swing:.0f} pts", "compare": "between the best and worst month",
     "help": "Points on the same index (100 = an average day). Monthly averages also contain five years of growth."},
    {"label": "SNAP days, extra sales", "value": f"{cal.loc[cats, 'snap_vs_other_days'].max():+.0%}",
     "compare": f"largest in {cal.loc[cats, 'snap_vs_other_days'].idxmax().title()}",
     "help": "SNAP is US food assistance; in California it can be spent on the 1st to 10th of each month."},
    {"label": "Price and sales", "value": f"{r.median():+.2f}", "compare": f"typical correlation; negative for {(r < 0).mean():.0%} of items",
     "help": "Rank correlation of each item's weekly units with its weekly price. Negative means dearer weeks sell less."},
])

# trend and weekly pattern of one total, last 52 weeks
pick = st.selectbox("Series to break down", TOTALS, format_func=lambda s: "California total" if s == "CA_TOTAL" else s.title(),
                    key="dp_series")
d = decomp[decomp["series"] == pick].sort_values("date")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=["Daily sales and the underlying trend", "Weekly pattern: extra units on each day"])
fig.add_scatter(x=d["date"], y=d["observed"], mode="lines", name="daily sales", line=dict(color=NEUTRAL, width=0.8), row=1, col=1,
                hovertemplate="%{x|%a %d %b %Y}: %{y:,.0f} units<extra>sales</extra>")
fig.add_scatter(x=d["date"], y=d["trend"], mode="lines", name="trend", line=dict(color=ACCENT, width=2.2), row=1, col=1,
                hovertemplate="%{x|%d %b %Y}: %{y:,.0f} units<extra>trend</extra>")
fig.add_scatter(x=d["date"], y=d["seasonal"], mode="lines", name="weekly pattern", line=dict(color="#C47F1A", width=1), row=2, col=1,
                hovertemplate="%{x|%a %d %b %Y}: %{y:+,.0f} units<extra>weekly pattern</extra>")
style(fig, height=490)
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0), margin=dict(t=30, b=40))
fig.update_yaxes(title_text="units per day", tickformat=",", row=1, col=1)
fig.update_yaxes(title_text="units vs trend", tickformat=",", row=2, col=1)
trend_change = d["trend"].tail(28).mean() / d["trend"].head(28).mean() - 1      # four-week averages: the trend's ends are less settled
wk = d["seasonal"].tail(7)
label = "California total" if pick == "CA_TOTAL" else pick.title()
chart(fig, f"{label}: the weekly pattern swings by about {wk.max() - wk.min():,.0f} units a day",
      f"From the first four weeks to the last four, the trend moved {trend_change:+.0%}; the weekly pattern repeats every week on top of it.")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    for cat in cats:
        fig.add_scatter(x=[x[:3] for x in DAYS], y=day[cat], mode="lines+markers", name=cat.title(),
                        line=dict(color=CAT_COLORS[cat]), hovertemplate="%{x}: %{y:.1f}<extra>" + cat.title() + "</extra>")
    fig.add_hline(y=100, line_dash="dot", line_color=NEUTRAL)
    style(fig, height=320).update_yaxes(title="sales index (100 = average day)")
    fig.update_xaxes(title="day of the week")
    chart(fig, f"Weekends are busiest: up to {peak:+.0f}% on {peak_day}",
          f"Midweek is up to {low:.0f}% below an average day. "
          + ("Hobbies drops back on Sunday." if "HOBBIES" in cats and day.loc["Sunday", "HOBBIES"] < day.loc["Saturday", "HOBBIES"] else ""))
with c2:
    fig = go.Figure()
    for cat in cats:
        fig.add_scatter(x=[MONTHS[m - 1] for m in month.index], y=month[cat], mode="lines+markers", name=cat.title(),
                        line=dict(color=CAT_COLORS[cat]), hovertemplate="%{x}: %{y:.1f}<extra>" + cat.title() + "</extra>")
    fig.add_hline(y=100, line_dash="dot", line_color=NEUTRAL)
    style(fig, height=320).update_yaxes(title="sales index (100 = average day)")
    fig.update_xaxes(title="month")
    best_m = month[cats].mean(axis=1).idxmax()
    chart(fig, f"The year matters less: at most {swing:.0f} points between months",
          f"The highest month on average is {MONTHS[best_m - 1]}. Monthly averages also include five years of growth; "
          f"notebook 06 separates the two.")

c1, c2 = st.columns(2)
with c1:
    e = cal.loc[cats].reset_index()
    fig = go.Figure()
    for col, lab, color in [("snap_vs_other_days", "SNAP days", ACCENT), ("event_vs_normal_no_christmas", "event days (not Christmas)", "#C47F1A")]:
        fig.add_bar(x=e["category"].str.title(), y=100 * e[col], name=lab, marker_color=color,
                    hovertemplate="%{x}: %{y:+.1f}%<extra>" + lab + "</extra>")
    fig.add_hline(y=0, line_color=NEUTRAL)
    fig.update_layout(barmode="group")
    style(fig, height=320).update_yaxes(title="sales vs a normal day of the same weekday (%)", ticksuffix="%")
    ev = cal.loc[cats, "event_vs_normal_no_christmas"]
    snap_up = bool((cal.loc[cats, "snap_vs_other_days"] > 0).all())
    ev_down = bool((cal.loc[cats, "event_vs_normal_no_christmas"] < 0).all())
    chart(fig, ("SNAP days lift sales" if snap_up else "SNAP days do not lift every category")
          + ("; event days do not" if ev_down else "; event days are mixed"),
          f"Event days sell {ev.min():+.1%} to {ev.max():+.1%} against normal days, even without Christmas. "
          f"SNAP days and the start of the month coincide, so this cannot separate the two.")
with c2:
    fig = go.Figure(go.Histogram(x=r, nbinsx=40, marker_color=ACCENT, hovertemplate="correlation %{x:.2f}: %{y} items<extra></extra>"))
    fig.add_vline(x=0, line_color=NEUTRAL)
    style(fig, height=320, legend=False).update_xaxes(title="correlation of weekly units with weekly price")
    fig.update_yaxes(title="number of items")
    chart(fig, f"Dearer weeks usually sell less, but the link is weak (typical {r.median():+.2f})",
          f"{len(r):,} items changed price at least once; for {(r < 0).mean():.0%} of them higher-price weeks sold less.")

with st.expander("Do the series hold a steady level?"):
    labels = ["stationary", "trend-stationary", "difference-stationary", "conflicting"]
    colors = [ACCENT, "#7FB2C5", "#C47F1A", "#B8B8B8"]
    s = stat[stat["category"].isin(cats + ["ALL"])]
    fig = go.Figure()
    for lab, c in zip(labels, colors):
        fig.add_bar(y=s["category"].str.title(), x=s[lab], name=lab, orientation="h", marker_color=c,
                    hovertemplate="%{y}: %{x:.1%} " + lab + "<extra></extra>")
    fig.update_layout(barmode="stack")
    style(fig, height=260).update_xaxes(tickformat=".0%", range=[0, 1], title="share of items")
    a = stat[stat["category"] == "ALL"].iloc[0]
    chart(fig, f"Only {a['stationary']:.0%} of items hold one steady level; {a['conflicting']:.0%} get mixed answers",
          "Two standard tests ask opposite questions. Mixed answers usually mean a level that holds for a while and then moves, "
          "which is what the Demand shifts page looks for.")

ev_down = bool((cal.loc[cats, "event_vs_normal_no_christmas"] < 0).all())
tells(f"The week is the pattern that matters: weekends run up to {peak:.0f}% above an average day and midweek up to "
      f"{low:.0f}% below, while months differ by at most {swing:.0f} points. SNAP days add up to "
      f"{cal.loc[cats, 'snap_vs_other_days'].max():.0%}, "
      + ("event days sell less than normal days, " if ev_down else "event days are mixed, ")
      + f"and price has a weak pull (typical correlation {r.median():+.2f}). So the weekly shape is the first thing a "
      f"forecast has to capture.")
