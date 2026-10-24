import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.ui import ACCENT, NEUTRAL, SECOND, chart, figure, header, insight, n_note, style

CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"]
CAT_COLORS = {"FOODS": ACCENT, "HOBBIES": SECOND, "HOUSEHOLD": "#009E73"}

header("Explore",
       "Before fitting anything we looked at how the 726 products sell through the week and the year, and what moves "
       "with their sales: price, SNAP days and events.")

day = get("weekday_index").set_index("weekday")
month = get("month_index").set_index("month")
cal = get("calendar_effects").set_index("category")
rho = get("price_correlation")["spearman_units_vs_price"]
quality = get("data_quality")

# the key figure: the weekly rhythm
fig = go.Figure()
for cat in CATS:
    fig.add_scatter(x=[d[:3] for d in day.index], y=day[cat], mode="lines+markers", name=cat.title(),
                    line=dict(color=CAT_COLORS[cat]), hovertemplate="%{x}: %{y:.0f}<extra>" + cat + "</extra>")
fig.add_hline(y=100, line_dash="dot", line_color=NEUTRAL)
style(fig).update_yaxes(title="sales index (100 = average day)")
fig.update_xaxes(title="day of the week")
wk_hi = day.loc[["Saturday", "Sunday"], ["FOODS", "HOUSEHOLD"]].max().max() - 100
chart(fig, f"Weekends sell up to {wk_hi:.0f}% more than an average day",
      "Average daily sales of all products in a category on each weekday, divided by the category's overall daily "
      "average. HOBBIES peaks on Saturday and drops back on Sunday.")

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    for cat in CATS:
        fig.add_scatter(x=month.index, y=month[cat], mode="lines+markers", name=cat.title(), line=dict(color=CAT_COLORS[cat]),
                        hovertemplate="month %{x}: %{y:.0f}<extra>" + cat + "</extra>")
    fig.add_hline(y=100, line_dash="dot", line_color=NEUTRAL)
    style(fig, height=320).update_yaxes(title="sales index (100 = average day)")
    fig.update_xaxes(title="month", dtick=1)
    swing = (month[CATS].max() - month[CATS].min()).max()
    chart(fig, f"The yearly swing is smaller: at most {swing:.0f} points between months",
          "Monthly averages also carry the growth of the totals over five years. The Decomposition page separates the two.")
with c2:
    fig = go.Figure(go.Histogram(x=rho, nbinsx=40, marker_color=ACCENT,
                                 hovertemplate="correlation %{x:.2f}: %{y} products<extra></extra>"))
    fig.add_vline(x=0, line_color="#4D4D4D")
    style(fig, height=320, legend=False).update_xaxes(title="correlation of weekly units with weekly price")
    fig.update_yaxes(title="number of products")
    chart(fig, f"Dearer weeks usually sell less: median correlation {rho.median():+.2f}",
          f"Spearman correlation per product, for the {len(rho)} products whose price changed at least once. "
          f"Negative for {(rho < 0).mean():.0%} of them.")

st.markdown("#### SNAP days and events")
st.caption("Each day compared with the average of the same weekday, so weekends do not bias the comparison. SNAP days "
           "are the 1st to 10th of each month in California.")
show = cal.reset_index()
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"category": "Category",
                            "snap_vs_other_days": st.column_config.NumberColumn("SNAP days vs other days", format="percent"),
                            "event_vs_normal_days": st.column_config.NumberColumn("event days vs normal days", format="percent"),
                            "event_vs_normal_no_christmas": st.column_config.NumberColumn("same, without Christmas",
                                                                                          format="percent")})

st.markdown("#### Is the data clean?")
q = quality[["category", "items", "raw_empty_cells", "missing_days_inside_series", "zero_days_kept_share",
             "outlier_days", "outlier_share"]]
st.dataframe(q, hide_index=True, width="stretch",
             column_config={"raw_empty_cells": "empty cells", "missing_days_inside_series": "missing days",
                            "zero_days_kept_share": st.column_config.NumberColumn("zero days", format="percent"),
                            "outlier_days": "extreme days",
                            "outlier_share": st.column_config.NumberColumn("extreme share", format="percent")})
st.caption("Extreme = above Q3 + 3 × IQR of the product's daily sales. Extreme days and zero days are kept: they are "
           "real sales.")
n_note(int(quality.loc[quality["category"] == "ALL", "items"].iloc[0]))

with st.expander("The same plots as saved by notebook 03"):
    figure("sales_and_zero_share_histograms.png", "Daily sales are skewed, and most products are rarely zero.")
    figure("daily_line_three_products.png", "One year of daily sales for one product in each category.")

foods_snap = cal.loc["FOODS", "snap_vs_other_days"]
insight(f"The week is the rhythm every model has to get right: weekends are far above midweek in every category, "
        f"while the months move much less. SNAP days lift FOODS by about {foods_snap:.0%}, and price has a weak but "
        f"steady pull on sales, so calendar and price are worth giving to a model that can use them.")
