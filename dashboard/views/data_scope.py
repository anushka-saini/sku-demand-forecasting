import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, value
from utils.kpis import abc_xyz_matrix
from utils.ui import ACCENT, NEUTRAL, apply_item_filters, chart, download, filters, filters_used, kpis, page, style, tells, vline

CAT_COLORS = {"FOODS": "#1F6F8B", "HOBBIES": "#C47F1A", "HOUSEHOLD": "#5E8C61"}

f = facts()
flt = filters()
items = get("items")
weekly = get("sales_weekly")
lz = get("leading_zero_example")
quality = get("data_quality")
window = f["scope"]["windows"]["W1"]

page("Data and scope", "What data is this, and which items did we study?")
filters_used(["category", "segment"])

n_ca, n_items = int(value(f, "scope", "ca_items")), int(value(f, "scope", "ax_ay_items"))
share_items, share_units = value(f, "scope", "ax_ay_share_of_items"), value(f, "scope", "ax_ay_share_of_units")
n_lz = int(value(f, "scope", "items_with_over_30_leading_zero_days"))
first_wk, last_wk = weekly["week_start"].min(), weekly["week_start"].max()
kpis([
    {"label": "Items in California", "value": f"{n_ca:,}", "compare": "four stores, three categories"},
    {"label": "Items studied", "value": f"{n_items:,}", "compare": f"{share_items:.1%} of items, {share_units:.1%} of units sold",
     "help": "AX and AY items: the top 70% of each category's units, with fairly steady daily sales."},
    {"label": "Sales history", "value": f"{(last_wk - first_wk).days / 365.25:.1f} years",
     "compare": f"{first_wk:%b %Y} to {last_wk:%b %Y}"},
    {"label": "Forecast test period", "value": f"{(pd.Timestamp(window['last_eval_day']) - pd.Timestamp(window['first_eval_day'])).days + 1} days",
     "compare": f"{pd.Timestamp(window['first_eval_day']):%d %b %Y} to {pd.Timestamp(window['last_eval_day']):%d %b %Y}"},
    {"label": "Items that start with months of zeros", "value": f"{n_lz:,}", "compare": f"{n_lz / n_ca:.0%} of all California items",
     "help": "More than 30 days without a sale before the first sale: the item was not on the shelf yet."},
])

st.markdown("The data is Walmart's public M5 set: daily units sold for every item and store, a calendar with holidays "
            "and food-assistance (SNAP) days, and weekly prices. We add the four California stores into one daily "
            "series per item.")

scope = apply_item_filters(items, {**flt, "shift": "All items"})
if scope.empty:
    st.info("No items match the category and item-group filters. Select at least one of each in the sidebar.")
    st.stop()

# what the sales look like: weekly units of the studied items, by category
w = weekly.merge(scope[["item_id", "category"]], on="item_id")
tot = w.groupby(["week_start", "category"], as_index=False)["units"].sum()
tot = tot[(tot["week_start"] > tot["week_start"].min()) & (tot["week_start"] < tot["week_start"].max())]   # drop part weeks
fig = go.Figure()
for cat, g in tot.groupby("category"):
    fig.add_scatter(x=g["week_start"], y=g["units"], mode="lines", name=cat.title(), line=dict(color=CAT_COLORS.get(cat, ACCENT), width=1.4),
                    hovertemplate="week of %{x|%d %b %Y}: %{y:,.0f} units<extra>" + cat.title() + "</extra>")
style(fig).update_yaxes(title="units sold per week", tickformat=",")
fig.update_xaxes(title="week")
cat_share = tot.groupby("category")["units"].sum() / tot["units"].sum()
yr = tot.assign(year=tot["week_start"].dt.year).groupby("year")["units"].sum()
full = yr.loc[yr.index[1]:yr.index[-2]] if len(yr) > 3 else yr
growth = full.iloc[-1] / full.iloc[0] - 1
top = cat_share.idxmax()
chart(fig, f"{top.title()} is {cat_share.max():.0%} of the units sold by these items",
      f"Yearly sales of the {len(scope):,} items in view changed by {growth:+.0%} from {full.index[0]} to {full.index[-1]} "
      f"(full years only).")

# trimming: one item before and after
item = lz["item_id"].iloc[0]
removed, kept = lz[~lz["kept_after_trim"]], lz[lz["kept_after_trim"]]
first_sale = kept["date"].min()
fig = go.Figure()
fig.add_scatter(x=removed["date"], y=removed["sales"], mode="lines", name="removed: not on sale yet",
                line=dict(color=NEUTRAL, width=1, dash="dot"), hovertemplate="%{x|%d %b %Y}: %{y} units<extra>removed</extra>")
fig.add_scatter(x=kept["date"], y=kept["sales"], mode="lines", name="kept", line=dict(color=ACCENT, width=1),
                hovertemplate="%{x|%d %b %Y}: %{y} units<extra>kept</extra>")
vline(fig, first_sale, f" first sale {first_sale:%d %b %Y}")
style(fig, height=280).update_yaxes(title="units sold per day")
fig.update_xaxes(title="date")
trimmed = int(quality.loc[quality["category"] == "ALL", "leading_zero_days_trimmed"].iloc[0])
chart(fig, f"Before and after trimming: {item} sold nothing for {len(removed):,} days because it was not on sale yet",
      f"We start every series at its first sale. Across the {n_items:,} items this removes {trimmed:,} days of zeros "
      f"that would otherwise look like a huge jump in demand at launch.")

with st.expander("How the items were chosen"):
    counts, share = abc_xyz_matrix(items)
    fig = go.Figure(go.Heatmap(z=share.to_numpy(), x=["X (steady)", "Y (variable)", "Z (erratic)"],
                               y=["A (top 70% of units)", "B (next 20%)", "C (last 10%)"],
                               text=[[f"{counts.iloc[i, j]:,} items<br>{share.iloc[i, j]:.0%} of units" for j in range(3)]
                                     for i in range(3)],
                               texttemplate="%{text}", showscale=False,
                               colorscale=[[0, "rgba(31,111,139,0.06)"], [1, "rgba(31,111,139,0.85)"]],
                               hovertemplate="%{y} / %{x}: %{z:.1%} of units<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    ax = counts.loc["A", "X"] + counts.loc["A", "Y"]
    chart(style(fig, height=300, legend=False), f"The {ax:,} AX and AY items carry {share.loc['A', 'X'] + share.loc['A', 'Y']:.0%} of units",
          "Rows rank items by volume, columns by how much daily sales vary. We model the top-left two cells.")

with st.expander("Is the data clean?"):
    q = quality[["category", "items", "raw_empty_cells", "missing_days_inside_series", "zero_days_kept_share",
                 "outlier_days", "outlier_share"]].copy()
    st.dataframe(q, hide_index=True, width="stretch",
                 column_config={"category": "Category", "items": st.column_config.NumberColumn("Items", format="%d"),
                                "raw_empty_cells": st.column_config.NumberColumn("Empty cells", format="%d"),
                                "missing_days_inside_series": st.column_config.NumberColumn("Missing days", format="%d"),
                                "zero_days_kept_share": st.column_config.NumberColumn("Zero-sale days kept", format="percent"),
                                "outlier_days": st.column_config.NumberColumn("Extreme days", format="%d"),
                                "outlier_share": st.column_config.NumberColumn("Extreme share", format="percent")})
    st.caption("Extreme = above the 75th percentile plus three times the interquartile range of the item's daily sales. "
               "Extreme and zero days are kept, because they are real sales.")
    download(q, "data_quality.csv", key="dl_quality")

allq = quality[quality["category"] == "ALL"].iloc[0]
gaps = int(allq["raw_empty_cells"]) + int(allq["missing_days_inside_series"])
tells((f"The data has no gaps: {int(allq['raw_empty_cells']):,} empty cells and {int(allq['missing_days_inside_series']):,} "
       f"missing days inside the series. " if gaps == 0 else
       f"The data has {gaps:,} empty cells or missing days inside the series. ")
      + f"The main cleaning step is trimming the {n_lz:,} items that start with months of zeros. The {n_items:,} items "
      f"studied are {share_items:.0%} of the range but {share_units:.0%} of what is sold.")
