import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.kpis import abc_xyz_matrix, filter_items
from utils.ui import chart, header, n_note, style

items = get("items")

header("Segmentation",
       "Items are grouped by how much they sell (ABC) and how erratic their sales are (XYZ). Only the high-volume, "
       "reasonably predictable AX and AY items were modelled.")

cats = st.multiselect("Category", sorted(items["category"].unique()), default=sorted(items["category"].unique()))
sub = filter_items(items, categories=cats)
if sub.empty:
    st.info("No items match the selected categories.")
    st.stop()

counts, share = abc_xyz_matrix(sub)
text = np.array([[f"n = {counts.iloc[i, j]:,}<br>{share.iloc[i, j]:.1%} of units" for j in range(3)] for i in range(3)])
fig = go.Figure(go.Heatmap(
    z=share.to_numpy(), x=["X (steady)", "Y (variable)", "Z (erratic)"], y=["A (top 70% of units)", "B", "C"],
    text=text, texttemplate="%{text}", colorscale=[[0, "rgba(0,114,178,0.05)"], [1, "rgba(0,114,178,0.85)"]],
    showscale=False, customdata=counts.to_numpy(),
    hovertemplate="%{y} / %{x}<br>items: %{customdata:,}<br>share of units: %{z:.2%}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
style(fig, height=340, legend=False)
ax_ay = share.loc["A", "X"] + share.loc["A", "Y"]
chart(fig, f"AX and AY items are {counts.loc['A', 'X'] + counts.loc['A', 'Y']:,} of {len(sub):,} items but "
           f"{ax_ay:.0%} of units sold",
      "Rows rank items by volume, columns by variability (coefficient of variation). Darker cells hold more of "
      "the units sold. The cell text gives the item count and share of units.")
n_note(len(sub))

st.markdown("#### Look inside a cell")
c1, c2 = st.columns(2)
abc = c1.selectbox("ABC class", ["A", "B", "C"])
xyz = c2.selectbox("XYZ class", ["X", "Y", "Z"])
cell = filter_items(sub, abc=abc, xyz=xyz).sort_values("total_sales", ascending=False)
n_note(len(cell), f"items in {abc}{xyz}")
if cell.empty:
    st.info(f"No {abc}{xyz} items in the selected categories.")
    st.stop()

table = cell[["item_id", "category", "total_sales", "mean_sales", "cv", "in_scope", "cusum_first_break",
              "first_break_week", "adf_p_value"]].copy()
table["first_break_week"] = table["first_break_week"].dt.date
st.caption("How to read this: one row per item. Click a row, then open it in the item explorer (only modelled "
           "AX/AY items have forecasts). Click a column header to sort.")
event = st.dataframe(
    table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    column_config={
        "item_id": "Item", "category": "Category",
        "total_sales": st.column_config.NumberColumn("Units sold", format="%d"),
        "mean_sales": st.column_config.NumberColumn("Mean units/day", format="%.1f"),
        "cv": st.column_config.NumberColumn("CV", format="%.2f"),
        "in_scope": st.column_config.CheckboxColumn("Modelled"),
        "cusum_first_break": st.column_config.CheckboxColumn("Break detected"),
        "first_break_week": st.column_config.DateColumn("First break week"),
        "adf_p_value": st.column_config.NumberColumn("ADF p-value", format="%.3f",
                                                     help="Stationarity test, only run for modelled items"),
    })
rows = event.selection.rows if event and event.selection else []
if rows:
    chosen = table.iloc[rows[0]]
    if chosen["in_scope"]:
        if st.button(f"Open {chosen['item_id']} in the item explorer", type="primary"):
            st.session_state["explorer_item"] = chosen["item_id"]
            st.switch_page("views/forecasts.py")
    else:
        st.info(f"{chosen['item_id']} is outside the modelled AX/AY scope, so there are no forecasts for it.")
