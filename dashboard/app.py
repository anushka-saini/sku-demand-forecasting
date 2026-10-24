"""Entry point: python -m streamlit run dashboard/app.py"""
import pandas as pd
import streamlit as st

from utils.data import get
from utils.derive import eval_origins
from utils.facts import facts
from utils.ui import CSS, SHIFT_GROUPS, footer, topbar

st.set_page_config(page_title="SKU Demand Forecasting", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/data_scope.py", title="Data and scope"),
    st.Page("views/demand_patterns.py", title="Demand patterns"),
    st.Page("views/accuracy.py", title="Forecast accuracy"),
    st.Page("views/item_explorer.py", title="Item explorer"),
    st.Page("views/demand_shifts.py", title="Demand shifts"),
    st.Page("views/stock_impact.py", title="Stock impact"),
    st.Page("views/method_limits.py", title="Method and limits"),
]
nav = st.navigation(pages, position="sidebar")

f = facts()
items = get("items")
topbar(int(f["scope"]["ax_ay_items"]["value"]), int(f["scope"]["ca_items"]["value"]))

# Filters live here, not on the pages, so they keep their values when you move between pages.
origins = eval_origins()
with st.sidebar:
    st.markdown("**Filters**")
    cats = sorted(items.loc[items["in_scope"], "category"].unique())
    st.multiselect("Category", cats, default=cats, key="g_cat", format_func=str.title)
    st.multiselect("Item group", ["AX", "AY"], default=["AX", "AY"], key="g_seg",
                   help="AX items sell a lot and steadily; AY items sell a lot with more day-to-day variation.")
    st.radio("Shift group", SHIFT_GROUPS, key="g_shift",
             help="Whether the detector had found a lasting change in the item's sales level by a forecast date.")
    st.select_slider("Forecast dates", options=origins, value=(origins[0], origins[-1]), key="g_window",
                     format_func=lambda d: f"{pd.Timestamp(d):%d %b %Y}",
                     help="The dates the forecasts were made. Each forecast covers the next 28 days.")
    st.caption("Each page says which of these filters it uses.")

nav.run()
footer(f)
