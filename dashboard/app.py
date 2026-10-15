"""Entry point: streamlit run dashboard/app.py"""
import streamlit as st

from utils.data import get
from utils.facts import facts
from utils.ui import CSS, footer, sidebar_filters, sidebar_glossary

st.set_page_config(page_title="SKU Demand Shifts & Inventory", layout="wide", initial_sidebar_state="auto")
st.markdown(CSS, unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/data_method.py", title="Data & method"),
    st.Page("views/segmentation.py", title="Segmentation"),
    st.Page("views/breaks.py", title="Shift detection"),
    st.Page("views/forecasts.py", title="Forecast comparison"),
    st.Page("views/retraining.py", title="Does retraining help?"),
    st.Page("views/inventory.py", title="Inventory simulator"),
    st.Page("views/planner.py", title="Planner: what to check Monday"),
    st.Page("views/replication.py", title="Replication: Texas & Wisconsin"),
    st.Page("views/limitations.py", title="Assumptions & limitations"),
]
nav = st.navigation(pages)

items = get("items")
sidebar_filters(sorted(items["category"].unique()), ["AX", "AY"])
sidebar_glossary()
nav.run()
footer(facts())
