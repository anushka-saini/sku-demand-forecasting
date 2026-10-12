"""Entry point: streamlit run dashboard/app.py"""
import streamlit as st

from utils.ui import CSS, sidebar_glossary

st.set_page_config(page_title="SKU Demand Breaks & Inventory", layout="wide", initial_sidebar_state="auto")
st.markdown(CSS, unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/data_method.py", title="Data & method"),
    st.Page("views/segmentation.py", title="Segmentation"),
    st.Page("views/breaks.py", title="Break detection"),
    st.Page("views/forecasts.py", title="Forecast comparison"),
    st.Page("views/inventory.py", title="Inventory decision"),
    st.Page("views/limitations.py", title="Assumptions & limitations"),
]
nav = st.navigation(pages)
sidebar_glossary()
nav.run()
