import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.ui import chart, figure, header, insight, n_note, style

header("Stationarity",
       "Before fitting classical models we test whether each series holds a steady level, and how much differencing "
       "it needs, with two tests that look at the question from opposite sides.")

summ = get("stationarity_summary")
st_items = get("stationarity_items")
totals = get("stationarity_totals")

all_row = summ[summ["category"] == "ALL"].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Stationary (ADF and KPSS agree)", f"{all_row['stationary']:.0%}", help="Share of the 726 products")
c2.metric("Need one difference (d = 1)", f"{all_row['d=1']:.0%}")
c3.metric("Need a seasonal difference (D = 1)", f"{all_row['D=1']:.0%}")
n_note(int(all_row["n_items"]))

labels = ["stationary", "trend-stationary", "difference-stationary", "conflicting"]
colors = ["#0072B2", "#56B4E9", "#E69F00", "#BBBBBB"]
fig = go.Figure()
for lab, c in zip(labels, colors):
    fig.add_bar(y=summ["category"], x=summ[lab], name=lab, orientation="h", marker_color=c,
                hovertemplate="%{y}: %{x:.1%} " + lab + "<extra></extra>")
fig.update_layout(barmode="stack")
fig.update_xaxes(tickformat=".0%", range=[0, 1], title="share of products")
top = all_row[labels].astype(float).idxmax()
chart(style(fig, height=300), f"Most products test as {top}",
      "ADF (null: the series wanders) and KPSS (null: it holds a level) at 5%, each with a constant and with a trend. "
      "Conflicting means the tests disagree in a way no single label explains.")

st.markdown("#### One product's test results")
pick = st.selectbox("Product", sorted(st_items["item_id"]), key="st_pick")
r = st_items[st_items["item_id"] == pick].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Label", r["label"])
c2.metric("ADF p (constant)", f"{r['adf_c']:.4f}", help="Below 0.05: the series does not wander")
c3.metric("KPSS p (constant)", f"{r['kpss_c']:.3f}", help="Below 0.05: the series does not hold one level")
c4.metric("Differencing", f"d = {r['d']}, D = {r['D']}")

st.markdown("#### The four totals")
st.dataframe(totals[["series", "label", "d", "D", "adf_c", "kpss_c"]], hide_index=True, width="stretch",
             column_config={"adf_c": st.column_config.NumberColumn("ADF p (constant)", format="%.4f"),
                            "kpss_c": st.column_config.NumberColumn("KPSS p (constant)", format="%.3f")})

with st.expander("Autocorrelation and differencing, as saved by notebook 05"):
    figure("acf_pacf_items.png", "Spikes at lags 7, 14 and 21: the weekly pattern.")
    figure("differencing.png", "The California total before and after differencing.")

conf = float(all_row["conflicting"])
insight(f"Only {all_row['stationary']:.0%} of products are cleanly stationary and {conf:.0%} get conflicting answers, "
        f"mostly because both tests reject at once: the series stays near a level that moves now and then. That is "
        f"why the models difference once (d = 1 for {all_row['d=1']:.0%}), carry the weekly pattern in seasonal terms "
        f"rather than a seasonal difference, and why we look for breaks later.")
