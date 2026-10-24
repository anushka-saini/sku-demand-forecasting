import streamlit as st

from utils.data import get
from utils.facts import facts, value
from utils.ui import figure, header, insight

cat = get("category_summary")
items = get("items")

header("Data",
       "Walmart's M5 data records daily unit sales of every product in its California stores from 2011 to 2016; this "
       "page shows what is in it and which products we chose to model.")

n_total = int(cat["total_items"].sum())
n_lz = int(cat["items_with_leading_zeros"].sum())
n_axay = int(cat["ax_ay_candidates"].sum())

st.markdown("#### Pipeline")
steps = [("Meet the data", "sales, calendar, prices"),
         ("Choose scope", f"California, 4 stores summed · {n_total:,} products, {n_axay} AX/AY kept"),
         ("Explore", "weekly and yearly patterns, price, SNAP, events"),
         ("Zeros and trimming", f"{n_lz:,} products had >30 zero days before their first sale"),
         ("Stationarity and decomposition", "ADF + KPSS, differencing, weekly and yearly patterns"),
         ("Classical models, breaks, LightGBM", "same backtest: 7 origins, 28-day forecasts"),
         ("Uncertainty and stock", "calibrated bands, reorder-point simulation over 168 days")]
boxes = "<span class='arrow'>→</span>".join(
    f"<div class='step'><b>{i}. {name}</b><span>{detail}</span></div>" for i, (name, detail) in enumerate(steps, 1))
st.markdown(f"<div class='pipeline'>{boxes}</div>", unsafe_allow_html=True)
f = facts()
c1, c2, c3 = st.columns(3)
c1.metric("Items modelled", f"{value(f, 'scope', 'ax_ay_items'):,}", help="AX and AY items in California")
c2.metric("Share of all California items", f"{value(f, 'scope', 'ax_ay_share_of_items'):.1%}",
          help=f"of {value(f, 'scope', 'ca_items'):,} items")
c3.metric("Share of units sold", f"{value(f, 'scope', 'ax_ay_share_of_units'):.1%}")
st.caption("How to read this: a minority of items carries most of the volume, so modelling the high-volume, "
           "reasonably steady items covers most of what a planner stocks.")

st.markdown("#### The three files")
c1, c2 = st.columns(2)
with c1:
    figure("snapshot_sales.png", "Sales: one row per product and store, one column per day.")
with c2:
    figure("snapshot_prices.png", "Prices: one row per product, store and week.")
figure("snapshot_calendar.png", "Calendar: dates, weekdays, events and SNAP days for each state.")

st.markdown("#### Category summary")
st.caption("How to read this: one row per category. 'Stationary' is the share of AX/AY items whose ADF test "
           "rejects a unit root at 5% (their mean level is stable enough to forecast with standard models).")
show = cat.rename(columns={"category": "Category", "total_items": "Items", "items_with_leading_zeros": "Items with >30 leading zero days",
                           "pct_items_with_leading_zeros": "% with leading zeros", "ax_ay_candidates": "AX/AY items kept",
                           "pct_stationary": "% stationary (ADF, 5%)"}).drop(columns=["items_tested"])
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"Items": st.column_config.NumberColumn(format="%d"),
                            "Items with >30 leading zero days": st.column_config.NumberColumn(format="%d"),
                            "AX/AY items kept": st.column_config.NumberColumn(format="%d"),
                            "% with leading zeros": st.column_config.NumberColumn(format="%.1f%%"),
                            "% stationary (ADF, 5%)": st.column_config.NumberColumn(format="%.1f%%")})

st.markdown("#### Method in one paragraph each")
st.markdown(
    "- **Break detection.** Weekly sales; the first 8 weeks after launch are skipped; the next 52 weeks set a "
    "baseline. A two-sided CUSUM then watches for a lasting shift. Its threshold is calibrated so that only about "
    "5% of simulated break-free series raise a false alarm. PELT, a second method, is a cross-check.\n"
    "- **Forecasting.** Seasonal naive (baseline), ETS, a global LightGBM and TSB (for intermittent items) were "
    "backtested at 7 rolling origins, 28 days apart, with 6 used for evaluation. At each origin, breaks are "
    "detected only from data up to that date (no lookahead).\n"
    "- **Inventory.** A daily-review reorder-point policy with lost sales was simulated on the 168 evaluation "
    "days. Safety stock was estimated from past forecast errors only.")

insight(f"A minority of products carries most of the volume: the {n_axay} high-volume, reasonably steady AX/AY products "
        f"are {value(f, 'scope', 'ax_ay_share_of_items'):.0%} of California's products but "
        f"{value(f, 'scope', 'ax_ay_share_of_units'):.0%} of the units sold. Modelling them covers most of what a planner "
        f"stocks, and every one of them has at least 500 days of sales history to learn from.")
