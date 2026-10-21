import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import get
from utils.m2 import COLORS, ORDER, name
from utils.ui import assumed, chart, header, style

header("Mid-sem demo walkthrough",
       "One item from raw data to a stock decision, in seven steps. Pick any of the 150 items in the Milestone 2 sample; "
       "every number below is read from the notebook outputs.")

sample = get("m2_sample")
raw = get("m2_demo_series")
decomp = get("m2_demo_decomp")
stat = get("m2_stationarity")
orders = get("m2_orders")
fc = get("m2_forecasts")
im = get("m2_item_metrics")
stock = get("m2_stock_items")

items = sorted(sample["item_id"])
# Start on a typical item: the one whose SARIMA error is closest to the sample median
sar = im[(im["model"] == "sarima") & (im["origin"] > im["origin"].min())].groupby("item_id")["mase"].mean()
typical = (sar - sar.median()).abs().idxmin()
item = st.selectbox("Item", items, index=items.index(typical), key="demo_item",
                    help="Starts on the item with the median SARIMA error; pick any other.")
info = sample[sample["item_id"] == item].iloc[0]
st.caption(f"{info['category']} · {'a break was detected at some origin' if info['break_item'] else 'no break detected'}")

# 1. raw series
r = raw[raw["item_id"] == item]
lead = int((~r["kept_after_trim"]).sum())
st.markdown("### 1. Raw daily sales (four California stores summed)")
fig = go.Figure()
fig.add_scatter(x=r["date"], y=r["sales_raw"], mode="lines", line=dict(color="#8C8C8C", width=0.8), name="raw")
chart(style(fig, legend=False), f"{lead} days of zeros before the first sale" if lead else "Sold from the first day of the data",
      "Grey is the raw file. Zeros before a product's first sale mean it was not on sale yet, not zero demand.")

# 2. trimmed series
st.markdown("### 2. Trimmed series (what the models see)")
k = r[r["kept_after_trim"]]
fig = go.Figure()
fig.add_scatter(x=k["date"], y=k["sales_raw"], mode="lines", line=dict(color="#0072B2", width=0.8), name="trimmed")
chart(style(fig, legend=False), f"{len(k):,} days from the first sale on {k['date'].min().date()}",
      "Zeros after the first sale are kept: they are real (stockouts, slow days).")

# 3. decomposition
st.markdown("### 3. Decomposition (STL, weekly period, last 52 weeks)")
d = decomp[decomp["series"] == item]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["trend", "weekly season", "residual"], vertical_spacing=0.08)
for row, part in enumerate(["trend", "seasonal", "resid"], start=1):
    fig.add_scatter(x=d["date"], y=d[part], mode="lines", line=dict(color="#0072B2", width=1.2), row=row, col=1, showlegend=False)
swing = d["seasonal"].tail(7)
chart(style(fig, height=460, legend=False), f"The weekly pattern swings by about {swing.max() - swing.min():.1f} units a day",
      "Trend = slow level; season = the repeating weekly shape; residual = what is left.")

# 4. stationarity
st.markdown("### 4. Stationarity verdict")
s = stat[stat["item_id"] == item].iloc[0]
o = orders[orders["series"] == item].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Verdict (ADF + KPSS)", s["label"])
c2.metric("ADF p (constant)", f"{s['adf_c']:.4f}", help="Below 0.05: reject a unit root")
c3.metric("KPSS p (constant)", f"{s['kpss_c']:.3f}", help="Below 0.05: reject stationarity")
c4.metric("Differencing at origin 0", f"d = {o['d']}, D = {o['D']}")
st.markdown(f"Chosen orders (lowest AICc at origin 0): **SARIMA {o['sarima_order']}{o['sarima_sorder']}**, "
            f"**ARIMA {o['arima_order']}**.")

# 5. forecasts
st.markdown("### 5. Forecasts from each model")
origins = sorted(fc["origin"].unique())[1:]
origin = st.select_slider("Forecast origin", origins, value=origins[-1], format_func=lambda x: str(x.date()), key="demo_origin")
g = fc[(fc["item_id"] == item) & (fc["origin"] == origin)]
fig = go.Figure()
a = g[g["model"] == "sarima"].sort_values("date")
fig.add_scatter(x=a["date"], y=a["actual"], mode="lines+markers", name="actual", line=dict(color="#7F7F7F", width=2.5))
for m in ["seasonal_naive", "hw_add", "sarima", "lightgbm"]:
    x = g[g["model"] == m].sort_values("date")
    fig.add_scatter(x=x["date"], y=x["forecast"], mode="lines", name=name(m), line=dict(color=COLORS[m]))
chart(style(fig), f"28 days ahead from {origin.date()}", "Each model only saw data up to the origin.")

# 6. metrics
st.markdown("### 6. Error metrics (mean over origins 1-6)")
e = im[(im["item_id"] == item) & im["origin"].isin(origins)].groupby("model")[["rmse", "mae", "mape", "mase"]].mean()
e = e.reindex(ORDER).dropna(how="all")
best = e["mase"].idxmin()
st.markdown(f"**{name(best)}** has the lowest MASE for this item ({e.loc[best, 'mase']:.3f}).")
e.index = e.index.map(name)
st.dataframe(e, width="stretch", column_config={c: st.column_config.NumberColumn(c.upper(), format="%.3f") for c in e.columns})

# 7. stock
st.markdown("### 7. What it means for stock (95% target, 7-day lead time)")
assumed("Lead time, holding cost (25% a year) and lost-sale cost (0.5 × price) are assumptions.")
sk = stock[stock["item_id"] == item].pivot(index="model", columns="method", values="total_cost").reindex(
    ["seasonal_naive", "ets", "hw_add", "sarima", "lightgbm"])
fr = stock[stock["item_id"] == item].pivot(index="model", columns="method", values="fill_rate")
fig = go.Figure()
for method, dash in [("normal formula", None), ("conformal", "/")]:
    fig.add_bar(x=[name(m) for m in sk.index], y=sk[method], name=f"{method} safety stock",
                marker=dict(color=[COLORS[m] for m in sk.index], pattern_shape=dash),
                customdata=fr.reindex(sk.index)[method],
                hovertemplate="%{x}: $%{y:,.0f}, fill rate %{customdata:.1%}<extra>" + method + "</extra>")
cheap = sk["conformal"].idxmin()
chart(style(fig), f"With calibrated safety stock, {name(cheap)} is cheapest for this item (${sk.loc[cheap, 'conformal']:,.0f})",
      "Simulated holding + lost-sale cost over 168 days. Hatched bars use safety stock sized from past forecast errors "
      "(conformal) instead of the textbook formula.")
