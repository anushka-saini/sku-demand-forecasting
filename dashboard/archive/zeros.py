import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.ui import ACCENT, NEUTRAL, chart, figure, header, insight, style, vline

header("Zeros and trimming",
       "Many products show months of zero sales before they were ever sold. Those days are not demand, so we cut them "
       "off before doing anything else.")

cat = get("category_summary")
lz = get("leading_zero_example")
quality = get("data_quality")

n_total = int(cat["total_items"].sum())
n_lz = int(cat["items_with_leading_zeros"].sum())
trimmed = int(quality.loc[quality["category"] == "ALL", "leading_zero_days_trimmed"].iloc[0])
c1, c2, c3 = st.columns(3)
c1.metric("Products with more than 30 zero days at the start", f"{n_lz:,}", help=f"of {n_total:,} California products")
c2.metric("Share of all products", f"{n_lz / n_total:.1%}")
c3.metric("Zero days removed from the 726 we model", f"{trimmed:,}")

item = lz["item_id"].iloc[0]
removed = lz[~lz["kept_after_trim"]]
kept = lz[lz["kept_after_trim"]]
first_sale = kept["date"].min()
fig = go.Figure()
fig.add_trace(go.Scatter(x=removed["date"], y=removed["sales"], mode="lines", name="removed (not on sale yet)",
                         line=dict(color=NEUTRAL, width=1, dash="dot"),
                         hovertemplate="%{x|%Y-%m-%d}: %{y} units (removed)<extra></extra>"))
fig.add_trace(go.Scatter(x=kept["date"], y=kept["sales"], mode="lines", name="kept", line=dict(color=ACCENT, width=1),
                         hovertemplate="%{x|%Y-%m-%d}: %{y} units<extra></extra>"))
vline(fig, first_sale, f" first sale {first_sale.date()}")
style(fig).update_yaxes(title="units per day (four California stores)")
fig.update_xaxes(title="date")
chart(fig, f"{item} had {len(removed)} days of zeros before it was ever sold",
      "Grey dotted: days before the first sale, which we remove. Blue: the series we keep. Zeros after the first sale "
      "(slow days, stockouts) stay in, because they are real.")

st.markdown("#### The same product decomposed before and after trimming")
c1, c2 = st.columns(2)
with c1:
    figure("foods_1_004_decomposition_before_trim.png", "Before: the trend climbs from zero, which looks like a huge shift at launch.")
with c2:
    figure("foods_1_004_decomposition.png", "After: the trend is flat and the weekly pattern is easy to see.")

st.markdown("#### How many products start with zeros, by category")
show = cat[["category", "total_items", "items_with_leading_zeros", "pct_items_with_leading_zeros", "ax_ay_candidates"]]
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"category": "Category", "total_items": "Products",
                            "items_with_leading_zeros": "More than 30 zero days at the start",
                            "pct_items_with_leading_zeros": st.column_config.NumberColumn("Share", format="%.1f%%"),
                            "ax_ay_candidates": "AX/AY products kept after trimming"})
with st.expander("When products first went on sale"):
    figure("first_sale_dates.png", "How many zero days products start with, and when they were first sold.")

insight(f"About half of California's products ({n_lz / n_total:.0%}) start with a long run of zeros, and products "
        f"launch all through the five years. Left in, those zeros look like a huge shift at launch and would mislead "
        f"both the break detector and every model, so every later step starts at each product's first sale.")
