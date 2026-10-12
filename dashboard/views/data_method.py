import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.ui import ACCENT, NEUTRAL, chart, header, style, vline

cat = get("category_summary")
lz = get("leading_zero_example")
items = get("items")

header("Data & method",
       "From raw Walmart sales to an inventory policy in seven steps. Each step's item count comes from the result files.")

n_total = int(cat["total_items"].sum())
n_lz = int(cat["items_with_leading_zeros"].sum())
n_axay = int(cat["ax_ay_candidates"].sum())

st.markdown("#### Pipeline")
steps = [("Raw M5 data", "sales, calendar, prices"),
         ("Scope", f"California, 4 stores summed · {n_total:,} items"),
         ("Trim pre-launch zeros", f"{n_lz:,} items had >30 leading zero days"),
         ("ABC-XYZ segmentation", f"{n_axay} AX/AY items kept"),
         ("Break detection", "weekly CUSUM, PELT cross-check"),
         ("Forecasting", "static vs adaptive vs placebo, 7 rolling origins"),
         ("Inventory policy", "safety stock and reorder point, 168 simulated days")]
boxes = "<span class='arrow'>→</span>".join(
    f"<div class='step'><b>{i}. {name}</b><span>{detail}</span></div>" for i, (name, detail) in enumerate(steps, 1))
st.markdown(f"<div class='pipeline'>{boxes}</div>", unsafe_allow_html=True)

# Leading-zero example
item = lz["item_id"].iloc[0]
removed = lz[~lz["kept_after_trim"]]
kept = lz[lz["kept_after_trim"]]
first_sale = kept["date"].min()
fig = go.Figure()
fig.add_trace(go.Scatter(x=removed["date"], y=removed["sales"], mode="lines", name="removed (not yet on sale)",
                         line=dict(color=NEUTRAL, width=1, dash="dot"),
                         hovertemplate="%{x|%Y-%m-%d}: %{y} units (removed)<extra></extra>"))
fig.add_trace(go.Scatter(x=kept["date"], y=kept["sales"], mode="lines", name="kept for analysis",
                         line=dict(color=ACCENT, width=1),
                         hovertemplate="%{x|%Y-%m-%d}: %{y} units<extra></extra>"))
vline(fig, first_sale, f" first sale {first_sale.date()}")
style(fig).update_yaxes(title="units per day (CA total)")
chart(fig, f"{item} showed {len(removed)} days of zero sales before it was ever sold, so they are not real demand",
      f"Grey dotted: days before the first sale, which were removed. Blue: the series used from then on. Treating "
      f"the grey part as demand would make the item look like it had a huge structural break at launch. Zeros "
      f"*after* the first sale (stockouts, delistings) are kept.")

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
