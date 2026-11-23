import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.final import FC_FILES, TABLES, fc_index, load, need_results
from utils.ui import chart, kpis, page, style, tells

need_results()
items = get("items")
split = load("split")
page("Data", "What is in the sample, and how is the sales history split into training, validation and test?")
st.caption("Daily unit sales of Walmart products in the four California stores, added together per product. "
           "The 726 high-volume products are the study set; the final model comparison runs on a sample of 150 of them.")

scope = items[items["in_scope"]]
idx = fc_index()
final_items = idx["item_id"].unique()
cats = sorted(scope["category"].unique())


@st.cache_data(show_spinner=False)
def zero_share():
    """Share of item-days with no sales, per category, over the days the 150 items were scored (Aug 2015 to May 2016)."""
    p = TABLES / FC_FILES["sales"]
    d = pd.read_parquet(p, columns=["item_id", "sales"])
    d["cat"] = d["item_id"].str.split("_").str[0]
    return (d["sales"] == 0).groupby(d["cat"]).mean(), float((d["sales"] == 0).mean())


zc, zall = zero_share()
kpis([
    {"label": "Products in the study", "value": f"{len(scope):,}", "compare": f"of {len(items):,} in California"},
    {"label": "Products in the final comparison", "value": f"{len(final_items)}", "compare": "a sample of the study set"},
    {"label": "Categories", "value": f"{len(cats)}", "compare": ", ".join(c.title() for c in cats)},
    {"label": "Days with no sales", "value": f"{zall:.1%}", "compare": "of item-days, final 150 items, Aug 2015 on",
     "help": "Days where a product sold nothing. They are kept as zero in every model."},
])

t = scope.groupby("category").size().rename("Study products").to_frame()
t["Final comparison"] = pd.Series([i.split("_")[0] for i in final_items]).value_counts()
t["Days with no sales"] = zc
t.index = [c.title() for c in t.index]
st.dataframe(t.style.format({"Days with no sales": "{:.1%}"}), width="stretch")
st.caption("Days with no sales are measured on the 150 final items from August 2015 on, the period the forecasts cover.")

# weekly sales of the 726 items, with the three parts of the history shaded
pick = st.radio("Category", ["All"] + [c.title() for c in cats], horizontal=True, key="data_cat")
w = get("sales_weekly").merge(scope[["item_id", "category"]], on="item_id")
if pick != "All":
    w = w[w["category"] == pick.upper()]
w = w.groupby("week_start")["units"].sum().reset_index()
w = w[w["week_start"] > w["week_start"].min() + pd.Timedelta(days=14)]          # the first weeks are still ramping up
s = split.set_index("part")
tr_end = pd.Timestamp(s.loc["train", "last_day"])
va0, va1 = pd.Timestamp(s.iloc[1]["first_day"]), pd.Timestamp(s.iloc[3]["last_day"])
te0, te1 = pd.Timestamp(s.iloc[4]["first_day"]), pd.Timestamp(s.iloc[4]["last_day"])
fig = go.Figure(go.Scatter(x=w["week_start"], y=w["units"], mode="lines", line=dict(color="#222222", width=1.6), name="weekly units"))
for a, z, c, lab in [(w["week_start"].min(), tr_end, "rgba(128,128,128,0.10)", "training"),
                     (va0, va1, "rgba(230,159,0,0.22)", "validation"), (te0, te1, "rgba(0,114,178,0.20)", "test")]:
    fig.add_vrect(x0=a, x1=z, fillcolor=c, line_width=0, annotation_text=lab, annotation_position="top left")
fig.update_yaxes(title="units a week, all study products", rangemode="tozero")
fig.update_xaxes(title="week")
chart(style(fig, height=380, legend=False), f"Weekly sales, {pick.lower()} products: training, validation and test",
      "Models learn on the grey part, are tuned on the validation part, and are scored once on the test part.", key="data_series")

sp = split.copy()
sp.columns = ["Part", "Forecast dates", "First day", "Last day"]
st.dataframe(sp, hide_index=True, width="stretch")
st.caption("Each forecast looks 28 days ahead from a forecast date. Sales before that date are all a model may see.")
tells(f"{len(final_items)} of the {len(scope)} study products are compared in full, and about {zall:.0%} of their item-days are zero. "
      "Zero days are kept as they are, so some may be stock-outs rather than lack of demand.")
