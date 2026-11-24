import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.final import FC_FILES, TABLES, fc_index, load, need_results
from utils.ui import chart, download, ink, kpis, page, style, tells

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

download(t.reset_index(names="Category"), "data_categories.csv", key="dl_data_cat")

# how the study items were chosen
st.markdown("#### How the 726 products were chosen: ABC and XYZ classes")
st.markdown("Products are classed inside each category, on their sales from their first sale on (leading zeros trimmed), with the four stores added up.\n"
            "- **Volume (ABC):** rank products by total units sold. **A** is the products that together make up the first 70% of the category's units, "
            "**B** the next 20% (70% to 90%), **C** the last 10%.\n"
            "- **Variability (XYZ):** the coefficient of variation of daily sales (standard deviation over the mean). **X** is 0.5 or below, "
            "**Y** is above 0.5 up to 1.0, **Z** is above 1.0.\n"
            "- **Kept:** classes **AX** and **AY**, the high-volume products with low to medium variability.")
ab = items.assign(abc_class=items["abc_class"].fillna("?"), xyz_class=items["xyz_class"].fillna("?"))
cp = st.radio("Show", ["All categories"] + [c.title() for c in cats], horizontal=True, key="data_abc")
ab = ab if cp == "All categories" else ab[ab["category"] == cp.upper()]
grid = ab.groupby(["abc_class", "xyz_class"]).size().unstack(fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
grid.index = [f"{c} (volume)" for c in grid.index]
grid.columns = [f"{c} (variability)" for c in grid.columns]
st.dataframe(grid.style.apply(lambda col: ["background-color: rgba(14,124,134,0.28)" if (i < 1 and col.name[0] in "XY") else "" for i in range(len(col))]),
             width="stretch")
kept = ab[ab["segment"].isin(["AX", "AY"])]
st.caption(f"Products per class. The shaded cells (AX and AY) are the study set: {len(kept)} of {len(ab)} products, "
           f"{kept['total_sales'].sum() / ab['total_sales'].sum():.0%} of the units. The classes use the whole sales history, including the test months, "
           "so the choice of products is not strictly out of sample.")
download(grid.reset_index(names="ABC"), "data_abc_xyz.csv", key="dl_data_abc")

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
fig = go.Figure(go.Scatter(x=w["week_start"], y=w["units"], mode="lines", line=dict(color=ink(), width=1.6), name="weekly units"))
for a, z, c, lab, pos in [(w["week_start"].min(), tr_end, "rgba(128,128,128,0.12)", "training", "top left"),
                          (va0, va1, "rgba(224,160,64,0.30)", "validation", "bottom left"), (te0, te1, "rgba(14,124,134,0.30)", "test", "top left")]:
    fig.add_vrect(x0=a, x1=z, fillcolor=c, line_width=0, annotation_text=lab, annotation_position=pos, annotation_font_color=ink())
fig.update_yaxes(title="units a week, all study products", rangemode="tozero")
fig.update_xaxes(title="week")
chart(style(fig, height=380, legend=False), f"Weekly sales, {pick.lower()} products: training, validation and test",
      "Models learn on the training part, are tuned on the validation part, and are scored once on the test part.", key="data_series")

sp = split.copy()
sp.columns = ["Part", "Forecast dates", "First day", "Last day"]
st.dataframe(sp, hide_index=True, width="stretch")
download(sp, "data_split.csv", key="dl_data_split")
st.caption("Each forecast looks 28 days ahead from a forecast date. Sales before that date are all a model may see.")
tells(f"{len(final_items)} of the {len(scope)} study products are compared in full, and about {zall:.0%} of their item-days are zero. "
      "Zero days are kept as they are, so some may be stock-outs rather than lack of demand.")
