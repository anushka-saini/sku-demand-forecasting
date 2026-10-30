import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.derive import all_item_scores, eval_origins, in_window, sample_forecasts, sample_metrics
from utils.models import BEST, BEST_SHORT, COLORS, NAMES, ORDER, SAMPLE_DUPLICATES, SHORT, name
from utils.ui import ACCENT, DARK, NEUTRAL, apply_item_filters, chart, download, empty, filters, filters_used, kpis, page, style, tells, vline

flt = filters()
items = get("items")
fd_all = get("forecast_daily")
weekly = get("sales_weekly")
bo = get("breaks_by_origin")
sample = get("classical_sample")
cmp_fc = sample_forecasts()
cmp_m = sample_metrics()
cov = get("coverage")
item_cov = get("item_coverage")

page("Item explorer", "How does the forecast look for one item, and how does that item compare with the rest?")
filters_used(["category", "segment", "shift", "window"])

pool = apply_item_filters(items, flt).sort_values("total_sales", ascending=False)
if pool.empty:
    st.info("No items match the sidebar filters. Widen the category, item group or shift group.")
    st.stop()
ids = pool["item_id"].tolist()
if st.session_state.get("ie_item") not in ids:        # the filters removed the last item picked
    st.session_state["ie_item"] = ids[0]
item = st.selectbox("Item (type to search; sorted by units sold)", ids, key="ie_item")
info = items.set_index("item_id").loc[item]
in_sample = item in set(sample["item_id"])
# models run on the 150-item sample only; on the sample the best model is the 07h contender it repeats
SAMPLE_MODELS = [m for m in ["sarima", "sarimax", "theta", "prophet", "autots", "ets_boxcox", "combination", "hw_add",
                             "hw_mul", "arima"] if m not in SAMPLE_DUPLICATES.values()]

fd = in_window(fd_all[fd_all["item_id"] == item], flt["window"])
models = [m for m in [BEST, "ets", "seasonal_naive", "tsb"] if m in set(fd["model"])]
if in_sample:
    models += [m for m in SAMPLE_MODELS if m in set(cmp_fc["model"])]
if st.session_state.get("ie_model") not in models:
    st.session_state["ie_model"] = models[0]
c1, c2 = st.columns([2, 1])
model = (c1.radio if len(models) <= 5 else c1.selectbox)("Forecasting method", models, format_func=lambda m: SHORT[m],
                                                         key="ie_model", **({"horizontal": True} if len(models) <= 5 else {}))
can_retrain = model in set(fd.loc[fd["variant"] == "adaptive", "model"])
retrained = c2.toggle("Version retrained after a shift", value=False, key="ie_retrained", disabled=not can_retrain,
                      help="Retrains the model only on sales after the latest detected shift. Only differs for items with a shift.")
variant = "adaptive" if (retrained and can_retrain) else "static"
if not in_sample:
    st.caption("SARIMA, SARIMAX, Theta, Prophet, AutoTS, ARIMA and Holt-Winters were run on a 150-item sample; this "
               "item is not in it.")

# scores for this item, and its rank among all items for the chosen static method where every item has one
allsc = all_item_scores(flt["window"])
top = allsc[(allsc["model"] == BEST) & (allsc["variant"] == "static")].set_index("item_id")["mase"]
rank = int(top.rank(method="min").get(item, np.nan)) if item in top.index else None
mine = allsc[allsc["item_id"] == item]
ibo = bo[(bo["item_id"] == item) & bo["is_eval_origin"]].sort_values("origin")
shift_known = bool(ibo["has_break"].any())
# a retrained version only differs from the static one when a shift was known; seasonal naive has no retrained version
item_rows = [{"method": NAMES[m] + (", retrained after a shift" if v == "adaptive" else ""), "MASE": r["mase"], "MAE": r["mae"]}
             for (m, v), r in mine.set_index(["model", "variant"]).iterrows()
             if v == "static" or (v == "adaptive" and shift_known and m != "seasonal_naive")]
if in_sample:
    cm = in_window(cmp_m[(cmp_m["item_id"] == item) & cmp_m["origin"].isin(set(eval_origins()))], flt["window"])
    for m, g in cm[cm["model"].isin(SAMPLE_MODELS)].groupby("model"):
        item_rows.append({"method": NAMES[m], "MASE": g["mase"].mean(), "MAE": g["mae"].mean()})
et = pd.DataFrame(item_rows).drop_duplicates("method").sort_values("MASE")

if model in set(fd["model"]):
    f = fd[(fd["model"] == model) & (fd["variant"] == variant)].sort_values("date")
    has_band = True
else:
    f = in_window(cmp_fc[(cmp_fc["item_id"] == item) & (cmp_fc["model"] == model)], flt["window"]).sort_values("date")
    f = f[f["origin"].isin(set(eval_origins()))]
    has_band = False
chosen = et[et["method"] == NAMES[model] + (", retrained after a shift" if variant == "adaptive" else "")]
chosen_mase = chosen["MASE"].iloc[0] if len(chosen) else np.nan
kpis([
    {"label": "Average sales", "value": f"{info['mean_sales']:,.1f}", "compare": f"units a day · {info['category'].title()} · {info['segment']}"},
    {"label": f"Error, {SHORT[model]}", "value": f"{chosen_mase:.3f}" if pd.notna(chosen_mase) else "n/a",
     "compare": "MASE over the forecast dates in view", "help": "Average error divided by the error of repeating last week. Lower is better."},
    {"label": f"Rank among all items ({BEST_SHORT})", "value": f"{rank:,} of {len(top):,}" if rank else "n/a",
     "compare": f"lower error than {1 - rank / len(top):.0%} of items" if rank else "",
     "help": f"1 = the item {BEST_SHORT} forecasts best. Uses {BEST_SHORT} because it was run on every item."},
    {"label": "Detected shift", "value": "Yes" if shift_known else "No",
     "compare": (f"latest change week of {ibo.loc[ibo['has_break'], 'last_change_week'].iloc[-1]:%d %b %Y}" if shift_known
                 else "none found by any forecast date")},
])

if f.empty:
    st.plotly_chart(empty("No forecasts are saved for this item and method in the chosen dates."), width="stretch")
else:
    fig = go.Figure()
    if has_band:
        fig.add_scatter(x=pd.concat([f["date"], f["date"][::-1]]), y=pd.concat([f["hi_90"], f["lo_90"][::-1]]), fill="toself",
                        fillcolor="rgba(31,111,139,0.14)", line=dict(width=0), name="prediction interval", hoverinfo="skip")
    fig.add_scatter(x=f["date"], y=f["actual"], mode="lines+markers", name="actual sales", line=dict(color=DARK, width=1),
                    marker=dict(size=3), hovertemplate="%{x|%a %d %b %Y}: %{y:,.0f} sold<extra></extra>")
    hover = ("%{x|%a %d %b %Y}: forecast %{y:,.1f} (interval %{customdata[0]:,.1f} to %{customdata[1]:,.1f})<extra></extra>"
             if has_band else "%{x|%a %d %b %Y}: forecast %{y:,.1f}<extra></extra>")
    fig.add_scatter(x=f["date"], y=f["forecast"], mode="lines", name=f"{SHORT[model]} forecast",
                    line=dict(color=COLORS[model], width=2), customdata=f[["lo_90", "hi_90"]] if has_band else None,
                    hovertemplate=hover)
    for o in sorted(f["origin"].unique()):
        fig.add_vline(x=o, line_width=0.8, line_dash="dot", line_color=NEUTRAL)
    style(fig, height=400).update_yaxes(title="units per day", tickformat=",")
    fig.update_xaxes(title="date")
    inside = ((f["actual"] >= f["lo_90"]) & (f["actual"] <= f["hi_90"])).mean() if has_band else np.nan
    allcov = cov[(cov["model"] == BEST) & (cov["variant"] == "static") & (cov["level"] == 90) & (cov["origins"] == "origins 2-6")]
    band_note = (f" The shaded band is a 90% interval from the item's own past errors; here it held the actual sales on "
                 f"{inside:.0%} of days, and across all items on {allcov['coverage_before'].iloc[0]:.1%}."
                 if has_band and len(allcov) else " No interval is saved for this method.")
    miss = (f["forecast"] - f["actual"]).sum() / max(f["actual"].sum(), 1)
    chart(fig, f"{item}: {SHORT[model]} forecast against what sold",
          f"Each dotted line is a forecast date; the forecast covers the next 28 days. Over this period the forecast was "
          f"{miss:+.0%} against actual sales in total." + band_note)

hist = weekly[weekly["item_id"] == item].sort_values("week_start")
fig = go.Figure(go.Scatter(x=hist["week_start"], y=hist["units"], mode="lines", line=dict(color=ACCENT, width=1.2),
                           name="weekly sales", hovertemplate="week of %{x|%d %b %Y}: %{y:,.0f} units<extra></extra>"))
if not f.empty:
    fig.add_vrect(x0=f["date"].min(), x1=f["date"].max(), fillcolor=NEUTRAL, opacity=0.12, line_width=0)
if pd.notna(info["first_break_week"]):
    vline(fig, info["first_break_week"], " first shift", color="#B26B00", dash="dash")
if pd.notna(info["last_break_week"]) and info["last_break_week"] != info["first_break_week"]:
    vline(fig, info["last_break_week"], " latest shift", color="#B26B00", dash="dot")
style(fig, height=280, legend=False).update_yaxes(title="units per week", tickformat=",")
fig.update_xaxes(title="week")
chart(fig, f"Sales history of {item}" + (" with its detected shifts" if pd.notna(info["first_break_week"]) else ""),
      ("Dashed lines mark where the detector dated a lasting change in the sales level. " if pd.notna(info["first_break_week"])
       else "The detector found no lasting change in this item's sales level. ")
      + "The grey band is the forecast period shown above.")

st.markdown("<div class='chart-title'>This item's error by method</div>", unsafe_allow_html=True)
st.dataframe(et, hide_index=True, width="stretch", column_config={"MASE": st.column_config.NumberColumn(format="%.3f"),
                                                                    "MAE": st.column_config.NumberColumn("MAE (units a day)", format="%.2f")})
c1, c2 = st.columns(2)
with c1:
    download(et, f"{item}_errors.csv", "Download these errors as CSV", key="dl_item_err")
with c2:
    if not f.empty:
        download(f.drop(columns=[c for c in ["category"] if c in f]), f"{item}_{model}_forecast.csv",
                 "Download the forecast as CSV", key="dl_item_fc")

if et.empty:
    tells(f"No scores are saved for {item} in the chosen forecast dates.")
else:
    first = et.iloc[0]
    text = f"For {item}, the most accurate method is {first['method']} (MASE {first['MASE']:.3f}). "
    if rank:
        text += (f"With {BEST_SHORT} it ranks {rank:,} of {len(top):,} items, so it is "
                 f"{'easier' if rank <= len(top) / 2 else 'harder'} to forecast than most. ")
    text += ("A shift had been detected by at least one forecast date." if shift_known else
             "No shift had been detected by any forecast date.")
    tells(text)
