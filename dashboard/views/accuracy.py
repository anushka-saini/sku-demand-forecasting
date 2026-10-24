import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.derive import METRIC_HELP, METRICS, horizon_scores, leaderboard, sample_scores
from utils.models import COLORS, ORDER, SHIFT_AWARE, SHORT, family, name
from utils.ui import SHIFT_GROUPS, chart, download, filters, filters_used, kpis, page, style, tells

flt = filters()
page("Forecast accuracy", "Which forecasting method is most accurate on the same test, and where do the simple ones win?")
filters_used(["category", "shift", "window"])

metric = st.radio("Error measure", list(METRICS), horizontal=True, key="acc_metric",
                  help="Lower is better for every measure.")
st.caption(f"{metric}: {METRIC_HELP[metric]}")
col = METRICS[metric]

scores = sample_scores(flt["window"])
if flt["categories"]:
    scores = scores[scores["category"].isin(flt["categories"])]
if flt["shift"] == SHIFT_GROUPS[1]:
    scores = scores[scores["break_item"]]
elif flt["shift"] == SHIFT_GROUPS[2]:
    scores = scores[~scores["break_item"]]
if scores.empty:
    st.info("No items match the filters. Widen the category or shift group in the sidebar.")
    st.stop()

lb = leaderboard(scores, metric)
few = lb[lb["n_items"] < 0.9 * lb["n_items"].max()]          # e.g. TSB, which only runs on items that often sell nothing
lb = lb[lb["n_items"] >= 0.9 * lb["n_items"].max()]
static = lb[lb["model"].isin(ORDER)].sort_values("value")
best, base = static.iloc[0], lb.set_index("model").loc["seasonal_naive", "value"]
second = static.iloc[1]
n_items = int(scores["item_id"].nunique())
wide = scores[scores["model"].isin(ORDER)].pivot_table(index="item_id", columns="model", values=col)
winner = wide.reindex(columns=ORDER).dropna(axis=1, how="all").idxmin(axis=1)
simple_win = (winner != "lightgbm").mean()
kpis([
    {"label": "Most accurate method", "value": SHORT[best["model"]], "compare": f"typical {metric} {best['value']:.3f}"},
    {"label": "Baseline: repeat last week", "value": f"{base:.3f}", "compare": f"{base / best['value'] - 1:.0%} more error than the best",
     "help": "Seasonal naive: next week's sales are forecast to equal last week's."},
    {"label": "Gap to the runner-up", "value": f"{second['value'] - best['value']:.3f}",
     "compare": f"{SHORT[second['model']]} is second ({(second['value'] - best['value']) / best['value']:.1%} behind)"},
    {"label": "Items where a simpler method is best", "value": f"{simple_win:.0%}",
     "compare": f"of {n_items:,} items, on {metric}"},
])

# leaderboard: every model and version that has this measure saved
lb = lb.assign(label=lb["model"].map(name), kind=lb["model"].map(family)).sort_values("value", ascending=False)
fig = go.Figure(go.Bar(x=lb["value"], y=lb["label"], orientation="h",
                       marker=dict(color=[COLORS.get(m, "#888") for m in lb["model"]],
                                   opacity=[0.55 if m in SHIFT_AWARE else 1 for m in lb["model"]]),
                       customdata=np.c_[lb["n_items"], lb["kind"], lb["beats_baseline"].fillna(-1)],
                       text=[f"{v:.3f}" for v in lb["value"]], textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}<br>" + metric + " %{x:.3f}<br>%{customdata[1]}, %{customdata[0]} items<extra></extra>"))
fig.add_vline(x=base, line_dash="dot", line_color="#8C8C8C")
style(fig, height=110 + 30 * len(lb), legend=False).update_xaxes(title=f"typical {metric} (median over items, lower is better)",
                                                                  range=[0, lb["value"].max() * 1.15])
missing = [name(m) for m in SHIFT_AWARE + ORDER if m not in set(lb["model"]) and m not in set(few["model"])]
chart(fig, f"{SHORT[best['model']]} is most accurate on {metric}, {second['value'] - best['value']:.3f} ahead of {SHORT[second['model']]}",
      f"Same {n_items:,} items and forecast dates for every bar. Paler bars are versions that react to a detected shift; "
      f"the dotted line is the baseline. Each bar is the typical item's error; the change for the same item is on the "
      f"Demand shifts page." + (f" Not saved for {metric}: {', '.join(missing)}." if missing else "")
      + "".join(f" {name(r.model)} runs on only {r.n_items} of these items and is left out." for r in few.itertuples()))

table = lb.sort_values("value")[["label", "kind", "value", "beats_baseline", "n_items"]].rename(
    columns={"label": "Method", "kind": "Type", "value": f"Typical {metric}", "beats_baseline": "Beats the baseline on",
             "n_items": "Items"})
table[f"vs baseline"] = table[f"Typical {metric}"] / base - 1
st.dataframe(table, hide_index=True, width="stretch",
             column_config={f"Typical {metric}": st.column_config.NumberColumn(format="%.3f"),
                            "Beats the baseline on": st.column_config.NumberColumn(format="percent"),
                            "vs baseline": st.column_config.NumberColumn(format="percent"),
                            "Items": st.column_config.NumberColumn(format="%d")})
download(table, f"forecast_accuracy_{col}.csv", key="dl_leader")

mape_skip = get("classical_item_metrics")["mape_excluded"].mean() if "mape_excluded" in get("classical_item_metrics") else np.nan
st.caption(f"Which measure to trust: we rank on MASE, because it has no units and compares items of any size. MAE and RMSE "
           f"are in units, so big sellers dominate them. MAPE has to skip the {mape_skip:.1%} of days with no sales, "
           f"which makes it unreliable here.")

# error by horizon
hz = horizon_scores(flt["window"])
sample = get("classical_sample")
hz = hz.merge(sample[["item_id", "break_item"]], on="item_id")
if flt["categories"]:
    hz = hz[hz["category"].isin(flt["categories"])]
if flt["shift"] == SHIFT_GROUPS[1]:
    hz = hz[hz["break_item"]]
elif flt["shift"] == SHIFT_GROUPS[2]:
    hz = hz[~hz["break_item"]]
hm = hz.groupby(["model", "week"])[col].median().unstack("week")
fig = go.Figure()
for m in ORDER:
    if m in hm.index:
        fig.add_scatter(x=[f"days {7 * (w - 1) + 1}-{7 * w}" for w in hm.columns], y=hm.loc[m], mode="lines+markers",
                        name=SHORT[m], line=dict(color=COLORS[m], dash="dot" if m == "seasonal_naive" else "solid"),
                        hovertemplate="%{x}: " + metric + " %{y:.3f}<extra>" + SHORT[m] + "</extra>")
style(fig, height=340).update_yaxes(title=f"typical {metric}")
fig.update_xaxes(title="how far ahead (days after the forecast date)")
rise = (hm.iloc[:, -1] - hm.iloc[:, 0]) / hm.iloc[:, 0]
grows = bool((rise > 0).all())
chart(fig, ("Error grows the further ahead the forecast looks" if grows else "Error does not simply grow with the horizon"),
      f"From the first to the fourth week of the forecast, {SHORT[rise.idxmax()]}'s error changes most ({rise.max():+.0%}) and "
      f"{SHORT[rise.idxmin()]}'s least ({rise.min():+.0%}). Static methods only; same items as above.")

# error by item group
st.markdown("<div class='chart-title'>Error by item group</div>", unsafe_allow_html=True)
grp = scores[scores["model"].isin(ORDER)].copy()
rows = []
for gname, mask in [("Foods", grp["category"] == "FOODS"), ("Hobbies", grp["category"] == "HOBBIES"),
                    ("Household", grp["category"] == "HOUSEHOLD"), ("With a detected shift", grp["break_item"]),
                    ("Without a detected shift", ~grp["break_item"])]:
    g = grp[mask]
    if g["item_id"].nunique() == 0:
        continue
    med = g.groupby("model")[col].median()
    rows.append({"Item group": gname, "Items": g["item_id"].nunique(), "Best method": SHORT[med.idxmin()],
                 **{SHORT[m]: med.get(m, np.nan) for m in ORDER}})
gt = pd.DataFrame(rows)
st.dataframe(gt, hide_index=True, width="stretch",
             column_config={SHORT[m]: st.column_config.NumberColumn(format="%.3f") for m in ORDER})
st.markdown(f"<div class='takeaway'>Typical {metric} per group. The best method changes with the group: "
            + "; ".join(f"{r['Item group']}: {r['Best method']}" for r in rows) + ".</div>", unsafe_allow_html=True)
download(gt, f"accuracy_by_group_{col}.csv", key="dl_groups")

# where the simple models win
counts = winner.map(SHORT).value_counts()
fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(), marker_color=[COLORS[k] for k in
                       [next(m for m in ORDER if SHORT[m] == lab) for lab in counts.index]],
                       text=counts.to_numpy(), textposition="outside", cliponaxis=False,
                       hovertemplate="%{x}: best for %{y} items<extra></extra>"))
style(fig, height=300, legend=False).update_yaxes(title="items where it is the most accurate", range=[0, counts.max() * 1.2])
fig.update_xaxes(title="method")
beats = {m: (wide[m] < wide["lightgbm"]).mean() for m in ["sarima", "ets"] if m in wide}
chart(fig, f"LightGBM is the most accurate for {counts.get('LightGBM', 0)} of {len(winner):,} items; classical methods for the rest",
      "Simple methods win often: " + ", ".join(f"{SHORT[m]} beats LightGBM on {v:.0%} of items" for m, v in beats.items()) + ".")

with st.expander("All 726 items: the methods run on every item"):
    mu = get("mase_unrounded")
    a = mu[mu["category"] == "ALL"].set_index("model")
    pw = get("pairwise")
    le = pw[(pw["model_a"] == "lightgbm") & (pw["model_b"] == "ets") & (pw["category"] == "ALL")].iloc[0]
    tsb = a.loc["tsb"]
    st.markdown(f"Typical MASE on all {int(a.loc['lightgbm', 'n_items']):,} items: LightGBM {a.loc['lightgbm', 'median_item_mase']:.3f}, "
                f"ETS {a.loc['ets', 'median_item_mase']:.3f}, seasonal naive {a.loc['seasonal_naive', 'median_item_mase']:.3f}. "
                f"LightGBM is better than ETS for {le['share_improved']:.0%} of items. TSB, built for items that often sell "
                f"nothing, runs only on the {int(tsb['n_items'])} such items and scores {tsb['median_item_mase']:.3f}.")

with st.expander("In progress: more advanced models"):
    wip = get("advanced_wip")
    n_wip = get("advanced_wip_items")["item_id"].nunique()
    new = [m for m in wip["model"] if m not in ORDER and m != "tsb"]
    names = {"xgboost": "XGBoost", "prophet": "Prophet", "gru": "a small neural network (GRU)"}
    st.markdown(f"Running now: {', '.join(names.get(m, m) for m in new)}, on {n_wip} items, with no tuning. "
                f"These numbers are not final and are not part of any result on this dashboard.")

vals = lb.set_index("model")["value"]
all_beat = bool((static.set_index("model")["value"].drop("seasonal_naive") < base).all())
base_of = {m: ("seasonal_naive" if m.startswith("seasonal") else m.split("_")[0]) for m in SHIFT_AWARE}
reacting = [(m, vals[m] - vals[base_of[m]]) for m in SHIFT_AWARE if m in vals and base_of[m] in vals]
better = [m for m, d in reacting if d < -0.0005]
tells(f"On {metric}, {SHORT[best['model']]} is the most accurate method ({best['value']:.3f})"
      + (f" and every fitted method beats the baseline ({base:.3f}). " if all_beat else f"; the baseline is {base:.3f}. ")
      + f"The lead is small: {SHORT[second['model']]} is {second['value'] - best['value']:.3f} behind, and a method other "
      f"than LightGBM is the best choice for {simple_win:.0%} of items. "
      + (f"Of the {len(reacting)} versions that react to shifts and have {metric} saved, none beats its static model by "
         f"more than 0.0005." if not better else
         f"{len(better)} of the {len(reacting)} versions that react to shifts beat their static model on {metric}: "
         + ", ".join(name(m) for m in better) + ". The Demand shifts page shows whether that gain is large enough to matter."))
