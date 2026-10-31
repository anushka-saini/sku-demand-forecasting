import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import ROOT, get
from utils.derive import (METRIC_HELP, METRICS, TS_KEYS, eval_origins, horizon_scores, leaderboard, sample_forecasts,
                          sample_scores)
from utils.models import (ASSUMES, BEST, BEST_SHORT, BLOCKS, COLORS, CONTENDERS, PICKED, SAME_ON_SAMPLE,
                          SAMPLE_DUPLICATES, SHIFT_AWARE, SHORT, family, name)
from utils.ui import SHIFT_GROUPS, chart, download, filters, filters_used, kpis, page, style, tells, verdict_box

sys.path.append(str(ROOT / "src"))
from model_verdict import LABEL, TESTED, against, verdict  # noqa: E402  (pure pandas, shared with notebook 07i)

flt = filters()
page("Forecast accuracy", "Which forecasting model is most accurate on the same test, and where do the simple ones win?")
filters_used(["category", "shift", "window"])

names = list(METRICS)
metric = st.radio("Error measure", names, index=names.index("RMSE"), horizontal=True, key="acc_metric",
                  help="Lower is better for every measure. The model for the stock work was chosen on RMSE, before the run.")
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

lb_all = leaderboard(scores, metric)
few = lb_all[lb_all["n_items"] < 0.9 * lb_all["n_items"].max()]          # e.g. TSB, which only runs on items that often sell nothing
lb_all = lb_all[lb_all["n_items"] >= 0.9 * lb_all["n_items"].max()]
vals = lb_all.set_index("model")["value"]
cont = vals.reindex(CONTENDERS).dropna().sort_values()
best, second = cont.index[0], cont.index[1]
base = vals["seasonal_naive"]
rank = list(cont.index).index(PICKED) + 1
n_items = int(scores["item_id"].nunique())
wide = scores[scores["model"].isin(CONTENDERS)].pivot_table(index="item_id", columns="model", values=col)
winner = wide.reindex(columns=CONTENDERS).dropna(axis=1, how="all").idxmin(axis=1)
other_win = (winner != PICKED).mean()
nth = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
kpis([
    {"label": f"Lowest typical {metric} of the eight", "value": SHORT[best], "compare": f"typical {metric} {cont.iloc[0]:.3f}"},
    {"label": "Used for the stock work", "value": SHORT[PICKED],
     "compare": (f"also the lowest on {metric}" if best == PICKED else f"{nth[rank]} on {metric} ({cont[PICKED]:.3f})")
     + "; picked on RMSE before the run",
     "help": "07h picked the model with the lowest median RMSE on the scored dates, a rule written down before the run."},
    {"label": "Baseline: repeat last week", "value": f"{base:.3f}", "compare": f"{base / cont.iloc[0] - 1:.0%} more error than the best",
     "help": "Seasonal naive: next week's sales are forecast to equal last week's."},
    {"label": f"Products where another model beats {SHORT[PICKED]}", "value": f"{other_win:.0%}",
     "compare": f"of {n_items:,} products, on {metric}"},
])

# the leaderboard: the eight models of 07h, plus any group the reader adds
GROUPS = {"Other classical models (07a)": ["ets", "hw_add", "hw_mul"],
          "Building blocks (07c)": BLOCKS,
          "Weighted combination (07h)": ["combination_weighted"],
          "Versions that react to a shift": SHIFT_AWARE}
extra = st.multiselect("Also show", list(GROUPS), key="acc_extra",
                       help="The chart starts with the eight models compared head to head. Every bar you add is scored "
                            "on the same products and forecast dates.")
shown = CONTENDERS + [m for g in extra for m in GROUPS[g]]
lb = lb_all[lb_all["model"].isin(shown)]
lb = lb.assign(label=[name(m) + ("  (used for stock)" if m == PICKED else "") for m in lb["model"]],
               kind=lb["model"].map(family)).sort_values("value", ascending=False)
fig = go.Figure(go.Bar(x=lb["value"], y=lb["label"], orientation="h",
                       marker=dict(color=[COLORS.get(m, "#888") for m in lb["model"]],
                                   opacity=[0.55 if m in SHIFT_AWARE else 1 for m in lb["model"]]),
                       customdata=np.c_[lb["n_items"], lb["kind"]],
                       text=[f"{v:.3f}" for v in lb["value"]], textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}<br>" + metric + " %{x:.3f}<br>%{customdata[1]}, %{customdata[0]} items<extra></extra>"))
fig.add_vline(x=base, line_dash="dot", line_color="#8C8C8C")
style(fig, height=110 + 30 * len(lb), legend=False).update_xaxes(title=f"typical {metric} (median over items, lower is better)",
                                                                  range=[0, lb["value"].max() * 1.15])
members = get("ts_combination_members")["member"].map(lambda m: SHORT.get({"ets": "ets_boxcox"}.get(m, m), m)).tolist()
missing = [name(m) for m in shown if m not in set(lb_all["model"]) and m not in set(few["model"]) and m not in SAMPLE_DUPLICATES]
chart(fig, f"{SHORT[best]} has the lowest typical {metric} of the eight, {cont.iloc[1] - cont.iloc[0]:.3f} below {SHORT[second]}",
      f"Same {n_items:,} products and forecast dates for every bar; the dotted line is the baseline. The eight are the "
      f"models compared head to head, each tuned on the validation dates (Sep to Nov 2015) only. The combination is the "
      f"plain average of {', '.join(members)}, the three best on validation."
      + (" Paler bars are versions that react to a detected shift; the change for the same item is on the Demand "
         "shifts page." if "Versions that react to a shift" in extra else "")
      + (f" Not saved for {metric}: {', '.join(missing)}." if missing else "")
      + "".join(f" {name(r.model)} runs on only {r.n_items} of these items and is left out." for r in few.itertuples()
                if r.model in shown)
      + (f" On these items {name(BEST)}, which the stock work uses, is the {SHORT[SAME_ON_SAMPLE]} bar itself, so it "
         f"is shown once." if SAME_ON_SAMPLE else ""))

head = get("ts_headline").set_index("model")
table = lb.sort_values("value")[["model", "label", "kind", "value", "beats_baseline", "n_items"]]
table["vs baseline"] = table["value"] / base - 1
table["Validation RMSE"] = table["model"].map(lambda m: head["val_rmse"].get({"ets_boxcox": "ets"}.get(m, m), np.nan)
                                             if m != "ets" else np.nan)       # 07h's 'ets' is the Box-Cox one
table["What it assumes"] = table["model"].map(lambda m: ASSUMES.get(m.split("_")[0] if m in SHIFT_AWARE else m, ""))
table = table.drop(columns="model").rename(columns={"label": "Model", "kind": "Type", "value": f"Typical {metric}",
                                                    "beats_baseline": "Beats the baseline on", "n_items": "Items"})
st.dataframe(table, hide_index=True, width="stretch",
             column_config={f"Typical {metric}": st.column_config.NumberColumn(format="%.3f"),
                            "Beats the baseline on": st.column_config.NumberColumn(format="percent"),
                            "vs baseline": st.column_config.NumberColumn(format="percent"),
                            "Validation RMSE": st.column_config.NumberColumn(format="%.3f"),
                            "Items": st.column_config.NumberColumn(format="%d")})
download(table, f"forecast_accuracy_{col}.csv", key="dl_leader")

ts = get("ts_tests")
ts = ts.assign(model=ts["model"].replace({"ets": "ets_boxcox"}))
ts = ts[ts["metric"].isin(["rmse", "mae", "mape"]) & ts["model"].isin(CONTENDERS)]
said = []
for k in ["rmse", "mae", "mape"]:
    t = ts[ts["metric"] == k]
    b = ", ".join(SHORT[m] for m in t.loc[t["better_than_sarima"], "model"]) or "none"
    w = ", ".join(SHORT[m] for m in t.loc[t["worse_than_sarima"], "model"]) if "worse_than_sarima" in t else ""
    said.append(f"{k.upper()}: better {b}; worse {w or 'none'}")
mape_skip = get("classical_item_metrics")["mape_excluded"].mean() if "mape_excluded" in get("classical_item_metrics") else np.nan
st.caption("Significantly different from SARIMA (paired Wilcoxon tests, Holm-corrected, all 150 products and the six "
           "scored dates, not filtered): " + " | ".join(said) + ". "
           f"Which measure to use: the stock model was picked on RMSE, a rule fixed before the run. MASE has no units, "
           f"so it compares items of any size, and it is the measure on all 726 items below and on the other pages. MAE "
           f"and RMSE are in units, so big sellers count more. MAPE has to skip the {mape_skip:.1%} of days with no "
           f"sales.")

# which model is best: 07i's tables for the eight, all 150 products and the six scored dates (the filters do not apply)
st.subheader("Which model is best")
key = lambda d, cols=("model",): d.assign(**{c: d[c].replace(TS_KEYS) for c in cols})
vm = key(get("verdict_metrics"))
vr = key(get("verdict_ranks")).set_index("model")
vt = key(get("verdict_tests"), ("model", "reference"))
vw = key(get("verdict_wins"))
vs = key(get("verdict_stock"))
med = vm[vm["group"] == "ALL"].set_index("model")
n_prod = int(med["n_products"].iloc[0])
st.caption(f"From notebook 07i: the eight models on all {n_prod} sample products and the {len(eval_origins())} scored "
           f"forecast dates. The sidebar filters do not apply to this section.")

vlabel = st.radio("Measure", list(LABEL.values()), index=0, horizontal=True, key="verdict_metric",
                  help="Re-sorts the chart. Lower is better for every measure.")
k = {v: c for c, v in LABEL.items()}[vlabel]
order = med[k].sort_values(ascending=False)
lead = order.index[-1]
fig = go.Figure(go.Bar(x=order.to_numpy(), y=[SHORT[m] for m in order.index], orientation="h",
                       marker_color=[COLORS[m] for m in order.index], text=[f"{v:.3f}" for v in order],
                       textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}: " + vlabel + " %{x:.3f}<extra></extra>"))
style(fig, height=110 + 34 * len(order), legend=False).update_xaxes(
    title=f"typical {vlabel} (median over products, lower is better)", range=[0, order.max() * 1.15])
if k in TESTED:
    t = against(vt, lead, k)
    tied = [SHORT[m] for m in t.index if t.loc[m, "result"] != "worse"]
    title = (f"{SHORT[lead]} has the lowest typical {vlabel}, but its lead over {', '.join(tied)} is a tie" if tied else
             f"{SHORT[lead]} has the lowest typical {vlabel}, and every gap to it is real")
    note = (f"Tested product by product against {SHORT[lead]}: " + ("; ".join(
        f"{SHORT[m]} {t.loc[m, 'result']} ({t.loc[m, 'gap_pct']:+.1%})" for m in t.sort_values("gap_pct").index)) + ".")
else:
    title = f"{SHORT[lead]} has the lowest typical {vlabel}; gaps on this measure were not tested"
    note = "Significance was tested on RMSE, MAE and MASE only."
chart(fig, title, f"Each product's {vlabel} averaged over the six forecast dates, then the median over products. {note} "
                  f"A gap is real when the 95% interval of the per-product difference leaves out zero and the "
                  f"Holm-corrected Wilcoxon test is below 0.05; otherwise it is a tie.", key="verdict_bars")

# ranks as a heat map
cols = [f"rank_{c}" for c in LABEL]
z = vr[cols + ["avg_rank"]].to_numpy(dtype=float)
hover = [[f"{SHORT[m]}<br>{lab}: {v:g}<br>first on {int(vr.loc[m, 'n_first'])} of 6 measures, top 2 on "
          f"{int(vr.loc[m, 'n_top2'])}, top 3 on {int(vr.loc[m, 'n_top3'])}"
          for lab, v in zip(list(LABEL.values()) + ["average rank"], row)] for m, row in zip(vr.index, z)]
fig = go.Figure(go.Heatmap(z=z, x=list(LABEL.values()) + ["Average rank"], y=[SHORT[m] for m in vr.index],
                           zmin=1, zmax=len(vr), colorscale=[[0, "#4F9BB3"], [1, "#F1F6F8"]], showscale=False,
                           text=[[f"{v:.2f}" if j == len(cols) else f"{v:.0f}" for j, v in enumerate(row)] for row in z],
                           texttemplate="%{text}", textfont=dict(color="#1B1B1B"), hovertext=hover, hoverinfo="text",
                           xgap=2, ygap=2))
style(fig, height=110 + 34 * len(vr), legend=False).update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(side="top")
top = vr.index[0]
chart(fig, f"{SHORT[top]} has the best average rank, {vr.loc[top, 'avg_rank']:.2f} over the six measures",
      f"1 = lowest median error on that measure. {SHORT[top]} is first on {int(vr.loc[top, 'n_first'])} measures and in "
      f"the top three on {int(vr.loc[top, 'n_top3'])}; "
      + "; ".join(f"{SHORT[m]} first on {int(vr.loc[m, 'n_first'])}" for m in vr.index[1:] if vr.loc[m, "n_first"])
      + ". Rows are sorted by average rank.", key="verdict_ranks")

# significance against the recommended model
cost = vs[vs["method"] == "conformal"].set_index("model")
rec, text = verdict(med, vr, vt, vw, cost, {m: SHORT[m] for m in med.index}, eval_origins(), n_prod)
rows = [m for m in vr.index if m != rec]
cell = {(m, c): against(vt, rec, c).loc[m] for m in rows for c in TESTED}
fig = go.Figure(go.Heatmap(
    z=[[{"better": -1, "tie": 0, "worse": 1}[cell[m, c]["result"]] for c in TESTED] for m in rows],
    x=[LABEL[c] for c in TESTED], y=[SHORT[m] for m in rows], zmin=-1, zmax=1, showscale=False,
    colorscale=[[0, "#CFE5EC"], [0.33, "#CFE5EC"], [0.34, "#E4E4E4"], [0.66, "#E4E4E4"], [0.67, "#F3D9C4"], [1, "#F3D9C4"]],
    text=[[f"{cell[m, c]['result']} {cell[m, c]['gap_pct']:+.1%}" for c in TESTED] for m in rows],
    texttemplate="%{text}", textfont=dict(color="#1B1B1B"), xgap=2, ygap=2,
    hovertext=[[f"{SHORT[m]} against {SHORT[rec]} on {LABEL[c]}: {cell[m, c]['result']}<br>median difference per "
                f"product {cell[m, c]['median_diff']:+.3f} (95% interval {cell[m, c]['ci_low']:+.3f} to "
                f"{cell[m, c]['ci_high']:+.3f}), Holm p = {cell[m, c]['p_holm']:.3f}" for c in TESTED] for m in rows],
    hoverinfo="text"))
style(fig, height=90 + 34 * len(rows), legend=False).update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(side="top")
res = pd.Series([cell[m, c]["result"] for m in rows for c in TESTED])
chart(fig, f"Against {SHORT[rec]}: {int((res == 'worse').sum())} of {len(res)} gaps are real, {int((res == 'tie').sum())} "
           f"are ties",
      f"Each cell: is the row model better, tied or worse than {SHORT[rec]}, product by product? The percentage is the "
      f"median difference per product as a share of {SHORT[rec]}'s typical error; positive means more error.",
      key="verdict_matrix")

st.markdown("<div class='chart-title'>How to read this: why the measures can disagree</div>", unsafe_allow_html=True)
st.markdown("- **RMSE** squares each miss, so a few large misses on busy days weigh heavily.\n"
            "- **MAE** counts every unit missed the same, so it rewards being close on ordinary days.\n"
            "- **MAPE** divides each miss by that day's sales and skips days with none, so slow days dominate it.\n"
            "- **MASE** divides by each product's own naive error, so every product counts the same, whatever its size.")

# accuracy and money side by side
side = med[["rmse", "mae", "mase"]].join(cost[["total_cost", "fill_rate", "avg_inventory_value"]]).sort_values("rmse")
side.insert(0, "Model", [name(m) for m in side.index])
st.markdown("<div class='chart-title'>Accuracy and stock cost side by side</div>", unsafe_allow_html=True)
st.dataframe(side.rename(columns={"rmse": "RMSE", "mae": "MAE", "mase": "MASE", "total_cost": "Total cost ($)",
                                  "fill_rate": "Fill rate", "avg_inventory_value": "Average stock ($)"}),
             hide_index=True, width="stretch",
             column_config={"RMSE": st.column_config.NumberColumn(format="%.3f"),
                            "MAE": st.column_config.NumberColumn(format="%.3f"),
                            "MASE": st.column_config.NumberColumn(format="%.3f"),
                            "Total cost ($)": st.column_config.NumberColumn(format="$%,.0f"),
                            "Fill rate": st.column_config.NumberColumn(format="percent"),
                            "Average stock ($)": st.column_config.NumberColumn(format="$%,.0f")})
st.markdown(f"<div class='takeaway'>Stock simulation on the same {n_prod} products and forecasts, 95% target, 7-day lead "
            f"time, safety stock sized from past errors, assumed costs; the cheapest is {SHORT[side['total_cost'].idxmin()]}"
            f" (${side['total_cost'].min():,.0f}).</div>", unsafe_allow_html=True)

# building blocks: the parts SARIMA is built from, and VARIMA on the category totals
blk = head.loc[[m for m in BLOCKS if m in head.index], list(LABEL)]
st.markdown("<div class='chart-title'>Building blocks</div>", unsafe_allow_html=True)
st.dataframe(blk.assign(Model=[name(m) for m in blk.index])[["Model"] + list(LABEL)].rename(columns=LABEL),
             hide_index=True, width="stretch",
             column_config={v: st.column_config.NumberColumn(format="%.3f") for v in LABEL.values()})
var = get("varima_scores").pivot(index="series", columns="model", values="mase")
lost = [name(m) for m in BLOCKS if m not in head.index] + ([] if "varima" in var else ["VARIMA"])
st.dataframe(pd.DataFrame({"Category total": [s.title() for s in var.index],
                           "VARIMA MASE": var.get("varima", np.nan).to_numpy(),
                           "SARIMA MASE": var["sarima"].to_numpy()}),
             hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["VARIMA MASE", "SARIMA MASE"]})
st.markdown(f"<div class='takeaway'>The building blocks are scored on the same {n_prod} products and dates as the "
            f"eight; the best of them, {name(blk['rmse'].idxmin())}, has a typical RMSE of {blk['rmse'].min():.3f} against "
            f"{med['rmse'].min():.3f} for {SHORT[med['rmse'].idxmin()]}. VARIMA forecasts the three category totals "
            f"together, so it has no per-product score; its MASE on the totals is in the second table."
            + (f" Not in the outputs: {', '.join(lost)}." if lost else "") + "</div>", unsafe_allow_html=True)

verdict_box(text)
c1, c2 = st.columns(2)
with c1:
    download(vr.reset_index(), "verdict_ranks.csv", "Download the rank table as CSV", key="dl_verdict_ranks")
with c2:
    download(vt, "verdict_tests.csv", "Download every paired test as CSV", key="dl_verdict_tests")

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
hm = hm.loc[[m for m in shown if m in hm.index]]
fig = go.Figure()
for m in hm.index:
    fig.add_scatter(x=[f"days {7 * (w - 1) + 1}-{7 * w}" for w in hm.columns], y=hm.loc[m], mode="lines+markers",
                    name=SHORT[m], line=dict(color=COLORS[m], dash="dot" if m == "seasonal_naive" else "solid"),
                    hovertemplate="%{x}: " + metric + " %{y:.3f}<extra>" + SHORT[m] + "</extra>")
style(fig, height=360).update_yaxes(title=f"typical {metric}")
fig.update_xaxes(title="how far ahead (days after the forecast date)")
rise = (hm.iloc[:, -1] - hm.iloc[:, 0]) / hm.iloc[:, 0]
grows = bool((rise > 0).all())
chart(fig, ("Error grows the further ahead the forecast looks" if grows else "Error does not simply grow with the horizon"),
      f"From the first to the fourth week of the forecast, {SHORT[rise.idxmax()]}'s error changes most ({rise.max():+.0%}) and "
      f"{SHORT[rise.idxmin()]}'s least ({rise.min():+.0%}). Every forecast date is a Sunday, so the last two days are always "
      f"a busy weekend. Static models only; same items as above.")

# error by item group
st.markdown("<div class='chart-title'>Error by item group</div>", unsafe_allow_html=True)
grp = scores[scores["model"].isin(CONTENDERS)].copy()
rows = []
for gname, mask in [("Foods", grp["category"] == "FOODS"), ("Hobbies", grp["category"] == "HOBBIES"),
                    ("Household", grp["category"] == "HOUSEHOLD"), ("With a detected shift", grp["break_item"]),
                    ("Without a detected shift", ~grp["break_item"])]:
    g = grp[mask]
    if g["item_id"].nunique() == 0:
        continue
    med = g.groupby("model")[col].median()
    rows.append({"Item group": gname, "Items": g["item_id"].nunique(), "Best model": SHORT[med.idxmin()],
                 **{SHORT[m]: med.get(m, np.nan) for m in CONTENDERS}})
gt = pd.DataFrame(rows)
st.dataframe(gt, hide_index=True, width="stretch",
             column_config={SHORT[m]: st.column_config.NumberColumn(format="%.3f") for m in CONTENDERS})
st.markdown(f"<div class='takeaway'>Typical {metric} per group, for the eight models. The best model changes with the "
            "group: " + "; ".join(f"{r['Item group']}: {r['Best model']}" for r in rows) + ".</div>", unsafe_allow_html=True)
download(gt, f"accuracy_by_group_{col}.csv", key="dl_groups")

# which model wins each product
counts = winner.value_counts().reindex(CONTENDERS).fillna(0).astype(int).sort_values(ascending=False)
counts = counts[counts > 0]
fig = go.Figure(go.Bar(x=[SHORT[m] for m in counts.index], y=counts.to_numpy(), marker_color=[COLORS[m] for m in counts.index],
                       text=counts.to_numpy(), textposition="outside", cliponaxis=False,
                       hovertemplate="%{x}: best for %{y} items<extra></extra>"))
style(fig, height=300, legend=False).update_yaxes(title="items where it is the most accurate", range=[0, counts.max() * 1.2])
fig.update_xaxes(title="model")
top = counts.index[0]
chart(fig, (f"No single model wins most products: {SHORT[top]} is best for {counts.iloc[0]} of {len(winner):,}"
            if counts.iloc[0] < len(winner) / 2 else f"{SHORT[top]} is best for {counts.iloc[0]} of {len(winner):,} products"),
      f"Each product counted once, for the model with its lowest {metric} of the eight. {SHORT[PICKED]} is best for "
      f"{counts.get(PICKED, 0)}. A model that wins many products can still have a worse typical error, when it misses by "
      f"more on the rest.")

with st.expander("All 726 items: the models run on every item"):
    mu = get("mase_unrounded")
    a = mu[mu["category"] == "ALL"].set_index("model")
    pw = get("pairwise")

    pv = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    def pair(x, y):
        r = pw[(pw["model_a"] == x) & (pw["model_b"] == y) & (pw["category"] == "ALL")]
        return r.iloc[0] if len(r) else None

    le, tsb = pair(BEST, "ets"), a.loc["tsb"]
    st.markdown(f"Typical MASE on all {int(a.loc[BEST, 'n_items']):,} items: {BEST_SHORT} {a.loc[BEST, 'median_item_mase']:.3f}, "
                f"ETS {a.loc['ets', 'median_item_mase']:.3f}, seasonal naive {a.loc['seasonal_naive', 'median_item_mase']:.3f}. "
                f"{BEST_SHORT} is better than ETS for {le['share_improved']:.0%} of items ({pv(le['p_value'])}). TSB, built "
                f"for items that often sell nothing, runs only on the {int(tsb['n_items'])} such items and scores "
                f"{tsb['median_item_mase']:.3f}.")
    runner = next((m for m in a.index if m not in ("seasonal_naive", "ets", BEST, "tsb")), None)   # 09b
    if runner is not None and pair(runner, BEST) is not None:
        rb, re_ = pair(runner, BEST), pair(runner, "ets")

        def compared(r, other):
            if r["p_value"] < 0.05:
                return f"significantly {'better' if r['median_diff'] < 0 else 'worse'} than {other}"
            return f"level with {other}"
        st.markdown(f"{SHORT[runner]}, the runner-up on the sample, was also fitted on every item (09b). Its typical MASE "
                    f"is {a.loc[runner, 'median_item_mase']:.3f}: {compared(rb, BEST_SHORT)} (smaller error for "
                    f"{rb['share_improved']:.0%} of items, median difference {rb['median_diff']:+.4f}, {pv(rb['p_value'])}) "
                    f"and {compared(re_, 'ETS')} (smaller error for {re_['share_improved']:.0%}, {pv(re_['p_value'])}). The "
                    f"stock work uses {BEST_SHORT}, picked by the rule fixed before the run.")
    keep = [m for m in ["seasonal_naive", "ets", BEST, runner, "tsb"] if m in a.index]
    t726 = pd.DataFrame({"Model": [name(m) for m in keep], "Items": a.loc[keep, "n_items"].astype(int).to_numpy(),
                         "Typical MASE": a.loc[keep, "median_item_mase"].to_numpy()})
    st.dataframe(t726, hide_index=True, width="stretch",
                 column_config={"Typical MASE": st.column_config.NumberColumn(format="%.4f")})

with st.expander("What tuning changed: SARIMA one tweak at a time"):
    abl = get("ts_ablation")
    sa = abl[abl["family"] == "SARIMA"].reset_index(drop=True)
    vk, sk = f"val_{col}", f"scored_{col}"
    fig = go.Figure()
    labels = [s + ("  (kept)" if k else "") for s, k in zip(sa["step"], sa["kept"])]
    fig.add_bar(y=labels[::-1], x=sa[vk][::-1], orientation="h", name="validation (chosen on)", marker_color="#56B4E9",
                hovertemplate="%{y}<br>validation " + metric + " %{x:.3f}<extra></extra>")
    fig.add_bar(y=labels[::-1], x=sa[sk][::-1], orientation="h", name="scored (not used to choose)", marker_color=COLORS["sarima"],
                hovertemplate="%{y}<br>scored " + metric + " %{x:.3f}<extra></extra>")
    lo = float(np.nanmin([sa[vk].min(), sa[sk].min()]))
    style(fig, height=140 + 38 * len(sa)).update_layout(barmode="group")
    fig.update_xaxes(title=f"typical {metric}", range=[lo * 0.9, float(np.nanmax([sa[vk].max(), sa[sk].max()])) * 1.03])
    kept = sa.loc[sa["kept"], "step"].tolist()
    chart(fig, "SARIMA, one tweak at a time: " + (f"{len(kept)} of {len(sa) - 1} tweaks passed" if kept else "no tweak passed"),
          "A tweak was kept only if it lowered both RMSE and MAE on the validation dates (Sep to Nov 2015). "
          + (f"Kept: {', '.join(s.lstrip('+ ') for s in kept)}; with it the model is SARIMAX, one of the eight. " if kept else "")
          + "The scored bars are shown for the record only.", key="acc_abl")
    tuned = abl[abl["kept"]][["family", "step"]].rename(columns={"family": "Model family", "step": "Kept tweak"})
    st.dataframe(tuned if len(tuned) else pd.DataFrame({"Kept tweak": ["none"]}), hide_index=True, width="stretch")
    st.markdown("Every tweak tried, including the ones that did not help, is in `outputs/tables/ts_sarima_ablation.csv` "
                "and `ts_other_tuning.csv`.")

with st.expander("Category totals: SARIMA in units and in thousands, and VARIMA"):
    tot = get("aggregate_metrics")
    raw = tot[tot["model"] == "sarima"].set_index("series")["mase"]
    rs = get("sarima_totals_rescaled")
    th = rs[rs["scale"] == "thousands"].groupby("series")["mase"].mean()
    vt = get("varima_scores").pivot(index="series", columns="model", values="mase")
    tt = pd.DataFrame({"Total": [s.replace("CA_TOTAL", "California").title() for s in raw.index],
                       "SARIMA in units": raw.to_numpy(), "SARIMA in thousands": th.reindex(raw.index).to_numpy(),
                       "VARIMA in thousands": vt["varima"].reindex(raw.index).to_numpy()})
    st.dataframe(tt, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tt.columns[1:]})
    better = [s.title() for s in vt.index if vt.loc[s, "varima"] < vt.loc[s, "sarima"]]
    st.markdown(f"MASE over the scored forecast dates. Fitted on the raw totals, SARIMA's numbers break down for some "
                f"windows (FOODS {raw['FOODS']:.3f}, worse than repeating last week). Dividing the totals by 1,000 before "
                f"fitting fixes most of it (FOODS {th['FOODS']:.3f}). The published number stays the one in units. VARIMA "
                f"fits the three category totals together, so each can learn from the others; it is ahead of SARIMA on "
                f"{' and '.join(better) if better else 'none of them'}. California as a whole is not part of VARIMA.")

with st.expander("Forecast against actual, any of the 150 products"):
    fc = sample_forecasts()
    ids = sorted(fc["item_id"].unique())
    if flt["categories"]:
        ids = [i for i in ids if i.split("_")[0] in flt["categories"]] or ids
    c1, c2 = st.columns([1, 1])
    typical = (wide[PICKED] - wide[PICKED].median()).abs().idxmin()          # open on a product with a typical error
    item = c1.selectbox("Product", ids, index=ids.index(typical) if typical in ids else 0, key="acc_item")
    evals = set(eval_origins())
    origins = [o for o in sorted(fc["origin"].unique()) if pd.Timestamp(o) in evals]
    origin = c2.select_slider("Forecast date", options=origins, value=origins[-1], key="acc_origin",
                              format_func=lambda o: f"{pd.Timestamp(o):%d %b %Y}")
    have = set(fc["model"].unique())
    avail = [m for m in CONTENDERS + GROUPS["Other classical models (07a)"] + BLOCKS + ["combination_weighted"] if m in have]
    picked = st.multiselect("Models", avail, default=[m for m in [PICKED, second, "seasonal_naive"] if m in avail],
                            format_func=name, key="acc_models")
    g = fc[(fc["item_id"] == item) & (fc["origin"] == origin)]
    hist = fc[(fc["item_id"] == item) & (fc["date"] <= origin) & (fc["date"] > origin - pd.Timedelta(days=42))]
    hist = hist.drop_duplicates("date").sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["actual"], mode="lines", name="actual, before", line=dict(color="#888", width=1.4))
    act = g.drop_duplicates("date").sort_values("date")
    fig.add_scatter(x=act["date"], y=act["actual"], mode="lines", name="actual", line=dict(color="#555", width=2.2))
    for m in picked:
        x = g[g["model"] == m].sort_values("date")
        fig.add_scatter(x=x["date"], y=x["forecast"], mode="lines", name=SHORT[m],
                        line=dict(color=COLORS.get(m, "#888"), width=1.8, dash="dot" if m == "seasonal_naive" else "solid"))
    style(fig, height=380).update_yaxes(title="units per day")
    errs = {SHORT[m]: float(np.sqrt(np.mean((g[g["model"] == m]["actual"] - g[g["model"] == m]["forecast"]) ** 2))) for m in picked}
    chart(fig, f"{item}, 28 days from {pd.Timestamp(origin):%d %b %Y}",
          ("RMSE in this window: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(errs.items(), key=lambda kv: kv[1])) + "."
           if errs else "Pick at least one model."), key="acc_explorer")

all_beat = bool((cont.drop("seasonal_naive") < base).all())
spread = cont.drop("seasonal_naive").max() - cont.iloc[0]
base_of = {m: ("seasonal_naive" if m.startswith("seasonal") else m.split("_")[0]) for m in SHIFT_AWARE}
base_of = {m: SAMPLE_DUPLICATES.get(b, b) for m, b in base_of.items()}
reacting = [(m, vals[m] - vals[base_of[m]]) for m in SHIFT_AWARE if m in vals and base_of[m] in vals]
improved = [m for m, d in reacting if d < -0.0005]
tells(f"On {metric}, {SHORT[best]} has the lowest typical error of the eight ({cont.iloc[0]:.3f})"
      + (f", and it is the model used for the stock work. " if best == PICKED else
         f"; {SHORT[PICKED]}, used for the stock work because it had the lowest RMSE, is {nth[rank]} at {cont[PICKED]:.3f}. ")
      + (f"Every fitted model beats repeating last week ({base:.3f}), " if all_beat else f"The baseline is at {base:.3f}, ")
      + f"but the spread between them is small: {spread:.3f} from the best to the worst fitted model. "
      f"A model other than {SHORT[PICKED]} is best for {other_win:.0%} of products, so the choice matters less than "
      f"having one of these models at all. "
      + ("" if not reacting else
         f"Of the {len(reacting)} versions that react to shifts and have {metric} saved, none beats its static model by "
         f"more than 0.0005." if not improved else
         f"{len(improved)} of the {len(reacting)} versions that react to shifts beat their static model on {metric}: "
         + ", ".join(name(m) for m in improved) + ". The Demand shifts page shows whether that gain is large enough to matter."))
