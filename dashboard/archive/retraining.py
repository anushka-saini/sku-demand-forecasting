import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts, lookup
from utils.kpis import per_item_variant_diff
from utils.verdicts import retraining
from utils.ui import ACCENT, DARK, NEUTRAL, SECOND, chart, filters, header, insight, n_note, style

f = facts()
var = get("variants")
by_cat = get("variants_by_cat")
tests = get("adaptive_tests")
errors = get("forecast_error")
sb = get("selection_bias")
cats, _, _ = filters()
v = retraining(f)
VAR_NAMES = {"truncate": "(a) restart from the shift", "decay": "(b) down-weight older data",
             "level_shift": "(c) add a 'shift happened' feature", "blend": "(d) average of static and (a)"}
PRACTICAL = 0.0074

header("Does retraining after a shift help?",
       "When a shift is detected, the forecast model can be retrained to focus on the new level. Four ways of doing "
       "that were compared against the unchanged (static) model, using only data available at each forecast date.")

st.markdown(f"<div class='headline'><b>{v['verdict']}.</b> {v['answer']}</div>", unsafe_allow_html=True)

trunc = var[(var["group"] == "all detected breaks") & (var["variant"] == "truncate")].iloc[0]
blend = var[(var["group"] == "all detected breaks") & (var["variant"] == "blend")].iloc[0]
st.markdown(
    f"**In plain English.** Restarting the model from the shift made typical error *worse* by {trunc['median_diff']:+.4f} "
    f"(on an error of about {lookup(f, 'forecasts', 'median_item_mase', 'lightgbm_ALL')['value']:.2f}). The gentlest "
    f"option, averaging old and new models, moved error by {blend['median_diff']:+.4f}, too small to matter and not "
    f"significant after correcting for testing four methods. A rule written down before the test required an improvement of at least "
    f"{PRACTICAL} with a corrected p-value below 0.05; nothing met it, even on the clearest, high-confidence shifts. "
    "A likely reason (not tested here) is that the static model's rolling averages already track level changes, so "
    "throwing away history loses more (seasonality, price effects) than it gains.")

v2 = var.copy()
v2["label"] = v2["variant"].map(VAR_NAMES)
fig = go.Figure()
for group, color, symbol in [("all detected breaks", ACCENT, "circle"), ("high-confidence", SECOND, "diamond")]:
    d = v2[v2["group"] == group]
    fig.add_trace(go.Scatter(
        x=d["median_diff"], y=d["label"], mode="markers", name=f"{group} (n = {int(d['n'].iloc[0])} items)",
        marker=dict(color=color, symbol=symbol, size=11),
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["median_diff"],
                     arrayminus=d["median_diff"] - d["ci_low"], color=color, thickness=1.5),
        customdata=d[["ci_low", "ci_high", "p_value", "p_holm", "share_improved", "rank_biserial"]],
        hovertemplate="%{y}<br>median change %{x:+.4f}<br>95% CI %{customdata[0]:+.4f} to %{customdata[1]:+.4f}<br>"
                      "p = %{customdata[2]:.4f}, Holm-adjusted %{customdata[3]:.4f}<br>items improved %{customdata[4]:.1%}"
                      "<br>effect size (rank-biserial) %{customdata[5]:+.2f}<extra>" + group + "</extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
fig.add_vrect(x0=-PRACTICAL, x1=0, fillcolor=NEUTRAL, opacity=0.12, line_width=0,
              annotation_text="too small to matter", annotation_position="top left")
style(fig, height=360).update_xaxes(title="median per-item change in error vs static (negative = retraining better)")
fig.update_yaxes(autorange="reversed")
chart(fig, "No retraining method improves forecasts; restarting from the shift makes them worse",
      "Each dot is the median per-item change in MASE vs the static model, with a 95% bootstrap CI. Blue: every item-date "
      "with a detected shift. Orange: only high-confidence shifts. The shaded band is smaller than the pre-registered "
      "minimum useful improvement.")

# Placebo and break vs no-break
st.divider()
st.markdown("#### Control: does a random training cut on items *without* a shift do the same?")
rows = tests[(tests["model"].isin(["lightgbm", "ets"])) & (tests["category"] == "ALL")].copy()
rows["label"] = rows["model"].map({"lightgbm": "LightGBM", "ets": "ETS"}) + " · " + rows["comparison"]
fig = go.Figure(go.Scatter(
    x=rows["median_diff"], y=rows["label"], mode="markers", marker=dict(color=ACCENT, size=10),
    error_x=dict(type="data", symmetric=False, array=rows["ci_high"] - rows["median_diff"],
                 arrayminus=rows["median_diff"] - rows["ci_low"], color=ACCENT),
    customdata=rows[["n", "p_value"]],
    hovertemplate="%{y}<br>%{x:+.4f}, n = %{customdata[0]}, p = %{customdata[1]:.4f}<extra></extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=260, legend=False).update_xaxes(title="median per-item change in error vs static")
fig.update_yaxes(autorange="reversed")
pl = rows[(rows["model"] == "lightgbm") & rows["comparison"].str.startswith("placebo")].iloc[0]
chart(fig, f"The placebo changes nothing ({pl['median_diff']:+.4f}, p = {pl['p_value']:.2f}), so shift detection adds no useful signal for retraining",
      "Placebo: items with no detected shift were given an equally short, random training window. If retraining helped "
      "only because recent data is better, the placebo would improve too.")

model = st.radio("Model for the distribution", ["lightgbm", "ets"], horizontal=True,
                 format_func={"lightgbm": "LightGBM", "ets": "ETS"}.get)
ad = per_item_variant_diff(errors, model, "adaptive")
pd_ = per_item_variant_diff(errors, model, "placebo")
lo, hi = np.percentile(np.r_[ad["diff"], pd_["diff"]], [1, 99])          # display range only
bins = dict(start=lo, end=hi, size=(hi - lo) / 60)
fig = go.Figure()
fig.add_trace(go.Histogram(x=ad["diff"].clip(lo, hi), xbins=bins, name=f"restart, items with a shift (n={len(ad)})",
                           marker=dict(color=SECOND), opacity=0.75, histnorm="percent"))
fig.add_trace(go.Histogram(x=pd_["diff"].clip(lo, hi), xbins=bins, name=f"placebo, items without (n={len(pd_)})",
                           marker=dict(color=NEUTRAL, pattern_shape="/"), opacity=0.6, histnorm="percent"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
fig.update_layout(barmode="overlay")
style(fig).update_xaxes(title="per-item change in error vs static (negative = better)")
fig.update_yaxes(title="% of items")
chart(fig, f"Both are centred near zero: median {ad['diff'].median():+.4f} (restart) vs {pd_['diff'].median():+.4f} (placebo)",
      "Each item's average change across the forecast dates. Trimmed to the 1st-99th percentile for drawing only; medians use all items.")

st.markdown("#### Are items with shifts simply harder to forecast?")
reg = sb[sb["kind"] == "regression"]
adj = reg[reg["variable"] == "lightgbm adjusted"].iloc[0]
fig = go.Figure(go.Scatter(
    x=reg["pct_effect"], y=reg["variable"], mode="markers", marker=dict(color=ACCENT, size=10),
    error_x=dict(type="data", symmetric=False, array=reg["ci_high_pct"] - reg["pct_effect"],
                 arrayminus=reg["pct_effect"] - reg["ci_low_pct"], color=ACCENT),
    customdata=reg[["p_value", "n_break", "n_no_break"]],
    hovertemplate="%{y}: %{x:+.1f}%, p = %{customdata[0]:.3f}<br>n = %{customdata[1]} with / %{customdata[2]} without<extra></extra>"))
fig.add_vline(x=0, line_color=DARK, line_width=1)
style(fig, height=240, legend=False).update_xaxes(title="static error of items with a shift vs without (%)")
chart(fig, f"Not detectably: {adj['pct_effect']:+.1f}% after adjusting for volume, variability and history (95% CI {adj['ci_low_pct']:+.1f}% to {adj['ci_high_pct']:+.1f}%)",
      "Raw = simple comparison. Adjusted = regression controlling for volume, variability, share of zero days, history length and category.")

st.markdown("#### By category")
bc = by_cat[by_cat["category"].isin(cats)] if cats else by_cat
bc = bc.assign(variant=bc["variant"].map(VAR_NAMES))
st.dataframe(bc[["group", "variant", "category", "n", "median_diff", "ci_low", "ci_high", "p_value"]], hide_index=True,
             width="stretch", column_config={
                 "group": "Shifts", "variant": "Method", "category": "Category", "n": "Items",
                 "median_diff": st.column_config.NumberColumn("Median change", format="%+.4f"),
                 "ci_low": st.column_config.NumberColumn("95% CI low", format="%+.4f"),
                 "ci_high": st.column_config.NumberColumn("95% CI high", format="%+.4f"),
                 "p_value": st.column_config.NumberColumn("p-value", format="%.4f")})
st.caption("Descriptive only: the pre-registered decision used all categories together. Rows with n below 20 are not reliable.")
if (bc["n"] < 20).any():
    st.warning(f"{int((bc['n'] < 20).sum())} rows have fewer than 20 items.")

insight(f"None of the four ways of reacting to a shift met the rule, and restarting from the shift made LightGBM "
        f"worse ({trunc['median_diff']:+.4f} MASE). A random short window on products without a shift changes almost "
        f"nothing ({pl['median_diff']:+.4f}), so the detected shifts give retraining no useful signal; our guess, not "
        f"tested, is that the static model's recent averages already follow a change in level.")
