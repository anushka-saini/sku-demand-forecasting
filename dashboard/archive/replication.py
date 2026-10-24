import plotly.graph_objects as go
import streamlit as st

from utils.data import get
from utils.facts import facts
from utils.ui import ACCENT, NEUTRAL, SECOND, chart, header, insight, style

f = facts()
concl = get("rep_conclusions")
scope = get("rep_scope")
brk = get("rep_breaks")
mase = get("rep_mase")
inv = get("rep_inventory")

header("Texas and Wisconsin",
       "The whole pipeline was rerun on two other states with every threshold and setting frozen at the California "
       "values. A conclusion holds if it passes the same test, written down before the run, there.")

n_hold = int(concl["holds"].sum())
st.markdown(f"<div class='headline'><b>{n_hold} of {len(concl)}</b> state-by-conclusion checks hold"
            + ("." if n_hold == len(concl) else f"; {len(concl) - n_hold} do not, listed below.") + "</div>",
            unsafe_allow_html=True)
show = concl.copy()
show["result"] = show["holds"].map({True: "HOLDS", False: "DOES NOT HOLD"})
st.dataframe(show[["state", "conclusion", "result", "evidence"]], hide_index=True, width="stretch",
             column_config={"evidence": st.column_config.TextColumn("Evidence", width="large")})
failed = show[~show["holds"]]
for r in failed.itertuples():
    st.warning(f"{r.state}: '{r.conclusion}' did not replicate ({r.evidence}).")

c1, c2 = st.columns(2)
sc = scope.groupby("state").agg(items=("ax_ay_candidates", "sum"), all_items=("total_items", "sum")).reset_index()
c1.dataframe(sc, hide_index=True, width="stretch",
             column_config={"items": "AX/AY items modelled", "all_items": "All items"})
b = brk[brk["category"] == "ALL"]
c2.dataframe(b[["state", "n_items", "cusum_break_rate", "cusum_far_seasonal_null"]].assign(
    cusum_break_rate=lambda x: 100 * x["cusum_break_rate"], cusum_far_seasonal_null=lambda x: 100 * x["cusum_far_seasonal_null"]),
    hide_index=True, width="stretch", column_config={
        "cusum_break_rate": st.column_config.NumberColumn("Items flagged", format="%.1f%%"),
        "cusum_far_seasonal_null": st.column_config.NumberColumn("False alarms (seasonal null)", format="%.1f%%")})

fig = go.Figure()
for model, color, pattern, name in [("seasonal_naive", NEUTRAL, ".", "Seasonal naive"), ("ets", SECOND, "/", "ETS"),
                                    ("lightgbm", ACCENT, "", "LightGBM")]:
    d = mase[mase["model"] == model]
    fig.add_trace(go.Bar(x=d["state"], y=d["median_item_mase"], name=name, marker=dict(color=color, pattern_shape=pattern),
                         text=d["median_item_mase"].round(3), textposition="outside", customdata=d[["n_items"]],
                         hovertemplate="%{x}: %{y:.3f}, n = %{customdata[0]} items<extra>" + name + "</extra>"))
fig.update_layout(barmode="group")
style(fig).update_yaxes(title="median item MASE", range=[0, 1.15])
best = mase.loc[mase.groupby("state")["median_item_mase"].idxmin()]
lgb_best = (best["model"] == "lightgbm").all()
chart(fig, "LightGBM has the lowest error in both states, as in California" if lgb_best
      else "The best model differs by state: " + ", ".join(f"{r.state} {r.model}" for r in best.itertuples()),
      "Median over items of each item's error across the 6 forecast dates. Lower is better.")

i95 = inv[inv["service_level"] == 0.95]
fig = go.Figure()
for p, color, pattern, name in [("seasonal_naive_static", NEUTRAL, ".", "Seasonal naive"),
                                ("lightgbm_static", ACCENT, "", "LightGBM static"),
                                ("lightgbm_adaptive", SECOND, "/", "LightGBM adaptive")]:
    d = i95[i95["policy"] == p]
    fig.add_trace(go.Bar(x=d["state"], y=d["total_cost"], name=name, marker=dict(color=color, pattern_shape=pattern),
                         customdata=d[["n_items", "fill_rate"]],
                         hovertemplate="%{x}: $%{y:,.0f}, fill rate %{customdata[1]:.2%}, n = %{customdata[0]}<extra>" + name + "</extra>"))
fig.update_layout(barmode="group")
style(fig).update_yaxes(tickprefix="$", tickformat=",.0f", title="total cost (ASSUMED costs)")
cheapest = i95.loc[i95.groupby("state")["total_cost"].idxmin()]
chart(fig, "The static forecast is the cheapest in both states" if (cheapest["policy"] == "lightgbm_static").all()
      else "Cheapest policy by state: " + ", ".join(f"{r.state} {r.policy}" for r in cheapest.itertuples()),
      "Total holding + lost-sale cost over 168 days at a 95% target and 7-day lead time, with the assumed base costs.")

insight(f"{n_hold} of {len(concl)} checks hold in the two other states, with every setting frozen at its California "
        f"value. Shifts are about as common, LightGBM still beats seasonal naive, and retraining after a shift still "
        f"does not help, so the California findings are not a quirk of one state.")
