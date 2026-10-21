import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import get
from utils.m2 import TOTALS
from utils.ui import chart, header, n_note, style

header("Stationarity and decomposition",
       "Milestone 2, notebook 12. Before fitting classical models we check whether each series is stationary, how "
       "much to difference it, and whether its weekly pattern is additive or multiplicative.")

summ = get("m2_stationarity_summary")
st_items = get("m2_stationarity")
totals = get("m2_stationarity_totals")
quality = get("m2_quality")
choice = get("m2_decomp_choice")
decomp = get("m2_demo_decomp")
sample = get("m2_sample")

all_row = summ[summ["category"] == "ALL"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stationary (ADF and KPSS agree)", f"{all_row['stationary']:.0%}", help="Share of the 726 items")
c2.metric("Need one difference (d = 1)", f"{all_row['d=1']:.0%}")
c3.metric("Need a seasonal difference (D = 1)", f"{all_row['D=1']:.0%}")
c4.metric("Empty sales cells", f"{int(quality.loc[quality['category'] == 'ALL', 'raw_empty_cells'].iloc[0])}")
n_note(int(all_row["n_items"]))

labels = ["stationary", "trend-stationary", "difference-stationary", "conflicting"]
colors = ["#0072B2", "#56B4E9", "#E69F00", "#BBBBBB"]
fig = go.Figure()
for lab, c in zip(labels, colors):
    fig.add_bar(y=summ["category"], x=summ[lab], name=lab, orientation="h", marker_color=c,
                hovertemplate="%{y}: %{x:.1%} " + lab + "<extra></extra>")
fig.update_layout(barmode="stack")
fig.update_xaxes(tickformat=".0%", range=[0, 1])
top = all_row[labels].astype(float).idxmax()
chart(style(fig, height=300), f"Most items test as {top}",
      "ADF (null: unit root) and KPSS (null: stationary) at 5%, each with a constant and with a trend. "
      "Conflicting = the tests disagree in a way no single label explains.")

st.markdown("#### Totals")
st.dataframe(totals[["series", "label", "d", "D", "adf_c", "kpss_c"]], hide_index=True, width="stretch",
             column_config={"adf_c": st.column_config.NumberColumn("ADF p (constant)", format="%.4f"),
                            "kpss_c": st.column_config.NumberColumn("KPSS p (constant)", format="%.3f")})

st.markdown("#### Data quality: missing values and outliers")
st.caption("Pre-launch zeros were trimmed in Milestone 1. Outliers are days above Q3 + 3 × IQR of the item's sales; "
           "they are flagged, not changed, because promotion spikes are real demand.")
q = quality[["category", "items", "raw_empty_cells", "leading_zero_days_trimmed", "missing_days_inside_series",
             "zero_days_kept_share", "outlier_days", "outlier_share"]]
st.dataframe(q, hide_index=True, width="stretch",
             column_config={"zero_days_kept_share": st.column_config.NumberColumn("zero days kept", format="percent"),
                            "outlier_share": st.column_config.NumberColumn("outlier share", format="percent")})

st.markdown("#### Decomposition: additive or multiplicative?")
st.caption("Rule: if the weekly swing grows with the weekly level (Spearman rho ≥ 0.5, p < 0.05), multiplicative; "
           "otherwise additive. Multiplicative cannot divide by zero, so items use sales + 1 and the totals' "
           "25 December (stores closed) is replaced by the same weekday a week before and after, for decomposition only.")
st.dataframe(choice[["series", "level", "zero_handling", "spearman_level_vs_swing", "rule_choice",
                     "resid_cv_additive", "resid_cv_multiplicative", "lower_residual"]],
             hide_index=True, width="stretch",
             column_config={"spearman_level_vs_swing": st.column_config.NumberColumn("rho", format="%.2f"),
                            "resid_cv_additive": st.column_config.NumberColumn("residual spread, additive", format="%.3f"),
                            "resid_cv_multiplicative": st.column_config.NumberColumn("residual spread, multiplicative",
                                                                                     format="%.3f")})

options = TOTALS + sorted(sample["item_id"])
pick = st.selectbox("Series to decompose (STL, period 7, last 52 weeks)", options, key="m2_decomp_pick")
d = decomp[decomp["series"] == pick]
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["observed", "trend", "weekly season", "residual"],
                    vertical_spacing=0.06)
for row, part in enumerate(["observed", "trend", "seasonal", "resid"], start=1):
    fig.add_scatter(x=d["date"], y=d[part], mode="lines", line=dict(color="#0072B2", width=1.2), row=row, col=1,
                    showlegend=False, hovertemplate="%{x|%Y-%m-%d}: %{y:,.1f}<extra></extra>")
verdict = st_items.loc[st_items["item_id"] == pick, "label"]
verdict = verdict.iloc[0] if len(verdict) else totals.loc[totals["series"] == pick, "label"].iloc[0]
chart(style(fig, height=620, legend=False), f"{pick}: tests as {verdict}",
      "STL splits the series into a slow trend, a repeating weekly pattern and what is left. A small, patternless "
      "residual means trend + season explain the series well.")
