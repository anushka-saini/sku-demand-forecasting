from plotly.subplots import make_subplots
import streamlit as st

from utils.data import get
from utils.models import TOTALS
from utils.ui import chart, figure, header, insight, style

header("Decomposition",
       "We split each series into a slow trend, a repeating weekly pattern and what is left, first by week and then by "
       "year, to see which patterns a model has to capture.")

choice = get("decomposition_choice")
decomp = get("demo_decomposition")
sample = get("classical_sample")

options = TOTALS + sorted(sample["item_id"])
pick = st.selectbox("Series to decompose (STL, weekly period, last 52 weeks)", options, key="dec_pick")
d = decomp[decomp["series"] == pick]
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["observed", "trend", "weekly pattern", "residual"],
                    vertical_spacing=0.06)
for row, part in enumerate(["observed", "trend", "seasonal", "resid"], start=1):
    fig.add_scatter(x=d["date"], y=d[part], mode="lines", line=dict(color="#0072B2", width=1.2), row=row, col=1,
                    showlegend=False, hovertemplate="%{x|%Y-%m-%d}: %{y:,.1f} units<extra></extra>")
fig.update_xaxes(title_text="date", row=4, col=1)
swing = d["seasonal"].tail(7)
chart(style(fig, height=620, legend=False), f"{pick}: the weekly pattern swings by about {swing.max() - swing.min():,.1f} units a day",
      "STL splits the series into a slow trend, a weekly pattern that repeats, and a residual. A small residual with no "
      "pattern means trend and season explain the series well.")

st.markdown("#### Additive or multiplicative?")
st.caption("Rule: if the weekly swing grows with the weekly level (Spearman rho at least 0.5, p below 0.05), "
           "multiplicative; otherwise additive. Products use sales + 1 for the multiplicative form, and the totals' "
           "25 December (stores closed) is replaced for decomposition only.")
st.dataframe(choice[["series", "level", "spearman_level_vs_swing", "rule_choice", "resid_cv_additive",
                     "resid_cv_multiplicative", "lower_residual"]],
             hide_index=True, width="stretch",
             column_config={"spearman_level_vs_swing": st.column_config.NumberColumn("rho", format="%.2f"),
                            "rule_choice": "rule picks",
                            "resid_cv_additive": st.column_config.NumberColumn("residual spread, additive", format="%.3f"),
                            "resid_cv_multiplicative": st.column_config.NumberColumn("residual spread, multiplicative",
                                                                                     format="%.3f"),
                            "lower_residual": "lower residual"})

st.markdown("#### The yearly pattern")
figure("yearly_seasonality.png", "STL on weekly totals with a 52-week period: the yearly pattern of each category.")

ca = choice[choice["series"] == "CA_TOTAL"].iloc[0]
insight(f"The weekly pattern is the part every model must get right; the form barely matters (residual spread "
        f"{ca['resid_cv_additive']:.3f} additive against {ca['resid_cv_multiplicative']:.3f} multiplicative for the "
        f"California total), so we fit both kinds of Holt-Winters. The yearly pattern is mostly a dip from "
        f"Thanksgiving to Christmas, which falls inside our first test window.")
