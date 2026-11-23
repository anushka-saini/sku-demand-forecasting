import plotly.graph_objects as go
import streamlit as st

from utils.final import COL, best_info, load, names, need_results, text
from utils.ui import chart, kpis, page, style, tells

need_results()
lb = load("leaderboard")
nm = names(lb)
info = best_info()
tx = text()
b = info["model"]
page("Diagnostics", "Where do the models fail, which gaps are real, and what do GARCH and VAR add?")
st.caption("The checks behind the Leaderboard: errors by type of day, significance tests, volatility of the errors, and a model for the category totals.")

t1, t2, t3, t4 = st.tabs(["Where models fail", "Significance tests", "GARCH", "VAR"])

with t1:
    seg = load("segments")
    ms = [m for m in [b, "seasonal_naive", "sarima"] if m in set(seg["model"])]
    s = seg[seg["model"].isin(ms)].pivot(index="segment", columns="model", values="mae")
    s = s.drop(index="all days", errors="ignore").sort_values(b, ascending=False)
    fig = go.Figure()
    for m in ms:
        fig.add_bar(y=s.index, x=s[m], orientation="h", name=nm[m], marker_color=COL.get(m))
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="MAE on those item-days, units a day")
    chart(style(fig, height=460), "Error by type of day", "Longest bars are where the best model misses most.", key="dg_seg")
    t = s.copy()
    t.columns = [nm[c] for c in t.columns]
    t["Item-days"] = seg[seg["model"] == b].set_index("segment")["item_days"]
    st.dataframe(t.round(3), width="stretch")
    st.caption("Event and SNAP days come from the M5 calendar; 'zero week before origin' means nothing sold in the 7 days before the forecast was made.")
    for line in tx["failure"]:
        st.markdown("- " + line)

with t2:
    dm = load("dm")
    show = dm.copy()
    show["model_a"], show["model_b"] = show["model_a"].map(nm), show["model_b"].map(nm)
    st.dataframe(show[["model_a", "model_b", "loss", "mean_loss_a", "mean_loss_b", "dm_stat", "p_value", "significant_5pct",
                       "share_items_a_better", "wilcoxon_p"]].round(4), hide_index=True, width="stretch")
    sig = dm[dm["significant_5pct"] & dm["a_better"]]
    st.caption("Diebold-Mariano on per-day loss differences over the 168 test days (Newey-West, 27 lags, small-sample correction), plus a "
               f"paired Wilcoxon test per item. {len(sig)} of {len(dm)} tests show the first model significantly better at 5%.")
    st.markdown(tx["dm"])

with t3:
    g, lm, dt = load("garch"), load("archlm"), load("daytype")
    n95, n80 = g[g["nominal"] == 0.95].set_index("interval"), g[g["nominal"] == 0.8].set_index("interval")
    gb, cs = "GARCH(1,1)", "constant std"
    st.markdown(f"A GARCH(1,1) was fitted to the forecast errors of **{nm[b]}**. It models the *size* of the errors and does not change the "
                "forecast, so MAE and RMSE are the same with or without it.")
    kpis([
        {"label": "Items with changing variance", "value": f"{(lm['arch_lm_p'] < 0.05).mean():.0%}", "compare": "ARCH-LM test rejects constant variance at 5%"},
        {"label": "80% band coverage, GARCH", "value": f"{n80.loc[gb, 'coverage']:.1%}", "compare": f"constant spread: {n80.loc[cs, 'coverage']:.1%}"},
        {"label": "95% band coverage, GARCH", "value": f"{n95.loc[gb, 'coverage']:.1%}", "compare": f"constant spread: {n95.loc[cs, 'coverage']:.1%}"},
        {"label": "Interval score, 80% (lower is better)", "value": f"{n80.loc[gb, 'interval_score']:.2f}",
         "compare": f"constant spread: {n80.loc[cs, 'interval_score']:.2f}", "tone": "good" if n80.loc[gb, "interval_score"] < n80.loc[cs, "interval_score"] else "bad"},
    ])
    fig = go.Figure()
    fig.add_bar(x=dt["day_type"], y=dt["mean_sigma_over_scale"], name="GARCH spread (sigma)", marker_color="#0072B2")
    fig.add_bar(x=dt["day_type"], y=dt["mean_abs_error_over_scale"], name="actual absolute error", marker_color="#D55E00")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="relative to the item's usual sales")
    chart(style(fig), "Fitted spread and real error by type of day",
          "Where the orange bar is clearly above the blue one, the spread under-states the error on those days.", key="dg_dt")
    st.dataframe(g.round(3), hide_index=True, width="stretch")
    st.markdown(tx["garch"])

with t4:
    var, tot, vs = load("var"), load("totals"), lb[lb["level"] == "item"].set_index("model")
    st.markdown("A VAR model forecasts the three category totals together (Foods, Hobbies, Household), and its forecast is then split down to the items.")
    st.markdown("**Settings chosen at each forecast date** (lag order by AIC, number of differences, and the ADF check per category)")
    st.dataframe(var, hide_index=True, width="stretch")
    st.markdown("**Category totals: error of each model on the three totals** (units a day, summed over the category)")
    tt = tot[["name", "mae", "rmse", "wape"]].rename(columns={"name": "Model", "mae": "MAE", "rmse": "RMSE", "wape": "WAPE"})
    st.dataframe(tt.round(3), hide_index=True, width="stretch")
    if "var" in vs.index:
        v, bb = vs.loc["var"], vs.loc[b]
        st.markdown(f"On the items, VAR has MAE {v['mae']:.2f} and RMSE {v['rmse']:.2f} (rank {int(v['rank'])} of {len(vs)}), against "
                    f"{bb['mae']:.2f} and {bb['rmse']:.2f} for {nm[b]}.")
tells("The failures are spikes and zero days, the tests say which gaps are real, GARCH widens or narrows the band but not the forecast, "
      "and VAR on the totals does not beat the item-level models.")
