"""Shared layout pieces: chart styling, headers, captions, glossary."""
import plotly.graph_objects as go
import streamlit as st

from utils.kpis import is_small

ACCENT = "#0072B2"        # Okabe-Ito blue: static / main series
SECOND = "#E69F00"        # Okabe-Ito orange: adaptive (always paired with a different dash/marker)
NEUTRAL = "#8C8C8C"       # seasonal naive / context
DARK = "#4D4D4D"

POLICY_STYLE = {
    "lightgbm_static": dict(color=ACCENT, dash="solid", symbol="circle", label="LightGBM static"),
    "lightgbm_adaptive": dict(color=SECOND, dash="dash", symbol="diamond", label="LightGBM adaptive"),
    "seasonal_naive_static": dict(color=NEUTRAL, dash="dot", symbol="square", label="Seasonal naive"),
}
VARIANT_STYLE = {"static": POLICY_STYLE["lightgbm_static"], "adaptive": POLICY_STYLE["lightgbm_adaptive"]}
CHART_HEIGHT = 380

GLOSSARY = {
    "Structural break": "A lasting change in an item's typical demand level, as opposed to normal week-to-week noise.",
    "CUSUM": "A running total of how far each week's sales sit above or below a baseline. When the total drifts past a "
             "threshold, a break is flagged.",
    "PELT": "A second break-finding method that looks at the whole series at once. Used here only as a cross-check.",
    "MASE": "Mean Absolute Scaled Error: average forecast error divided by the error of simply repeating last week. "
            "Below 1 is better than that simple rule; lower is better.",
    "Static vs adaptive": "Static models train on all history. Adaptive models retrain only on data after the most "
                          "recently detected break.",
    "Placebo": "A control: items without a break get a random, equally short training window, to check whether any "
               "gain comes from the break or just from using recent data.",
    "Service level": "The target chance of not running out during a replenishment cycle (90 / 95 / 99%). It sets how "
                     "much safety stock is held.",
    "Fill rate": "Share of customer demand actually served from stock (measured in the simulation).",
    "Safety stock": "Extra units held to cover forecast error during the lead time.",
    "Reorder point": "Stock level that triggers a new order: expected demand over the lead time plus safety stock.",
    "ABC-XYZ": "ABC ranks items by units sold (A = top 70%). XYZ ranks by variability (X = steady, Z = erratic). "
               "AX/AY items are high-volume and reasonably predictable.",
}


def style(fig, height=CHART_HEIGHT, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      font=dict(size=13))
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def chart(fig, takeaway, how_to_read, key=None, **kwargs):
    """Every chart: a takeaway title, a how-to-read line, and the figure (tooltips are set per trace)."""
    st.markdown(f"#### {takeaway}")
    st.caption(f"How to read this: {how_to_read}")
    return st.plotly_chart(fig, width="stretch", key=key, **kwargs)


def header(title, intro):
    st.title(title)
    st.markdown(f"<p class='intro'>{intro}</p>", unsafe_allow_html=True)


def n_note(n, what="items"):
    """Show n, and warn when a subgroup is small."""
    if is_small(n):
        st.warning(f"Only n = {n} {what}. Too few for reliable conclusions (fewer than 20).")
    else:
        st.caption(f"n = {n:,} {what}")


def assumed(text):
    st.markdown(f"<span class='assumed'>ASSUMED</span> {text}", unsafe_allow_html=True)


def derived(text):
    st.markdown(f"<span class='derived'>DERIVED</span> {text}", unsafe_allow_html=True)


CSS = """
<style>
.block-container {max-width: 1200px; padding-top: 2.2rem;}
p.intro {font-size: 1.05rem; opacity: 0.85; max-width: 60rem;}
.headline {font-size: 1.3rem; line-height: 1.5; border-left: 4px solid #0072B2; padding: 0.6rem 1rem;
           margin: 0.5rem 0 1.5rem 0; background: rgba(0,114,178,0.07); border-radius: 4px;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; padding: 0.8rem 1rem;}
.assumed, .derived {font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; padding: 0.1rem 0.45rem;
                    border-radius: 3px; border: 1px solid currentColor;}
.assumed {color: #B45F00;}
.derived {color: #0072B2;}
h4 {margin-top: 1.4rem; margin-bottom: 0.1rem;}
.pipeline {display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.4rem; margin: 0.5rem 0 1rem 0;}
.pipeline .step {flex: 1 1 130px; border: 1px solid rgba(128,128,128,0.35); border-radius: 6px;
                 padding: 0.55rem 0.65rem; font-size: 0.85rem; line-height: 1.35;}
.pipeline .step b {display: block; margin-bottom: 0.2rem;}
.pipeline .step span {opacity: 0.75;}
.pipeline .arrow {align-self: center; opacity: 0.5;}
.verdict {border: 1px solid rgba(128,128,128,0.3); border-radius: 8px; padding: 0.9rem 1rem; height: 100%;}
.verdict .q {font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.35rem;}
.verdict .badge {display: inline-block; font-weight: 700; font-size: 1.15rem; padding: 0.1rem 0.6rem; border-radius: 4px;
                 border: 2px solid currentColor; margin-right: 0.4rem;}
.verdict .yes {color: #0072B2;} .verdict .no {color: #B45F00;} .verdict .partly {color: #7F7F7F;}
.verdict .strength {font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.8;}
.verdict .ans {margin: 0.55rem 0 0.35rem 0; line-height: 1.4;}
.verdict .cav {font-size: 0.82rem; opacity: 0.75;}
.footer {font-size: 0.78rem; opacity: 0.7; border-top: 1px solid rgba(128,128,128,0.3); margin-top: 2.5rem; padding-top: 0.6rem;}
</style>
"""


def sidebar_filters(categories, segments):
    """Global filters, kept in session state across pages (widgets live in the entry point)."""
    with st.sidebar:
        st.markdown("**Filters (all pages)**")
        st.multiselect("Category", categories, default=categories, key="g_cat")
        st.multiselect("Segment", segments, default=segments, key="g_seg", help="Only AX and AY items were modelled.")
        st.radio("Target service level", [0.90, 0.95, 0.99], index=1, key="g_sl", horizontal=True,
                 format_func=lambda x: f"{x:.0%}")


def filters():
    """(categories, segments, service level) from the global sidebar filters."""
    return (st.session_state.get("g_cat") or None, st.session_state.get("g_seg") or None,
            st.session_state.get("g_sl", 0.95))


def sidebar_glossary():
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Glossary**")
        for term, text in GLOSSARY.items():
            with st.expander(term):
                st.write(text)
        st.caption("Data: M5 (Walmart), California stores, 2011-01-29 to 2016-05-22. All numbers are read from "
                   "pre-computed result files in outputs/tables/.")


def vline(fig, x, text, color=DARK, dash="dot", row=None, col=None):
    """Vertical marker with a text label (shape + annotation; works with datetime axes)."""
    kw = dict(row=row, col=col) if row else {}
    fig.add_vline(x=x, line_width=1.5, line_dash=dash, line_color=color, **kw)
    fig.add_annotation(x=x, y=1, yref="y domain" if not row else "paper", text=text, showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=color), bgcolor="rgba(0,0,0,0)")
    return fig


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=200, legend=False)


def assumed_chart(fig):
    """Distinct style for charts driven by ASSUMED costs: dashed frame and a persistent label."""
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#B45F00", width=1.5, dash="dash"), layer="below")
    fig.add_annotation(text="ASSUMED COSTS", xref="paper", yref="paper", x=1, y=1, xanchor="right", yanchor="bottom",
                       showarrow=False, font=dict(size=10, color="#B45F00"))
    return fig


def verdict_card(v):
    cls = {"Yes": "yes", "No": "no", "Partly": "partly"}.get(v["verdict"], "yes")
    st.markdown(f"<div class='verdict'><div class='q'>{v['question']}</div>"
                f"<span class='badge {cls}'>{v['verdict']}</span><span class='strength'>{v['strength']} evidence</span>"
                f"<div class='ans'>{v['answer']}</div><div class='cav'>Caveat: {v['caveat']}</div></div>",
                unsafe_allow_html=True)


def footer(f):
    """Scope line on every page, built from the facts file."""
    sc = f["scope"]
    w = sc["windows"]
    rep = f.get("replication_scope", {})
    rep_txt = ", ".join(f"{k} ({v['value']} items)" for k, v in rep.items())
    st.markdown(
        f"<div class='footer'>Scope: California, {sc['ax_ay_items']['value']} high-volume (AX/AY) items, sales summed "
        f"across stores. Forecast evaluation {w['W1']['first_eval_day']} to {w['W1']['last_eval_day']}; inventory also "
        f"checked on two earlier 168-day windows. Replicated in {rep_txt}. M5 ends {w['W1']['last_eval_day']}, so there "
        f"is no COVID-era data.</div>", unsafe_allow_html=True)
