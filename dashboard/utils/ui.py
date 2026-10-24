"""Shared layout pieces: header, KPI cards, charts, the closing box, filters, downloads."""
import html
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

FIGS = Path(__file__).resolve().parents[2] / "outputs" / "figures"

ACCENT = "#1F6F8B"          # the one interface accent: links, the closing box, single-series charts
NEUTRAL = "#8C8C8C"
DARK = "#5B6670"
GOOD, BAD = "#2E7D5B", "#B0463C"
CHART_HEIGHT = 360

CSS = """
<style>
.block-container {max-width: 1240px; padding-top: 3.4rem; padding-bottom: 3rem;}
.topbar {display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.4rem 0.9rem; padding: 0.15rem 0 0.7rem 0;
         margin-bottom: 0.6rem; border-bottom: 1px solid rgba(128,128,128,0.25);}
.topbar .brand {font-weight: 700; font-size: 1.02rem; letter-spacing: 0.01em;}
.topbar .sub {font-size: 0.86rem; opacity: 0.7;}
h1 {font-size: 2.0rem !important; margin-bottom: 0.1rem !important;}
p.question {font-size: 1.06rem; opacity: 0.85; margin: 0 0 1.1rem 0; max-width: 62rem;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 0.8rem; margin: 0.3rem 0 1.4rem 0;}
.kpi {border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; padding: 0.8rem 0.95rem 0.75rem 0.95rem;}
.kpi .label {font-size: 0.8rem; opacity: 0.72; display: flex; gap: 0.35rem; align-items: center;}
.kpi .info {display: inline-flex; width: 1.0rem; height: 1.0rem; border-radius: 50%; border: 1px solid currentColor;
            font-size: 0.66rem; align-items: center; justify-content: center; cursor: help; opacity: 0.8;}
.kpi .value {font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin-top: 0.2rem; font-variant-numeric: tabular-nums;}
.kpi .compare {font-size: 0.8rem; opacity: 0.75; margin-top: 0.15rem;}
.kpi .compare.good {color: #2E7D5B; opacity: 1;} .kpi .compare.bad {color: #B0463C; opacity: 1;}
.tells {border-left: 4px solid #1F6F8B; background: rgba(31,111,139,0.08); border-radius: 4px; padding: 0.7rem 1rem;
        margin: 1.6rem 0 0.4rem 0; max-width: 62rem;}
.tells .head {font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 700; color: #1F6F8B;
              margin-bottom: 0.2rem;}
.simulated {border: 1px dashed #B26B00; border-radius: 4px; padding: 0.45rem 0.8rem; font-size: 0.9rem; margin: 0.2rem 0 1rem 0;}
.simulated b {color: #B26B00;}
.takeaway {font-size: 0.92rem; margin: -0.4rem 0 1.4rem 0; opacity: 0.85;}
.chart-title {font-size: 1.08rem; font-weight: 600; margin: 1.0rem 0 0.1rem 0;}
@media (prefers-color-scheme: dark) {
  .kpi .compare.good {color: #5CC08F;} .kpi .compare.bad {color: #E8857B;}
  .tells .head {color: #6FB6CF;} .tells {border-left-color: #6FB6CF; background: rgba(111,182,207,0.10);}
  .simulated b {color: #E0A040;}
}
.footer {font-size: 0.78rem; opacity: 0.65; border-top: 1px solid rgba(128,128,128,0.3); margin-top: 2.5rem; padding-top: 0.6rem;}
.filters-used {font-size: 0.8rem; opacity: 0.7; margin: -0.6rem 0 1rem 0;}
</style>
"""


def topbar(items, ca_items):
    st.markdown(f"<div class='topbar'><span class='brand'>SKU Demand Forecasting</span>"
                f"<span class='sub'>Walmart M5 · California stores · {items:,} of {ca_items:,} items</span></div>",
                unsafe_allow_html=True)


def page(title, question):
    st.title(title)
    st.markdown(f"<p class='question'>{html.escape(question)}</p>", unsafe_allow_html=True)


def kpis(cards):
    """cards: list of dicts with label, value, compare (optional), tone ('good'/'bad'/None) and help (optional)."""
    parts = []
    for c in cards:
        info = (f"<span class='info' title='{html.escape(c['help'], quote=True)}'>i</span>" if c.get("help") else "")
        tone = f" {c['tone']}" if c.get("tone") else ""
        comp = f"<div class='compare{tone}'>{html.escape(c['compare'])}</div>" if c.get("compare") else ""
        parts.append(f"<div class='kpi'><div class='label'>{html.escape(c['label'])}{info}</div>"
                     f"<div class='value'>{html.escape(c['value'])}</div>{comp}</div>")
    st.markdown(f"<div class='kpis'>{''.join(parts)}</div>", unsafe_allow_html=True)


def style(fig, height=CHART_HEIGHT, legend=True):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=12, b=8), hovermode="closest", separators=".,",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
                      font=dict(size=13))
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.4)", automargin=True)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.16)", zerolinecolor="rgba(128,128,128,0.4)", automargin=True)
    return fig


def chart(fig, title, takeaway, key=None):
    """One chart: a title that states the finding, the chart, then a one-line takeaway under it."""
    st.markdown(f"<div class='chart-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    st.markdown(f"<div class='takeaway'>{html.escape(takeaway)}</div>", unsafe_allow_html=True)


def tells(sentences):
    """The closing 'What this tells us' box, built from the page's computed values."""
    st.markdown(f"<div class='tells'><div class='head'>What this tells us</div>{html.escape(sentences)}</div>",
                unsafe_allow_html=True)


def simulated(text):
    st.markdown(f"<div class='simulated'><b>Simulated.</b> {html.escape(text)}</div>", unsafe_allow_html=True)


def download(df, filename, label="Download as CSV", key=None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)


def empty(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=180, legend=False)


def vline(fig, x, text, color=DARK, dash="dot"):
    fig.add_vline(x=x, line_width=1.5, line_dash=dash, line_color=color)
    fig.add_annotation(x=x, y=1, yref="y domain", text=text, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color=color))
    return fig


# ---------------------------------------------------------------- filters (widgets live in app.py, so they persist)
SHIFT_GROUPS = ["All items", "With a detected shift", "Without a detected shift"]


def filters():
    s = st.session_state
    return {"categories": s.get("g_cat") or [], "segments": s.get("g_seg") or [],
            "shift": s.get("g_shift", SHIFT_GROUPS[0]), "window": s.get("g_window")}


def filters_used(names):
    """A small line under the title saying which sidebar filters this page applies."""
    f = filters()
    parts = []
    if "category" in names:
        parts.append("category: " + (", ".join(c.title() for c in f["categories"]) or "none selected"))
    if "segment" in names:
        parts.append("item group: " + (", ".join(f["segments"]) or "none selected"))
    if "shift" in names:
        parts.append(f["shift"].lower())
    if "window" in names and f["window"]:
        a, b = f["window"]
        parts.append(f"forecast dates {a:%d %b %Y} to {b:%d %b %Y}")
    st.markdown(f"<div class='filters-used'>Filters in use: {html.escape('; '.join(parts))}</div>", unsafe_allow_html=True)


def apply_item_filters(items, f, shift_col="has_shift"):
    """Filter the item table by the sidebar's category, item group and shift group."""
    d = items[items["in_scope"]] if "in_scope" in items else items
    if f["categories"]:
        d = d[d["category"].isin(f["categories"])]
    if f["segments"] and "segment" in d:
        d = d[d["segment"].isin(f["segments"])]
    if f["shift"] == SHIFT_GROUPS[1]:
        d = d[d[shift_col].astype(bool)]
    elif f["shift"] == SHIFT_GROUPS[2]:
        d = d[~d[shift_col].astype(bool)]
    return d


def footer(f):
    sc, w = f["scope"], f["scope"]["windows"]["W1"]
    st.markdown(f"<div class='footer'>California, {sc['ax_ay_items']['value']:,} high-volume items, sales added across "
                f"the four stores. Forecasts tested on {w['first_eval_day']} to {w['last_eval_day']}. All numbers are "
                f"read from the saved results in outputs/ when the page loads.</div>", unsafe_allow_html=True)
