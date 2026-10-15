import streamlit as st

from utils.data import get
from utils.facts import facts, lookup
from utils.kpis import BAND_FLAG, BIAS_FLAG, build_watchlist
from utils.ui import assumed, derived, filters, header, n_note

f = facts()
cal_items = get("calibrated_items")
item_bias = get("item_bias")
item_cov = get("item_coverage")
items = get("items")
hc_items = get("hc_items")
cats, segs, sl = filters()

header("What should I check on Monday?",
       "Items whose simulated fill rate fell below the target service level, with the likely reasons. Forecast: "
       "LightGBM static, 7-day lead time.")

method = st.radio("Safety-stock method", ["conformal", "normal formula"], horizontal=True,
                  format_func={"conformal": "Calibrated (recommended)", "normal formula": "Textbook normal formula"}.get)
wl = build_watchlist(cal_items, item_bias, item_cov, items, hc_items, method, sl)
fact_n = lookup(f, "planner", f"{method}|{sl}")
if len(wl) != fact_n["value"]:
    st.error(f"Watchlist has {len(wl)} items but results_facts.json says {fact_n['value']}. Re-run scripts/build_facts.py.")
    st.stop()
total = len(wl)
if cats:
    wl = wl[wl["category"].isin(cats)]
if segs:
    wl = wl[wl["segment"].isin(segs)]

other = lookup(f, "planner", f"{'normal formula' if method == 'conformal' else 'conformal'}|{sl}")["value"]
st.markdown(f"<div class='headline'><b>{total} of {fact_n['n']} items</b> fell below the {sl:.0%} target with this "
            f"method ({other} with the {'textbook formula' if method == 'conformal' else 'calibrated method'}).</div>",
            unsafe_allow_html=True)
derived(f"Reasons are display rules, not tests: bias beyond ±{BIAS_FLAG:.0%}, a calibrated 90% band covering fewer than "
        f"{BAND_FLAG:.0%} of days, a detected shift, or intermittent demand.")
assumed("Lead time (7 days) and the costs behind 'total cost' are assumptions.")

if wl.empty:
    st.info("No items below target for the current filters.")
else:
    st.caption("How to read this: sorted by units of lost sales. Click a column header to re-sort.")
    first = ["item_id", "reasons", "fill_rate", "lost_units"]
    show = wl[first + [c for c in wl.columns if c not in first]].copy()
    for c in ["fill_rate", "forecast_bias", "band_coverage_90"]:
        show[c] = 100 * show[c]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "item_id": "Item", "category": "Category", "segment": "Segment",
        "fill_rate": st.column_config.NumberColumn("Fill rate", format="%.1f%%"),
        "lost_units": st.column_config.NumberColumn("Lost units", format="%.0f"),
        "stockout_days": st.column_config.NumberColumn("Stockout days", format="%d"),
        "total_cost": st.column_config.NumberColumn("Total cost (ASSUMED)", format="$%.0f"),
        "mean_safety_stock": st.column_config.NumberColumn("Safety stock (avg units)", format="%.1f"),
        "forecast_bias": st.column_config.NumberColumn("Forecast bias", format="%+.1f%%"),
        "band_coverage_90": st.column_config.NumberColumn("90% band coverage", format="%.0f%%"),
        "shift_detected": st.column_config.CheckboxColumn("Shift flagged"),
        "high_confidence_shift": st.column_config.CheckboxColumn("High-confidence shift"),
        "intermittent": st.column_config.CheckboxColumn("Intermittent"),
        "reasons": st.column_config.TextColumn("Why it may be short", width="large")})
    n_note(len(wl), "items on the watchlist")
    st.download_button("Download watchlist (CSV)", wl.to_csv(index=False).encode(),
                       file_name=f"watchlist_{method.replace(' ', '_')}_{int(sl * 100)}.csv", mime="text/csv")

    counts = {"under-forecast": wl["reasons"].str.contains("under-forecast").sum(),
              "band misses often": wl["reasons"].str.contains("band covers").sum(),
              "shift flagged": wl["shift_detected"].sum(),
              "intermittent": wl["intermittent"].sum()}
    st.markdown("**Most common reasons on this list:** " + ", ".join(f"{k} {int(v)}" for k, v in counts.items()))
