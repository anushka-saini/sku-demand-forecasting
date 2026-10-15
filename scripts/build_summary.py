"""
Build a static results summary (docs/results_summary.html, and a PDF copy when a browser is available) from
outputs/results_facts.json. It is the fallback link for when the hosted dashboard is asleep.

Run: python scripts/build_summary.py
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
FACTS = ROOT / "outputs" / "results_facts.json"
HTML_OUT = ROOT / "docs" / "results_summary.html"
PDF_OUT = ROOT / "docs" / "results_summary.pdf"

CSS = """
:root{--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--accent:#0072B2;--warn:#B45F00;--bg:#fff}
@media (prefers-color-scheme:dark){:root{--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--accent:#58a6ff;--warn:#f0a050;--bg:#0d1117}}
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);max-width:960px;margin:2rem auto;padding:0 16px;line-height:1.5}
h1{font-size:1.7rem;margin-bottom:.2rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}
.muted{color:var(--muted);font-size:.9rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:1rem}
.card{border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.q{font-size:.85rem;color:var(--muted)} .badge{font-weight:700;font-size:1.1rem;border:2px solid currentColor;border-radius:4px;padding:0 8px;display:inline-block;margin:.3rem .4rem .3rem 0}
.yes{color:var(--accent)} .no{color:var(--warn)} .partly{color:var(--muted)} .strength{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.cav{font-size:.82rem;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.9rem} td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.assumed{color:var(--warn);font-weight:600;font-size:.75rem;border:1px solid currentColor;border-radius:3px;padding:0 5px}
"""


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def build(f):
    from utils.verdicts import all_verdicts
    e = html.escape
    v = all_verdicts(f)
    sc, br, fc, inv = f["scope"], f["breaks"], f["forecasts"], f["inventory"]
    adj = br["adjusted_share_with_real_shift"]["CUSUM"]
    trunc = fc["adaptation_variants"]["all detected breaks|truncate"]
    cov = fc["band_coverage"]["lightgbm_static_90_origins2-6"]
    base = inv["static_vs_naive_base_95"]
    calc = inv["conformal_vs_normal_cost_change"]["0.95"]
    w = sc["windows"]["W1"]
    cards = "".join(
        f"<div class='card'><div class='q'>{e(x['question'])}</div>"
        f"<span class='badge {({'Yes': 'yes', 'No': 'no', 'Partly': 'partly'}).get(x['verdict'], 'yes')}'>{e(x['verdict'])}</span>"
        f"<span class='strength'>{e(x['strength'])} evidence</span><div>{e(x['answer'])}</div>"
        f"<div class='cav'>Caveat: {e(x['caveat'])}</div></div>" for x in v)
    rows = [
        ("Items modelled", f"{sc['ax_ay_items']['value']} high-volume California items ({pct(sc['ax_ay_share_of_items']['value'])} of items, {pct(sc['ax_ay_share_of_units']['value'])} of units)"),
        ("False alarms (seasonal null)", f"CUSUM {pct(br['seasonal_null_false_alarm_rate']['CUSUM']['value'])}, PELT {pct(br['seasonal_null_false_alarm_rate']['PELT']['value'])}"),
        ("Items with a real shift", f"{pct(adj['value'])} (95% CI {pct(adj['ci95'][0])} to {pct(adj['ci95'][1])}), n = {adj['n']}"),
        ("High-confidence breaks", f"{int(br['high_confidence_breaks']['ALL']['value'])} of {int(br['high_confidence_breaks']['ALL']['n'])}"),
        ("Median forecast error (MASE)", f"LightGBM {fc['median_item_mase']['lightgbm_ALL']['value']:.3f}, ETS {fc['median_item_mase']['ets_ALL']['value']:.3f}, seasonal naive {fc['median_item_mase']['seasonal_naive_ALL']['value']:.3f}"),
        ("Retraining from the shift", f"{trunc['value']:+.4f} MASE (95% CI {trunc['ci95'][0]:+.4f} to {trunc['ci95'][1]:+.4f}), n = {trunc['n']}"),
        ("Calibrated 90% band coverage", f"{pct(cov['value'])} (was {pct(cov['coverage_before'])})"),
        ("Static LightGBM vs naive, 95% target <span class='assumed'>ASSUMED COSTS</span>", f"{pct(abs(base['inventory_change']))} less inventory, {pct(abs(base['value']))} lower cost"),
        ("Calibrated safety stock, 95% target <span class='assumed'>ASSUMED COSTS</span>", f"total cost {calc['value']:+.1%}"),
    ]
    table = "".join(f"<tr><th>{k}</th><td>{e(val)}</td></tr>" for k, val in rows)
    rep = "".join(f"<tr><td>{e(k.split('|')[0])}</td><td>{e(k.split('|')[1])}</td><td>{'holds' if r['holds'] else '<b>does not hold</b>'}</td><td>{e(r['evidence'])}</td></tr>"
                  for k, r in f["replication"].items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demand Shifts Results</title><style>{CSS}</style></head><body>
<h1>Do demand shifts matter for forecasting and stock?</h1>
<p class="muted">Walmart M5, California ({sc['ax_ay_items']['value']} high-volume items, sales summed across stores), forecast evaluation {w['first_eval_day']} to {w['last_eval_day']}, replicated in Texas and Wisconsin. M5 has no COVID-era data. Generated from outputs/results_facts.json.</p>
<div class="cards">{cards}</div>
<h2>Key numbers</h2><table>{table}</table>
<h2>Replication (Texas and Wisconsin, every setting frozen at the California value)</h2>
<table><tr><th>State</th><th>Conclusion</th><th>Result</th><th>Evidence</th></tr>{rep}</table>
<p class="muted">Lead time, holding and stockout costs are assumptions, not observed data. Full results: docs/RESULTS.md. Interactive version: dashboard/app.py.</p>
</body></html>"""


def to_pdf():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:
                browser = p.chromium.launch(channel="msedge")
            page = browser.new_page()
            page.goto(HTML_OUT.as_uri())
            page.emulate_media(color_scheme="light")
            page.pdf(path=str(PDF_OUT), format="A4", margin={"top": "15mm", "bottom": "15mm", "left": "12mm", "right": "12mm"})
            browser.close()
        return True
    except Exception as ex:                       # no browser available: keep the HTML only
        print(f"PDF not written ({type(ex).__name__}); the HTML summary is still up to date.")
        return False


if __name__ == "__main__":
    HTML_OUT.write_text(build(json.loads(FACTS.read_text())), encoding="utf-8")
    print("wrote", HTML_OUT.relative_to(ROOT))
    if "--no-pdf" not in sys.argv and to_pdf():
        print("wrote", PDF_OUT.relative_to(ROOT))
