"""Checks that the leaderboard numbers are identical in every place they are shown, and lists any mismatch.

Truth = outputs/tables/final_leaderboard.csv. Compared against: results_facts.json, the final report (PDF), the project
explanation (PDF), the slides (PPTX) and the dashboard's leaderboard page (run with Streamlit's test runner).

    python scripts/check_numbers.py [--skip-dashboard]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
REPORT = ROOT / "outputs" / "report"
lb = pd.read_csv(T / "final_leaderboard.csv")
lb = lb[lb["level"] == "item"].reset_index(drop=True)
problems = []


def pdf_text(p):
    import pymupdf
    return "\n".join(pg.get_text() for pg in pymupdf.open(p))


def row_of(text, name, nxt=None):
    """The text of one table row: from the model's name to the next line that starts a new row."""
    i = text.find(name)
    return text[i:i + 160] if i >= 0 else ""


def check_row_values(label, text, strict=True):
    for _, x in lb.iterrows():
        # the name as printed in each document may be shortened; look for the MAE and RMSE strings near the name
        pass


# ---- facts file
facts = json.loads((ROOT / "outputs" / "results_facts.json").read_text())["final"]["metrics"]
for _, x in lb.iterrows():
    for k in ["mae", "rmse", "mape_nonzero", "wape"]:
        v = facts[f"{x['model']}|{k}"]["value"]
        if abs(v - x[k]) > 1e-9:
            problems.append(f"facts: {x['model']} {k} {v} != {x[k]}")

# ---- the numbers as printed in each document: every model's MAE, RMSE, MAPE, WAPE must appear in one table row
def find_row(text, strings):
    """True if one line-group of the text contains all of the strings (a table row split over lines)."""
    flat = re.sub(r"\s+", " ", text)
    return all(s in flat for s in strings)


def check_doc(label, text, name_col="name"):
    """For every model, some occurrence of its name must be followed (within one table row) by its MAE, RMSE, MAPE and WAPE."""
    flat = re.sub(r"\s+", " ", text)
    for _, x in lb.iterrows():
        nm = x[name_col]
        want = {"MAE": f"{x['mae']:.2f}", "RMSE": f"{x['rmse']:.2f}", "MAPE": f"{x['mape_nonzero']:.3f}", "WAPE": f"{x['wape']:.3f}"}
        ok = False
        seen = False
        for m in re.finditer(re.escape(nm), flat):
            seen = True
            seg = flat[m.start():m.start() + 240]
            if all(v in seg for v in want.values()):
                ok = True
                break
        if not seen:
            problems.append(f"{label}: model '{nm}' not found")
        elif not ok:
            problems.append(f"{label}: no table row for '{nm}' has MAE {want['MAE']}, RMSE {want['RMSE']}, MAPE {want['MAPE']}, WAPE {want['WAPE']}")


rep = REPORT / "final_report.pdf"
exp = ROOT / "docs" / "Project_Explanation.pdf"
if rep.exists():
    check_doc("report", pdf_text(rep))
if exp.exists():
    check_doc("explanation", pdf_text(exp))


# ---- other headline numbers: stock costs, interval coverage, significance tests must appear in both PDFs
stock = pd.read_csv(T / "final_stock_summary.csv").set_index(["method", "model"])
gar = pd.read_csv(T / "final_garch_intervals.csv").set_index(["interval", "nominal"])
dmt = pd.read_csv(T / "final_dm_tests.csv")
keys = {}
for meth in ("conformal", "normal formula"):
    for m in ("seasonal_naive", "sarima", "combination", "xgboost"):
        keys[f"stock cost {meth} {m}"] = f"{stock.loc[(meth, m), 'total_cost']:,.0f}"
keys["GARCH 80% coverage"] = f"{gar.loc[('GARCH(1,1)', 0.8), 'coverage'] * 100:.1f}%"
keys["constant 80% coverage"] = f"{gar.loc[('constant std', 0.8), 'coverage'] * 100:.1f}%"
pvals = {f"DM p vs {x['model_b']}": float(x["p_value"]) for _, x in dmt.iterrows() if x["model_b"] in ("sarima", "combination") and x["loss"].startswith("abs")}
for label, path in [("report", rep), ("explanation", exp)]:
    if path.exists():
        flat = re.sub(r"\s+", " ", pdf_text(path))
        for k, v in keys.items():
            if v not in flat:
                problems.append(f"{label}: {k} = {v} not found")
        for k, v in pvals.items():
            if not any(f"{v:.{d}f}" in flat for d in (2, 3, 4)):
                problems.append(f"{label}: {k} = {v:.4f} not found at 2, 3 or 4 decimals")

# ---- slides: top 7 rows of the table on the results slide
sl = REPORT / "final_slides.pptx"
if sl.exists():
    from pptx import Presentation
    prs = Presentation(sl)
    cells = []
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_table:
                cells += [[c.text for c in r.cells] for r in sh.table.rows]
    top = lb.head(7)
    found = 0
    for _, x in top.iterrows():
        for r in cells:
            if len(r) >= 6 and r[1].split(" (")[0] in x["name"] and r[2] == f"{x['mae']:.2f}" and r[3] == f"{x['rmse']:.2f}" and r[4] == f"{x['mape_nonzero']:.3f}" and r[5] == f"{x['wape']:.3f}":
                found += 1
                break
        else:
            problems.append(f"slides: row for {x['name']} not found with matching MAE/RMSE/MAPE/WAPE")

# ---- dashboard
if "--skip-dashboard" not in sys.argv:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300)
    at.run()
    at.switch_page("views/final_leaderboard.py").run()
    at.radio(key="lb_fam").set_value(at.radio(key="lb_fam").options[1]).run()      # all models
    df = at.dataframe[0].value
    df = df.set_index("Model")
    for _, x in lb.iterrows():
        if x["name"] not in df.index:
            problems.append(f"dashboard: {x['name']} missing")
            continue
        r = df.loc[x["name"]]
        for col, k, tol in [("MAE", "mae", 5e-4), ("RMSE", "rmse", 5e-4), ("MAPE (non-zero days)", "mape_nonzero", 5e-4), ("WAPE", "wape", 5e-4)]:
            if abs(float(r[col]) - x[k]) > tol:
                problems.append(f"dashboard: {x['name']} {col} {r[col]} != {x[k]:.4f}")

print("checked", len(lb), "models in: facts JSON, report PDF, explanation PDF, slides, dashboard" if "--skip-dashboard" not in sys.argv else "models in: facts JSON, report PDF, explanation PDF, slides")
print("MISMATCHES:", len(problems))
for p in problems[:40]:
    print(" -", p)
sys.exit(1 if problems else 0)
