"""Rules for the time-series work: SARIMA still gives its 07a numbers, and the new files use no course labels."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import EVAL_ORIGINS  # noqa: E402
from metrics import mae, rmse  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
# built from pieces so this file does not match itself
WORDS = ["M" + "1", "M" + "2", "M" + "3", "mile" + "stone", "mid" + "-sem", "mid" + "sem", "pha" + "se", "syn" + "opsis"]
PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in WORDS) + r")\b", re.IGNORECASE)
NEW = ["docs/MODEL_COVERAGE.md", "docs/ENVIRONMENT.md", "docs/PREREGISTRATION_TS.md", "docs/TS_MODELS.md",
       "requirements-extra.txt", "src/ts_models.py", "notebooks/07[c-i]_*.ipynb", "tests/test_ts_*.py",
       "notebooks/09b_*.ipynb", "dashboard/views/accuracy.py", "outputs/tables/ts_*.csv",
       "outputs/tables/runner_up_726_*.csv", "outputs/tables/verdict_*.csv", "src/model_verdict.py"]


def new_files():
    for pattern in NEW:
        yield from (p for p in ROOT.glob(pattern) if p.is_file())


def text_of(path):
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8", errors="ignore")
    parts = []
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        parts.append("".join(cell["source"]))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", "")))
            parts.append("".join(out.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def test_new_files_have_no_course_words():
    hits = []
    for p in new_files():
        for n, line in enumerate(text_of(p).splitlines(), 1):
            m = PATTERN.search(line)
            if m:
                hits.append(f"{p.relative_to(ROOT).as_posix()}:{n}: {m.group(0)}")
    assert not hits, "\n".join(hits[:40])


NEW_NOTEBOOKS = ["07c_ar_ma_arma_arima", "07d_sarima_family", "07e_varima", "07f_theta_prophet_autots",
                 "07h_time_series_comparison", "09b_runner_up_all_products", "07i_model_verdict"]


@pytest.mark.parametrize("name", NEW_NOTEBOOKS)
def test_new_notebooks_were_run_cleanly(name):
    path = ROOT / "notebooks" / f"{name}.ipynb"
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    assert all(c.get("execution_count") for c in code), "run it top to bottom with nbconvert"
    assert not [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert "written" in "".join(code[-1]["source"]), "the last cell should list the files written"
    assert "To be written after the run" not in "\n".join("".join(c["source"]) for c in cells)


def test_sarima_still_gives_the_07a_numbers():
    path = TABLES / "classical_forecasts.parquet"
    if not path.exists():
        pytest.skip("results not generated")
    fc = pd.read_parquet(path)
    fc = fc[(fc["model"] == "sarima") & fc["origin"].isin(EVAL_ORIGINS)]
    per = fc.groupby(["item_id", "origin"]).apply(lambda g: pd.Series({"rmse": rmse(g["actual"], g["forecast"]),
                                                                       "mae": mae(g["actual"], g["forecast"])}))
    per = per.groupby("item_id").mean()
    assert len(per) == 150
    assert round(per["rmse"].median(), 3) == 4.445
    assert round(per["mae"].median(), 3) == 3.522


def test_new_sarima_base_equals_07a():
    """07d refits plain SARIMA at every origin; at the scored origins it must be the 07a forecast."""
    new, old = TABLES / "ts_sarima_forecasts.parquet", TABLES / "classical_forecasts.parquet"
    if not new.exists() or not old.exists():
        pytest.skip("results not generated")
    a = pd.read_parquet(new, filters=[("model", "==", "sarima_base")])
    a = a[a["origin"].isin(EVAL_ORIGINS)]
    b = pd.read_parquet(old, filters=[("model", "==", "sarima")])
    x = a.merge(b, on=["item_id", "origin", "date"])
    assert len(x) == 150 * len(EVAL_ORIGINS) * 28
    assert np.allclose(x["forecast_x"], x["forecast_y"])
