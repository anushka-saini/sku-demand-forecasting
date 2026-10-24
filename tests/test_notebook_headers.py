"""Every notebook opens with a title and a short paragraph; the restructured ones have been run cleanly."""
import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"
# the *_ORIGINAL notebook is the owner's backup copy, not part of the project
NOTEBOOKS = sorted(p for p in NB.glob("*.ipynb") if "_ORIGINAL" not in p.name)
ORDER = ["01_meet_the_data", "02_scope", "03_explore", "04_zeros_and_trimming", "05_stationarity", "06_decomposition",
         "07a_classical_forecasting", "07b_classical_residuals", "08a_break_detection", "08b_detector_calibration",
         "08c_high_confidence_breaks", "08d_shift_aware_classical", "09a_lightgbm", "09b_lightgbm_after_breaks",
         "10_model_comparison", "11a_uncertainty", "11b_inventory_simulation", "11c_calibrated_safety_stock",
         "11d_inventory_other_seasons", "11e_classical_forecasts_and_stock", "11f_texas_and_wisconsin",
         "12_advanced_models_wip", "90_powerbi_export", "91_dashboard_export"]
RESTRUCTURED = [NB / f"{n}.ipynb" for n in ORDER if (NB / f"{n}.ipynb").exists()]


def cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def text(cell):
    return "".join(cell["source"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_opens_with_a_title_and_a_paragraph(path):
    first = cells(path)[0]
    assert first["cell_type"] == "markdown"
    lines = text(first).strip().splitlines()
    assert lines[0].startswith("# "), "first line should be the title"
    assert len(" ".join(lines[1:]).strip()) > 80, "the title should be followed by a short opening paragraph"


@pytest.mark.parametrize("path", RESTRUCTURED, ids=lambda p: p.name)
def test_restructured_notebooks_were_run_cleanly(path):
    code = [c for c in cells(path) if c["cell_type"] == "code"]
    assert all(c.get("execution_count") for c in code), "run it top to bottom with nbconvert"
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{path.name} has error outputs"
    assert "written" in text(code[-1]), "the last cell should list the files written"
