"""Every notebook starts with the standard header; Milestone 2 notebooks (12+) also follow the 8-section template."""
import json
import re
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"
FIELDS = ["Milestone", "Task", "Stage", "Aim", "Inputs", "Outputs", "Hypotheses", "Expected runtime"]
SECTIONS = ["Setup and config", "Load data", "Method", "Run", "Results", "Checks", "Interpretation", "What comes next"]
# The *_ORIGINAL notebook is the owner's untouched backup copy of notebook 00, not part of the pipeline.
NOTEBOOKS = sorted(p for p in NB.glob("*.ipynb") if "_ORIGINAL" not in p.name)


def cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def text(cell):
    return "".join(cell["source"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_header_has_every_field(path):
    first = cells(path)[0]
    assert first["cell_type"] == "markdown"
    missing = [f for f in FIELDS if f"| **{f}** |" not in text(first)]
    assert not missing, f"{path.name} header is missing: {missing}"


M2 = [p for p in NOTEBOOKS if int(re.match(r"(\d+)", p.name).group(1)) >= 12]


@pytest.mark.parametrize("path", M2, ids=lambda p: p.name)
def test_milestone2_notebooks_follow_the_template(path):
    md = [text(c) for c in cells(path) if c["cell_type"] == "markdown"]
    found = [m.group(1) for t in md for m in [re.match(r"## \d\. (.+)", t.splitlines()[0])] if m]
    assert found == SECTIONS, f"{path.name} sections: {found}"
    assert "WRITTEN" in text(cells(path)[-1]), "last cell should list the files written"


@pytest.mark.parametrize("path", M2, ids=lambda p: p.name)
def test_milestone2_notebooks_were_run(path):
    code = [c for c in cells(path) if c["cell_type"] == "code"]
    assert all(c.get("execution_count") for c in code), "run it top to bottom with nbconvert"
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{path.name} has error outputs"
