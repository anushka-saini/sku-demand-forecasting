"""Every result number in README.md must come from outputs/results_facts.json."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FACTS = ROOT / "outputs" / "results_facts.json"
README = ROOT / "README.md"

pytestmark = pytest.mark.skipif(not FACTS.exists(), reason="results_facts.json not built")


def test_key_results_block_matches_facts():
    import update_readme
    text = README.read_text(encoding="utf-8")
    expected = update_readme.updated_readme(text, json.loads(FACTS.read_text()))
    assert text == expected, "README key results are stale: run python scripts/update_readme.py"


def test_no_result_numbers_outside_generated_block():
    """Percentages and decimals may only appear inside the generated block (dates and versions are allowed)."""
    import update_readme
    text = README.read_text(encoding="utf-8")
    head, rest = text.split(update_readme.START, 1)
    _, tail = rest.split(update_readme.END, 1)
    outside = re.sub(r"```.*?```", "", head + tail, flags=re.S)            # code blocks are commands
    outside = re.sub(r"\d{4}-\d{2}-\d{2}", "", outside)                    # dates
    stray = re.findall(r"\d+\.\d+|\d+(?:\.\d+)?\s?%", outside)
    assert not stray, f"numbers outside the generated block: {stray}"
