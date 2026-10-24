"""Every dashboard page loads without an error, and no page or dashboard file uses course labels or filler words."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from utils.data import SCHEMA  # noqa: E402

PAGES = ["overview", "data_scope", "demand_patterns", "accuracy", "item_explorer", "demand_shifts", "stock_impact",
         "method_limits"]
# built from pieces so this file does not match itself
COURSE = ["M" + "1", "M" + "2", "M" + "3", "mile" + "stone", "mid" + "-sem", "mid" + "sem", "pha" + "se", "syn" + "opsis"]
FILLER = ["rob" + "ust", "lever" + "age", "del" + "ve", "compre" + "hensive", "key " + "takeaways"]
COURSE_RE = re.compile(r"\b(" + "|".join(COURSE) + r")\b", re.IGNORECASE)
FILLER_RE = re.compile(r"\b(" + "|".join(FILLER) + r")\b", re.IGNORECASE)


def page_text(at):
    """Every piece of text the page shows: titles, markdown, captions, labels, messages."""
    parts = []
    for group in [at.title, at.header, at.subheader, at.markdown, at.caption, at.info, at.warning, at.success, at.text]:
        parts += [str(e.value) for e in group]
    for group in [at.radio, at.selectbox, at.multiselect, at.toggle, at.select_slider, at.expander]:
        parts += [str(getattr(e, "label", "")) for e in group]
    return "\n".join(parts)


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_cleanly(page):
    if not SCHEMA["items"][0].exists():
        pytest.skip("results not generated")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    text = page_text(at)
    assert len(text) > 500, "the page shows almost no text"
    assert not COURSE_RE.search(text), COURSE_RE.findall(text)
    assert not FILLER_RE.search(text), FILLER_RE.findall(text)
    assert "What this tells us" in "\n".join(str(e.value) for e in at.markdown), "every page ends with a closing box"


def test_dashboard_files_have_no_course_or_filler_words():
    hits = []
    for p in (ROOT / "dashboard").rglob("*.py"):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for rx in (COURSE_RE, FILLER_RE):
                m = rx.search(line)
                if m:
                    hits.append(f"{p.relative_to(ROOT).as_posix()}:{n}: {m.group(0)}")
    assert not hits, "\n".join(hits)
