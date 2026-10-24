"""No course or planning labels in notebooks, docs, dashboard, code or tests: the files describe the work only."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# built from pieces so this file does not match itself
WORDS = ["M" + "1", "M" + "2", "M" + "3", "mile" + "stone", "mid" + "-sem", "mid" + "sem", "phase" + " 1", "syn" + "opsis"]
PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in WORDS) + r")\b", re.IGNORECASE)
# the user's own untouched copy of the first notebook
SKIP = {"notebooks/00_data_scoping_eda_ORIGINAL.ipynb"}


def files():
    for pattern in ["notebooks/*.ipynb", "docs/**/*.md", "dashboard/**/*.py", "src/*.py", "scripts/*.py",
                    "tests/*.py", "powerbi/*.md", "*.md", "Makefile", "requirements.txt", "run_all.bat"]:
        for p in ROOT.glob(pattern):
            if p.is_file() and p.relative_to(ROOT).as_posix() not in SKIP:
                yield p


def text_of(path):
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8", errors="ignore")
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        parts.append("".join(cell["source"]))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", "")))
            data = out.get("data", {})
            parts.append("".join(data.get("text/plain", "")))      # images are base64 and can match by chance
    return "\n".join(parts)


def test_no_course_words():
    hits = []
    for p in files():
        for n, line in enumerate(text_of(p).splitlines(), 1):
            m = PATTERN.search(line)
            if m:
                hits.append(f"{p.relative_to(ROOT).as_posix()}:{n}: {m.group(0)}")
    assert not hits, "\n".join(hits[:40])
