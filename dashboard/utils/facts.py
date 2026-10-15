"""Loader for outputs/results_facts.json (built by scripts/build_facts.py)."""
import json
from pathlib import Path

import streamlit as st

from utils.data import ROOT, DataError

FACTS_PATH = ROOT / "outputs" / "results_facts.json"


def read_facts(path=FACTS_PATH):
    path = Path(path)
    if not path.exists():
        raise DataError(f"Missing {path.name}. Run `python scripts/build_facts.py` to create it from outputs/tables/.")
    return json.loads(path.read_text())


def lookup(facts, *keys):
    """facts['a']['b']... with a clear error naming the missing key."""
    d = facts
    for i, k in enumerate(keys):
        if not isinstance(d, dict) or k not in d:
            raise DataError(f"results_facts.json has no entry {' / '.join(keys[:i + 1])}. Re-run scripts/build_facts.py.")
        d = d[k]
    return d


def value(facts, *keys):
    return lookup(facts, *keys)['value']


@st.cache_data(show_spinner=False)
def _load():
    return read_facts()


def facts():
    try:
        return _load()
    except DataError as e:
        st.error(f"**Data problem:** {e}")
        st.stop()
