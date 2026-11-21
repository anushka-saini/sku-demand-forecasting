"""The final results hang together: the leaderboard, the facts file, the report and the slides show the same numbers."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
REPORT = ROOT / "outputs" / "report"
pytestmark = pytest.mark.skipif(not (T / "final_leaderboard.csv").exists(), reason="notebook 13 not run")


def lb():
    d = pd.read_csv(T / "final_leaderboard.csv")
    return d[d["level"] == "item"].sort_values("rank")


def test_leaderboard_is_complete_and_ranked():
    d = lb()
    assert len(d) == 21 and d["model"].is_unique
    assert list(d["rank"]) == list(range(1, 22))
    for c in ["mae", "rmse", "mape_nonzero", "wape"]:
        assert d[c].notna().all()
    nn = d[d["family"] == "Deep learning"]
    assert len(nn) == 3 and nn["mae_std"].notna().all()


def test_best_model_is_the_top_row_and_beats_seasonal_naive():
    d = lb()
    info = json.loads((T / "final_best_model.json").read_text())
    assert info["model"] == d.iloc[0]["model"]
    n = d[d["model"] == "seasonal_naive"].iloc[0]
    assert d.iloc[0]["mae"] < n["mae"] and d.iloc[0]["rmse"] < n["rmse"]


def test_json_and_csv_leaderboards_match():
    j = pd.DataFrame(json.loads((T / "final_leaderboard.json").read_text())).set_index("model")
    c = pd.read_csv(T / "final_leaderboard.csv").set_index("model")
    assert (j["mae"] - c["mae"]).abs().max() < 1e-9


def test_facts_file_matches_the_leaderboard():
    facts = json.loads((ROOT / "outputs" / "results_facts.json").read_text())["final"]
    for m, x in lb().set_index("model").iterrows():
        assert abs(facts["metrics"][f"{m}|mae"]["value"] - x["mae"]) < 1e-9
        assert abs(facts["metrics"][f"{m}|rmse"]["value"] - x["rmse"]) < 1e-9


def test_combinations_use_validation_only_weights():
    w = pd.read_csv(T / "final_stack_weights.csv")
    assert (w["weight"] >= 0).all()
    f = pd.read_parquet(T / "final_forecasts.parquet")
    assert set(f[f["split"] == "validation"]["origin"]).isdisjoint(set(f[f["split"] == "test"]["origin"]))


def test_report_and_slides_quote_the_leaderboard():
    pdf, pptx = REPORT / "final_report.pdf", REPORT / "final_slides.pptx"
    if not (pdf.exists() and pptx.exists()):
        pytest.skip("report not built")
    best = lb().iloc[0]
    want = [f"{best['mae']:.2f}", f"{best['rmse']:.2f}"]
    import pymupdf
    text = "\n".join(p.get_text() for p in pymupdf.open(pdf))
    from pptx import Presentation
    stext = "\n".join(sh.text_frame.text for s in Presentation(pptx).slides for sh in s.shapes if sh.has_text_frame)
    for w in want:
        assert w in text, f"{w} missing from the report"
        assert w in stext, f"{w} missing from the slides"
    assert len(pymupdf.open(pdf)) <= 10
    assert len(Presentation(pptx).slides) in range(12, 15)


def test_no_markdown_files_were_added_for_this_work():
    for p in list((ROOT / "outputs").rglob("*.md")):
        pytest.fail(f"unexpected markdown file {p}")
