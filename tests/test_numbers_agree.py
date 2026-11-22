"""The leaderboard numbers are identical in the facts file, the report, the project explanation and the slides
(scripts/check_numbers.py; the dashboard is checked there too when it is run without --skip-dashboard)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_numbers_agree_across_files():
    if not (ROOT / "outputs" / "report" / "final_report.pdf").exists() or not (ROOT / "docs" / "Project_Explanation.pdf").exists():
        pytest.skip("report or explanation not built")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_numbers.py"), "--skip-dashboard"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-1500:]
