"""
Run the whole project on Windows without `make`: the notebooks in order, then the facts file, then the tests.
Same steps as the Makefile.

  python scripts/run_all.py                 everything (about 1.5 hours)
  python scripts/run_all.py --m2            only the Milestone 2 notebooks (12-16), facts and tests
  python scripts/run_all.py --m1-eda        only the Milestone 1 EDA notebooks (00, 00a, 00b)
  python scripts/run_all.py --skip-notebooks   facts and tests only
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

M1_EDA = ["00_data_scoping_eda", "00a_data_audit", "00b_cross_category_scoping"]
M1 = ["00b_cross_category_scoping", "01_structural_break_detection", "02_forecasting_static_vs_adaptive",
      "03_inventory_decision_layer", "04_powerbi_export", "05_dashboard_export", "06_detector_calibration",
      "07_high_confidence_breaks", "08_adaptation_variants", "09_forecast_calibration", "10_inventory_robustness",
      "11_replication_tx_wi"]
M2 = ["12_stationarity_decomposition", "13_classical_models", "14_model_diagnostics", "15_shift_aware_classical",
      "16_classical_stock_link"]


def run(cmd, label):
    """Run one step and stop everything if it fails."""
    print(f"== {label}", flush=True)
    t0 = time.time()
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"FAILED: {label}")
    print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    args = sys.argv[1:]
    if "--skip-notebooks" in args:
        notebooks = []
    elif "--m2" in args:
        notebooks = M2
    elif "--m1-eda" in args:
        notebooks = M1_EDA
    else:
        notebooks = M1 + M2
    for nb in notebooks:
        run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=20000", f"notebooks/{nb}.ipynb"], nb)
    run([PY, "scripts/build_facts.py"], "facts")
    run([PY, "scripts/update_readme.py"], "README key results")
    run([PY, "-m", "pytest", "tests", "-q"], "tests")


if __name__ == "__main__":
    main()
