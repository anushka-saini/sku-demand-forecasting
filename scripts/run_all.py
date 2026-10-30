"""
Run the project on Windows without `make`: the notebooks in order, then the facts file, the README numbers and the
tests. Same steps as the Makefile.

  python scripts/run_all.py                          everything, in order
  python scripts/run_all.py --only 05_stationarity   only these notebooks (comma-separated), then facts and tests
  python scripts/run_all.py --from 07a_classical_forecasting   from this notebook to the end
  python scripts/run_all.py --skip-notebooks         facts, README and tests only
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# nbconvert's default 'python3' kernel can be another Python install (here the system one, with other numpy/scipy
# versions), and SARIMA fits differ between them. This kernel runs the notebooks in the Python that runs this script.
KERNELS = Path(tempfile.mkdtemp(prefix="run_all_"))
(KERNELS / "kernels" / "run_all").mkdir(parents=True)
(KERNELS / "kernels" / "run_all" / "kernel.json").write_text(json.dumps(
    {"argv": [PY, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "run_all", "language": "python"}))
ENV = {**os.environ, "JUPYTER_PATH": str(KERNELS)}

# the learning path: data, classical and other time-series models, breaks, the best model on all products,
# comparison, stock, exports
ORDER = ["01_meet_the_data", "02_scope", "03_explore", "04_zeros_and_trimming", "05_stationarity", "06_decomposition",
         "07a_classical_forecasting", "07b_classical_residuals",
         "07c_ar_ma_arma_arima", "07d_sarima_family", "07e_varima", "07f_theta_prophet_autots",
         "07h_time_series_comparison", "08a_break_detection", "08b_detector_calibration",
         "08c_high_confidence_breaks", "08d_shift_aware_classical", "09_best_model_all_products", "09b_runner_up_all_products",
         "10_model_comparison", "11a_uncertainty", "11b_inventory_simulation", "11c_calibrated_safety_stock",
         "11d_inventory_other_seasons", "11e_classical_forecasts_and_stock",
         "07i_model_verdict",      # reads 07h's scores, and 09's and 11e's files for its stock check
         "11f_texas_and_wisconsin",
         "90_powerbi_export", "91_dashboard_export"]


def run(cmd, label):
    """Run one step and stop everything if it fails."""
    print(f"== {label}", flush=True)
    t0 = time.time()
    if subprocess.run(cmd, cwd=ROOT, env=ENV).returncode != 0:
        sys.exit(f"FAILED: {label}")
    print(f"   done in {(time.time() - t0) / 60:.1f} min", flush=True)


def pick(args):
    """Which notebooks to run, in path order, keeping only the ones that exist."""
    names = [n for n in ORDER if (ROOT / "notebooks" / f"{n}.ipynb").exists()]
    if "--skip-notebooks" in args:
        return []
    if "--only" in args:
        wanted = args[args.index("--only") + 1].split(",")
        return [n for n in names if n in wanted]
    if "--from" in args:
        start = args[args.index("--from") + 1]
        return names[names.index(start):]
    return names


def main():
    for nb in pick(sys.argv[1:]):
        run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=20000", "--ExecutePreprocessor.kernel_name=run_all",
             f"notebooks/{nb}.ipynb"], nb)
    run([PY, "scripts/build_facts.py"], "facts")
    run([PY, "scripts/update_readme.py"], "README key results")
    run([PY, "-m", "pytest", "tests", "-q"], "tests")


if __name__ == "__main__":
    main()
