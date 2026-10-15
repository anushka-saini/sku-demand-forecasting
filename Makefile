# Reproduce everything from the raw M5 files (see docs/DATA.md for the download).
# On Windows, run from Git Bash or install make (e.g. `choco install make`); or run the commands by hand.

PY ?= python
RUN = $(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=20000

NOTEBOOKS = \
	notebooks/00b_cross_category_scoping.ipynb \
	notebooks/01_structural_break_detection.ipynb \
	notebooks/02_forecasting_static_vs_adaptive.ipynb \
	notebooks/03_inventory_decision_layer.ipynb \
	notebooks/04_powerbi_export.ipynb \
	notebooks/05_dashboard_export.ipynb \
	notebooks/06_detector_calibration.ipynb \
	notebooks/07_high_confidence_breaks.ipynb \
	notebooks/08_adaptation_variants.ipynb \
	notebooks/09_forecast_calibration.ipynb \
	notebooks/10_inventory_robustness.ipynb \
	notebooks/11_replication_tx_wi.ipynb

.PHONY: all notebooks facts test dashboard

all: notebooks facts test

notebooks:
	@for nb in $(NOTEBOOKS); do echo "== $$nb"; $(RUN) $$nb || exit 1; done

facts:
	$(PY) scripts/build_facts.py

test:
	$(PY) -m pytest tests -q

dashboard:
	$(PY) -m streamlit run dashboard/app.py
