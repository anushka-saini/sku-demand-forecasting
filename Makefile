# Reproduce everything from the raw M5 files (see docs/DATA.md for the download).
# On Windows without make, use: python scripts/run_all.py   (same steps; see its --help text)

PY ?= python
RUN = $(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=20000

# in learning order; the same list as ORDER in scripts/run_all.py
NOTEBOOKS = \
	notebooks/01_meet_the_data.ipynb \
	notebooks/02_scope.ipynb \
	notebooks/03_explore.ipynb \
	notebooks/04_zeros_and_trimming.ipynb \
	notebooks/05_stationarity.ipynb \
	notebooks/06_decomposition.ipynb \
	notebooks/07a_classical_forecasting.ipynb \
	notebooks/07b_classical_residuals.ipynb \
	notebooks/07c_ar_ma_arma_arima.ipynb \
	notebooks/07d_sarima_family.ipynb \
	notebooks/07e_varima.ipynb \
	notebooks/07f_theta_prophet_autots.ipynb \
	notebooks/07h_time_series_comparison.ipynb \
	notebooks/08a_break_detection.ipynb \
	notebooks/08b_detector_calibration.ipynb \
	notebooks/08c_high_confidence_breaks.ipynb \
	notebooks/08d_shift_aware_classical.ipynb \
	notebooks/09_best_model_all_products.ipynb \
	notebooks/09b_runner_up_all_products.ipynb \
	notebooks/10_model_comparison.ipynb \
	notebooks/11a_uncertainty.ipynb \
	notebooks/11b_inventory_simulation.ipynb \
	notebooks/11c_calibrated_safety_stock.ipynb \
	notebooks/11d_inventory_other_seasons.ipynb \
	notebooks/11e_classical_forecasts_and_stock.ipynb \
	notebooks/11f_texas_and_wisconsin.ipynb \
	notebooks/90_powerbi_export.ipynb \
	notebooks/91_dashboard_export.ipynb

.PHONY: all notebooks facts test dashboard

all: notebooks facts test

notebooks:
	@for nb in $(NOTEBOOKS); do echo "== $$nb"; $(RUN) $$nb || exit 1; done

facts:
	$(PY) scripts/build_facts.py
	$(PY) scripts/update_readme.py

test:
	$(PY) -m pytest tests -q

dashboard:
	$(PY) -m streamlit run dashboard/app.py
