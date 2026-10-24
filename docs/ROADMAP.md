# Roadmap

Where the work stands. The notebooks run in the order below; each one starts by saying what question it answers
and what it hands on, and ends with what we found and what comes next.

## Done

| Stage | Notebooks | Main outputs | Dashboard page |
|---|---|---|---|
| 1 Meet the data | `01_meet_the_data` | file snapshots | Data |
| 2 Choose scope | `02_scope` | `data/interim/*_candidates.csv`, the 726 AX/AY products | Data, Segmentation |
| 3 Explore | `03_explore` | `explore_*.csv`, `data_quality.csv`, seasonal and price plots | Explore |
| 4 Zeros at the start | `04_zeros_and_trimming` | before and after decomposition of `FOODS_1_004` | Zeros and trimming |
| 5 Stationarity | `05_stationarity` | `stationarity_*.csv`, ACF and differencing plots | Stationarity |
| 6 Decomposition | `06_decomposition` | `decomposition_choice.csv`, weekly and yearly plots | Decomposition |
| 7 Classical forecasting | `07a_classical_forecasting`, `07b_classical_residuals` | `classical_*.csv`, `diagnostics_*.csv`, `parameters_*.csv` | Classical models |
| 8 Structural breaks | `08a_break_detection`, `08b_detector_calibration`, `08c_high_confidence_breaks`, `08d_shift_aware_classical` | `breaks_by_origin.csv`, `calibration_*.csv`, `hc_*.csv`, `shift_aware_*.csv` | Breaks |
| 9 Machine learning | `09a_lightgbm`, `09b_lightgbm_after_breaks` | `backtest_forecasts.parquet`, `backtest_errors.csv`, `adaptation_variants_*.csv` | Does retraining help?, Forecast explorer |
| 10 Comparison | `10_model_comparison` | `comparison_*.csv`, `model_pairwise_tests.csv`, `forecast_bias.csv` | ML and comparison, Overview |
| 11 Uncertainty and stock | `11a_uncertainty` to `11f_texas_and_wisconsin` | `conformal_*.csv`, `inventory_*.csv`, `robustness_*.csv`, `stock_*.csv`, `replication_*.csv` | Uncertainty and stock, Planner, Texas and Wisconsin |
| Exports | `90_powerbi_export`, `91_dashboard_export` | `pbi_*.csv`, `outputs/tables/app/`, demo files | every page |

## In progress

Stage 12, advanced models (`12_advanced_models_wip`). XGBoost, Prophet and a small GRU are set up on the same
backtest and scores, but only on 30 products and with no tuning. The notebook is a start: its numbers are not a
result, and none of it is on the dashboard's main pages.

## What remains

For stage 12: run XGBoost on all 726 products and the others on the 150-product sample; tune each one on an origin
before the scored ones; give Prophet the SNAP and event inputs it can take; give the GRU price and calendar inputs
and more data. Only then compare them with the models in stage 10.

Beyond that, the questions the current results leave open: a break detector that tells a sudden step from a slow
drift, slow-moving and erratic products (the Z and C groups), store-level series, and real cost data for the stock
simulation.

## Where to start reading

`README.md` for the question and the headline answers; this file for the path; `docs/RESULTS.md` for every result
with its numbers; `docs/MODEL_CHOICES.md` for why each model is there; `docs/RUN_LOG.md` for every run. Shared code
lives in `src/` (`config.py` holds the paths and backtest settings), tests in `tests/`, and
`python scripts/run_all.py` reruns everything.
