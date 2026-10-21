# Roadmap: every notebook, in order

## Read this repo in 10 minutes
1. **The question** (1 min): `README.md`, first section. Do lasting changes in demand matter for forecasting and stock?
2. **The answers** (2 min): `docs/RESULTS.md` (Milestone 1) and `docs/midsem/INTERIM_FINDINGS.md` (Milestone 2).
3. **What was promised in advance** (2 min): `docs/PREREGISTRATION.md` and `docs/PREREGISTRATION_M2.md`. Every run, including failures, is in `docs/RUN_LOG.md`.
4. **How it flows** (3 min): the table below. Each notebook's first cell has the same header (milestone, task, stage, aim, inputs, outputs, hypotheses, runtime).
5. **See it** (2 min): `python -m streamlit run dashboard/app.py`, then the *Overview* and *M2: Mid-sem demo walkthrough* pages.

Code that notebooks share lives in `src/` (`config.py` holds the Milestone 2 paths and constants). Tests are in `tests/`; `python scripts/run_all.py` reruns everything on Windows.

## Names used in this repo
| Term | Meaning |
|---|---|
| **Milestone 1 / 2 / 3** | The course milestones (CSET369). Milestone 1 = notebooks 00-11; Milestone 2 = notebooks 12-16 (mid-sem, classical models); Milestone 3 = later. |
| **Stage 1-5** | Steps of the pipeline: 1 data preparation, 2 shift detection, 3 forecasting, 4 stock decisions, 5 reporting. Older notebooks called these "Phase 1-5"; the markdown now says "Stage" so it does not clash with the course milestones. |
| **M1-W1 ... M1-W11** | Milestone 1 work packages (one per notebook group). W6-W11 are the pre-registered Tasks 1-6 of `docs/PREREGISTRATION.md`. |
| **M1-T1 ... M1-T3** | Milestone 1 safeguards added with Milestone 2: the recap (`docs/M1_RECAP.md`), the facts test (`tests/test_m1_facts.py`) and the re-run of notebooks 00, 00a, 00b. |
| **M2-T0 ... M2-T11** | Milestone 2 tasks (`docs/PREREGISTRATION_M2.md`, `docs/M2_DECISIONS.md`). |

## Notebooks
| Notebook | Milestone | Task | Stage | Aim | Main inputs | Main outputs | Dashboard page |
|---|---|---|---|---|---|---|---|
| `00_data_scoping_eda` | 1 | M1-W1 | 1 | First look at CA FOODS, the pre-launch zero problem, ABC-XYZ, ADF | raw files | first-pass `item_segmentation.csv`, `scoped_sales_candidates.csv`, `adf_results.csv`, 2 figures | Data & method (via 05) |
| `00a_data_audit` | 1 | M1-W1 | 1 | Snapshots and first-look plots for the synopsis slides | raw files | `outputs/figures/data_audit/*.png` | none (slides) |
| `00b_cross_category_scoping` | 1 | M1-W1 | 1 | Trim, segment, ADF for all categories: the 726 items | raw files | `data/interim/*_candidates.csv` | Data & method, Segmentation |
| `01_structural_break_detection` | 1 | M1-W2 | 2 | Calibrated weekly CUSUM, PELT, power, event alignment | candidates | `cusum_breaks.csv`, `break_calibration.csv` (frozen threshold) | Shift detection |
| `02_forecasting_static_vs_adaptive` | 1 | M1-W3 | 3 | Backtest naive, ETS, TSB, LightGBM; static vs adaptive vs placebo | candidates, prices | `backtest_forecasts.parquet`, `backtest_errors.csv`, `breaks_by_origin.csv` | Forecast comparison, Does retraining help? |
| `03_inventory_decision_layer` | 1 | M1-W4 | 4 | Reorder-point simulation and cost sensitivity | forecasts | `inventory_*` tables | Inventory simulator, Planner |
| `04_powerbi_export` | 1 | M1-W5 | 5 | Star-schema tables for Power BI | processed files | `pbi_*.csv` | (several, via 05) |
| `05_dashboard_export` | 1 | M1-W5 | 5 | Compact files for the app | pbi tables | `outputs/tables/app/*` | all Milestone 1 pages |
| `06_detector_calibration` | 1 | M1-W6 | 2 | False alarms on a seasonal null; power for steps and drifts | candidates, threshold | `calibration_*.csv` | Shift detection |
| `07_high_confidence_breaks` | 1 | M1-W7 | 2 | Breaks that are confirmed, large and lasting | candidates, threshold | `hc_breaks_*.csv` | Shift detection |
| `08_adaptation_variants` | 1 | M1-W8 | 3 | Four ways to adapt LightGBM to a break | forecasts, breaks | `adaptation_variants_*.csv` | Does retraining help? |
| `09_forecast_calibration` | 1 | M1-W9 | 3, 4 | Paired tests, bias, conformal bands and safety stock | forecasts | `conformal_*`, `inventory_calibrated_*` | Forecast comparison, Inventory simulator, Planner |
| `10_inventory_robustness` | 1 | M1-W10 | 4 | Three 168-day windows | candidates, forecasts | `robustness_*.csv` | Inventory simulator |
| `11_replication_tx_wi` | 1 | M1-W11 | 1-4 | Texas and Wisconsin with frozen settings | raw files | `replication_*.csv` | Replication |
| `12_stationarity_decomposition` | 2 | M2-T2, M2-T3 | 1 | ADF + KPSS on 726 items, d and D, ACF/PACF, outliers, decomposition and the additive/multiplicative rule | candidates, raw files | `outputs/m2/tables/stationarity_*`, `decomposition_choice.csv`, demo files | M2: Stationarity & decomposition; M2: Demo |
| `13_classical_models` | 2 | M2-T4, M2-T5 | 3 | Holt-Winters, ARIMA, SARIMA on the Milestone 1 backtest; RMSE, MAE, MAPE, sMAPE, WAPE, MASE; H1, H2, H4 | candidates, saved forecasts, breaks | `classical_*`, `aggregate_metrics.csv` | M2: Classical models; M2: Demo; Overview |
| `14_model_diagnostics` | 2 | M2-T6 | 3 | Assumptions, residual checks, parameter table | orders, candidates | `diagnostics_*`, `parameters_*` | M2: Classical models |
| `15_shift_aware_classical` | 2 | M2-T7 | 3 | Static vs restart vs level-shift dummy vs placebo SARIMA (H3) | orders, breaks, scores | `shift_aware_*` | M2: Classical models; Overview |
| `16_classical_stock_link` | 2 | M2-T8 | 4 | Classical forecasts in the stock simulation, normal and conformal safety stock | classical forecasts, prices | `stock_*` | M2: Demo |

`00_data_scoping_eda_ORIGINAL.ipynb` is the owner's untouched backup of notebook 00 and is not part of the pipeline.

## Dependency chain
```
raw files ─► 00b ─► 01 ─► 02 ─► 03 ─► 04 ─► 05 ─► dashboard (Milestone 1 pages)
              │      │     │     └──────────────► 09, 10
              │      └──── threshold ─► 06, 07, 11
              │            └──► 08
              └─► 12 ─► 13 ─► 14
                         ├──► 15 (also uses 02's breaks)
                         └──► 16 (also uses 02's forecasts) ─► dashboard (M2 pages)
00 and 00a stand alone (00's first-pass file is read by 05).
```
