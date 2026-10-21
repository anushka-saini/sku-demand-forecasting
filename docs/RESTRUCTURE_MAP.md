# Restructure map

The notebooks used to be numbered in the order we happened to build them. They now follow the order you would learn
the data in: meet it, scope it, explore it, clean it, test it, decompose it, forecast it with classical models,
look for breaks, try machine learning, compare everything, then turn forecasts into stock decisions. Advanced
models come last and are a start only.

Nothing in the analysis changes. Old notebooks are renamed, merged or split; the code that produces each result is
moved, not rewritten, and every result file is checked against a hashed backup after each stage.

| Current notebook | New position and name | Why |
|---|---|---|
| `00a_data_audit` (file snapshots) | `01_meet_the_data` | the first thing anyone needs: what is in the three files |
| `00a_data_audit` (line and weekday plots) | `03_explore` | these are exploration plots |
| `00a_data_audit` (zero-days histogram) | `04_zeros_and_trimming` | it shows the leading-zero problem |
| `00b_cross_category_scoping` | `02_scope` | California, 3,049 products, the 726 AX/AY items; adds the 500-points check |
| *(new)* | `03_explore` | summary statistics, histograms, seasonal plots, correlations with price, SNAP and events |
| `12_stationarity_decomposition` (data quality, outliers) | `03_explore` | missing values and outliers belong with exploration |
| `00_data_scoping_eda` | `04_zeros_and_trimming` | it is mostly the story of the leading zeros; adds a before/after decomposition |
| `12_stationarity_decomposition` (ADF, KPSS, ACF, differencing) | `05_stationarity` | one topic per notebook |
| `12_stationarity_decomposition` (decomposition) | `06_decomposition` | one topic per notebook; adds yearly seasonality |
| `02_forecasting_static_vs_adaptive` (seasonal naive, ETS, TSB, static) + `13_classical_models` (Holt-Winters, ARIMA, SARIMA) | `07a_classical_forecasting` | every classical model in one place, before any machine learning |
| `14_model_diagnostics` | `07b_classical_residuals` | the residual checks get their own notebook |
| `01_structural_break_detection` | `08a_break_detection` | breaks come after the classical models |
| `06_detector_calibration` | `08b_detector_calibration` | checks on the detector |
| `07_high_confidence_breaks` | `08c_high_confidence_breaks` | checks on the detector |
| `02_forecasting_static_vs_adaptive` (ETS and TSB retrained after a break) + `15_shift_aware_classical` | `08d_shift_aware_classical` | the shift-aware classical models together |
| `02_forecasting_static_vs_adaptive` (LightGBM, scoring, bands) | `09a_lightgbm` | machine learning after the classical models |
| `08_adaptation_variants` | `09b_lightgbm_after_breaks` | the shift-aware LightGBM variants |
| `13_classical_models` (tables with every model) + `09_forecast_calibration` (paired tests, bias) | `10_model_comparison` | one honest comparison of everything |
| `09_forecast_calibration` (conformal bands) | `11a_uncertainty` | uncertainty before stock |
| `03_inventory_decision_layer` | `11b_inventory_simulation` | the stock simulation |
| `09_forecast_calibration` (calibrated safety stock) | `11c_calibrated_safety_stock` | uses the bands from 11a |
| `10_inventory_robustness` | `11d_inventory_other_seasons` | the simulation in two more seasons |
| `16_classical_stock_link` | `11e_classical_forecasts_and_stock` | classical forecasts in the same simulation |
| `11_replication_tx_wi` | `11f_texas_and_wisconsin` | the whole pipeline again on two other states; fits best after everything it re-checks |
| *(new)* | `12_advanced_models_wip` | XGBoost, Prophet and a small GRU on a few items; work in progress |
| `04_powerbi_export` | `90_powerbi_export` | an export, not a step in the analysis; runs after everything |
| `05_dashboard_export` | `91_dashboard_export` | an export for the dashboard; runs after everything |
| `00_data_scoping_eda_ORIGINAL` | left where it is | the owner's own backup copy with uncommitted edits; not touched |

Result tables that lived in a separate folder move into `outputs/tables/`, and their figures into
`outputs/figures/`, so every notebook writes to the same two places.
