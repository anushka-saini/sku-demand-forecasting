# Traceability: objective → method → code → result

Each Milestone 2 requirement and project objective, the method used, the code that does it, and the table or
figure that shows the result. Paths are relative to the repository root.

| Objective / requirement | Method | Code | Result table | Figure | Dashboard |
|---|---|---|---|---|---|
| Data preprocessing | trim pre-launch zeros (Milestone 1); count empty cells, missing days, outliers (Q3 + 3 IQR, flagged not changed) | `notebooks/00b_cross_category_scoping.ipynb`, `notebooks/12_stationarity_decomposition.ipynb`, `src/stationarity.py` (`count_outliers`) | `outputs/m2/tables/data_quality.csv` | `outputs/m2/figures/m2_outliers_example.png` | M2: Stationarity & decomposition |
| Stationarity checks | ADF + KPSS with constant and trend, 4-way label; `ndiffs` / `nsdiffs` | `src/stationarity.py` (`check_series`, `classify`), notebook 12 | `stationarity_items.csv`, `stationarity_summary.csv`, `stationarity_aggregates.csv` | `m2_stationarity_shares.png`, `m2_differencing.png` | M2: Stationarity & decomposition |
| ACF / PACF | statsmodels `plot_acf`, `plot_pacf`, 28 lags | notebook 12 | – | `m2_acf_pacf_items.png`, `m2_acf_pacf_aggregates.png` | – |
| Seasonal decomposition, additive and multiplicative | classical `seasonal_decompose` (period 7) both ways, plus STL; swing rule for the choice; zeros handled explicitly | notebook 12, `src/stationarity.py` (`swing_rule`, `fix_christmas`) | `decomposition_choice.csv` | `m2_decomposition_aggregate.png`, `m2_decomposition_item.png` | M2: Stationarity & decomposition; M2: Demo |
| Holt-Winters (additive, damped, multiplicative) | statsmodels `ExponentialSmoothing`; damped = Milestone 1 ETS | `src/classical.py` (`fit_hw`, `hw_forecast`), notebook 13 | `classical_metrics_summary.csv` | `m2_forecast_examples.png`, `m2_mase_by_model.png` | M2: Classical models |
| ARIMA and SARIMA | statsmodels `SARIMAX`; AICc grid at origin 0; 730-day window; seasonal-naive fallback | `src/classical.py` (`choose_order`, `sarima_forecast`, `backtest_series`), notebook 13 | `classical_orders.csv`, `classical_run_stats.csv` | same | M2: Classical models |
| Model assumptions, parameters, justification | Ljung-Box, Jarque-Bera, AR/MA roots; parameter table | notebook 14, `docs/midsem/MODEL_JUSTIFICATION.md` | `diagnostics_summary.csv`, `diagnostics_items.csv`, `parameters_summary.csv`, `parameters_items.csv` | `m2_residual_acf.png`, `m2_diagnostics_pass.png` | M2: Classical models |
| Error metrics RMSE, MAE, MAPE (+ sMAPE, WAPE, MASE) | per item per origin, then median and mean over items; MAPE on non-zero days | `src/metrics.py` (`rmse`, `mae`, `mape`, `mape_excluded`, `smape`, `wape`, `all_errors`), notebook 13 | `classical_metrics_item_origin.csv`, `classical_metrics_summary.csv`, `aggregate_metrics.csv` | `m2_mase_by_model.png` | M2: Classical models; M2: Demo |
| Compare approaches (H1, H2) | paired Wilcoxon, bootstrap CI, Holm | notebook 13, `src/metrics.py` (`paired_test`, `holm_adjust`) | `classical_tests.csv` | `m2_mase_by_model.png` | M2: Classical models; Overview |
| Additive vs multiplicative (H4) | median MASE on zero-heavy items; within-5% check on totals | notebook 13 | `classical_h4.csv` | – | M2: Classical models |
| Core research question for classical models (H3) | static vs restart vs level-shift dummy vs placebo SARIMA, breaks known at the origin only | notebook 15 | `shift_aware_tests.csv`, `shift_aware_item_origin.csv` | `m2_shift_aware.png` | M2: Classical models; Overview |
| Retail stock planning | Milestone 1 reorder-point simulation with classical forecasts; normal and conformal safety stock | notebook 16, `src/pipeline.py` (`simulate_inventory`) | `stock_summary.csv`, `stock_tests.csv`, `stock_items.csv` | `m2_stock_cost.png` | M2: Demo |
| No lookahead | fits use data up to the origin only; orders from origin 0 | `tests/test_m2_methods.py::test_backtest_has_no_lookahead` | – | – | – |
| Live demo of the modules | dashboard walkthrough | `dashboard/views/demo.py`, `docs/midsem/DEMO_SCRIPT.md` | – | – | M2: Mid-sem demo walkthrough |
| Interim observations and conclusions | – | `docs/midsem/INTERIM_FINDINGS.md` | all of the above | – | Overview |
