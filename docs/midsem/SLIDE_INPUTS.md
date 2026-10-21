# Slide inputs: every figure and number, with its source

All figures: white background, 150 dpi, titles that state the finding (computed from the data when the notebook
runs). Regenerate by running the notebook named in the "made by" column.

## Figures
| Slide topic | File | Made by |
|---|---|---|
| Dataset snapshot | `outputs/figures/data_audit/snapshot_all_files.png` | notebook 00a |
| Line plot, zero values, weekday pattern | `outputs/figures/data_audit/plot_line_one_product.png`, `plot_zero_values.png`, `plot_seasonal_weekday.png` | notebook 00a |
| Stationarity shares | `outputs/m2/figures/m2_stationarity_shares.png` | notebook 12 |
| ACF / PACF, items and totals | `outputs/m2/figures/m2_acf_pacf_items.png`, `m2_acf_pacf_aggregates.png` | notebook 12 |
| Before / after differencing | `outputs/m2/figures/m2_differencing.png` | notebook 12 |
| Outlier rule | `outputs/m2/figures/m2_outliers_example.png` | notebook 12 |
| Decomposition: additive, multiplicative, STL | `outputs/m2/figures/m2_decomposition_aggregate.png`, `m2_decomposition_item.png` | notebook 12 |
| Forecast plots | `outputs/m2/figures/m2_forecast_examples.png` | notebook 13 |
| Error by model | `outputs/m2/figures/m2_mase_by_model.png` | notebook 13 |
| Residual diagnostics | `outputs/m2/figures/m2_residual_acf.png`, `m2_diagnostics_pass.png` | notebook 14 |
| Shift-aware SARIMA (H3) | `outputs/m2/figures/m2_shift_aware.png` | notebook 15 |
| Stock cost by model | `outputs/m2/figures/m2_stock_cost.png` | notebook 16 |
| Dashboard pages | `docs/screenshots/stationarity.png`, `classical.png`, `demo.png` (and `dark/`) | `scripts/screenshots.py` |

## Numbers
| Number | Value | Source file (in `outputs/m2/tables/` unless stated) | Facts key (`outputs/results_facts.json`) |
|---|---|---|---|
| Items in the Milestone 2 sample | 150 (75 with a break) | `m2_sample_items.csv` | `m2.sample` |
| Items stationary / trend- / difference-stationary / conflicting | 16.9% / 8.4% / 15.7% / 59.0% of 726 | `stationarity_summary.csv` | `m2.stationarity` |
| Recommended d = 1; D = 1 | 94.4%; 0% | `stationarity_summary.csv` | `m2.stationarity["d=1|ALL"]`, `["D=1|ALL"]` |
| Empty sales cells; missing days; outlier share | 0; 0; 0.2% | `data_quality.csv` | – |
| Median MASE: naive / ARIMA / HW mult. / HW add. / ETS / SARIMA / LightGBM | 0.973 / 0.794 / 0.761 / 0.757 / 0.756 / 0.755 / 0.747 | `classical_metrics_summary.csv` | `m2.metrics["<model>|ALL|mase"]` |
| Median RMSE, MAE, MAPE per model | see INTERIM_FINDINGS table | `classical_metrics_summary.csv` | `m2.metrics["<model>|ALL|rmse"]` etc. |
| MAPE: share of days left out | 5.4% | `classical_metrics_summary.csv` | `m2.mape_excluded_share` |
| H1: SARIMA − naive; HW − naive | −0.223 (CI −0.237 to −0.200); −0.218 (−0.233 to −0.207) | `classical_tests.csv` | `m2.tests["H1|..."]` |
| H2: LightGBM − SARIMA | −0.007 (CI −0.014 to −0.002), Holm p = 0.036 | `classical_tests.csv` | `m2.tests["H2|lightgbm-sarima"]` |
| H4: HW add vs mult., zero-heavy items | 0.763 vs 0.762 | `classical_h4.csv` | `m2.h4["zero-heavy"]` |
| SARIMA on FOODS total (raw units / in thousands, post-hoc) | 1.976 / 0.834 | `aggregate_metrics.csv`, `sarima_totals_rescaled.csv` | `m2.aggregate_mase["FOODS|sarima"]` |
| Ljung-Box pass: SARIMA / HW add. | 82% / 47% | `diagnostics_summary.csv` | `m2.diagnostics["sarima|item"]` |
| Normality pass (max over models) | 0.7% | `diagnostics_summary.csv` | `m2.diagnostics` |
| H3: restart − static, where it differs | +0.0062 (CI +0.0015 to +0.0107), Holm p = 0.002 | `shift_aware_tests.csv` | `m2.shift_aware["only where the variant differs|restart"]` |
| H3: share of break item-origins equal to static | 44% | `shift_aware_item_origin.csv` | – |
| Stock cost, calibrated: SARIMA / LightGBM | $7,857 / $8,089 | `stock_summary.csv` | `m2.stock["conformal|sarima"]` |
| SARIMA vs ETS, calibrated | −14.1% (CI −$2,878 to −$70) | `stock_tests.csv` | `m2.stock_tests["conformal|sarima-ets"]` |
| Fallback rate, item fits | 0% for every model | `classical_run_stats.csv` | `m2.run_stats` |
| Milestone 1 synopsis numbers (3,049 products, 1,565 leading-zero products, 726 items, file sizes) | – | checked by `tests/test_m1_facts.py` | `scope` |
