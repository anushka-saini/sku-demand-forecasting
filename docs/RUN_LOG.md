# Run log

Every run of the pre-registered analyses (`docs/PREREGISTRATION.md`), including failures and deviations.

| # | Date | Task | What was run | Outcome |
|---|---|---|---|---|
| 1 | 2026-10-13 | 0 (audit) | Recomputed every headline number from `outputs/tables/` | All matched at the stated rounding. One label was wrong: the "45% false alarms on unmodified series" is the no-injection control of the power test on real series, not a false-alarm rate. |
| 2 | 2026-10-13 | 1 | `notebooks/06_detector_calibration.ipynb`, first and only run (5.7 min), frozen thresholds h = 14.385 and penalty = 31.03 | Completed. Results in `outputs/tables/calibration_*.csv`. H1.1, H1.2, H1.3 supported; H1.4 not supported. |
| 3 | 2026-10-13 | 2 | `notebooks/07_high_confidence_breaks.ipynb`, first run | **Failed** with a KeyError before any result table was written. Cause: a coding error (yes/no columns had mixed types because items without a break carry no value). No definition or threshold was involved. |
| 4 | 2026-10-13 | 2 | Same notebook after casting those columns to true/false | Completed. Results in `outputs/tables/hc_breaks_*.csv`. 171 of 488 CUSUM breaks are high-confidence. |
| 5 | 2026-10-15 | 3 | `notebooks/08_adaptation_variants.ipynb`, first and only run | Completed. The refactored LightGBM reproduced the saved static forecasts exactly (max difference 0). No variant met the "helps" rule in either group; H3 supported. |
| 6 | 2026-10-15 | 4 | `notebooks/09_forecast_calibration.ipynb`, first and only run (1.3 min) | Completed. The reusable inventory simulation reproduced notebook 03 (max total-cost difference $0.0005). Conformal coverage met the ±2-point criterion for both LightGBM variants on origins 2–6. |
| 7 | 2026-10-15 | 5 | `notebooks/10_inventory_robustness.ipynb`, first and only run (6 min) | Completed. Two new backtests (W2, W3), plus inventory for three windows on a 10-level service grid. |
| 8 | 2026-10-15 | 6 | `notebooks/11_replication_tx_wi.ipynb`, first and only run (13.1 min) | Completed. Reusable scoping reproduced the CA item set exactly (726 items, identical rows and units). TX 529 items, WI 598 items. All 7 conclusions hold in both states. |
| 9 | 2026-10-15 | 5 | `notebooks/10_inventory_robustness.ipynb`, re-run after a **bug fix** (6.1 min) | Run 7 classified intermittent items on data before 2014-12-07 for all three windows; notebook 03 used data before each window's warm-up date. That switched some items' safety-stock rule, so W1 did not match notebook 03 (up to $2,538 per item). Fixed by classifying per window; W1 now reproduces notebook 03 exactly (6,507 rows, max difference 0). W3 is unchanged. W1 static − naive at 95% moved from −9.8% (CI reaching 0) to −10.9% (CI −$16,847 to −$357); W1 adaptive − static from +16.1% to +12.6%. Results in `docs/RESULTS.md` use this run. |
| 10 | 2026-10-21 | M2 step 0 | Timing run, 5 items (seed 42, one per category + 2 random), first scored origin only, last 730 days; scratch script, nothing saved to the repo | Completed. Per item: SARIMA order grid (36 fits) 19.0 s, SARIMA refit 1.2 s, ARIMA grid 1.5 s, Holt-Winters 0.4–0.5 s, SARIMAX with level dummy 1.9 s, ADF+KPSS 0.1 s. Frequent statsmodels convergence warnings. nsdiffs (OCSB, m = 7) gave D = 0 for all 5. Baseline pytest: 118 passed. |
| 11 | 2026-10-21 | M2-T4 | `notebooks/13_classical_models.ipynb`, first run (43.4 min, sharing the CPU with run 12 and a test run) | **Failed** after all fits had finished and been checkpointed (150 items + 4 totals), in the aggregate-scoring cell: the totals' checkpoints carried the series name (`FOODS`) instead of the `_agg_FOODS` label the cell looked for. Coding error; no definition involved. |
| 12 | 2026-10-21 | M2-T2 | `notebooks/12_stationarity_decomposition.ipynb`, first run | **Hung** for about 30 min with idle joblib workers while run 11's joblib pool was also active (two loky pools at once on Windows). Stopped by hand; only `data_quality.csv` had been written. |
| 13 | 2026-10-21 | M2-T4 | Notebook 13 re-run after fixing the label (read from checkpoints, 0.9 min) | Completed. Same 150-item sample as run 11 (file compared). H1 and H2 supported. SARIMA on the totals: 5 of 28 fits fell back to seasonal naive and several diverged quietly (FOODS MASE 1.976). |
| 14 | 2026-10-21 | M2-T2 | Notebook 12 with the stationarity loop made serial (no joblib) | **Failed** (3.5 min): `fix_christmas` wrote a fractional average into an integer series (TypeError). Fixed by converting to float. |
| 15 | 2026-10-21 | M2-T2, T3 | Notebook 12 again (4.1 min) | Completed. 16.9% stationary, 8.4% trend-stationary, 15.7% difference-stationary, 59.0% conflicting; d = 1 for 94%, D = 0 for all. |
| 16 | 2026-10-21 | M2-T6 | `notebooks/14_model_diagnostics.ipynb`, first and only run (6.9 min) | Completed. Includes a **post-hoc** check, not pre-registered: SARIMA on the totals refitted in thousands of units (FOODS 1.976 → 0.834). The same raw-unit refit gave a different result from run 13 for HOUSEHOLD (4.710 vs 1.145): notebook 13 fits in joblib worker processes, notebook 14 in the main process, and these ill-conditioned fits are sensitive to that. Each notebook repeats exactly (confirmed in run 24). The pre-registered numbers are unchanged. |
| 17 | 2026-10-21 | M2-T7 | `notebooks/15_shift_aware_classical.ipynb`, first and only run (4.9 min) | Completed. H3 supported: neither restart nor the level dummy helps; restart is worse where it differs from static (+0.0062, Holm p = 0.002). |
| 18 | 2026-10-21 | M2-T8 | `notebooks/16_classical_stock_link.ipynb`, first run (0.6 min) | **Failed**: a duplicate `category` column after a merge (KeyError). Coding error. |
| 19 | 2026-10-21 | M2-T8 | Notebook 16 after removing the extra merge (1.0 min) | Completed. LightGBM and seasonal naive reproduce Milestone 1's item results exactly (300 rows, max cost difference $0.0000). |
| 20 | 2026-10-21 | M2-T4 | Notebook 13 re-run to add the degenerate-forecast count (from checkpoints, about 1 min) | Completed; no number changed. Degenerate windows: totals SARIMA 4 all-zero + 1 above 3× peak; items 1-8 per model out of 1,050. |
| 21 | 2026-10-21 | M1-T3 | Notebooks 00 (after the cell-order fix), 00a, 00b re-run top to bottom (1.7 + 0.5 + 5.1 min) | Completed, no errors. All 33 files they write (data/interim and outputs/figures) are byte-identical to the copies taken before (MD5). |
| 22 | 2026-10-21 | M2-T4 | Notebook 13 re-run after shortening the box-plot axis labels (from checkpoints, 0.8 min) | Completed; no number changed. |
| 23 | 2026-10-21 | M2-T7 | Notebook 15 re-run after moving the figure legend (refits, 5.3 min) | Completed; every number in `shift_aware_tests.csv` identical to run 17. |
| 24 | 2026-10-22 | M2 (all) | `python scripts/run_all.py --m2`: notebooks 12-16, facts, README, tests (18.7 min) | Completed. 202 tests passed. Every Milestone 2 headline number identical to runs 13-19, including notebook 14's post-hoc table. |

## Deviations from the pre-registration
- **Requirements (Task 7, not part of the analysis):** four packages that are never imported (pmdarima, seaborn, openpyxl, scikit-learn) were dropped from `requirements.txt` when pinning versions.
- No analysis definition, threshold or decision rule was changed.

## Notes on Task 1 (run 2)
- 718 of 726 items had at least 104 post-burn-in weeks and used the seasonal null. 8 used the reference-window fallback and are reported separately.
- Injection test: 3,397 hosts used; 193 excluded because the host alarmed before the injection date.
- The deseasonalized diagnostic raised the break rate to 97% (CUSUM) and 98% (PELT). The frozen thresholds were calibrated on raw series. Removing the seasonal component shrinks the reference-window standard deviation, so the same thresholds are much looser on deseasonalized data. No false-alarm rate was pre-registered for deseasonalized series, so this number cannot be read as "97% of items have real shifts". A matched null for deseasonalized series would be a post-hoc addition and has not been run.

## Notes on Task 2 (run 4)
- "Weeks 1–26 after the change week" was implemented as the 26 weeks following the change week (the change week itself is excluded from both the pre and post windows). "Weeks 27–52 after" follow directly. A break needs 52 weeks after the change week to be assessable.
- No hypothesis was pre-registered for Task 2; it is descriptive.
- 38.9% of change weeks fall in the launch ramp (within 26 weeks of monitoring start), against 13.8% expected if change weeks were spread evenly over each item's monitoring period. This repeats the earlier finding that breaks cluster right after monitoring starts.

## Notes on Task 3 (run 5)
- The per-origin high-confidence subset covers 1,172 of 2,869 break item-origins (254 items).
- **Blend, all breaks:** the bootstrap 95% CI of the median difference lies below zero (−0.0018 to −0.0001), but the Wilcoxon test is not significant after Holm (p = 0.22), and the effect is below the 0.0074 practical threshold. Under the pre-registered rule it does not help. The two tests measure different things (median vs signed ranks), so this is reported as is.

## Notes on Task 4 (run 6)
- The "before" bands are the existing per-item empirical quantiles. At 90% for LightGBM static on origins 2–6 they cover 85.5%. The 84.5% reported earlier covers all six origins, including origin 1, which has only the warm-up to calibrate on.

## Notes on Task 5 (run 7)
- **Items:** the eligibility rule (60 full weeks before the window's warm-up origin) keeps 723 items in W1 (not 726), 718 in W2 and 709 in W3. W1 uses the existing notebook 02 forecasts.
- **Thresholds:** CUSUM h was recalibrated by the frozen procedure before each window: W2 12.81, W3 12.31.
- **Run 9 supersedes run 7.** The bug was in how the inputs were set up, not in any pre-registered definition or threshold.
- **Per-item medians:** the median per-item adaptive − static cost difference is 0. Items without a break have identical static and adaptive forecasts, and they are about a third of all items. The totals and their CIs carry the comparison.
- **Grid edge:** the simulated cheapest service level was 0.995, the top of the grid, in every scenario. The true optimum may lie above the grid.

## Notes on Task 6 (run 8)
- The frozen forecasting threshold recomputed by the CA procedure is 13.830, the value used in notebook 02.
- R3 holds in TX on the median (0.7342 vs 0.7393), but the paired difference is not significant (p = 0.60). In WI it is (p < 0.001).
- R6 in WI: adaptive costs +168 more in total, with a 95% CI of -2,393 to +2,460, so not distinguishable from static. In TX the CI is wholly above zero (+252 to +4,285).
