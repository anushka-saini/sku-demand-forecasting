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
