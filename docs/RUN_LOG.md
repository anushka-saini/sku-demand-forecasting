# Run log

Every run of the pre-registered analyses (`docs/PREREGISTRATION.md`), including failures and deviations.

| # | Date | Task | What was run | Outcome |
|---|---|---|---|---|
| 1 | 2026-10-13 | 0 (audit) | Recomputed every headline number from `outputs/tables/` | All matched at the stated rounding. One label was wrong: the "45% false alarms on unmodified series" is the no-injection control of the power test on real series, not a false-alarm rate. |
| 2 | 2026-10-13 | 1 | `notebooks/06_detector_calibration.ipynb`, first and only run (5.7 min), frozen thresholds h = 14.385 and penalty = 31.03 | Completed. Results in `outputs/tables/calibration_*.csv`. H1.1, H1.2, H1.3 supported; H1.4 not supported. |
| 3 | 2026-10-13 | 2 | `notebooks/07_high_confidence_breaks.ipynb`, first run | **Failed** with a KeyError before any result table was written. Cause: a coding error (yes/no columns had mixed types because items without a break carry no value). No definition or threshold was involved. |
| 4 | 2026-10-13 | 2 | Same notebook after casting those columns to true/false | Completed. Results in `outputs/tables/hc_breaks_*.csv`. 171 of 488 CUSUM breaks are high-confidence. |

## Deviations from the pre-registration
None so far.

## Notes on Task 1 (run 2)
- 718 of 726 items had at least 104 post-burn-in weeks and used the seasonal null. 8 used the reference-window fallback and are reported separately.
- Injection test: 3,397 hosts used; 193 excluded because the host alarmed before the injection date.
- The deseasonalized diagnostic raised the break rate to 97% (CUSUM) and 98% (PELT). The frozen thresholds were calibrated on raw series. Removing the seasonal component shrinks the reference-window standard deviation, so the same thresholds are much looser on deseasonalized data. No false-alarm rate was pre-registered for deseasonalized series, so this number cannot be read as "97% of items have real shifts". A matched null for deseasonalized series would be a post-hoc addition and has not been run.

## Notes on Task 2 (run 4)
- "Weeks 1–26 after the change week" was implemented as the 26 weeks following the change week (the change week itself is excluded from both the pre and post windows). "Weeks 27–52 after" follow directly. A break needs 52 weeks after the change week to be assessable.
- No hypothesis was pre-registered for Task 2; it is descriptive.
- 38.9% of change weeks fall in the launch ramp (within 26 weeks of monitoring start), against 13.8% expected if change weeks were spread evenly over each item's monitoring period. This repeats the earlier finding that breaks cluster right after monitoring starts.
