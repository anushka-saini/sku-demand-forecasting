# Pre-registration: robustness and replication analyses

Written 2026-10-13, before any of the analyses below were run.

## What was already known when this was written
This is not a blind pre-registration. The California (CA) results in `README.md` already existed and had been seen:
- break rates of 64–69%
- CUSUM–PELT agreement of 83.7%, with 64.8% of CUSUM breaks having a PELT changepoint within ±8 weeks
- median MASE of 0.976 / 0.755 / 0.744 (seasonal naive / ETS / LightGBM)
- truncation-based adaptation changing LightGBM MASE by +0.0033
- the 168-day inventory simulation

None of the quantities defined below (null false-alarm rates on the new null, high-confidence break counts, the new adaptation variants, calibrated coverage, multi-window costs, Texas and Wisconsin results) had been computed.

## Rules
1. Thresholds, subgroup definitions and success criteria in this file are not changed after results are seen. Any deviation is recorded in `docs/RUN_LOG.md` with the reason, and the original analysis is still reported.
2. Every run is logged in `docs/RUN_LOG.md`, including failed and abandoned runs.
3. No lookahead: anything used to make a forecast or an inventory decision at an origin uses only data up to that origin. Offline diagnostics that use the whole series are labelled "offline".
4. No tuning on evaluation windows. Hyperparameters for new methods are fixed here.
5. Raw data in `data/raw/` is never modified.
6. Random seed 42 everywhere unless stated. Two-sided tests, α = 0.05.

## Frozen settings (from the existing CA analysis)
| Setting | Value |
|---|---|
| Detection series | weekly sums, full Saturday–Friday weeks |
| Burn-in / reference window | 8 weeks / 52 weeks |
| CUSUM | two-sided, k = 1.0 σ_ref |
| CUSUM threshold h | 14.385 (full series); 13.83 (data before 2015-11-08, used in forecasting) |
| PELT | l2 cost, min segment 8 weeks, penalty 31.03 |
| Forecast horizon / origin spacing | 28 days / 28 days |
| LightGBM | features and parameters as in `src/forecasting.py` (300 rounds) |
| Scope rule | ABC by units within category (70/90%), XYZ by CV (0.5/1.0), keep AX and AY |
| Inventory base case | lead time 7 days, holding 25%/yr, stockout 0.5 × price, review cycle 7 days |

---

## Task 1: detector calibration

**Why.** The existing null (4-week block bootstrap of the 52-week reference window) has no annual seasonal pattern and no autocorrelation beyond 4 weeks. The threshold was set on that same null, so its 4.8% false-alarm rate is true by construction.

**Note on seasonality.** Detection runs on weekly totals, so day-of-week seasonality is summed out. The seasonality that matters at this resolution is the annual pattern.

**New null ("seasonal residual bootstrap").** For each item with at least 104 post-burn-in weeks:
1. Decompose the post-burn-in weekly series with robust STL (period 52).
2. Null series = mean of the reference window + the STL seasonal component + a moving-block bootstrap of the STL remainder (block length 13 weeks).
3. The STL trend is dropped, so the null has no level shift or drift by construction.

Items with fewer than 104 weeks use the existing reference-window null and are reported separately. Each item gets 20 null series.

**How rates are measured.**
- **False-alarm rate (FAR):** share of null series on which the detector raises at least one alarm, using the frozen thresholds with no recalibration. CUSUM: any alarm in the monitoring period. PELT: any changepoint after burn-in. Reported overall and by category.
- **Observed rate:** share of real items with an alarm under the same rule.
- **Adjusted share of items with a real shift:** (observed − FAR) / (1 − FAR). The 95% CI is a bootstrap over items (2,000 resamples; each resample recomputes both the observed rate and the FAR from the resampled items' null series).

**Deseasonalized detection (offline).** Rerun CUSUM and PELT on each series minus its STL seasonal component, with the frozen thresholds. Report the break rate and the share of items whose flag changes.

**Injection test.** Hosts are the null series (so the truth is known). Injection date t0 is uniform in the monitoring period, at least 10 weeks after monitoring starts and at least 39 weeks before the end. The sign is random. Trials where the host alarms before t0 are excluded and counted.
- **Shapes:** sudden step; slow drift (linear ramp reaching full size after 26 weeks, then flat).
- **Sizes:** 0% (control), 5%, 10%, 20%, 30% of the item's reference-window mean.
- **CUSUM:** power = share alarming at or after t0; power within 26 weeks; median delay among detections; share with the correct sign.
- **PELT:** hit = a changepoint within ±8 weeks of t0 for steps, or between t0 − 8 and t0 + 34 weeks for drifts.

**Expectations (hypotheses).**
- H1.1: CUSUM FAR on the new null is above 5%.
- H1.2: the adjusted share of items with a real shift is below the observed 67%.
- H1.3: power for 5% and 10% shifts is not distinguishable from the 0% control (difference under 10 percentage points).
- H1.4: at equal size, drifts are detected later than steps.

## Task 2: high-confidence breaks

A CUSUM break is **high-confidence** if all three hold:
1. **Agreement:** a PELT changepoint lies within ±56 days of the CUSUM estimated change week.
2. **Effect size:** |post − pre| / pre ≥ 20%, where pre = mean of the 26 weeks before the change week and post = mean of weeks 1–26 after it.
3. **Persistence:** (later − pre) / (post − pre) ≥ 0.5, where later = mean of weeks 27–52 after the change. The shift keeps at least half its size.

Breaks with fewer than 52 weeks of data after the change cannot be assessed for persistence. They are counted as "not assessable" and are not high-confidence.

**Per-origin version (for Task 3, no lookahead).** Computed from data up to the origin only:
- agreement, using PELT run on data up to the origin with the frozen penalty
- effect size, using up to 26 post-change weeks (at least 8)
- persistence, with "later" taken as the last 8 weeks before the origin, required when at least 34 weeks have passed since the change

**Reported.**
- **Counts:** per category at each step of the filter.
- **Direction:** up or down.
- **Position of the change week:**
  - weeks since first sale
  - in the launch ramp (within 26 weeks of monitoring start)
  - near the series end (within 26 weeks of the last week)
- **Possible delisting or stockout artifact:** a downward break where the item's last 8 weeks are all zero, or where the post mean is below 10% of the pre mean.

## Task 3: adaptation variants

Model: LightGBM (the best static model). ETS is not rerun, because variants (b) and (c) do not apply to it. Same 7 origins (1 warm-up + 6 evaluation) and the same break detection at each origin as the existing analysis.

| Variant | Definition (applies to items with a break detected by the origin; other items unchanged) |
|---|---|
| (a) truncate | train only on rows from the latest change week (existing "adaptive") |
| (b) decay weights | sample weight 1 from the change week on; before it, 0.5^(days before change / 182) |
| (c) level-shift feature | two added features: `post_break` (1 from the change week) and `weeks_since_break` (capped at 104; 0 before or if none) |
| (d) blend | 0.5 × static forecast + 0.5 × truncate forecast |

**Evaluation.** Per-item mean of (variant − static) MASE over the evaluation origins at which the condition holds, for two groups: all detected breaks, and the per-origin high-confidence subset.
- **Tests:** paired Wilcoxon signed-rank; rank-biserial effect size; bootstrap 95% CI of the median; Holm correction across the 4 variants within each group. Raw and adjusted p-values are both reported.
- **A variant "helps" if** Holm-adjusted p < 0.05, the median difference is negative, and it is at least 0.0074 MASE in size (1% of the static median of 0.744).
- **Small subgroups:** fewer than 20 items are reported descriptively only.
- **Hypothesis H3:** no variant helps on all detected breaks. For the high-confidence subset no direction is predicted.

## Task 4: forecast statistics and calibration

- **LightGBM vs ETS:** paired Wilcoxon on per-item mean MASE over the 6 evaluation origins (static variants, 726 items), with effect size and CI.
- **Bias:** pooled bias (total forecast / total actual − 1) and the median per-item bias with a bootstrap CI, per model and category.
- **Calibration method (split conformal on rolling-origin residuals):**
  - For item i at origin k, the calibration set is the residuals of all items in the same category from origins earlier than k, each divided by that item's MASE scale at its origin.
  - Lower and upper offsets are the α/2 and 1 − α/2 quantiles of those normalized signed residuals, with the finite-sample correction ⌈(n+1)q⌉/n, multiplied back by item i's scale.
  - Bands are clipped at zero.
- **Coverage:** at 80% and 90% before (existing per-item empirical quantile bands) and after, on evaluation origins 2–6. Origin 1 has only the warm-up origin to calibrate on and is reported separately.
- **Success criterion:** overall coverage within ±2 percentage points of nominal at both levels.
- **Inventory rerun:** safety stock = item scale × the service-level quantile of normalized lead-time (7-day) error sums from earlier origins in the same category. Rerun the simulation for the three existing policies at 90/95/99% and compare fill rate, achieved cycle service level, inventory and cost with the existing normal-formula results.

## Task 5: inventory robustness

- **Windows.** Three consecutive 168-day windows:

  | Window | Dates | Seasonal coverage |
  |---|---|---|
  | W1 | 2015-12-07 to 2016-05-22 | winter–spring; includes Christmas, no summer |
  | W2 | 2015-06-22 to 2015-12-06 | summer–autumn; includes Thanksgiving |
  | W3 | 2015-01-05 to 2015-06-21 | winter–spring; no Christmas |

  Each has a 28-day warm-up origin before it. The CUSUM threshold is recalibrated by the same procedure on data before the earliest origin of each window (same procedure, not the same value, to avoid lookahead). Items need a full reference window before the first origin. The AX/AY scope was defined on the full history, and that is a stated limitation.
- **Policies.** LightGBM static, LightGBM truncate-adaptive, seasonal naive. Base lead time and costs.
- **Uncertainty.** Bootstrap over items (5,000 resamples) for the total-cost differences adaptive − static and static − naive, per window and service level. Per-item median difference and the share of the absolute gap from the 10 largest items are also reported.
- **Service-level grid.** 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995.
- **Newsvendor comparison.**
  - Critical ratio = m / (m + h × 7/365), where m = stockout cost multiple and h = annual holding rate. A marginal unit is assumed to be held for one 7-day order cycle.
  - For each of the 12 cost scenarios, compare the critical-ratio level with the cheapest simulated level on the grid.
  - Report the stockout-to-holding ratio m/h at which 99% stops being cheaper than 95%: (H99 − H95) / (S95 − S99), where H and S are the holding and stockout cost bases.
- **Service level definition.** The target is a cycle service level (the z-quantile for no stockout during a lead time). Both achieved measures are reported:
  - cycle service level: share of replenishment cycles with no lost sales
  - fill rate: share of demand served

## Task 6: replication on Texas and Wisconsin

Run the full pipeline per state with every setting above frozen at the CA value, including the CUSUM thresholds and the PELT penalty (not recalibrated). The only state-specific inputs are the store filter, the SNAP column and the scope rule applied to that state's items. Models: seasonal naive, ETS (static), LightGBM (static and truncate-adaptive).

A CA headline conclusion "holds" in a state if:
| # | Conclusion | Criterion |
|---|---|---|
| R1 | Breaks are common | break rate (full series) between 50% and 80% |
| R2 | LightGBM beats seasonal naive | lower median MASE and paired Wilcoxon p < 0.05 |
| R3 | LightGBM is at least as good as ETS | median MASE not higher than ETS |
| R4 | Truncate-adaptation does not help | fails the "helps" rule of Task 3 |
| R5 | Static LightGBM is cheaper than naive | lower total cost at 95%, base case |
| R6 | Adaptive is not cheaper than static | bootstrap 95% CI of (adaptive − static) total cost does not lie wholly below 0 |
| R7 | 99% is the cheapest of 90/95/99 | at base costs |

The null false-alarm rate of the frozen thresholds is also reported per state.

## Task 7: reproducibility
`scripts/build_facts.py` writes every headline number, with n and CI where relevant, to `outputs/results_facts.json` from files in `outputs/tables/`. Numbers that currently exist only in notebook output or are computed in the dashboard (bias, band coverage, leading-zero counts, unrounded MASE) get a tracked source table first.
