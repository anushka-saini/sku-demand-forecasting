# Results: pre-registered robustness and replication analyses

Analyses defined in [`PREREGISTRATION.md`](PREREGISTRATION.md) before they were run. Every run, including one failure, is in [`RUN_LOG.md`](RUN_LOG.md). Every number below is in `outputs/results_facts.json`, which `scripts/build_facts.py` builds from `outputs/tables/`.

## Summary

| Question | Before | After the checks | Verdict |
|---|---|---|---|
| Do demand shifts exist and can we detect them? | 64–69% of items have a break; 4.8% false alarms | On a stricter null the false-alarm rate is 11.9%. The chance-adjusted share with a real shift is 63.5% (95% CI 59.8–67.2%). Only 35.0% of breaks are high-confidence. Shifts under ~30% of mean sales are mostly missed. | Partly. Real shifts are common, but the detector is coarse. |
| Which forecast is best? | LightGBM 0.744 median MASE vs ETS 0.755, naive 0.976 | LightGBM beats ETS in a paired test (p < 0.001) and beats naive in CA, TX and WI. It under-forecasts by 3.6% in total. | LightGBM, with a known low bias. |
| Does retraining after a shift help? | Truncation: +0.0033 MASE (slightly worse) | None of 4 adaptation methods helps, on all breaks or on high-confidence breaks. Truncation is also worse in TX and WI. | No. Holds across variants and states. |
| What does it mean for stock? | Static LightGBM: 16.6% less inventory and 10.1% lower cost than naive | Holds in all 3 time windows at 95% (CIs below 0), and in TX and WI. Calibrated safety stock cuts cost by a further 31% and moves fill rate from 97.7% to 99.0%. | A better static forecast plus calibrated safety stock is what pays. |

## 1. Detector calibration (Task 1)

Why the old number was optimistic. The old 4.8% false-alarm rate was measured on the same simple null that set the threshold, so it was true by construction. The new null keeps each item's annual seasonal pattern and the remainder's autocorrelation (STL seasonal + 13-week block bootstrap of the remainder), with no shift. It uses 20 null series per item, for the 718 items with at least two years of data.

| | False-alarm rate on seasonal null | Items flagged | Chance-adjusted share with a real shift (95% CI) |
|---|---|---|---|
| CUSUM | 11.9% (14,360 series) | 67.8% | 63.5% (59.8–67.2%) |
| PELT | 16.1% (14,360 series) | 81.8% | 78.3% (74.9–81.6%) |

By category, the CUSUM adjusted share is 65.5% for FOODS, 59.5% for HOBBIES and 62.7% for HOUSEHOLD.

Power. Known shifts were injected into null series (3,397 trials per row):

| Shift (% of mean) | Step detected | Step within 26 wks | Step median delay | Drift detected | Drift median delay |
|---|---|---|---|---|---|
| 0 (control) | 7.0% | 1.9% | – | – | – |
| 5% | 7.5% | 1.8% | 54 wks | 7.5% | 54 wks |
| 10% | 9.3% | 2.1% | 53 wks | 9.0% | 54 wks |
| 20% | 18.3% | 4.7% | 45 wks | 17.0% | 55 wks |
| 30% | 35.1% | 11.7% | 36 wks | 32.2% | 50 wks |

Hypotheses
- H1.1 (false-alarm rate above 5%): supported.
- H1.2 (adjusted share below observed): supported.
- H1.3 (5–10% shifts undetectable): supported.
- H1.4 (drifts detected later than steps at every size): not supported. It holds at 20% and 30% but not at 5%, where both delays are 54 weeks.

Correction. The earlier "45% false alarms on unmodified series" was never a false-alarm rate. It is how often CUSUM alarms on *real* series after a random date with nothing injected. On clean null series the same control gives 7.0%, which suggests most of that 45% was real drift.

Not interpretable: running the detector on deseasonalized series flags 97% of items. The thresholds were set for raw series and are much looser on deseasonalized data. No matching null was pre-registered, so this is not evidence that 97% of items shift.

## 2. High-confidence breaks (Task 2)

A break counts as high-confidence if PELT agrees within ±8 weeks, the level changes by at least 20%, and at least half the change persists in weeks 27–52.

| | CUSUM breaks | High-confidence | Share |
|---|---|---|---|
| All | 488 | 171 | 35.0% |
| FOODS | 234 | 67 | 28.6% |
| HOBBIES | 68 | 29 | 42.6% |
| HOUSEHOLD | 186 | 75 | 40.3% |

- Direction: 64.9% of high-confidence breaks are drops.
- Size: their median level change is 49.8%.
- Launch-ramp cluster: 38.9% of breaks fall within 26 weeks of monitoring start, against 13.8% expected by chance. High-confidence breaks show the same cluster (39.8%).
- Delisting or stockout: 20 breaks (4.1%) look like a delisting or stockout, and 10 of them pass the high-confidence filter.

## 3. Adaptation variants (Task 3)

LightGBM, same 6 evaluation origins, breaks detected using only data up to each origin. The values are the median per-item change in MASE vs static (negative = better), with 95% CIs and Holm-adjusted p-values across the 4 variants. "Helps" requires p_holm < 0.05, a negative median, and a size of at least 0.0074.

| Variant | All breaks (n = 488) | p_holm | High-confidence (n = 254) | p_holm |
|---|---|---|---|---|
| (a) truncate to post-break | +0.0033 (+0.0015 to +0.0053) | <0.001 | +0.0033 (+0.0008 to +0.0074) | 0.017 |
| (b) decay weights | +0.0011 (+0.0004 to +0.0024) | 0.138 | +0.0010 (−0.0008 to +0.0026) | 1.000 |
| (c) level-shift feature | +0.0001 (−0.0012 to +0.0006) | 0.608 | +0.0007 (−0.0008 to +0.0019) | 1.000 |
| (d) blend static + truncate | −0.0010 (−0.0018 to −0.0001) | 0.220 | −0.0009 (−0.0023 to +0.0004) | 0.905 |

No variant helps. Truncation is significantly *worse*. The blend's CI is just below zero for all breaks, but the effect is about a seventh of the practical threshold and not significant after correction.

## 4. Forecast statistics and calibration (Task 4)

LightGBM vs ETS: the median per-item MASE difference is −0.0091 (CI −0.0115 to −0.0066), p < 0.001. LightGBM is better on 61.6% of items. The difference is largest in FOODS (−0.0178); HOBBIES (p = 0.053) and HOUSEHOLD (p = 0.049) are marginal.

Bias (total forecast ÷ total actual − 1, evaluation origins):

| Model | Pooled bias | Median item bias (95% CI) |
|---|---|---|
| Seasonal naive | −0.8% | +0.1% (−0.5% to +0.8%) |
| ETS | −0.9% | −0.7% (−1.3% to −0.1%) |
| LightGBM static | −3.6% | −2.5% (−3.0% to −1.7%) |
| LightGBM adaptive | −4.2% | −1.9% (−2.6% to −1.2%) |

LightGBM's under-forecast is concentrated in FOODS (−4.9% pooled).

Coverage of the 90% / 80% bands, LightGBM static, origins 2–6:

| | Before (per-item empirical quantiles) | After (split conformal, category-pooled) |
|---|---|---|
| 90% band | 85.5% | 89.9% |
| 80% band | 75.6% | 79.8% |

The conformal bands meet the ±2-point criterion for both LightGBM variants, including origin 1. Seasonal naive misses it at origin 1 (87.1% / 76.8%), which has only the warm-up origin to calibrate on.

Inventory with conformal safety stock (7-day lead time, LightGBM static; service-level target = cycle service level):

| Target | Method | Fill rate | Achieved cycle service level | Avg. inventory | Total cost |
|---|---|---|---|---|---|
| 95% | normal formula | 97.69% | 82.7% | $161,205 | $68,708 |
| 95% | conformal | 99.00% | 91.7% | $195,327 | $47,306 |
| 99% | normal formula | 98.48% | 89.0% | $181,180 | $53,492 |
| 99% | conformal | 99.70% | 97.3% | $262,176 | $38,743 |

At a 95% target, calibrated safety stock holds more inventory but cuts total cost by 31% (assumed costs). It is cheaper for 71.9% of items (median −$8.49 per item, CI −$10.09 to −$6.79). Achieved cycle service level still falls short of the target, but the gap shrinks from 12.3 to 3.35 points. Adaptive remains costlier than static under calibrated safety stock: $55,824 vs $47,306 at 95%.

## 5. Inventory robustness (Task 5)

*Corrected after a bug fix (run 9 in the run log): intermittent items are now classified on data before each window's own warm-up date. Window 1 now reproduces the stock simulation (`11b_inventory_simulation`) exactly.*

Three 168-day windows, each with its own warm-up and CUSUM threshold. Results at a 95% target and 7-day lead time:

| Window | Season | Items | Adaptive − static total cost (95% CI) | Static − naive total cost (95% CI) |
|---|---|---|---|---|
| W1 Dec 2015–May 2016 | includes Christmas | 723 | +12.6%: +$8,492 (+$3,625 to +$14,325) | −10.9%: −$8,298 (−$16,847 to −$357) |
| W2 Jun–Dec 2015 | includes Thanksgiving | 718 | +9.1%: +$6,349 (+$2,712 to +$10,376) | −13.6%: −$10,930 (−$20,055 to −$2,214) |
| W3 Jan–Jun 2015 | no Christmas | 709 | +6.7%: +$4,474 (+$1,054 to +$8,689) | −13.5%: −$10,350 (−$18,880 to −$2,943) |

- Adaptive vs static: adaptive costs significantly more in every window.
- Static vs naive: static LightGBM is significantly cheaper in every window.
- How concentrated the adaptive gap is: the 10 largest items account for 39%, 34% and 34% of the absolute per-item gap. The earlier "82%" was the 10 items' share of the *net* gap, a different measure.

Newsvendor check. The critical ratio m / (m + h·7/365) ranges from 0.974 to 0.999 across the 12 cost scenarios. The simulated cheapest target on a grid from 0.50 to 0.995 was 0.995, the top of the grid, in every scenario and window, so the true optimum may be higher. The simulation favours higher targets than the critical ratio because the achieved cycle service level runs well below the target.

Break-even ratio. For static LightGBM, 99% is cheaper than 95% whenever the stockout cost multiple exceeds about 0.25–0.27 × the annual holding rate: 0.268 in W1, 0.254 in W2, 0.264 in W3. At 25% a year holding cost, 99% stops being cheaper only if a lost sale costs less than about 6.3–6.7% of its price.

## 6. Replication on Texas and Wisconsin (Task 6)

Every threshold and setting is frozen at the CA value. The reusable pipeline reproduced the CA item set exactly. TX has 529 AX/AY items and WI has 598. The CUSUM false-alarm rate on the seasonal null is 11.2% in TX and 11.7% in WI (CA: 11.9%).

| CA conclusion | TX | WI |
|---|---|---|
| R1 breaks are common (50–80%) | holds: 66.4% | holds: 64.9% |
| R2 LightGBM beats naive | holds: 0.734 vs 0.948, p < 0.001 | holds: 0.779 vs 1.025, p < 0.001 |
| R3 LightGBM ≥ ETS | holds, weakly: 0.734 vs 0.739, paired p = 0.60 | holds: 0.779 vs 0.805, p < 0.001 |
| R4 truncation does not help | holds: +0.0046 (+0.0022 to +0.0077) | holds: +0.0071 (+0.0052 to +0.0096) |
| R5 static cheaper than naive (95%) | holds: $47,727 vs $56,254 | holds: $54,896 vs $60,474 |
| R6 adaptive not cheaper than static | holds: +$2,234 (+$252 to +$4,285) | holds: +$168 (−$2,393 to +$2,460) |
| R7 99% cheapest of 90/95/99 | holds | holds |

All 7 conclusions replicate in both states. Two are weak: R3 in TX (no significant difference from ETS), and R6 in WI, where adaptive and static cost about the same rather than adaptive costing more.

## 7. Reproducibility (Task 7)
- `scripts/build_facts.py`: computes every headline number, with n and CI, from `outputs/tables/` into `outputs/results_facts.json`.
- `Makefile`: `make all` runs the notebooks, then the facts script, then the tests; `make test` runs the tests only.
- `requirements.txt`: pinned to the versions used. Four unused packages were dropped.
- `.github/workflows/tests.yml`: runs pytest on every push.
- `docs/DATA.md`: download instructions. M5 is not committed.
- Tests: 79 pass, including new tests for the calibration, break-quality and pipeline functions and for the facts script.

## What held up
- LightGBM is the most accurate model, in CA, TX and WI.
- Retraining after a detected break does not help, under four different adaptation methods, on high-confidence breaks only, and in two other states.
- A better static forecast lowers inventory cost relative to the naive baseline, significantly in all three CA windows, and in TX and WI.
- With assumed costs anywhere near the tested range, a 99% target beats 90% and 95%.

## What did not hold or needed correcting
- False-alarm rate: 4.8% was optimistic. The realistic figure is 11.9% (CUSUM) and 16.1% (PELT).
- The 45% figure was mislabelled as a false-alarm rate.
- Breaks: only about a third of detected breaks are high-confidence, and small shifts (≤10%) are invisible to the detector.
- The adaptive cost gap: it is not just "a few items". Its total is significantly positive in all three CA windows and in TX, with 34–39% of the absolute gap from the top 10 items. In WI it is indistinguishable from zero.
- The original safety stock was badly under-calibrated: a 95% target achieved an 82.7% cycle service level. Conformal calibration fixes most of this and lowers cost.
- H1.4 (drift slower than step at every size) was not supported.

## Updated limitations
- Coarse detector: breaks are judged against each item's first year. Gradual drift and steps are only partly separable, and the launch-ramp cluster (38.9% of breaks) survives every filter.
- Scope: AX/AY items only, summed across stores. The scope is defined on full history, which is a mild lookahead in item selection, but not in any forecast or decision.
- Costs are assumed: holding 15–35% a year, stockout 0.25–2× price, ordering cost ignored. The cheapest target sits at the edge of the tested grid.
- Service level: the target is a cycle service level. Achieved cycle service level stays below target even with conformal safety stock (91.7% at 95%).
- Evaluation length: 3 × 168 days in CA and 168 days in TX and WI. Results for other periods, or for slower-moving items, are untested.
- The dashboard (10 pages) shows these results, including the seasonal-null false-alarm rates, the retraining variants, calibrated safety stock, all three windows and the replication.
