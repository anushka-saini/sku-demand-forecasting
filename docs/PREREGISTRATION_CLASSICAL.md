# Pre-registration: classical models, stationarity and decomposition

Written 2026-10-21, before any classical model was scored. The notebook names and file paths have since been
updated to the current layout (`docs/RESTRUCTURE_MAP.md`); the rules, definitions and thresholds are as written.

## What was already known when this was written
This is not blind. Everything in `docs/RESULTS.md` had been seen, including: median MASE 0.976 / 0.755 / 0.744
(seasonal naive / ETS / LightGBM) on all 726 items, no adaptation variant helping LightGBM, and the inventory results.

One timing run was done first (`docs/RUN_LOG.md`, run 10): 5 items, one origin. It recorded run times and the SARIMA
orders picked by the grid. **No forecast was scored**, so no accuracy number for any new model has been seen.

## Rules
1. The rules in `docs/PREREGISTRATION.md` still apply (no lookahead, no tuning on evaluation windows, raw data never
   modified, every run logged, seed 42, two-sided tests, α = 0.05).
2. Nothing in the earlier analysis, the saved forecasts or the published numbers is changed. New work goes in new
   notebooks, new modules and new result files.
3. Definitions and thresholds here are not changed after results are seen. Any deviation is logged with its reason,
   and the original analysis is still reported.

## Frozen backtest design (the same as for LightGBM and ETS)
| Setting | Value |
|---|---|
| Origins | 7, 28 days apart, last test day 2016-05-22 (`pipeline.make_origins`) |
| Horizon | 28 days |
| Warm-up | origin 0 (2015-11-08): its errors seed bands and safety stock only |
| Scored | origins 1–6 (test days 2015-12-07 to 2016-05-22) |
| Series | trimmed daily CA sales summed over the 4 stores (`data/interim/*_candidates.csv`) |
| Breaks | online CUSUM at each origin, as saved in `data/processed/breaks_by_origin.csv` (h = 13.83) |

## Item subset
- **Sample:** 150 items, 50 per category. Within each category, 25 items with a detected break at any scored origin
  and 25 without (the same definition as for the LightGBM backtest: 488 against 238 items). Drawn with `numpy.random.default_rng(42)`,
  sorted item ids, without replacement. The list is saved to `outputs/tables/classical_sample_items.csv` before any fit.
- **Aggregates:** FOODS, HOBBIES, HOUSEHOLD totals and the CA total, each summed over the 726 items.
- 150 instead of the minimum 60 was chosen from the timing run (about 25–35 min with 12 workers).

## Step 2: stationarity (offline, full trimmed series, all 726 items and the 4 aggregates)
Tests at 5%: ADF (`autolag='AIC'`) and KPSS (`nlags='auto'`), each with a constant (`c`) and with a trend (`ct`).
| Class | Rule |
|---|---|
| stationary | ADF(c) rejects a unit root and KPSS(c) does not reject stationarity |
| trend-stationary | KPSS(c) rejects, KPSS(ct) does not reject, and ADF(ct) rejects |
| difference-stationary | KPSS(c) rejects and ADF(ct) does not reject |
| conflicting | anything else |

Recommended orders: `d = pmdarima.ndiffs(test='kpss', max_d=2)`, `D = pmdarima.nsdiffs(m=7, test='ocsb', max_D=1)`.
Shares reported per category.

**Outliers:** a day is flagged if it is above Q3 + 3 × IQR of that item's daily sales. Flagged days are counted and
reported but **not changed**; models see the raw values. **Missing values:** counted (none expected after trimming);
calendar gaps checked.

## Step 3: decomposition (offline)
Classical additive and multiplicative decomposition (period 7) and STL (period 7, robust) on the 4 aggregates and
6 example items (2 per category, the first two of the sample in sorted order).

**Choice rule.** For each series, compute the weekly mean (level) and the within-week range (max − min) for every
full week. If the Spearman correlation between level and range is ≥ 0.5 with p < 0.05, the seasonal swing grows with
the level and multiplicative is preferred; otherwise additive. Reported alongside: the residual standard deviation
divided by the series mean for each form.

**Zeros.** Multiplicative decomposition is undefined with zeros. Items: computed on sales + 1, labelled as such.
Aggregates: 25 December (stores closed) is replaced by the mean of the same weekday one week before and after,
for decomposition only; forecasting always uses the unmodified series.

## Step 4: classical models
| Model | Specification | Scope |
|---|---|---|
| Holt-Winters additive | additive trend, additive weekly season (m = 7), not damped | 150 items, 4 aggregates |
| Holt-Winters damped additive | **the existing ETS** (full history), relabelled; not refitted | reused |
| Holt-Winters multiplicative | additive trend, multiplicative season (m = 7); on items only with sales + 1, and only for H4 | 4 aggregates; 150 items (H4) |
| ARIMA | (p, d, q), no seasonal part | 150 items, 4 aggregates |
| SARIMA | (p, d, q)(P, D, Q)₇ | 150 items, 4 aggregates |
| Seasonal naive, ETS, LightGBM | saved forecasts, not retrained | reused |

- **Fit window:** the last 730 days up to each origin. Two years contain the yearly cycle twice; ARMA terms are short
  memory, so older data adds little, and it keeps fits fast and stable. The reused ETS and LightGBM use all history;
  this difference is stated wherever they are compared.
- **Orders:** grid p, q ∈ {0, 1, 2}, P, Q ∈ {0, 1}; d and D from `ndiffs` / `nsdiffs` on the fit window. Aggregates:
  p, q ∈ {0..3}, P, Q ∈ {0..2}. Chosen by lowest AICc **at origin 0**, then fixed for that series; parameters are
  re-estimated at every origin. ARIMA uses the same rule without the seasonal part.
- **Fallback:** if a fit raises an error or gives a non-finite forecast, that item-origin uses the seasonal-naive
  forecast. Fallback rate and non-convergence rate (convergence warning, forecast kept) are reported per model.
- All forecasts are clipped at 0. Parallel with joblib; results checkpointed per item in `data/processed/classical_fits/`.
- **No lookahead test:** a forecast at an origin must be unchanged when every value after that origin is replaced by
  noise. This is a unit test on the backtest function.

## Step 5: assumptions and diagnostics
On the residuals of each item's origin-0 fit: Ljung-Box at lags 7 and 14 (pass = p > 0.05 at both) and Jarque-Bera
normality (pass = p > 0.05). Shares passing are reported per model. Failing normality does not invalidate point
forecasts; it is reported because it matters for interval forecasts and the normal safety-stock formula.

## Step 6: metrics
Per item per origin, on the 28 test days: RMSE, MAE, MASE (existing scale: in-sample seasonal-naive MAE up to the
origin, all history), MAPE on days with actual > 0 only (share of excluded days reported), sMAPE = mean of
2|a − f| / (|a| + |f|) over days where |a| + |f| > 0, WAPE = Σ|a − f| / Σ|a|. Item score = mean over origins 1–6;
reported as median and mean across items. Aggregates are scored the same way as single series.

## Hypotheses and decision rules
All paired tests: Wilcoxon signed-rank on per-item MASE differences (origins 1–6 averaged), bootstrap 95% CI of the
median difference (5,000 resamples), `metrics.paired_test`.

**H1.** SARIMA and Holt-Winters additive beat seasonal naive. Family of 2 tests, Holm-corrected. Supported for a
model if its median difference vs naive is < 0 and Holm p < 0.05.

**H2.** LightGBM has the lowest median MASE of {seasonal naive, ETS, Holt-Winters additive, ARIMA, SARIMA,
LightGBM} on the 150 items. Paired tests of LightGBM against each of the 5 others, Holm-corrected (family of 5).
Reported either way.

**H3.** Shift-aware classical variants do not improve on static SARIMA. On item-origins with a break detected before
the origin (the saved `has_break`):
- **restart:** SARIMA, same order, fit only on data from the latest change week (`adaptive_start`, at least 56 days);
- **dummy:** SARIMAX, same order, exogenous level-shift dummy = 1 from the latest change week, and 1 over the horizon.

Control (descriptive, outside the Holm family): **placebo** = SARIMA fit from the saved `placebo_start` on items
without a break, exactly as for LightGBM.
A variant **helps** if Holm p < 0.05 (family of 2), median difference < 0, and |median difference| ≥ 1% of the median
static SARIMA MASE on the same items. H3 is supported if neither variant helps.

**H4 (descriptive, no test).** Among the 150 items, "zero-heavy" = share of zero days before origin 0 above the
sample median. H4a: additive Holt-Winters has lower median MASE than multiplicative on zero-heavy items (multiplicative
failures count as fallbacks and are reported). H4b: on the 4 aggregates, multiplicative MASE is within 5% of additive.

## Step 8: stock link (descriptive)
SARIMA and Holt-Winters additive forecasts go into the existing simulation (`pipeline.simulate_inventory`) on the
150 items, 7-day lead time, 95% target, base costs (25%/yr, 0.5 × price), normal-formula and conformal safety stock
(the conformal rule from `11c_calibrated_safety_stock`, pooled within category over the 150 items). Compared with ETS, LightGBM static and seasonal
naive on the same items: per-item total cost and fill rate, paired Wilcoxon and bootstrap 95% CI of the total
difference. No decision rule; descriptive.

## Not part of this plan
XGBoost, Prophet and a GRU are a later start (`12_advanced_models_wip`), run only after these checks pass, and
never reported as results of this plan.
