# Pre-registration: the time-series models

Written 2026-10-28, before any of the new models in 07c to 07h was scored. The rules, the list of tweaks and the
decision rules below are fixed. If something has to change after results are seen, the change and its reason are
added at the end of this file, and the original plan is still reported.

## What we already knew
This is not blind. We had seen every result in `docs/RESULTS.md`, including the 07a scores on the 150-product sample
(median RMSE / MAE: SARIMA 4.445 / 3.522, ETS 4.498 / 3.511, seasonal naive 6.039 / 4.720), and the 30-product
Prophet trial in notebook 12 (untuned, MASE 0.908, the weakest model there).

Before writing this we did three things that scored nothing:
1. Re-ran the SARIMA scoring from the saved forecasts and refitted all 1,050 SARIMA fits from scratch. Both give
   RMSE 4.445 and MAE 3.522, and the refitted forecasts equal the saved ones exactly.
2. Checked that the new wrappers in `src/ts_models.py` give the same SARIMA and ETS forecasts as 07a (they do, to the
   last digit, on three products).
3. Timed one fit of each new model on three products and three origins. No error was computed.

## Fixed backtest
| Setting | Value |
|---|---|
| Products | the same 150 as 07a (`outputs/tables/classical_sample_items.csv`) |
| Scored origins | the same 6 as 07a: 2015-12-06 to 2016-04-24, 28 days apart, 28-day horizon |
| Validation origins | 2015-09-13, 2015-10-11 and the warm-up origin 2015-11-08, each with its 28-day window |
| Fit window | the last 730 days up to the origin, as in 07a; damped ETS uses all history, as in 07a |
| Metric code | `evaluation.score` and `metrics.all_errors`, unchanged from 07a |
| Product score | mean over the origins in question; reported as the median across products |

The validation windows run from 2015-09-14 to 2015-12-06, so they end the day before the first scored day.
**Scored origins are never used to choose anything**: not a transform, an order, a regressor, a weight, a model list
or a profile route. In code, every choice goes through `ts_models.for_tuning()`, which only returns validation rows,
and a test checks that.

One known weakness: the SARIMA orders were chosen in 07a on data up to 2015-11-08, so at the two earlier validation
origins they have seen a few weeks past their own origin. This affects the validation numbers only, never the scored
ones, and we keep the 07a orders so SARIMA stays comparable with 07a.

## Metrics
RMSE, MAE and MAPE are the main metrics. MASE, sMAPE and WAPE are reported next to them. MAPE is computed only on
days with sales above zero (a zero day would divide by zero), and the share of days it leaves out is reported with
it. All metrics are per product and origin over the 28 days, then averaged over origins, then the median is taken
across products.

## The contenders
| # | Model | Settings before tuning |
|---|---|---|
| 1 | Seasonal naive | repeat the last week (baseline, not tuned) |
| 2 | Damped Holt-Winters (ETS) | additive damped trend, additive weekly season, all history (the 07a ETS) |
| 3 | SARIMA | the 07a orders, 730 days, statsmodels defaults |
| 4 | SARIMAX | SARIMA plus regressors (see the ablation) |
| 5 | Theta | statsmodels `ThetaModel`, period 7, `deseasonalize=True`, `method="auto"`, 730 days |
| 6 | Prophet | weekly and yearly seasonality, SNAP, event and relative-price regressors, 730 days |
| 7 | AutoTS | time-series models only, settings below, no tuning by us |
| 8 | Combination | equal-weight average of the strongest contenders, chosen on validation |

Building blocks, fitted but not tuned and not in the headline: AR(p), MA(q), ARMA(p, q) with d = 0 and orders 0 to 7
by AICc on the warm-up window; ARIMA (the 07a model); ARIMAX (ARIMA orders plus the SARIMAX regressors); VARIMA on the
three category totals.

## The tweaks we will try, and nothing else
| Tweak | What it is | Tried on |
|---|---|---|
| (a) transform | log1p, or Box-Cox on sales + 1 with lambda by maximum likelihood on each training window; forecasts turned back with the plain inverse (a median, not a mean) | ETS, SARIMA |
| (b) clipping | training days above Q3 + 3 IQR of the training window are set to that fence; the scored days are never changed | ETS, SARIMA, Theta, Prophet |
| (c) Fourier terms | K sine and cosine pairs with period 365.25 days, K in {1, 2, 3}; for Prophet, yearly order 3 instead of its default 10 | SARIMAX, Prophet |
| (d) Christmas closure dummy | 1 on 25 December, when the stores close | SARIMAX, Prophet |
| (e) convergence fix | lbfgs with up to 500 steps instead of 50; if that still does not converge, the simpler order (1, d, 1)(0, D, 1)₇ | SARIMA, SARIMAX |
| (f) combining models | equal weights, and inverse-error weights as a second version | the combination |

The SARIMAX regressors are, in this order of testing: Fourier terms, the Christmas dummy, SNAP and event days, and
relative price. SNAP is the California flag. An event day is any day with an M5 event other than Christmas. Relative
price is the product's median California price that day divided by its own average price over the past 365 days.
Prices are set by the retailer and are known ahead, as in the M5 competition, so the future prices are used for the
forecast days. The calendar is known ahead too.

## How a tweak is judged
Tweaks are added one at a time on top of what has been kept so far, in this order:

**SARIMA family:** (a) log1p and Box-Cox each on the base, the better of the two goes forward if it helps; (b)
clipping; (c) Fourier with K = 1, 2, 3, the best K goes forward if it helps; (d) Christmas dummy; SNAP and event
days; relative price; (e) the convergence fix.
**ETS:** (a) log1p and Box-Cox, then (b) clipping.
**Theta:** (b) clipping.
**Prophet:** (b) clipping, then (d) the Christmas regressor, then (c) yearly order 3.

**Rule.** A tweak is kept if it lowers both the median RMSE and the median MAE on the validation origins. If the two
disagree, it is not kept. MAPE is reported but does not decide.

**One exception, fixed now.** No validation window contains 25 December, so validation cannot show what the
Christmas dummy does on the day itself; it can only show its effect on the rest of the fit. The stores close every
year, so we keep the dummy unless it makes validation RMSE or MAE worse.

Every variant tried goes into the ablation table with both its validation and its scored numbers. The scored numbers
are shown for honesty; they are not used to keep or drop anything.

The **SARIMA contender** is the base SARIMA plus the kept (a), (b) and (e). The **SARIMAX contender** is that model
plus the kept regressors. If no regressor is kept, SARIMAX uses all of them and we say so. ARIMAX uses the same
regressors as SARIMAX on the 07a ARIMA orders.

## AutoTS
Wide format, the 150 products together, one run per origin on data up to that origin only. `forecast_length=28`,
daily frequency, `ensemble=None`, `max_generations=2`, `num_validations=2`, `validation_method="backwards"`,
`random_seed=42`. Model list, time-series models only: ConstantNaive, LastValueNaive, AverageValueNaive,
SeasonalNaive, ETS, ARIMA, Theta, UnobservedComponents, DynamicFactor, VAR, VECM. No regression, tree, boosting,
motif or neural models. A smoke test on 5 products and one origin comes first and is timed. If the full run would take
more than 8 hours it is cut down, and the cut is written into the notebook. With ensembles off, AutoTS picks one
model for all products at each origin. We save that choice and, for each product, the model that scored best for it
in AutoTS's own internal validation.

## The combination
The members are the three contenders (from 2 to 7) with the lowest validation median RMSE, ties broken by MAE. A
contender only counts if it has validation forecasts for all 150 products. The headline combination is their plain
daily average. The second version weights each member per product by 1 / (its mean squared error on that product
over the validation origins). We check whether the combination beats its best member on the scored origins and say
so plainly either way.

## Demand profiles and routing
tsfresh features (the efficient set; the minimal set if efficient takes over 30 minutes) on each of the 726 products'
sales up to 2015-11-08. Features with missing or infinite values are dropped. The rest are standardised and reduced
with PCA to 90% of the variance. Then k-means with k from 2 to 6, seed 42, keeping the k with the best silhouette.
Routing picks, per profile, the contender with the lowest validation median RMSE among the sample products in that
profile. A profile with fewer than 10 sample products uses the overall validation winner. The routing is a selection
strategy, not a ninth model.

## Tests
Paired Wilcoxon signed-rank tests on per-product scores (scored origins averaged), each model against the 07a
SARIMA, separately for RMSE, MAE and MAPE. Holm correction within each metric across all models compared. α = 0.05,
two-sided, bootstrap 95% CI of the median difference (`metrics.paired_test`, seed 42).

## What we expect
- No contender beats SARIMA by much. The weekly pattern is most of the signal, and ETS and SARIMA already capture it.
  We would be surprised by a gain above 3% in median RMSE.
- log1p and Box-Cox lower MAE and MAPE but raise RMSE, because they forecast a median. So under our rule they are
  probably not kept.
- Clipping helps a little. Fourier terms do little over 28 days, because the yearly swing is about 10%.
- SNAP helps FOODS products. 03 found FOODS sells about 10% more on SNAP days.
- The convergence fix changes little on products. It matters more on the totals.
- Theta lands near ETS. Prophet with regressors does better than in notebook 12 but stays behind SARIMA.
- AutoTS lands between seasonal naive and ETS, because one model is picked for all 150 products.
- The combination is the best or close to it, as is usual for averaged forecasts.
- VARIMA does not beat SARIMA on the totals. The categories move together mostly through the calendar.

## Why LightGBM, XGBoost and the GRU are left out
This comparison is about time-series models only. LightGBM, XGBoost and the GRU stay in the repo with their own
results (09a, 09b, 10, 12), but they are not refitted, tuned or compared here.

## Changes after the start
1. **Profiles (07g), decided before any tsfresh feature was computed.** Each product's 730-day window is divided by
   its own mean before the features are extracted, and the log of the mean is added back as one feature. Without
   this, most tsfresh features (sums, energies, quantiles) only measure how much a product sells, and the profiles
   would just be size bands. The window is the last 730 days up to 2015-11-08, the same as the fits use.
2. **No machine-learning models in the project, decided before any 07h result was seen.** LightGBM (09a, 09b) and
   notebook 12 (XGBoost, GRU) are removed from the project and the dashboard. Where LightGBM was the forecast behind
   a later result (the comparison in 10, the stock work in 11a to 11f), it is replaced by the contender with the
   lowest median RMSE in 07h, whatever that turns out to be, and those notebooks are run again. The old and new
   numbers are reported side by side.
3. **A code defect in the price regressor, fixed after the first 07d run.** For products whose price never changed
   in the 730-day window, relative price is a constant column. After differencing it cannot be estimated, the fit
   failed, and 10.6% of the "+ relative price" fits fell back to seasonal naive. Regressors that do not vary in a
   training window are now dropped before the fit (`ts_models.drop_constant`). This changes only the relative-price
   step, which was rerun; no other step had a constant regressor. The first run's numbers for that step are in
   `docs/RUN_LOG.md`.
4. **A guard against blow-ups in 09, added after its first run.** SARIMA on all 726 products gave four
   product-origins with forecasts 9 to 294 times the most the product had sold in its training window (for example
   HOUSEHOLD_1_103 at 2016-02-28: 145,651 units forecast over 28 days against 105 sold), which pushed the pooled bias
   to +10% and the HOUSEHOLD total's MASE to 4.7. A forecast above 3 times the training maximum now falls back to
   seasonal naive, like a failed fit (`best_model.CAP`). The same four are caught at any cap from 1.5 to 5, none of
   them is a sample product, so 07a and 07h are unchanged; only those four products were refitted.
5. **Profile routing removed, decided after 07h.** The tsfresh profiles (change 1) found no real groups: the best
   split had two profiles, one of them a single product with no sample products, so routing sent every product to
   the validation winner, SARIMAX, and its forecasts were SARIMAX's exactly. A copy of another contender adds nothing
   to the comparison, so routing and its notebook were removed and 07h was run again without it. Every other number
   in 07h was unchanged; the Holm-corrected p-values moved a little, because there is one model fewer in each family,
   and no test changed from significant to not or back.
6. **The blow-up guard applied in 07h too, decided after the effect on the ranking was known.** Change 4 added the
   guard in 09 and checked that SARIMA, then the winner, had no blow-up on the sample. It did not check the other
   contenders. Fitting the combination on all 726 products showed that SARIMAX, one of its members, had two
   blow-ups on sample products at scored origins: FOODS_3_203 at 2015-12-06 (193 units a day against a training
   maximum of 38) and HOUSEHOLD_1_217 at 2016-01-03 (2,263 against 23). 07h now applies the same guard, with the same
   code, to every model at every origin before anything is scored. It looks only at the training data and changes
   only those two forecasts. Nothing at the validation origins changes, so every tuning choice and the combination's
   members stay as they were. With the guard the combination has the lowest scored median RMSE (4.392 against
   SARIMA's 4.445) and is the winner by the rule of change 2, so 09, 10 and 11a to 11f were run again with it. We
   say plainly that this was decided after we knew it would change the winner. The case for it is that the guard is
   a fit-failure rule fixed in change 4, before this was known, and a forecast of 2,263 units a day for a product
   that never sold more than 23 is a failed fit; leaving it in would score a model on a numerical failure. The
   SARIMA results are kept in `docs/RUN_LOG.md` and next to the new ones in `docs/RESULTS.md`, and
   `09b_runner_up_all_products` fits the runner-up, now SARIMA, on all 726 products as a check.
