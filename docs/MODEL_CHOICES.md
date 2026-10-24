# Model choices

Why each model is in the project, how we set it up, and the decisions we made on the way. The numbers behind these
choices are in the notebooks named below.

## What the models have to handle

Each series is the daily units sold of one product, summed over the four California stores, starting at the
product's first sale (the zeros before launch are removed in `04_zeros_and_trimming`). Two things drive every choice.
There is a strong weekly pattern: weekends sell clearly more than midweek (`03_explore`), so every model needs a
period-7 season, and ARIMA, which has none, is only there to show what the weekly terms add. And most levels are
steady for long stretches with the occasional lasting shift (`05_stationarity`, `08a_break_detection`), so we start
with low-order models and test shift-aware versions separately (`08d_shift_aware_classical`).

## The models

| Model | Why it is here | How it is set up | What we checked |
|---|---|---|---|
| Seasonal naive | The baseline: next week is the same as last week | nothing to set | nothing to check |
| Damped Holt-Winters (ETS) | Exponential smoothing with a trend that fades out, so a short trend is not run on for 28 days | additive trend and season, period 7, damped, all history | residual autocorrelation (`07b`) |
| Holt-Winters additive | The same without damping; fast and easy to explain | additive trend and season, period 7, last 730 days | residual autocorrelation; the additive season is backed by the swing rule in `06_decomposition` |
| Holt-Winters multiplicative | The weekly swing as a share of the level, for series whose swing grows with the level | totals as they are; products use sales + 1 when the window has a zero day | swing rule; residuals |
| ARIMA(p, d, q) | Short-term autocorrelation without a season | d from a KPSS-based `ndiffs`; p, q from 0 to 2 by lowest AICc at the warm-up origin | stationarity after differencing, residuals, roots |
| SARIMA(p, d, q)(P, D, Q)7 | The standard classical model for a seasonal series | d from `ndiffs`, D from `nsdiffs` (OCSB); p, q from 0 to 2 and P, Q from 0 to 1 (totals up to 3 and 2), lowest AICc at the warm-up origin; the order is then fixed and the parameters refitted at every origin | as ARIMA, plus the seasonal roots and a normality check |
| TSB | Built for intermittent demand: it forecasts how often a product sells and how much when it does | used only for the 26 intermittent products | compared with the others on those products only |
| LightGBM | One gradient-boosted model for all products, with calendar, price and lagged sales features | lags of at least 28 days, Tweedie loss, retrained at each origin | no distribution assumed; no information after the origin by design |

XGBoost, Prophet and a small GRU are tried on 30 products in `12_advanced_models_wip`. That is a start, not a result.

## Why these settings

We use AICc rather than AIC because it corrects for small samples. With 730 days and at most seven parameters the
correction is small, but it costs nothing.

Orders are chosen once, at the warm-up origin, and then kept. Choosing at every origin would use more data, but it
makes the origins harder to compare, and choosing at a scored origin would let later data shape the warm-up
forecasts that the bands and safety stock learn from.

The classical models fit on the last 730 days: two yearly cycles. ARMA terms only remember a few weeks, so older
data adds little and slows fitting. Seasonal naive, ETS and LightGBM use all history. We say so wherever they are
compared.

When nothing is differenced, SARIMA gets a constant. Without one, statsmodels' SARIMAX forecasts drift towards zero.

If a fit fails or gives a forecast that is not a finite number, that product and origin falls back to seasonal
naive, so a failed fit never drops a product from the comparison. The fallback and non-convergence rates are in
`outputs/tables/classical_run_stats.csv`.

MASE is the main score. RMSE and MAE grow with the size of the product, and MAPE cannot handle days with no sales
(it skips them, and we report how many it skipped). MASE has no units, so products of different size can be
compared and pooled.

## Other decisions

We fit the item-level classical models on 150 products, 50 per category, half with a detected break and half
without, drawn with seed 42. 150 took about 40 minutes on 12 cores, so all 726 would take around three hours at
that rate.

On the totals, classical models are fitted on each total. Seasonal naive, ETS and LightGBM were never fitted on
totals, so for them we add up the product forecasts.

Stores close on 25 December, so the totals are almost zero that day, which breaks a multiplicative decomposition.
For decomposition only, we replace that day with the average of the same weekday a week before and after.
Forecasting always uses the real series.

The restarted and placebo windows in the shift-aware SARIMA are also capped at 730 days, so a restart never has
more data than the static model. When the latest change is older than the window, the restart equals the static
model; we count those as a difference of zero in the planned test and add a second, descriptive test on only the
cases where the two differ.

`nsdiffs` asked for no seasonal differencing on most series, so in SARIMA the weekly pattern is carried by the
seasonal AR and MA terms.

## How to read the residual checks

A Ljung-Box pass means no autocorrelation is left in the residuals, so the model has used the dependence in the
series. A fail means a better model is possible, not that the forecasts are useless.

Normality almost never holds for daily retail counts, which are skewed and full of small values. It matters for
prediction intervals and for the textbook safety-stock formula, which is why the stock simulation also uses the
calibrated safety stock from `11c_calibrated_safety_stock`.

Roots outside the unit circle confirm that a fitted SARIMA is stationary and invertible.
