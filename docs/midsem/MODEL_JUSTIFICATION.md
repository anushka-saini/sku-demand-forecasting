# Model justification (Milestone 2)

Why each model, how its settings were chosen, and how its assumptions were checked. Numbers are in
`docs/midsem/INTERIM_FINDINGS.md` and the tables named below.

## The data the models see
Daily units sold per product, summed over the four California stores, starting at each product's first sale
(pre-launch zeros removed in Milestone 1). Two things shape every model choice:
* **A strong weekly pattern** (weekend sales are 12-28% above an average day, notebook 00a), so every model needs a
  period-7 season, and a model without one (ARIMA) is included only as a reference.
* **Mostly stationary levels with occasional lasting shifts** (notebooks 01, 12), so low-order models with a
  stable level are a reasonable start, and shift-aware variants are tested separately (H3).

## Models
| Model | Why it is here | Settings and how they were chosen | Assumptions checked (notebook 14) |
|---|---|---|---|
| Seasonal naive | The benchmark every model must beat: "next week = last week" | none | none |
| Holt-Winters additive | Classical exponential smoothing with a level, a trend and a fixed-size weekly pattern; fast and easy to explain to planners | additive trend, additive season, m = 7; smoothing parameters by maximum likelihood (statsmodels) | residuals uncorrelated (Ljung-Box lags 7, 14); additive season supported by the swing rule (notebook 12) |
| Holt-Winters damped additive | The trend fades out, so a short-term trend is not run forward for 28 days | this is the Milestone 1 ETS (notebook 02), full history, reused not refitted | same residual checks (refitted at origin 0 only for the check) |
| Holt-Winters multiplicative | The weekly pattern as a percentage of the level, which fits series whose swing grows with the level | totals: as is; items: sales + 1 when the window has a zero (needed for H4) | swing rule; residual checks |
| ARIMA(p, d, q) | Short-term autocorrelation without a season; shows what the weekly terms add | d from `ndiffs` (KPSS); p, q ∈ {0, 1, 2} by lowest AICc at origin 0 | stationarity after d differences (notebook 12), white-noise residuals, stationary and invertible roots |
| SARIMA(p, d, q)(P, D, Q)₇ | The standard classical model for seasonal series | d from `ndiffs`, D from `nsdiffs` (OCSB); p, q ∈ {0, 1, 2}, P, Q ∈ {0, 1} (totals: p, q ≤ 3, P, Q ≤ 2) by lowest AICc at origin 0; order fixed, parameters re-estimated at each origin | as ARIMA, including the seasonal roots; normality (Jarque-Bera) for intervals |
| LightGBM | The Milestone 1 best model, as the bar to clear | reused from notebook 02 | no distributional assumption; no lookahead by design |

## Why these choices
* **AICc, not AIC:** with a 730-day window and up to 7 parameters the small-sample correction is minor but never hurts; it is the default in `auto_arima`.
* **Order chosen once, at origin 0:** choosing at every origin would use more data but make the comparison across origins less clean; choosing at a scored origin would leak later data into the warm-up forecasts (decision O2).
* **730-day window:** two yearly cycles; ARMA terms only remember a few weeks; older data adds little and slows fitting. The reused models use all history, which is stated in every comparison.
* **A constant when nothing is differenced:** without it SARIMAX forecasts drift to zero (decision D6).
* **Fallback to seasonal naive:** a failed fit should not remove an item from the comparison; fallback and non-convergence rates are reported (`outputs/m2/tables/classical_run_stats.csv`).
* **MASE as the headline metric:** RMSE and MAE depend on the item's size; MAPE breaks on zero days (it leaves them out, and the share left out is reported); MASE is scale-free and was the Milestone 1 metric, so results stay comparable.

## Assumption checks: how to read them
* **Ljung-Box pass** means no autocorrelation is left in the residuals, i.e. the model captured the dependence in the series. A fail means a better model is possible, not that the forecasts are unusable.
* **Normality** almost never holds for daily retail counts (skewed, many small values). It matters for interval forecasts and for the textbook safety-stock formula, which is why Milestone 1's calibrated (conformal) safety stock is used in notebook 16.
* **Roots outside the unit circle** confirm the fitted SARIMA is stationary and invertible.
