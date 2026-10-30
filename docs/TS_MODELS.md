# Time-series models

Every time-series model in the project: what it assumes, why it could suit these products, how it was set up and
tuned, and how it did. All scores are on the same backtest: the 150 sample products, six scored origins
(2015-12-06 to 2016-04-24), 28 days ahead, each product's mean over the origins and then the median across products,
with the 07a metric code. Settings were chosen on three earlier validation origins (2015-09-13, 2015-10-11,
2015-11-08) only; the plan is in `docs/PREREGISTRATION_TS.md`. A tweak was kept only if it lowered both validation
RMSE and MAE.

## The headline

| Model | Kind | RMSE | MAE | MAPE | MASE | Validation RMSE |
|---|---|---|---|---|---|---|
| SARIMA | contender, the winner | 4.445 | 3.522 | 0.603 | 0.755 | 4.697 |
| Damped Holt-Winters (ETS), Box-Cox | contender | 4.513 | 3.543 | 0.560 | 0.744 | 4.606 |
| Combination, inverse-error weights | second version | 4.520 | 3.576 | 0.591 | 0.748 | 4.559 |
| Combination (SARIMAX, ETS, Theta) | contender | 4.523 | 3.567 | 0.589 | 0.748 | 4.608 |
| Theta | contender | 4.541 | 3.510 | 0.602 | 0.756 | 4.691 |
| ARMA(p, q) | building block | 4.542 | 3.577 | 0.617 | 0.768 | 4.824 |
| SARIMAX (one yearly Fourier pair) | contender | 4.646 | 3.677 | 0.612 | 0.761 | 4.600 |
| AR(p) | building block | 4.761 | 3.922 | 0.640 | 0.786 | 5.241 |
| ARIMAX | building block | 4.801 | 3.866 | 0.637 | 0.801 | 4.939 |
| ARIMA | building block | 4.822 | 3.775 | 0.638 | 0.794 | 5.026 |
| AutoTS | contender | 4.848 | 3.885 | 0.626 | 0.831 | 4.742 |
| Prophet (with a Christmas regressor) | contender | 4.913 | 3.859 | 0.622 | 0.760 | 4.774 |
| MA(q) | building block | 5.288 | 4.299 | 0.641 | 0.839 | 5.497 |
| Seasonal naive | contender, the floor | 6.039 | 4.720 | 0.753 | 0.973 | 6.165 |

SARIMA as it was in 07a scores RMSE 4.445 and MAE 3.522. No SARIMA tweak passed on validation, so the SARIMA
contender is the 07a model, and the best model beats it by nothing. The winner, chosen by the rule written down
before the table was seen (lowest scored median RMSE), is SARIMA; `09` fits it on all 726 products and 10 and 11 use
it for the stock work.

Paired Wilcoxon tests against the 07a SARIMA, Holm-corrected within each metric (`ts_tests.csv`): the combination is
significantly better on MAE (better for 68% of products) and MAPE, but not on RMSE (p = 0.079, better for 63%); ETS is
significantly better on MAPE. SARIMAX and AutoTS are significantly worse on RMSE, Prophet on MAE and MAPE, and every
building block on RMSE.

## How the best model is decided

`07i_model_verdict` reads the scores 07h saved for the eight contenders and decides which is best. It refits nothing.

1. Six measures, each the median across the 150 products of a product's average over the six scored dates: RMSE,
   MAE, MAPE (days with sales only), sMAPE, WAPE and MASE, for all products and for each category.
2. A rank on each measure, the average rank, and how often each model is first, in the top two and in the top three.
3. Paired tests on RMSE, MAE and MASE: for every pair of models, a Wilcoxon test on the 150 per-product differences,
   Holm-corrected within each reference model and measure, and a bootstrap 95% interval for the median difference. A
   gap is real only when the interval leaves out zero and the Holm p-value is below 0.05. Otherwise it is a tie.
4. Product wins: for each measure, how many products each model forecasts best.
5. Stock: every contender goes through 11e's simulation on the same products, with the textbook safety stock and
   with safety stock sized from past errors. SARIMA and seasonal naive must match 11e to the cent.

The rule for the pick: RMSE decides, because it was fixed before the run as the measure for choosing the stock model.
Every model whose RMSE gap to the leader is a tie shares the lead. A shared lead is broken on the average rank, then
on stock cost. The verdict sentences are written by `src/model_verdict.py` from the saved tables, so the notebook and
the dashboard (Forecast accuracy page, "Which model is best") always say the same thing.

Limits: one evaluation window of six forecast dates, 150 products, and settings tuned on earlier validation dates
only. Models this close can swap places in another window.

## Model by model

**Seasonal naive** (`07a`). Assumes next week repeats last week. It suits these products because the weekly rhythm
is their strongest pattern (03), and it is the floor every other model has to clear. No settings.

**Damped Holt-Winters, ETS** (`07a`, tuned in `07d`). Assumes a slowly moving level, a trend that flattens out, and
a fixed weekly pattern added on top. Damping stops a short trend running on for 28 days. Tried: log1p, Box-Cox,
clipping. Kept: Box-Cox (validation RMSE 4.668 to 4.606, MAE 3.720 to 3.682). Best MAPE of all models.

**AR, MA, ARMA** (`07c`). Assume sales move around a fixed mean, with today depending on the last p days (AR), the
last q surprises (MA) or both. They have no weekly term, so they show what SARIMA's seasonal part adds. Orders from 0
to 7 by AICc at the warm-up origin; most products took lag 7 (AR(7) and MA(7) for 115 of 150), which is the weekly
pattern again. ARMA does surprisingly well (4.542) because its lag-7 terms keep the weekly pattern; on the 33 products
that test as stationary it matches SARIMA (4.414 against 4.444). 94% of the long ARMA fits stopped before the
optimiser converged.

**ARIMA** (`07a`, `07c`). ARMA after differencing, so the level can wander; no weekly term. Worse than ARMA (4.822),
because differencing throws away the fixed mean that most of these products have over 28 days.

**SARIMA** (`07a`, tuned in `07d`). Assumes that after differencing, today depends linearly on the last few days and
on the same weekday a week ago, with stable coefficients and constant error variance. The ACF spikes at lags 7 and 14
(05, 07c) are why it suits. Orders from 07a (d from KPSS, D from OCSB, AICc). Tried one at a time: log1p (val 4.775),
Box-Cox (4.646, MAE unchanged), clipping extreme days (4.700), yearly Fourier pairs K = 1 (4.600, kept), K = 2 (4.610)
and K = 3 (4.662), a Christmas dummy (4.670), SNAP and event days (4.676), relative price (4.776), and a convergence
fix (4.625). Only Fourier K = 1 passed, and it became SARIMAX.

**SARIMAX** (`07d`). SARIMA errors around a regression whose effects stay fixed over time; here one yearly Fourier
pair. Better on validation (4.600 against 4.697), worse on the scored origins (4.646 against 4.445): the yearly curve
helps in the autumn and pulls the wrong way in winter and spring. The coefficients of the full regressor set are
sensible (SNAP +8.1% for a typical FOODS product, Christmas -77% to -95%, events about -3%), but small next to daily
noise.

**ARIMAX** (`07d`). ARIMA errors around the same regressors. Scored 4.801: the regressors do not make up for the
missing weekly terms.

**VARIMA** (`07e`). Assumes the three category totals move together, so each one's past helps forecast the others.
First differences, weekday dummies, and a small order grid chosen by AICc. Fitted on the totals in thousands of
units, since 07b showed SARIMA breaking down numerically on raw totals. AICc and BIC both chose VARMA(1, 1). Against SARIMA on
each total (same scale), MASE is 0.678 against 0.699 for HOBBIES, 0.891 against 0.928 for HOUSEHOLD and 0.837 against
0.834 for FOODS. Only one Granger link survives the Holm correction, FOODS to HOUSEHOLD, which fits where VARIMA gains
most. It works on the totals only, so it is not a contender for the products.

**Theta** (`07f`). Assumes a series is a long-run straight line plus short-run smoothing, after the weekly pattern is
divided out. Few settings and robust. Clipping did not help. Scored 4.541 and the best MAE of all (3.510).

**Prophet** (`07f`). Assumes sales are a smooth trend plus weekly and yearly curves plus regressor effects, with days
independent given those. Regressors SNAP, event and relative price; the Christmas regressor was kept, clipping and a
third yearly order were not. Wins the most single products on RMSE (49 of 150) but has a poor median (4.913): where
it misses, it misses by more.

**AutoTS** (`07f`). Whatever single time-series model wins AutoTS's own backtest on the 150 products at each origin
(models: naive variants, ETS, ARIMA, Theta, UnobservedComponents, DynamicFactor, VAR, VECM; ensembles off). The smoke
test projected 10.8 hours, over the 8-hour limit, so each validation round scored candidates on a random 30 of the
150 products (`subset=30`); the run took 1.3 hours. It picked ETS at four origins, ARIMA at two and three other models
once each, and scored 4.848.

**Combination** (`07h`). Assumes the members' errors partly cancel. Members are the three contenders with the lowest
validation RMSE: SARIMAX (4.600), ETS (4.606) and Theta (4.691). Plain average 4.523; inverse-error weights from the
validation origins 4.520.

**tsfresh profiles and routing** (removed). Assumed that products with a similar demand shape want the same model.
730 features of each mean-scaled series, PCA to 90% of the variance, k-means with k from 2 to 6 by silhouette. The
silhouette chose two groups (0.813), but one group was a single sparse product; three to six groups scored 0.06 to
0.12. The products did not form distinct profiles, so routing sent everything to the validation winner, SARIMAX, and
its forecasts were SARIMAX's exactly. A copy of another contender adds nothing, so its notebook was removed and 07h
was run again without it (`docs/PREREGISTRATION_TS.md`, change 5).

## What helped and what did not

Helped on validation and kept: Box-Cox for ETS; one yearly Fourier pair for SARIMA (as SARIMAX); a Christmas regressor
for Prophet. Of these, only Prophet's Christmas regressor also helped on the scored origins.

Did not help: log1p and Box-Cox for SARIMA, clipping for every model, two or three Fourier pairs, SNAP and event days,
relative price, the convergence fix, Prophet's third yearly order, AutoTS's search, routing by profile (removed).

The combination is the most consistent improvement product by product, but not on the median RMSE.

## Why machine-learning models are not here

LightGBM, XGBoost and a small GRU learn one model across many products from engineered features. They are machine
learning, not time-series models, and were removed from the project. The winner above takes LightGBM's place in the
stock notebooks.
