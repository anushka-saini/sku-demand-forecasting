# Interim findings (Milestone 2, mid-sem)

Classical time-series models on the Walmart M5 California data, with the same backtest as Milestone 1: 7 forecast
origins 28 days apart, the first a warm-up, origins 1-6 scored (test days 2015-12-07 to 2016-05-22). Item-level
results use a 150-item sample (50 per category; 75 with a detected break). Hypotheses were written down before any
model was scored (`docs/PREREGISTRATION_M2.md`). Every number below comes from `outputs/m2/tables/` and is also in
`outputs/results_facts.json` (key `m2`).

## 1. Preprocessing and stationarity (notebook 12)
* **Data quality:** no empty sales cells and no missing days inside any series after Milestone 1's trim of pre-launch zeros (119,156 leading zero days removed across the 726 items). 8.1% of the kept days are real zeros; 0.2% of days are flagged as outliers (above Q3 + 3 IQR) and kept, because spikes are real demand.
* **ADF and KPSS together** label only **16.9%** of the 726 items stationary, 8.4% trend-stationary, 15.7% difference-stationary and **59.0% conflicting**. In 92% of the conflicting items ADF rejects a unit root *and* KPSS rejects a stable level: the signature of a series that is stationary around a level that shifts now and then. This is the behaviour Milestone 1's break detector looks for. (Milestone 1's ADF-only test called 85% of candidates stationary; the two results are consistent, ADF alone just cannot see level shifts.)
* **Differencing:** `ndiffs` recommends d = 1 for 94% of items; `nsdiffs` recommends D = 0 for every item. All four totals are difference-stationary.
* **Decomposition:** additive by the swing rule for FOODS, HOBBIES and the CA total; multiplicative for HOUSEHOLD and 4 of 6 example items. The residual spread barely differs between the forms (CA total 0.066 additive vs 0.065 multiplicative), so the choice matters little here. Multiplicative needs positive data: items use sales + 1, totals replace 25 December (stores closed) for decomposition only.

## 2. Classical models vs the baselines (notebook 13)
Median item MASE (lower is better; below 1 beats repeating last week):

| Model | MASE | RMSE (units) | MAE (units) | MAPE* |
|---|---|---|---|---|
| Seasonal naive | 0.973 | 6.04 | 4.72 | 0.753 |
| ARIMA | 0.794 | 4.82 | 3.78 | 0.638 |
| Holt-Winters multiplicative | 0.761 | 4.55 | 3.59 | 0.595 |
| Holt-Winters additive | 0.757 | 4.56 | 3.56 | 0.595 |
| Damped Holt-Winters (ETS, Milestone 1) | 0.756 | 4.50 | 3.51 | 0.595 |
| SARIMA | 0.755 | 4.45 | 3.52 | 0.603 |
| LightGBM (Milestone 1) | **0.747** | 4.48 | 3.55 | **0.574** |

\* MAPE uses only days with sales; 5.4% of days are left out.

* **H1 supported.** SARIMA − seasonal naive: −0.223 MASE (95% CI −0.237 to −0.200); Holt-Winters additive − naive: −0.218 (−0.233 to −0.207); both Holm p < 0.001. Both beat naive on about 97% of items.
* **H2 supported.** LightGBM has the lowest median MASE and is significantly better than all five others after Holm. Its edge over SARIMA is small: −0.007 (−0.014 to −0.002), p = 0.036.
* **The weekly season matters:** ARIMA without it is clearly worse (0.794) than SARIMA.
* **H4 partly supported.** On zero-heavy items additive Holt-Winters is *not* better than multiplicative (0.763 vs 0.762): H4a not supported, though these high-volume items have few zeros (median 6% of days). On the totals multiplicative is within 5% of additive everywhere: H4b supported.
* **Totals:** LightGBM summed over items is the best model on three of the four totals (Holt-Winters additive wins HOBBIES). SARIMA fitted directly on the totals is unreliable: several fits diverged quietly (FOODS MASE 1.976; 5 of 28 fits fell back to naive). A post-hoc check (notebook 14) shows the problem is numerical, from the large values: in thousands of units FOODS gives 0.834.
* **Run statistics:** no item fit raised an error (0% fallback); 30% of item SARIMA fits gave a convergence warning, with the forecast kept. 1-8 of 1,050 item forecast windows per model were degenerate (all zero).

## 3. Assumptions and parameters (notebook 14)
* SARIMA leaves white-noise residuals (Ljung-Box, lags 7 and 14) for **82%** of items; Holt-Winters for 47-49%, ARIMA and damped Holt-Winters for 29%.
* Normality (Jarque-Bera) holds for at most **0.7%** of items: daily sales are skewed counts. That does not hurt point forecasts but explains why the textbook safety-stock formula under-delivers.
* Every item ARIMA/SARIMA fit is stationary and invertible.
* Typical SARIMA: d = 1, D = 0, seasonal (1, 0, 1)₇ with a seasonal AR near 1 (median 0.996) and seasonal MA near −1, i.e. close to a seasonal difference. Holt-Winters uses small level smoothing (about 0.07-0.10) and a fixed weekly pattern.

## 4. Does a shift-aware classical model help? (notebook 15)
* **H3 supported:** neither variant meets the pre-registered "helps" rule.
* Restarting SARIMA from the latest detected break makes it **worse** where it differs from static: +0.0062 MASE (95% CI +0.0015 to +0.0107, Holm p = 0.002; 47 items). A level-shift dummy: +0.0010 (−0.0003 to +0.0058).
* 44% of break item-origins have a break older than the 730-day window, so the variant equals the static model; counted as 0 in the pre-registered test (median change 0.0000 for both variants).
* The placebo (random cut, items without a break) is also slightly worse (+0.0025). The classical family gives the same answer as LightGBM in Milestone 1.

## 5. What it means for stock (notebook 16; 95% target, 7-day lead time, assumed costs)
| Model | Calibrated safety stock: total cost | fill rate | Textbook formula: total cost | cycle service level |
|---|---|---|---|---|
| SARIMA | **$7,857** | 99.2% | $10,631 | 85.8% |
| LightGBM | $8,089 | 99.1% | $11,972 | 84.5% |
| Seasonal naive | $8,974 | 98.9% | $11,437 | 86.7% |
| Damped Holt-Winters (ETS) | $9,143 | 98.8% | $12,354 | 83.8% |
| Holt-Winters additive | $9,312 | 98.8% | $12,425 | 82.9% |

* With calibrated (conformal) safety stock SARIMA is cheapest; vs ETS −14.1% (bootstrap CI −$2,878 to −$70), vs LightGBM −2.9% (−$1,433 to +$804, not distinguishable).
* The textbook formula costs 27-48% more for every model and misses the 95% target (cycle service level 83-87%).
* On this sample seasonal naive is cheaper than LightGBM under the textbook formula, unlike the 726-item Milestone 1 result: totals over 150 items are dominated by a few items, so rankings of close models are not stable.

## Conclusions for retail stock planning
1. **Classical models are a strong, explainable baseline.** SARIMA and Holt-Winters cut error by about a quarter versus repeating last week, and SARIMA comes within 0.007 MASE of LightGBM.
2. **The weekly season is a key signal**: ARIMA without it is much worse, and lag-7 autocorrelation is clear in every series (0.22-0.48 for the six example items, 0.79 for the CA total, last two years).
3. **Reacting to detected breaks does not help** classical models either. Re-estimating parameters at every origin already follows the level.
4. **How safety stock is sized matters more than which good model is used:** calibrated safety stock lowers cost by a quarter or more for every model.
5. **Fit models at item level.** SARIMA on large totals is numerically fragile; summing good item forecasts works better.

## What did not work, and caveats
* SARIMA on totals in raw units: divergent fits, and results that differ between a joblib worker and the main process (notebooks 13, 14).
* Multiplicative Holt-Winters on items needs a +1 shift, so H4a is a weak test on these high-volume items.
* 150-item sample, California only; classical models use the last 730 days while the reused models use all history; costs are assumptions.
* Milestone 3 groundwork (XGBoost, Prophet, GRU) was not started.
