# Mid-sem demo script (5 minutes)

**Before you start:** `python -m streamlit run dashboard/app.py` and open http://localhost:8501. Keep
`notebooks/13_classical_models.ipynb` open in a second tab in case a question needs code. Everything shown is read
from saved results, so the demo never waits on a model fit.

| Time | Where | What to show | What to say |
|---|---|---|---|
| 0:00-0:30 | **Overview** | the Milestone 2 cards | "Milestone 2 asks whether classical time-series models are good enough for retail stock planning, and whether they gain from reacting to demand shifts. Both answers are computed from the result files." |
| 0:30-1:15 | **M2: Stationarity & decomposition** | KPI row, stacked bar, decomposition selector on `CA_TOTAL` | "ADF and KPSS together: only about one item in six is cleanly stationary; most are 'conflicting', which is what a series looks like when its level shifts now and then. One difference is recommended for 94% of items, no seasonal difference. The weekly pattern is clear; the additive/multiplicative rule depends on whether the weekly swing grows with the level." |
| 1:15-3:30 | **M2: Mid-sem demo walkthrough** | step 1 → 7 on the default item, then switch to one more item | 1 "Raw sales: the zeros before the first sale are not demand." 2 "Trimmed series is what models see." 3 "STL: trend, weekly season, residual." 4 "Stationarity verdict and the SARIMA order chosen by AICc at the warm-up origin only, so nothing leaks." 5 "Each model forecasts 28 days from an origin, using only the past; move the slider." 6 "RMSE, MAE, MAPE and MASE for this item." 7 "Same forecasts in the stock simulation: textbook vs calibrated safety stock." |
| 3:30-4:30 | **M2: Classical models** | metrics table, test table, then the H3 chart at the bottom | "Across 150 items SARIMA and Holt-Winters cut error by about a quarter versus repeating last week, and SARIMA is within 0.007 of LightGBM. Pre-registered paired tests, Holm-corrected. Restarting SARIMA after a detected break makes it worse, the same result as LightGBM in Milestone 1." |
| 4:30-5:00 | back to **Overview** | – | "For stock planning: a good classical model plus safety stock sized from real forecast errors gets most of the benefit. What did not work: SARIMA on large totals is numerically fragile, so we fit at item level." |

## Likely questions
* **Why MASE as the headline?** Scale-free, works with zero days (MAPE does not), and it is the Milestone 1 metric, so results stay comparable. RMSE, MAE, MAPE are on the same page.
* **How were the orders chosen?** d by KPSS (`ndiffs`), D by OCSB (`nsdiffs`), then the lowest AICc over p, q ≤ 2 and P, Q ≤ 1 on data up to the warm-up origin. The order is fixed; parameters are re-estimated at each origin.
* **Why only 150 items?** Runtime: about 40 minutes on 12 cores for the full order search. The sample is stratified by category and by break status, seed 42.
* **Are the assumptions met?** SARIMA residuals pass Ljung-Box for 82% of items; normality fails almost everywhere (skewed counts), which affects intervals and the textbook safety stock, not point forecasts.
* **Is anything lookahead?** No: fits use data up to each origin only; a unit test replaces all later data with noise and checks the forecast is unchanged.
