# Model coverage

Which of the required time-series models and tools the project already has, where they live, and what is missing.
Written before the new time-series work started, so the "status" column describes the repo as we found it. The
right-hand column says what the new notebooks (07c to 07h) add. Nothing that already exists is redone.

| Model or tool | Where it is now | Status before this work | Gap | Filled by |
|---|---|---|---|---|
| AR(p) | nowhere | missing | no pure autoregressive model fitted | 07c, orders 0-7 by AICc |
| MA(q) | nowhere | missing | no pure moving-average model fitted | 07c, orders 0-7 by AICc |
| ARMA(p, q) | nowhere | missing | no undifferenced ARMA fitted | 07c, p and q 0-7 by AICc |
| ARIMA | 07a (150 products, 7 origins) | done | none; orders p, q 0-2, d from KPSS | reused as is in 07c and 07h |
| SARIMA | 07a (150 products, 7 origins) | done, RMSE 4.445, MAE 3.522 | not tuned beyond the order search; totals diverge in raw units | tuned in 07d |
| Holt-Winters / ETS | 07a (damped ETS on 726, additive and multiplicative HW on 150) | done | no transform or clipping tried | transform and clipping tried in 07d; damped ETS is the contender |
| ARIMAX | nowhere | missing | no ARIMA with calendar or price regressors | 07d, as a building block |
| SARIMAX | 08d (one regressor: a marker for a detected level shift) | partial | no calendar, SNAP, price or Fourier regressors | 07d |
| VARIMA | nowhere | missing | no multivariate model on the category totals | 07e, statsmodels VARMAX |
| Theta | nowhere (the word appears in 07a only in passing) | missing | not fitted | 07f, statsmodels ThetaModel |
| Prophet | 12 (30 products, default settings, no regressors) | trial only | not on the 150 sample, no SNAP, event or price regressors | 07f; notebook 12 stays as it is |
| AutoTS | nowhere | missing | not installed | 07f, time-series models only |
| tsfresh | nowhere | missing | not installed | tried for demand profiles and routing, then removed: every product was routed to SARIMAX (`PREREGISTRATION_TS.md`, change 5) |
| Darts | nowhere | missing | optional | not used unless it installs cleanly (see docs/ENVIRONMENT.md) |

Related checks that already exist and are reused, not repeated:

| Check | Where it is |
|---|---|
| ADF and KPSS stationarity per product | 05 (`outputs/tables/stationarity_items.csv`) |
| ACF and PACF of a few products and the totals | 05 (`acf_pacf_items.png`, `acf_pacf_aggregates.png`) |
| Ljung-Box on SARIMA, ARIMA and HW residuals | 07b (`diagnostics_items.csv`) |
| SARIMA on the totals refitted in thousands (FOODS 1.976 to 0.834) | 07b (`sarima_totals_rescaled.csv`) |
| SNAP and event effects on the totals (FOODS about +10% on SNAP days) | 03 (`explore_calendar_effects.csv`) |

LightGBM, XGBoost and the GRU were removed from the project: they are machine-learning models, not time-series
models. The winner of 07h (SARIMA) takes LightGBM's place in 09 to 11.
