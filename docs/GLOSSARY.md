# Glossary

The dashboard explains each term once, next to where it appears. This page collects them in one place. It replaces the
glossary that used to sit in the dashboard's sidebar.

| Term | Meaning |
|---|---|
| ABC-XYZ | ABC ranks items by units sold (A = the top 70% of a category's units, B the next 20%, C the rest). XYZ ranks them by how much daily sales vary (X steady, Y variable, Z erratic). We model the AX and AY items: high volume and fairly predictable. |
| Backtest | Standing at past dates, forecasting forward with only the data available then, and scoring the forecast against what really sold. |
| Bias | Total forecast divided by total sales, minus 1. Negative means the forecast was too low. |
| CUSUM | The shift detector: a running total of how far each week's sales sit above or below a baseline. When the total drifts past a threshold, it raises an alarm. |
| PELT | A second method that finds the set of change points that best fits the whole series. Used only as a cross-check, because it looks at the future. |
| Structural break, shift | A lasting change in an item's normal sales level, as opposed to normal week-to-week noise or a one-day spike. |
| High-confidence shift | A shift confirmed by PELT within 8 weeks, at least 20% in size, and with at least half of it still there a year later. |
| Static and retrained | Static models train on all history. Retrained (adaptive) models train only on data after the latest detected shift. |
| Placebo | A control: items without a shift get a random, equally short training window, to check whether any gain comes from the shift or just from using recent data. |
| MASE | Mean absolute scaled error: average forecast error divided by the error of repeating last week, measured on the item's own history. Below 1 beats that simple rule. Lower is better. |
| MAE, RMSE | Average size of the miss in units; RMSE counts large misses more. |
| MAPE, sMAPE, WAPE | Percentage errors. MAPE skips days with no sales; sMAPE treats over- and under-forecasting alike; WAPE is the total miss divided by total sales. |
| Seasonal naive | The baseline forecast: next week's sales equal last week's. |
| ETS, Holt-Winters | Exponential smoothing: a level, a trend and a weekly pattern, each updated as new sales arrive. ETS here has a trend that fades out. |
| ARIMA, SARIMA | Models built on how today's sales relate to recent days' sales and errors. SARIMA adds weekly seasonal terms. |
| TSB | A method for items that often sell nothing: it forecasts how often an item sells and how much when it does. |
| LightGBM | A machine-learning model of many small decision trees, trained on all items at once with sales history, price and calendar inputs. |
| Prediction interval, band | A range the actual sales should fall in a stated share of the time (here 90%). |
| Conformal band | An interval built from pooled past errors with a small-sample correction, so it covers close to its stated share. |
| Service level | The target chance of not running out during one order cycle (90, 95 or 99%). It sets how much safety stock is held. |
| Cycle service level | The share of order cycles that actually got through without a stockout, measured in the simulation. |
| Fill rate | The share of customer demand served from stock, measured in the simulation. |
| Safety stock | Extra units held to cover forecast error while waiting for a delivery. |
| Reorder point | The stock level that triggers a new order: expected demand over the lead time plus safety stock. |
| Lead time | Days between placing an order and receiving it. Assumed in the simulation. |
| SNAP | The US food-assistance programme. In California, benefits can be spent on the 1st to 10th of each month. |
| Stationary | A series that keeps coming back to one level, with a stable spread. |
