# Data dictionary

All files are in `outputs/tables/` and produced by `notebooks/04_powerbi_export.ipynb`.
Units are item-level daily (or weekly) sales summed over the 4 CA stores. Money is in USD at CA median sell price.

## pbi_dim_item: one row per CA item (3,049)
| column | type | description |
|---|---|---|
| item_id | text | M5 item id (key) |
| category / dept_id | text | FOODS / HOBBIES / HOUSEHOLD; department e.g. FOODS_3 |
| abc_class | text | A/B/C by cumulative share of units within category (70% / 90% cut-offs) |
| xyz_class | text | X/Y/Z by coefficient of variation of daily sales (cut-offs 0.5 / 1.0) |
| segment | text | abc_class + xyz_class, e.g. AX |
| mean_sales, cv, total_sales | number | daily mean, CV and total units after leading zeros were trimmed |
| in_scope | bool | TRUE for the 726 AX/AY items that were modelled |
| cusum_first_break | bool | full-series CUSUM (notebook 01) raised an alarm |
| first_break_week | date | estimated change week of that first break |
| first_break_direction | int | +1 demand up, −1 demand down |
| first_break_shift_sd | number | size of the shift in reference-window standard deviations |
| pelt_n_changepoints | int | number of PELT changepoints after burn-in (cross-check) |
| break_at_last_origin | bool | online CUSUM had found ≥1 break before the last forecast origin (2016-04-24) |
| n_breaks_at_last_origin | int | number of breaks found by then |
| last_break_week | date | most recent estimated change week |
| intermittent | bool | ADI > 1.32 before the first origin (TSB and empirical safety stock used) |
| break_group | text | "break detected" / "no break" / "out of scope" (slicer) |

## pbi_dim_date: one row per day, 2011-01-29 to 2016-06-19
| column | description |
|---|---|
| date | key |
| wm_yr_wk, week_start | Walmart week id and its Saturday start date |
| weekday, month, year | calendar parts |
| event_name_1, event_type_1, is_event | M5 calendar event (first event only) |
| snap_CA | 1 on CA SNAP days (always days 1–10 of the month) |
| is_eval_period | TRUE for 2015-12-07 to 2016-05-22 |
| forecast_origin | origin whose 28-day window contains this date |

## pbi_dim_policy: one row per model × variant
| column | description |
|---|---|
| policy | key, e.g. `best_adaptive` |
| model | seasonal_naive, ets, best (the best time-series model from 07h, SARIMA), tsb |
| variant | static (all history), adaptive (retrained from the last detected break), placebo (random truncation of no-break items, control) |
| used_in_inventory | TRUE for the policies simulated in notebook 03 |

## pbi_fact_sales_weekly: item × week (in-scope items, full Saturday–Friday weeks)
item_id, week_start, units.

## pbi_fact_breaks: item × forecast origin
| column | description |
|---|---|
| origin | last date of training data |
| has_break, n_breaks | online CUSUM result using data up to the origin only |
| last_change_week, last_alarm_week | estimated change week and the week the alarm fired |
| adaptive_window_days | training-window length used by the adaptive variant |
| is_eval_origin | FALSE for the warm-up origin 2015-11-08 |

## pbi_fact_forecast_daily: item × day × model × variant (eval origins; placebo excluded)
| column | description |
|---|---|
| date, origin | forecast target day and the origin the forecast was made at |
| actual, forecast | units sold / forecast units |
| lo_90, hi_90 | empirical 90% band: forecast + 5th/95th percentile of that item's errors at earlier origins |
| abs_error | abs(actual − forecast) |

## pbi_fact_forecast_error: item × origin × model × variant
| column | description |
|---|---|
| mase | MAE over the 28 days ÷ in-sample seasonal-naive (lag 7) MAE. Below 1 beats seasonal naive in-sample |
| rmsse | M5-style root mean squared scaled error (lag-1 scale) |
| mae | mean absolute error, units/day |
| mase_scale | denominator of MASE |
| dollar_weight | units × price over the 28 days before the origin (WRMSSE weight) |
| has_break | a break was detected before this origin |

## pbi_fact_inventory_kpi: item × policy × service level × lead time (simulation over 168 eval days)
| column | description |
|---|---|
| service_level | target used to size safety stock: 0.90 / 0.95 / 0.99 |
| lead_time | days from order to arrival (ASSUMPTION: 3 / 7 / 14) |
| demand_units, lost_units | total demand and unmet demand (lost sales) |
| fill_rate | 1 − lost_units / demand_units |
| stockout_days | days with any lost sales |
| avg_inventory_units, avg_inventory_value | mean end-of-day stock, units and $ |
| n_orders | orders placed |
| mean_safety_stock, final_rop | average safety stock and the reorder point on the last day (units) |
| holding_value_days | Σ(price × on-hand) / 365. Multiply by the annual holding rate to get holding cost |
| lost_value | Σ(price × lost units). Multiply by the stockout multiple to get stockout cost |
| holding_cost, stockout_cost, total_cost | at the base assumptions (25% holding, 0.5× price per lost unit) |
| break_item, intermittent, category | convenience copies of item attributes |

## pbi_fact_inventory_daily: item × day × variant × service level (policy model only, lead time 7)
demand, sales, lost, on_hand (end of day), order_qty, rop (reorder point that day), policy.

## pbi_fact_cost_sensitivity: scenario × policy × service level (all in-scope items summed)
lead_time, holding_rate, stockout_multiple (ASSUMPTIONS), policy, service_level, total_cost.
