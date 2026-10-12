# DAX measures

Create a blank table called `_Measures` and paste these in. Table names match `star_schema.md`.
Measures that depend on the cost parameters use the what-if tables `Holding Rate` and `Stockout Multiple`.

## Page 1: Risk & segmentation

```DAX
Items = DISTINCTCOUNT ( pbi_dim_item[item_id] )

Items In Scope = CALCULATE ( [Items], pbi_dim_item[in_scope] = TRUE () )

Items With Break =
CALCULATE ( [Items], pbi_dim_item[in_scope] = TRUE (), pbi_dim_item[break_at_last_origin] = TRUE () )

Break Rate % = DIVIDE ( [Items With Break], [Items In Scope] )

Share Of Units % =
DIVIDE (
    SUM ( pbi_dim_item[total_sales] ),
    CALCULATE ( SUM ( pbi_dim_item[total_sales] ), REMOVEFILTERS ( pbi_dim_item[abc_class], pbi_dim_item[xyz_class], pbi_dim_item[segment] ) )
)

Median Shift (sd) =
CALCULATE ( MEDIAN ( pbi_dim_item[first_break_shift_sd] ), pbi_dim_item[cusum_first_break] = TRUE () )

Weekly Units = SUM ( pbi_fact_sales_weekly[units] )

-- Draw a line on the item timeline at the most recent break week
Last Break Week = SELECTEDVALUE ( pbi_dim_item[last_break_week] )
```

## Page 2: Forecast comparison

```DAX
Actual Units   = SUM ( pbi_fact_forecast_daily[actual] )
Forecast Units = SUM ( pbi_fact_forecast_daily[forecast] )
Band Low 90    = SUM ( pbi_fact_forecast_daily[lo_90] )
Band High 90   = SUM ( pbi_fact_forecast_daily[hi_90] )

-- Median over items of each item's mean MASE across the eval origins
Median MASE =
MEDIANX (
    VALUES ( pbi_dim_item[item_id] ),
    CALCULATE ( AVERAGE ( pbi_fact_forecast_error[mase] ), pbi_fact_forecast_error[is_eval_origin] = TRUE () )
)

MASE Static =
CALCULATE ( [Median MASE], pbi_dim_policy[variant] = "static" )

MASE Adaptive =
CALCULATE ( [Median MASE], pbi_dim_policy[variant] = "adaptive" )

-- Item-level paired difference, then median (matches notebook 02). Negative = adaptive better.
MASE Change Adaptive vs Static =
MEDIANX (
    FILTER ( VALUES ( pbi_dim_item[item_id] ), CALCULATE ( MAX ( pbi_dim_item[break_at_last_origin] ) ) = TRUE () ),
    CALCULATE ( AVERAGE ( pbi_fact_forecast_error[mase] ), pbi_dim_policy[variant] = "adaptive", pbi_fact_forecast_error[is_eval_origin] = TRUE () )
        - CALCULATE ( AVERAGE ( pbi_fact_forecast_error[mase] ), pbi_dim_policy[variant] = "static", pbi_fact_forecast_error[is_eval_origin] = TRUE () )
)

% Items Better With Adaptive =
VAR t =
    ADDCOLUMNS (
        FILTER ( VALUES ( pbi_dim_item[item_id] ), CALCULATE ( MAX ( pbi_dim_item[break_at_last_origin] ) ) = TRUE () ),
        "d",
            CALCULATE ( AVERAGE ( pbi_fact_forecast_error[mase] ), pbi_dim_policy[variant] = "adaptive", pbi_fact_forecast_error[is_eval_origin] = TRUE () )
                - CALCULATE ( AVERAGE ( pbi_fact_forecast_error[mase] ), pbi_dim_policy[variant] = "static", pbi_fact_forecast_error[is_eval_origin] = TRUE () )
    )
RETURN DIVIDE ( COUNTROWS ( FILTER ( t, [d] < 0 ) ), COUNTROWS ( t ) )

Band Coverage % =
DIVIDE (
    COUNTROWS ( FILTER ( pbi_fact_forecast_daily,
        pbi_fact_forecast_daily[actual] >= pbi_fact_forecast_daily[lo_90]
            && pbi_fact_forecast_daily[actual] <= pbi_fact_forecast_daily[hi_90] ) ),
    COUNTROWS ( pbi_fact_forecast_daily )
)

-- Dollar-weighted RMSSE (item level)
WRMSSE =
DIVIDE (
    SUMX ( FILTER ( pbi_fact_forecast_error, pbi_fact_forecast_error[is_eval_origin] ),
           pbi_fact_forecast_error[rmsse] * pbi_fact_forecast_error[dollar_weight] ),
    SUMX ( FILTER ( pbi_fact_forecast_error, pbi_fact_forecast_error[is_eval_origin] ),
           pbi_fact_forecast_error[dollar_weight] )
)
```

## Page 3: Inventory & cost trade-off

```DAX
Selected Lead Time = SELECTEDVALUE ( LeadTime[lead_time], 7 )

-- All KPI measures are evaluated at the selected lead time
Demand Units =
CALCULATE ( SUM ( pbi_fact_inventory_kpi[demand_units] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Lost Units =
CALCULATE ( SUM ( pbi_fact_inventory_kpi[lost_units] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Fill Rate = 1 - DIVIDE ( [Lost Units], [Demand Units] )

Stockout Days =
CALCULATE ( SUM ( pbi_fact_inventory_kpi[stockout_days] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Avg Inventory Value =
CALCULATE ( SUM ( pbi_fact_inventory_kpi[avg_inventory_value] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Reorder Point (last day) =
CALCULATE ( SUM ( pbi_fact_inventory_kpi[final_rop] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Safety Stock (avg) =
CALCULATE ( SUM ( pbi_fact_inventory_kpi[mean_safety_stock] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

-- ASSUMED cost parameters (what-if sliders)
Holding Rate Value      = SELECTEDVALUE ( 'Holding Rate'[Holding Rate], 0.25 )
Stockout Multiple Value = SELECTEDVALUE ( 'Stockout Multiple'[Stockout Multiple], 0.5 )

Holding Cost =
[Holding Rate Value]
    * CALCULATE ( SUM ( pbi_fact_inventory_kpi[holding_value_days] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Stockout Cost =
[Stockout Multiple Value]
    * CALCULATE ( SUM ( pbi_fact_inventory_kpi[lost_value] ), pbi_fact_inventory_kpi[lead_time] = [Selected Lead Time] )

Total Cost = [Holding Cost] + [Stockout Cost]

Total Cost Static   = CALCULATE ( [Total Cost], pbi_dim_policy[variant] = "static", pbi_dim_policy[model] <> "seasonal_naive" )
Total Cost Adaptive = CALCULATE ( [Total Cost], pbi_dim_policy[variant] = "adaptive" )
Cost Change Adaptive vs Static % = DIVIDE ( [Total Cost Adaptive] - [Total Cost Static], [Total Cost Static] )

-- Item flag: simulated fill rate below the target service level
Stockout Risk Flag =
VAR target = SELECTEDVALUE ( ServiceLevel[service_level] )
RETURN IF ( NOT ISBLANK ( target ) && [Fill Rate] < target, 1, 0 )

Items At Risk =
SUMX ( VALUES ( pbi_dim_item[item_id] ), [Stockout Risk Flag] )

-- Daily simulation (lead time 7 only)
On Hand Units  = SUM ( pbi_fact_inventory_daily[on_hand] )
Daily ROP      = SUM ( pbi_fact_inventory_daily[rop] )
Daily Demand   = SUM ( pbi_fact_inventory_daily[demand] )
Daily Lost     = SUM ( pbi_fact_inventory_daily[lost] )
```

Note: `pbi_fact_inventory_daily` contains lead time 7 only. Put "Lead time 7 days" in the subtitle of any visual that uses it.
