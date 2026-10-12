# Star schema and relationships

All tables are produced by `notebooks/04_powerbi_export.ipynb` in `outputs/tables/`.
Load each CSV with **Get Data → Text/CSV**. In Power Query set the types: dates as *Date*, `has_break` / `in_scope` / `is_*` as *True/False*, `service_level` as *Decimal*.

```
                     pbi_dim_date (date)
                           │1
            ┌──────────────┼───────────────────┐
            *              *                   │
 pbi_fact_forecast_daily   pbi_fact_inventory_daily
            *  *                *  *  *
            │  └──────┐  ┌──────┘  │  └─────────────┐
            │1        │1 │1        │                │
       pbi_dim_item ──┘ pbi_dim_policy        ServiceLevel (calc. table)
        │1 │1 │1 │1        │1  │1                  │1      │1
        *  *  *  *         *   *                   *       *
  fact_sales_weekly        fact_forecast_error   fact_inventory_kpi
  fact_breaks              fact_inventory_kpi    fact_cost_sensitivity
  fact_forecast_error      fact_cost_sensitivity
  fact_inventory_kpi
```

## Relationships (all single-direction, dimension filters fact)

| From (one side) | To (many side) | Cardinality |
|---|---|---|
| `pbi_dim_item[item_id]` | `pbi_fact_sales_weekly[item_id]` | 1 : * |
| `pbi_dim_item[item_id]` | `pbi_fact_breaks[item_id]` | 1 : * |
| `pbi_dim_item[item_id]` | `pbi_fact_forecast_daily[item_id]` | 1 : * |
| `pbi_dim_item[item_id]` | `pbi_fact_forecast_error[item_id]` | 1 : * |
| `pbi_dim_item[item_id]` | `pbi_fact_inventory_kpi[item_id]` | 1 : * |
| `pbi_dim_item[item_id]` | `pbi_fact_inventory_daily[item_id]` | 1 : * |
| `pbi_dim_date[date]` | `pbi_fact_forecast_daily[date]` | 1 : * |
| `pbi_dim_date[date]` | `pbi_fact_inventory_daily[date]` | 1 : * |
| `pbi_dim_policy[policy]` | `pbi_fact_forecast_daily[policy]` | 1 : * |
| `pbi_dim_policy[policy]` | `pbi_fact_forecast_error[policy]` | 1 : * |
| `pbi_dim_policy[policy]` | `pbi_fact_inventory_kpi[policy]` | 1 : * |
| `pbi_dim_policy[policy]` | `pbi_fact_inventory_daily[policy]` | 1 : * |
| `pbi_dim_policy[policy]` | `pbi_fact_cost_sensitivity[policy]` | 1 : * |
| `ServiceLevel[service_level]` | `pbi_fact_inventory_kpi[service_level]` | 1 : * |
| `ServiceLevel[service_level]` | `pbi_fact_inventory_daily[service_level]` | 1 : * |
| `ServiceLevel[service_level]` | `pbi_fact_cost_sensitivity[service_level]` | 1 : * |

`pbi_fact_sales_weekly` and `pbi_fact_breaks` are at weekly / origin grain, so they are **not** related to `pbi_dim_date`. Use their own `week_start` / `origin` columns on the axis.

`pbi_fact_cost_sensitivity` is aggregated over all items, so it has no item relationship. Item slicers do not filter it, and the visual using it should say so in its subtitle.

## Calculated and parameter tables (create in Power BI)

```DAX
ServiceLevel = DISTINCT ( pbi_fact_inventory_kpi[service_level] )

LeadTime = DISTINCT ( pbi_fact_inventory_kpi[lead_time] )   -- disconnected; used by measures

-- Modeling > New parameter > Numeric range (what-if). These are ASSUMPTIONS, not data.
Holding Rate      = GENERATESERIES ( 0.10, 0.40, 0.05 )     -- default 0.25
Stockout Multiple = GENERATESERIES ( 0.25, 2.00, 0.25 )     -- default 0.50
```

Mark `pbi_dim_date` as the date table (Table tools → Mark as date table → `date`).
