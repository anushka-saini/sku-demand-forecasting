# Dashboard layout spec

Canvas 16:9 (1280 × 720). Every page has the same left slicer panel:
**Category**, **Department**, **Segment (AX / AY)**, **Break group** (`pbi_dim_item[break_group]`).
Each visual below is listed with the planner question it answers. The question goes in the visual's subtitle.

---

## Page 1: Risk & Segmentation

Page filter: none (the ABC-XYZ matrix needs all items). The other visuals filter `in_scope = TRUE`.

| # | Visual | Fields | Planner question |
|---|---|---|---|
| 1.1 | KPI cards ×4 | `Items In Scope`, `Items With Break`, `Break Rate %`, `Median Shift (sd)` | How many of my important, forecastable items have changed behaviour? |
| 1.2 | Matrix (heat-map formatting) | Rows `abc_class`, columns `xyz_class`, values `Items` and `Share Of Units %` | Where are my items and my volume concentrated by value and variability? |
| 1.3 | Clustered bar | Axis `category`, values `Break Rate %`, legend `first_break_direction` (+1 up / −1 down) | Which category's demand is shifting most, and is it shifting up or down? |
| 1.4 | Table | `item_id`, `segment`, `mean_sales`, `cv`, `last_break_week`, `n_breaks_at_last_origin`, `pelt_n_changepoints`; conditional icon on `break_at_last_origin` | Which specific items should I review first? |
| 1.5 | Line chart (drill-through target) | Axis `pbi_fact_sales_weekly[week_start]`, value `Weekly Units`; constant line at `Last Break Week` | What did this item's demand look like before and after the detected break? |

Text box: "Breaks are flagged by CUSUM against each item's first year of sales. About half of the alarms reflect gradual drift rather than abrupt steps (see notebook 01). Only a small share line up with price changes, and none line up with holidays more than chance."

---

## Page 2: Forecast Comparison

Page filter: `pbi_dim_item[in_scope] = TRUE`. Slicer: **Model** (`pbi_dim_policy[model]`), **Origin** (`pbi_dim_date[forecast_origin]`).

| # | Visual | Fields | Planner question |
|---|---|---|---|
| 2.1 | KPI cards ×4 | `MASE Static`, `MASE Adaptive`, `MASE Change Adaptive vs Static`, `% Items Better With Adaptive` | Does retraining after a break make forecasts more accurate? |
| 2.2 | Clustered column | Axis `pbi_dim_policy[model]`, legend `variant`, value `Median MASE` | Which model should we trust, and does it beat the seasonal-naive baseline (MASE < 1)? |
| 2.3 | Column chart | Axis `category`, values `MASE Static`, `MASE Adaptive` | Is the answer different for Foods, Hobbies or Household? |
| 2.4 | Line + band (item drill-down) | Axis `pbi_dim_date[date]`, lines `Actual Units` and `Forecast Units` (legend `variant`), error band `Band Low 90` to `Band High 90` (Line chart → error bars, or Deneb) | For this item, how far off were static vs adaptive forecasts, and how uncertain? |
| 2.5 | Card | `Band Coverage %` | Can I trust the 90% band? (It should be close to 90%.) |
| 2.6 | Scatter | X `MASE Static`, Y `MASE Adaptive`, details `item_id`, colour `category`, plus a y = x reference line | Which break items gained or lost from adapting? |

---

## Page 3: Inventory & Cost Trade-off

Slicers: **Service level** (`ServiceLevel[service_level]`, single select, default 0.95), **Lead time** (`LeadTime[lead_time]`, default 7), **Policy** (`pbi_dim_policy[policy]` filtered to `used_in_inventory = TRUE`; single select for cards 3.1/3.4/3.5), **Holding Rate** and **Stockout Multiple** what-if sliders. Without a single policy selected, `Total Cost` sums all three simulated policies, so set the card visuals to require one.
Banner text: "Lead time and cost parameters are assumptions, not data. Move the sliders to test them."

| # | Visual | Fields | Planner question |
|---|---|---|---|
| 3.1 | KPI cards ×5 | `Fill Rate`, `Stockout Days`, `Avg Inventory Value`, `Total Cost`, `Cost Change Adaptive vs Static %` | At my chosen service level, what do I get and what does it cost? |
| 3.2 | Line chart | X `ServiceLevel[service_level]`, Y `Holding Cost` and `Stockout Cost` (legend `pbi_dim_policy[policy]`) | Where is the balance between holding stock and losing sales? |
| 3.3 | Scatter | X `Avg Inventory Value`, Y `Fill Rate`, legend `policy`, details `service_level` | How much extra inventory buys each point of fill rate? |
| 3.4 | Table | `item_id`, `Reorder Point (last day)`, `Safety Stock (avg)`, `Fill Rate`, `Stockout Days`, `Stockout Risk Flag` (red icon), sorted by flag then `Lost Units` | Which items need a higher reorder point right now? |
| 3.5 | Line chart (item drill-down) | Axis `date`, `On Hand Units`, `Daily ROP`, `Daily Demand`; columns `Daily Lost` | How did stock evolve for this item, and when did it run out? |
| 3.6 | Matrix | Rows `stockout_multiple`, columns `holding_rate`, values: `total_cost` from `pbi_fact_cost_sensitivity` at the selected service level and lead time, for static vs adaptive | Does the conclusion change if my cost assumptions are wrong? |

---

## Formatting conventions
* Static = blue, adaptive = orange, seasonal naive = grey. Use these on every page.
* Show MASE to 3 decimals and costs in $ with no decimals.
* Every page footer: "Data: M5 (Walmart), CA stores, 2011-01-29 to 2016-05-22. Evaluation window 2015-12-07 to 2016-05-22."
