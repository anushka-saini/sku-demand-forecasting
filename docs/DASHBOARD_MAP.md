# Dashboard map: old pages to new pages

The dashboard moves from 17 pages, one per notebook topic, to 8 pages built for a retail planning manager. Nothing
is lost. Content that still helps a manager moves into one of the new pages; reference material (the glossary, long
method notes, data-cleaning detail) moves into `docs/`. The old page files are kept, unchanged, in
`dashboard/archive/` so their code can still be read.

## New structure

| New page | File | What it answers |
|---|---|---|
| Overview | `views/overview.py` | What is the answer, in one screen? |
| Data and scope | `views/data_scope.py` | What data is this, and which items did we study? |
| Demand patterns | `views/demand_patterns.py` | How do sales move through the week, the year, events and price changes? |
| Forecast accuracy | `views/accuracy.py` | Which forecasting method is most accurate, and where do simple methods win? |
| Item explorer | `views/item_explorer.py` | How does the forecast look for one item I care about? |
| Demand shifts | `views/demand_shifts.py` | How often does demand shift for good, and does reacting to it help? |
| Stock impact | `views/stock_impact.py` | What do the forecasts mean for stock, cost and fill rate? |
| Method and limits | `views/method_limits.py` | How was this tested, and what does it not prove? |

## Where each old page goes

| Old page (file) | What it showed | New home |
|---|---|---|
| Overview (`overview.py`) | verdict cards, model table, key numbers | Overview: KPI cards, answer sentences, model chart. Verdict rules move to a collapsed box on Method and limits. |
| Data (`data_method.py`) | file snapshots, pipeline, scope metrics, category table | Data and scope: scope cards, sales chart. Pipeline steps move to Method and limits. File snapshots stay in `outputs/figures/` and `docs/DATA.md`. |
| Explore (`explore.py`) | weekday and month index, price correlation, SNAP and events, data quality | Demand patterns (weekday, month, events, price). Data quality moves to a collapsed box on Data and scope. |
| Zeros and trimming (`zeros.py`) | one item before and after trimming, counts | Data and scope: a short before/after chart and two numbers. The decomposition images stay in `outputs/figures/`. |
| Stationarity (`stationarity.py`) | ADF/KPSS labels, one item's tests, totals | Demand patterns: one collapsed box with the label shares. Per-item tests stay in `outputs/tables/stationarity_items.csv`. |
| Decomposition (`decomposition.py`) | STL for one series, additive vs multiplicative, yearly figure | Demand patterns: trend and weekly pattern for a chosen total or item. The additive/multiplicative table moves to `docs/MODEL_CHOICES.md` (already there). |
| Classical models (`classical.py`) | metrics by category, tests, totals, one item's forecasts, residual checks | Forecast accuracy (leaderboard, groups, totals) and Item explorer (one item). Residual checks move to a collapsed box on Method and limits. |
| Breaks (`breaks.py`) | break shares, timing, false alarms, power, PELT, funnel, events, shift-aware SARIMA | Demand shifts. PELT agreement and event alignment go into a collapsed box there. |
| ML and comparison (`comparison.py`) | box plot, overview table, winners, LightGBM vs ETS, bias | Forecast accuracy. |
| Does retraining help? (`retraining.py`) | four LightGBM variants, placebo, selection check | Demand shifts ("Did reacting help?"). |
| Forecast explorer (`forecasts.py`) | one item's LightGBM/ETS forecasts and breaks | Item explorer, extended with every model and the item's rank. |
| Uncertainty and stock (`inventory.py`) | band coverage, simulation, calibrated stock, seasons, newsvendor | Stock impact (simulation, cost controls, calibrated safety stock, seasons). Band coverage appears there and on Item explorer. |
| Planner (`planner.py`) | items below target with reasons, CSV | Stock impact: "Items to check" table with a CSV download. |
| One product, start to finish (`demo.py`) | one item from raw data to stock | Item explorer covers the forecast part; the raw-to-trimmed step is on Data and scope. |
| Segmentation (`segmentation.py`) | ABC-XYZ heat map and drill-down | Data and scope: the ABC-XYZ grid in a collapsed box. |
| Texas and Wisconsin (`replication.py`) | 14 checks, scope, breaks | Method and limits: one line in the text plus a collapsed table. |
| Assumptions and limitations (`limitations.py`) | scope, assumed parameters, what failed, how to rerun, glossary | Method and limits. The glossary moves to `docs/GLOSSARY.md`; terms are explained where they appear. |

## Filters

The sidebar holds the only filters, and they keep their values when you change page:

| Filter | Values | Used on |
|---|---|---|
| Category | FOODS, HOBBIES, HOUSEHOLD | every page that shows items |
| Item group | AX (steady), AY (more variable) | Data and scope, Item explorer, Demand shifts, Stock impact |
| Shift group | all items, items with a detected shift, items without | Forecast accuracy, Item explorer, Stock impact |
| Forecast dates | the scored forecast dates (time window) | Forecast accuracy, Item explorer |

There is no store-group filter: the saved outputs add the four California stores into one series per item, so no
store-level results exist.
