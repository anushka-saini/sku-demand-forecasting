# Dashboard

A Streamlit and Plotly app for a retail planning manager who has five minutes and no statistics background. It reads
only `outputs/results_facts.json` and the files in `outputs/tables/`. Nothing is modelled in the app: every number,
and every sentence that states a result, is computed from those saved files when the page loads, so if the outputs
change the text follows.

## Run locally
From the project folder, with the environment active:
```bash
pip install -r dashboard/requirements.txt      # first time only
python -m streamlit run dashboard/app.py
```
Open http://localhost:8501 (add `--server.port 8502` if that port is busy). If a results file or a column is missing,
the page stops with a message that names the file and says to rerun the notebooks.

## Pages
A fixed sidebar holds the navigation and the filters; a slim header carries the project name. Each page opens with
the question it answers, shows its key numbers in cards, then its charts (each with a one-line takeaway underneath),
and closes with a "What this tells us" box written from the page's own values. Longer material sits in collapsed
sections.

| Page | Question | Main content | Files it reads |
|---|---|---|---|
| Overview | What is the answer? | five KPI cards, the answer in three sentences, what to do with it, the model chart | `results_facts.json`, `model_mase_unrounded.csv`, `comparison_overview.csv`, `inventory_calibrated_vs_normal.csv`, test tables |
| Data and scope | What data is this, and which items? | scope cards, weekly sales by category, the trimming before/after; collapsed: ABC-XYZ grid, data quality | `app/items.csv`, `pbi_fact_sales_weekly.csv`, `app/leading_zero_example.csv`, `data_quality.csv` |
| Demand patterns | How do sales move? | trend and weekly pattern of a chosen total, weekday and month index, SNAP and events, price; collapsed: stationarity | `demo_decomposition.parquet`, `explore_*.csv`, `stationarity_summary.csv` |
| Forecast accuracy | Which method is most accurate? | error-measure toggle, leaderboard (static and shift-aware), error by horizon, by item group, where simple methods win; collapsed: all 726 items, work in progress | `comparison_metrics_item_origin.csv`, `comparison_forecasts.parquet`, `pbi_fact_forecast_error.csv`, `shift_aware_item_origin.csv`, `adaptation_variants_item_origin_mase.csv`, `advanced_wip_*.csv` |
| Item explorer | How does one item look? | search box, forecast against actual with the prediction interval, sales history with shifts marked, the item's error and rank, CSV downloads | `app/forecast_daily.parquet`, `comparison_forecasts.parquet`, `pbi_fact_sales_weekly.csv`, `pbi_fact_breaks.csv`, `pbi_fact_forecast_error.csv` |
| Demand shifts | How often, and does reacting help? | share flagged, size, timing, detection delay, detector sensitivity, every way of reacting with its change in error | `app/items.csv`, `hc_breaks_items.csv`, `calibration_*.csv`, `adaptation_variants_tests.csv`, `adaptive_vs_static_tests.csv`, `shift_aware_tests.csv` |
| Stock impact | What does it mean for stock? | lead time, service level and cost controls, cost and fill-rate cards, cost by policy, textbook vs past-error safety stock, other half-years, items to check | `pbi_fact_inventory_kpi.csv`, `inventory_cost_sensitivity.csv`, `inventory_calibrated_*.csv`, `robustness_cost_differences.csv` |
| Method and limits | How was it tested, what does it not prove? | test design, how the future was kept out, assumptions, limits; collapsed: other states, residual checks, verdict rules | `pbi_fact_breaks.csv`, `replication_conclusions.csv`, `diagnostics_summary.csv` |

## Filters
They live in the sidebar and keep their values when you change page. Each page says which ones it uses.

| Filter | Values |
|---|---|
| Category | FOODS, HOBBIES, HOUSEHOLD |
| Item group | AX (steady), AY (more variable) |
| Shift group | all items, items with a shift detected by a forecast date, items without |
| Forecast dates | any range of the six scored forecast dates |

There is no store-group filter: the saved outputs add the four California stores into one series per item.

## Conventions
- One colour per method across the whole dashboard; a version that reacts to shifts keeps its method's colour, paler.
- Chart titles state the finding; axes carry units; hover shows exact values.
- Tables only where values need looking up, each with a CSV download.
- Anything that depends on assumed costs is marked as simulated, with the assumptions next to it.
- Technical terms are explained once, next to where they appear. The full list is in `docs/GLOSSARY.md`.

## Tests
`tests/test_dashboard_pages.py` loads every page and fails on any error, on course labels, or on filler words in the
page text or the dashboard code. `tests/test_dashboard.py`, `test_dashboard_v2.py` and `test_classical_dashboard.py`
check the loaders, the verdict rules and the planner list.

## Deploy on Streamlit Community Cloud
1. Push the repository to GitHub. The app needs `outputs/results_facts.json`, `outputs/tables/*.csv`, the parquet files
   it reads and `outputs/tables/app/`, all committed (the two large `pbi_fact_*_daily.csv` files are not needed).
2. At share.streamlit.io choose Create app, then Deploy a public app from GitHub, and pick the repository and branch.
3. Set the main file path to `dashboard/app.py` and, under Advanced settings, Python 3.12.
4. Deploy. Cloud installs `dashboard/requirements.txt`, the file next to the entry point.

Free apps sleep when idle, so the first visitor after a quiet period waits for the app to wake.

## Screenshots
`python scripts/screenshots.py` starts the app, opens every page in a headless browser through Playwright, and saves
the pictures to `docs/screenshots/`; add `--dark` for `docs/screenshots/dark/`.

## Earlier pages
`docs/DASHBOARD_MAP.md` lists the 17 earlier pages and where each part now lives. Their code is kept, unused, in
`dashboard/archive/`.
