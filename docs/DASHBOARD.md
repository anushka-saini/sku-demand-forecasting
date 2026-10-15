# Dashboard

A Streamlit + Plotly app for two readers: someone with three minutes who wants the story, and an inventory planner who wants to act. It reads only `outputs/results_facts.json` and the CSV/Parquet files in `outputs/tables/`. Nothing is modelled in the app, and selectors only filter precomputed rows.

## Run locally
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
Open http://localhost:8501. If a results file or column is missing, the page stops with a message naming the file and the script or notebook that creates it.

## Pages
| # | Page | Planner question it answers |
|---|---|---|
| 1 | Overview | Four verdicts (Yes / Partly / No, evidence strength, caveat) and four key numbers |
| 2 | Data & method | What data, what was kept, and why pre-launch zeros were trimmed |
| 3 | Segmentation | Where volume and variability sit (ABC-XYZ), with click-through to an item |
| 4 | Shift detection | How often demand shifts, plus a validation panel: false alarms by null, power for steps vs drifts, agreement with PELT, alignment with price changes and holidays |
| 5 | Forecast comparison | Which model to trust (paired tests, CIs, win rates), whether forecasts and bands are honest (bias, coverage), and an item explorer |
| 6 | Does retraining help? | Four retraining methods, placebo control, break vs no-break items, high-confidence subset |
| 7 | Inventory simulator | Fill rate, inventory and cost by policy under chosen lead time and **assumed** costs; calibrated safety stock; where the cost gap comes from; other seasons; newsvendor check |
| 8 | Planner | "What should I check on Monday?" A watchlist of items below target, with reasons and a CSV download |
| 9 | Replication | Whether each conclusion holds in Texas and Wisconsin |
| 10 | Assumptions & limitations | Scope, assumed parameters, what did not work, how to reproduce, glossary |

**Global filters** (sidebar, kept across pages): category, segment, target service level.

**Honesty conventions**
- **ASSUMED** values (lead time, holding and stockout costs) carry an orange tag, and charts that depend on them have a dashed orange frame.
- **DERIVED** display rules (for example, the watchlist reasons) carry a blue tag.
- Every chart title states its takeaway, and every chart has a "How to read this" line.
- Aggregates show n, with a warning below 20.
- A scope footer appears on every page.

## How the verdicts are decided
These rules live in `dashboard/utils/verdicts.py` and are tested in `tests/test_dashboard_v2.py`.

| Question | Rule |
|---|---|
| Do shifts exist and can we detect them? | *Yes* if the chance-adjusted share's 95% CI is above 0 **and** a sudden 20% shift is detected at least 80% of the time; *Partly* if only the first holds; *No* otherwise. *Strong* if it also replicates in TX and WI. |
| Which forecast is best? | The model with the lowest median MASE. *Strong* if it beats every other model significantly in CA, TX and WI; otherwise *moderate*. |
| Does retraining help? | *No* if no method meets the pre-registered "helps" rule on all breaks or on high-confidence breaks; *Partly* if one helps only on high-confidence breaks. *Strong* if truncation also fails in TX and WI. |
| Does a better forecast cut stock cost? | *Yes* if static LightGBM is cheaper than seasonal naive with a 95% CI below 0 in at least 2 of 3 windows, and cheaper in TX and WI. *Strong* if all 3 windows qualify. |

## Deploy on Streamlit Community Cloud
1. Push the repository to GitHub. The app needs `outputs/results_facts.json`, `outputs/tables/*.csv` and `outputs/tables/app/`, all committed (the two large `pbi_fact_*_daily.csv` files are not needed).
2. At share.streamlit.io choose **Create app → Deploy a public app from GitHub**, then pick the repository and branch.
3. Set **Main file path** to `dashboard/app.py` and, under **Advanced settings**, Python 3.12.
4. Deploy. Cloud installs `dashboard/requirements.txt`, the file next to the entry point.

**Free apps sleep when idle.** The first visitor after a quiet period waits for the app to wake. For a link that always works, use the static results summary, which is regenerated from the facts file:
- [`docs/results_summary.html`](results_summary.html), and a PDF copy, [`docs/results_summary.pdf`](results_summary.pdf), both built by `python scripts/build_summary.py`
- [`docs/RESULTS.md`](RESULTS.md), which GitHub renders

## Screenshots
`python scripts/screenshots.py` starts the app, opens every page in a headless browser through Playwright (Chromium, or the installed Edge as a fallback), and saves full-page PNGs to `docs/screenshots/`.

## Data the app reads
| File (in `outputs/`) | Grain | Used on |
|---|---|---|
| `results_facts.json` | one entry per headline number (value, n, CI, source) | verdicts, KPI cards, footer, scope |
| `tables/app/items.csv` | CA item | segmentation, break flags, explorer |
| `tables/app/category_summary.csv`, `app/leading_zero_example.csv` | category / one item's days | Data & method |
| `tables/app/forecast_daily.parquet` | item × day × model × variant | item explorer |
| `tables/app/selection_bias.csv` | variable / model | retraining page |
| `tables/pbi_fact_forecast_error.csv` | item × origin × model × variant | per-item retraining effects |
| `tables/pbi_fact_breaks.csv`, `pbi_fact_sales_weekly.csv` | item × origin / item × week | explorer |
| `tables/pbi_fact_inventory_kpi.csv` | item × policy × service level × lead time | inventory simulator, cost-gap chart |
| `tables/inventory_cost_sensitivity.csv` | scenario × policy × service level | cost scenarios (assumed) |
| `tables/inventory_calibrated_vs_normal.csv`, `inventory_calibrated_items.csv` | policy / item × method × service level | calibrated safety stock, planner |
| `tables/item_bias.csv`, `item_band_coverage.csv` | item × model × variant | planner reasons |
| `tables/calibration_*.csv`, `break_calibration.csv`, `break_power_check.csv`, `break_method_agreement_summary.csv`, `break_event_alignment.csv`, `hc_breaks_*.csv` | method / shift size / event | shift detection |
| `tables/model_mase_unrounded.csv`, `model_pairwise_tests.csv`, `forecast_bias.csv`, `conformal_coverage.csv` | model × category | forecast comparison |
| `tables/adaptation_variants_*.csv`, `adaptive_vs_static_tests.csv` | variant × group | retraining page |
| `tables/robustness_*.csv` | window × policy × service level | inventory simulator (seasons, newsvendor) |
| `tables/replication_*.csv` | state | replication page |
