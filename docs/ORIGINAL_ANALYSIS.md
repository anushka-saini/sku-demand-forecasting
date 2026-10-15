# Original analysis (first version)

> This is the project's first write-up, kept for history. Where it differs from [RESULTS.md](RESULTS.md), the pre-registered robustness results take precedence (in particular the false-alarm rate, the meaning of the 44.7% figure, and safety-stock calibration). Links below are relative to the repository root.


**Research question:** Do SKU-level demand series show statistically detectable structural breaks? Does ignoring them degrade forecasts? Does adapting the model after a break improve inventory decisions compared with a static model?

**Short answer from this project:**
- **Breaks:** they are common but the detector is coarse. 67.8% of high-volume items alarm. After subtracting the 11.9% false-alarm rate measured on a seasonality-preserving null, an estimated 63.5% (95% CI 59.8–67.2%) have a real shift. Only 35% of breaks pass a strict high-confidence filter, and shifts under ~30% of mean sales are mostly missed.
- **Forecast accuracy:** break items are not measurably harder to forecast once item characteristics are controlled for.
- **Adaptation:** retraining after a break did **not** help. None of four adaptation methods improved forecasts, including on high-confidence breaks only, and the result repeats in Texas and Wisconsin. A good static model (LightGBM) clearly beat the seasonal-naive baseline on both accuracy and inventory cost.

> **Robustness update.** A set of pre-registered checks ([`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md)) re-tested every claim. Full results are in [`docs/RESULTS.md`](docs/RESULTS.md), and every number is reproducible from `outputs/results_facts.json` (`python scripts/build_facts.py`). Corrections to the sections below:
> - The 4.8% false-alarm rate holds only on the null used to set the threshold. On a stricter null it is 11.9% (CUSUM) and 16.1% (PELT).
> - "With no shift injected, CUSUM still alarms 44.7%" was measured on real series and is **not** a false-alarm rate.
> - The original safety stock was under-calibrated. A 95% target achieved an 82.7% cycle service level. Conformal safety stock raises that to 91.7% and lowers total cost by 31% (assumed costs).
> - The adaptive cost penalty is significant in all three CA windows tested, not only driven by a few items.

---

## Data
[M5 Forecasting (Walmart, Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy): `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- **Sales:** 2011-01-29 to 2016-05-22 (d_1941).
- **No COVID-era data:** nothing here relates to COVID.
- **Raw files:** place them in `data/raw/` (not committed).

## Scope (deliberate limits)
- **Stores:** California only; sales summed over the 4 CA stores to one series per item.
- **Items:** only AX/AY items (high volume, low-to-medium variability): FOODS 337, HOBBIES 106, HOUSEHOLD 283 = **726 items**.
- **ABC classification:** uses units, not revenue.
- **Adaptation:** simple retraining from the most recent detected break, nothing more elaborate.
- **Inventory inputs:** lead times and costs are **assumed**, not data, and are covered by a sensitivity analysis.

## Method

| Phase | Notebook | What it does |
|---|---|---|
| 0 | `00_data_scoping_eda`, `00b_cross_category_scoping` | Scoping, trims pre-launch leading zeros (≈50% of items had them), ABC-XYZ segmentation, ADF tests |
| 1–2 | `01_structural_break_detection` | Weekly CUSUM with burn-in, fixed 52-week reference window and a **surrogate-calibrated threshold**; PELT cross-check; injected-shift power test; validation against price changes, SNAP and calendar events |
| 3 | `02_forecasting_static_vs_adaptive` | Rolling-origin backtest (1 warm-up + 6 evaluation origins, 28-day horizon). Seasonal naive, ETS, global LightGBM, TSB. Static vs adaptive vs **placebo**. MASE, item-level WRMSSE, Wilcoxon tests, selection-bias regression, empirical 90% bands |
| 4 | `03_inventory_decision_layer` | Daily-review reorder-point simulation with lost sales on the 168 eval days. Normal safety stock for regular items and empirical-quantile safety stock for intermittent ones. 90/95/99% service levels × 3 lead times × 12 cost scenarios |
| 5 | `04_powerbi_export` + `powerbi/` | Star-schema CSVs, data dictionary, DAX measures, page layout |
| 6 | `05_dashboard_export` + `dashboard/` | Compact copies of the results for the Streamlit dashboard |

**Key design choices**
- **No lookahead in break detection.** The CUSUM threshold for Phase 3 is calibrated on data before the first origin. At every origin, breaks are detected only from full weeks up to that origin. After an alarm, the CUSUM restarts from the estimated change point, so later breaks can be found too.
- **Calibrated thresholds, not textbook ones.** The textbook threshold (h = 5σ) assumes independent data. On these autocorrelated retail series it gave a 100% break rate. The CUSUM threshold and the PELT penalty were instead chosen so that only ≈5% of **break-free surrogate series** alarm. The surrogates block-bootstrap each item's own reference year.
- **Placebo control.** Items *without* a break are given a random training truncation with the same length distribution as the adaptive windows. If the placebo gains as much as the adaptive model, any gain comes from "training on recent data", not from detecting the break.
- **Same MASE denominator for every variant.** The scale is the in-sample seasonal-naive MAE over all history up to the origin.

---

## Results
All numbers below are produced by the notebooks. Tables are in `outputs/tables/`, figures in `outputs/figures/`.

### Phase 2: break detection
| | FOODS | HOBBIES | HOUSEHOLD |
|---|---|---|---|
| Items with a CUSUM break (full series) | 234 / 337 (69.4%) | 68 / 106 (64.2%) | 186 / 283 (65.7%) |
| Share of breaks that are downward | 64.1% | 75.0% | 61.3% |
| Median detection delay (weeks) | 16 | 16 | 16.5 |

**Calibration and agreement**
- **CUSUM false alarms:** 4.8% on 14,520 independent surrogate series (calibrated h = 14.38).
- **PELT false alarms:** 4.8% on 7,260 surrogates.
- **CUSUM vs PELT:** they agree on 83.7% of items (Cohen's κ = 0.62). 64.8% of CUSUM breaks have a PELT changepoint within ±8 weeks.

**Power test.** A shift of known size was injected into real series at a random date, 1,264 trials per size. The size-0 row is the control: nothing injected.

| Injected shift (σ of reference year) | ≈ % of mean level (median) | CUSUM detection | median delay (wks) | PELT hit within ±8 wks |
|---|---|---|---|---|
| 0 (control) | 0% | 44.7% | 42 | 10.3% |
| 0.5 | 18% | 46.4% | 33 | 16.8% |
| 1 | 36% | 57.8% | 23 | 40.3% |
| 2 | 72% | 86.6% | 13 | 82.4% |
| 3 | 108% | 95.0% | 8 | 93.8% |

**Validation against real events** (alignment is not causation)
- **Sustained price changes** (≥5% and held for 4 weeks): 5.7% of breaks fall within ±4 weeks of one, against 4.2% expected by chance (p = 0.044). It's a real but tiny excess. In the 28 aligned cases, price and sales moved in opposite directions 89% of the time.
- **SNAP:** the CA schedule is fixed (days 1–10 of every month, all years), so it cannot cause a sustained shift.
- **Calendar events:** no excess alignment (49.2% vs 47.2% expected by chance, p = 0.20).

### Phase 3: forecasting (eval origins 2015-12-06 to 2016-04-24, 726 items)
**Median item MASE, static models** (<1 beats in-sample seasonal naive)

| | Seasonal naive | ETS | LightGBM |
|---|---|---|---|
| FOODS | 0.957 | 0.771 | **0.751** |
| HOBBIES | 1.004 | 0.755 | **0.752** |
| HOUSEHOLD | 0.978 | 0.744 | **0.730** |
| All | 0.976 | 0.755 | **0.744** |
| Item-level WRMSSE | 1.036 | 0.828 | **0.815** |

- **LightGBM vs seasonal naive:** median paired MASE difference −0.226 (95% CI −0.234 to −0.215). LightGBM was better on 93.5% of items (Wilcoxon p < 0.001, rank-biserial −0.93).
- **Intermittent items (26):** median MASE was LightGBM 0.886, ETS 0.962, TSB 0.986, seasonal naive 1.147. TSB did not beat the general models.
- **90% bands:** the empirical bands covered 84.5% of LightGBM actuals (static), so they are somewhat too narrow.

**Adaptive − static MASE on break items** (per-item mean over origins with a detected break; positive = adaptive worse)

| Model | Category | n | Median diff | 95% CI | Items improved | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| LightGBM | All | 488 | +0.0033 | +0.0015 to +0.0053 | 43.2% | <0.0001 |
| LightGBM | FOODS | 234 | +0.0054 | +0.0017 to +0.0087 | 40.6% | 0.0001 |
| LightGBM | HOBBIES | 67 | +0.0036 | +0.0008 to +0.0082 | 37.3% | 0.006 |
| LightGBM | HOUSEHOLD | 187 | +0.0011 | −0.0024 to +0.0046 | 48.7% | 0.16 |
| ETS | All | 488 | +0.0024 | +0.0004 to +0.0038 | 44.5% | 0.0009 |
| Placebo, LightGBM | no-break items | 261 | +0.0009 | −0.0015 to +0.0035 | 48.3% | 0.35 |
| Placebo, ETS | no-break items | 261 | +0.0004 | −0.0030 to +0.0038 | 48.7% | 0.81 |

- **Direction:** adaptation is *statistically* worse, but the effect is tiny: +0.003 MASE on a base of ≈0.74, under 0.5%.
- **Placebo:** shortening the training window at random changed nothing, so detecting a break added no useful information for retraining.
- **Does ignoring breaks degrade forecasts?** Not detectably. Break items and no-break items are similar in volume and CV (|SMD| ≤ 0.12) but differ in history length (SMD 0.45). After adjusting for volume, CV, zero share, history length and category, break items have +2.8% LightGBM MASE (95% CI −1.3% to +7.2%, p = 0.18).

### Phase 4: inventory simulation
Base case: lead time 7 days, holding cost 25%/yr, stockout cost 0.5 × price. The simulation covers 168 days and all 726 items.

| Policy | Target SL | Fill rate | Stockout days | Avg inventory ($) | Total cost ($) |
|---|---|---|---|---|---|
| Seasonal naive | 95% | 97.3% | 3,026 | 193,251 | 76,429 |
| LightGBM static | 95% | **97.7%** | **2,850** | **161,205** | **68,708** |
| LightGBM adaptive | 95% | 97.4% | 3,070 | 161,278 | 77,200 |
| Seasonal naive | 99% | 98.0% | 2,069 | 219,760 | 64,402 |
| LightGBM static | 99% | **98.5%** | **1,810** | **181,180** | **53,492** |
| LightGBM adaptive | 99% | 98.2% | 2,005 | 181,384 | 59,586 |

- **Static LightGBM vs seasonal naive:** at 95%, static LightGBM holds 16.6% less inventory and costs 10.1% less, with a higher fill rate. Across all 108 cost / lead-time / service-level scenarios its median cost change is −11.3% (range −33.6% to +1.5%).
- **Adaptive vs static:** adaptive cost more in all 108 scenarios (median +11.8%, range +4.2% to +20.0%). The cause is a slightly larger under-forecast bias on break items: −3.9% vs −2.9%.
- **Caveat on the adaptive result:** the total-cost gap is concentrated. 10 items account for 82% of the extra cost, and item by item 252 items got worse while 236 improved. For 7-day lead time at 95%, the per-item Wilcoxon test is not significant (median +$0.28, 95% CI −$0.14 to +$0.69, p = 0.31).
- **Service level:** 99% was the cheapest target in every cost scenario tested. Holding a unit for about a week costs about 0.5% of its price, while a lost sale costs 25–200% of price, so extra safety stock almost always pays for itself.
- **Achieved fill rates fall short of the targets** (98.5% at a 99% target). The normal safety-stock formula assumes unbiased, independent errors. Here LightGBM's forecasts are 3.6% low in total and its 90% bands cover only 84.5% of days.

---

## What did not work (honest list)
1. **The first CUSUM flagged 100% of items.** It took mean and std from the whole series, ran on daily data from day 1, and used the textbook threshold. The fix (weekly data, burn-in, reference window, surrogate-calibrated h) gives 64–69%.
2. **CUSUM cannot separate steps from drift.** In the injected-shift test, 44.7% of series alarm after the injection date even when nothing is injected. In addition, 17.4% of breaks fall within 4 weeks of monitoring start, against ≈2% expected if breaks were spread evenly. A longer burn-in did not remove that cluster. Many "breaks" are slow drift away from the first year's level.
3. **A step-vs-trend BIC test was not discriminating.** It preferred "step" for 95% of *no-break* items, because choosing the best split location gives the step model an unfair advantage. It is kept in notebook 01 as a negative result.
4. **Detection is slow.** The median delay is ~16 weeks, and shifts smaller than ~1σ (≈36% of mean volume) are hard to tell apart from background drift.
5. **Adaptation by retraining did not help** (Phase 3) and raised simulated inventory cost (Phase 4). A likely explanation (not tested here) is that LightGBM's rolling-mean features and ETS's smoothing already track level changes, so dropping history mostly removes useful seasonal and price information.
6. **TSB** did not beat LightGBM on the 26 intermittent items.
7. **Uncertainty bands and safety stock are too narrow.** LightGBM's 90% bands cover 84.5% of days, and fill rates fall below the service targets.

## Limitations
- **Scope:** CA only, AX/AY items only, summed across stores. Store-level stockouts and substitution are hidden.
- **Prices and calendar** are treated as known in advance, which is the M5 convention.
- **Adaptation:** only one mechanism (retrain from the last break) was tested. Others, such as level-shift dummies, intervention models or down-weighting old data, might behave differently.
- **Cost parameters are assumptions.** Ordering cost is ignored, the unit value is the sell price, and lost sales are assumed instead of backorders.
- **Short evaluation:** 6 origins × 28 days = 168 days, including the Christmas period.
- **Reproducibility checks:** notebooks 01 and 03 were each run twice and gave identical results. Notebook 02 (~30 min) was run once. Seeds are fixed and LightGBM runs in deterministic mode.

## Dashboard (Streamlit)

An explainable demo of the results, for a 3-minute read or for a planner to act on. It reads only pre-computed
files in `outputs/tables/` (no model is trained in the app), so it works from a fresh clone.

| Overview | Break detection |
|---|---|
| ![Overview](docs/screenshots/overview.png) | ![Break detection](docs/screenshots/breaks.png) |
| **Forecast comparison** | **Inventory decision** |
| ![Forecast comparison](docs/screenshots/forecasts.png) | ![Inventory decision](docs/screenshots/inventory.png) |

More screenshots are in `docs/screenshots/`. The walkthrough script, interview questions and captions are in
[`docs/dashboard_demo.md`](docs/dashboard_demo.md).

**Pages:** Overview (headline + 4 KPIs) → Data & method → Segmentation (ABC-XYZ, click through to an item) →
Break detection → Forecast comparison (with item explorer) → Inventory decision (service-level and lead-time
selectors, cost trade-off, item risk table) → Assumptions & limitations.

### Run locally
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
Open http://localhost:8501. The first load of each page takes a second or two while files are cached.

### Deploy on Streamlit Community Cloud
1. Push the repo to GitHub (the app's data in `outputs/tables/` and `outputs/tables/app/` must be committed; about 22 MB in total).
2. On share.streamlit.io choose **Create app → Deploy a public app from GitHub**, pick the repo and branch.
3. Set **Main file path** to `dashboard/app.py`. In **Advanced settings** choose Python 3.12.
4. Deploy. Cloud installs `dashboard/requirements.txt` (the file next to the entrypoint), not the heavier root file.

### Dashboard data dictionary
| File (in `outputs/tables/`) | Grain | Used for |
|---|---|---|
| `app/items.csv` | one row per CA item (3,049) | segmentation, break flags, ADF p-value (`adf_p_value`, modelled items only) |
| `app/category_summary.csv` | category | leading-zero counts (>30 zero days before first sale), AX/AY counts, % stationary |
| `app/leading_zero_example.csv` | day, one item | before/after trimming example (`kept_after_trim`) |
| `app/forecast_daily.parquet` | item × day × model × variant | actual, forecast, empirical 90% band (`lo_90`, `hi_90`), `horizon` (days after origin) |
| `app/inventory_policy.csv` | item × policy × service level × lead time | safety stock, reorder point, fill rate, costs, `stockout_risk_flag` (**derived**: fill rate < target) |
| `app/selection_bias.csv` | variable / model | covariate balance (SMD) and break-vs-no-break regression |
| `pbi_fact_forecast_error.csv` | item × origin × model × variant | MASE per window (`has_break`, `is_eval_origin`) |
| `pbi_fact_breaks.csv` | item × origin | online break detection at each forecast date |
| `pbi_fact_sales_weekly.csv` | item × week | weekly history in the item explorer |
| `adaptive_vs_static_tests.csv` | comparison × model × category | Wilcoxon tests, median difference, bootstrap 95% CI |
| `model_mase_by_category.csv` | category | median MASE per model |
| `break_calibration.csv`, `break_method_agreement_summary.csv` | method | false-alarm rates, CUSUM-PELT agreement |
| `break_power_check.csv`, `break_event_alignment.csv` | shift size / event | injected-shift test, alignment vs chance |
| `inventory_policy_summary.csv`, `inventory_adaptive_vs_static_tests.csv`, `inventory_cost_sensitivity.csv` | policy × service level (× scenario) | inventory KPIs, paired cost tests, assumed-cost grid |

Missing files or columns stop the page with a message naming the file and the notebook that creates it.

## How to reproduce
```bash
python -m venv sku_env && sku_env\Scripts\activate      # Windows
pip install -r requirements.txt
# put the three M5 CSVs in data/raw/
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_scoping_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/00b_cross_category_scoping.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_structural_break_detection.ipynb   # ~6 min
jupyter nbconvert --to notebook --execute --inplace notebooks/02_forecasting_static_vs_adaptive.ipynb --ExecutePreprocessor.timeout=10000  # ~35 min on 12 cores
jupyter nbconvert --to notebook --execute --inplace notebooks/03_inventory_decision_layer.ipynb  # ~3 min
jupyter nbconvert --to notebook --execute --inplace notebooks/04_powerbi_export.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_dashboard_export.ipynb
python -m pytest tests -q
```

## Repository layout
```
data/{raw,interim,processed}/   not committed
notebooks/                      00, 00b, 01-05 (see table above)
src/
  segmentation.py               trim_leading_zeros, ABC-XYZ
  diagnostics.py                ADF tests
  break_detection.py            weekly CUSUM, online multi-break CUSUM, surrogate calibration, PELT, power test, event alignment
  forecasting.py                seasonal naive, ETS, TSB, LightGBM features/training, adaptive & placebo windows
  metrics.py                    MASE, RMSSE/WRMSSE, Wilcoxon + rank-biserial + bootstrap CI, SMD
  inventory.py                  safety stock (normal / empirical), reorder-point simulation, KPIs & costs
tests/                          58 tests (src functions, dashboard helpers, every dashboard page renders)
outputs/tables, outputs/figures results; pbi_*.csv feed Power BI
powerbi/                        star_schema.md, data_dictionary.md, dax_measures.md, layout_spec.md
dashboard/                      Streamlit app (app.py, views/, utils/, requirements.txt)
docs/                           dashboard screenshots, demo script
```
