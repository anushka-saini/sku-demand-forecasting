# Milestone 2: every file added or changed

For review before committing. No git command that changes anything was run for this work (the `m2` branch was
created earlier, under the previous brief). "Existing numbers" means every value in `outputs/tables/`,
`data/interim/`, `data/processed/` (other than the new `data/processed/m2/` checkpoints) and the Milestone 1 part
of `outputs/results_facts.json`; all are unchanged (see the checks at the end).

## Added
| File | What it is |
|---|---|
| `src/config.py` | paths, seeds, backtest constants and plot style for notebooks 12+ |
| `src/stationarity.py` | ADF/KPSS, the 4-way label, d and D, outlier rule, swing rule, Christmas fix |
| `src/classical.py` | Holt-Winters, ARIMA, SARIMA, AICc order search, fallback, backtest, item sample |
| `notebooks/12_stationarity_decomposition.ipynb` | M2-T2, M2-T3 |
| `notebooks/13_classical_models.ipynb` | M2-T4, M2-T5 (H1, H2, H4) |
| `notebooks/14_model_diagnostics.ipynb` | M2-T6 |
| `notebooks/15_shift_aware_classical.ipynb` | M2-T7 (H3) |
| `notebooks/16_classical_stock_link.ipynb` | M2-T8 |
| `outputs/m2/tables/*` (28 files) | results of notebooks 12-16 |
| `outputs/m2/figures/*` (13 files) | slide-ready figures |
| `dashboard/views/stationarity.py`, `classical.py`, `demo.py` | the three new pages |
| `dashboard/utils/m2.py` | model names and colours for those pages |
| `scripts/run_all.py` | Windows runner (notebooks, facts, README, tests) |
| `tests/test_m2_methods.py` | metrics, stationarity classifier, no lookahead, fallback (17 tests) |
| `tests/test_m2_dashboard.py` | M2 verdict rules, facts, new pages render (7 tests) |
| `tests/test_notebook_headers.py` | header fields, 8-section template, notebooks were run (29 tests) |
| `tests/test_m1_facts.py` | Milestone 1 synopsis numbers against the raw data (8 tests) |
| `docs/PREREGISTRATION_M2.md` | hypotheses H1-H4, metrics, thresholds, subsets, seeds (written before results) |
| `docs/M2_DECISIONS.md`, `docs/M2_CHANGES.md` | decisions and this list |
| `docs/ROADMAP.md`, `docs/M1_RECAP.md` | every notebook in one table; Milestone 1 promises → files |
| `docs/midsem/INTERIM_FINDINGS.md`, `MODEL_JUSTIFICATION.md`, `TRACEABILITY.md`, `DEMO_SCRIPT.md`, `SLIDE_INPUTS.md` | mid-sem documents |
| `docs/screenshots/{stationarity,classical,demo}.png` and `dark/` copies | new page screenshots |
| `data/processed/m2/` (not in git; ignored with `data/`) | per-item checkpoints of notebook 13 |

## Changed
| File | Change |
|---|---|
| `src/metrics.py` | added `rmse`, `mae`, `mape`, `mape_excluded`, `smape`, `wape`, `all_errors`; existing functions untouched |
| `notebooks/00_data_scoping_eda.ipynb` | header cell; cells reordered so the saved files are reproduced; two premature save cells and two empty cells removed; `.head` → `.head()`; "10 leading zeros" → 403; "Phase" → "Stage"; re-run (all files it writes identical) |
| `notebooks/00a_data_audit.ipynb`, `00b_cross_category_scoping.ipynb` | header cell; "Phase" → "Stage"; re-run (all files identical). 00a already had an uncommitted deletion of its title cell before this work |
| `notebooks/01` … `11` | **markdown only**: header cell inserted, "Phase" → "Stage"; notebook 03's garbled characters fixed. Code cells and saved outputs are byte-identical (checked) |
| `dashboard/app.py` | three pages added to the navigation |
| `dashboard/utils/data.py` | M2 result files added to the schema |
| `dashboard/utils/verdicts.py` | `classical_vs_naive`, `classical_shift`, `m2_verdicts`; the four existing verdicts untouched |
| `dashboard/views/overview.py` | Milestone 2 card row and two lines in "How the verdicts are decided" |
| `scripts/build_facts.py` | `build_m2()` adds the `m2` section; Milestone 1 facts untouched |
| `scripts/update_readme.py` | Milestone 2 paragraph in the generated block |
| `scripts/screenshots.py` | three new pages; `--pages` option |
| `outputs/results_facts.json` | new `m2` key; everything else identical to the committed file |
| `README.md` | generated block (Milestone 2 paragraph), page count, repository map, `run_all.py` line, three gallery rows |
| `docs/screenshots/overview.png`, `dark/overview.png` | regenerated (Overview gained the Milestone 2 cards) |
| `docs/RUN_LOG.md` | runs 10-24 |
| `Makefile` | notebooks 12-16; `update_readme.py` in the facts step; Windows note |
| `requirements.txt` | `pmdarima==2.1.1` pinned again |

## Not touched
* `notebooks/00_data_scoping_eda_ORIGINAL.ipynb` shows as modified in `git status` because of the owner's own uncommitted edit from before this work.
* `data/raw/` and every Milestone 1 result file.

## Checks that existing numbers did not change
* `outputs/results_facts.json` without the `m2` key equals the committed version (compared with `git show HEAD:`).
* `git status` shows no change under `outputs/tables/` or `data/`.
* Notebooks 00, 00a, 00b re-run: all 33 files they write are MD5-identical to the backup (`docs/RUN_LOG.md`, run 21).
* Notebooks 01-11: code cells and outputs byte-identical to before the markdown edits.
* Reused forecasts: MASE recomputed in notebook 13 equals Milestone 1's per item-origin (max difference 1e-15); stock results for LightGBM and seasonal naive in notebook 16 equal Milestone 1's (max difference $0.0000).
