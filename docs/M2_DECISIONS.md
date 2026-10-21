# Milestone 2: decisions and mismatches

Every choice made while doing Milestone 2, with the reason. "Owner" = decided by the project owner; "default" =
the sensible default taken without asking, as the brief requested.

## Decided by the owner (2026-10-21)
| # | Decision | Why |
|---|---|---|
| O1 | Item-level classical models on **150 items**: 50 per category, 25 with a detected break and 25 without, seed 42 | Timing (run 10) put 150 items at about 25-35 min; the brief's minimum was 60 |
| O2 | ARIMA/SARIMA orders chosen **at origin 0** (the warm-up), then fixed | Choosing at origin 1 would let data after origin 0 shape the origin-0 forecasts that seed bands and safety stock |
| O3 | `scripts/run_all.py` as a Windows runner; the Makefile is kept and extended | `make` is not installed in PowerShell or Git Bash |
| O4 | Notebook 00 fixed (cell order, the "10 leading zeros" text, real figure 403) with its saved files kept identical | Notebook 05 reads 00's untrimmed first-pass file; a naive re-run would have overwritten it |

## Defaults taken
| # | Decision | Why |
|---|---|---|
| D1 | **Git is read-only.** The `m2` branch was created on 2026-10-21 under the earlier brief; since the new brief, no git command that changes anything has been run | Hard rule 1 |
| D2 | `PREREGISTRATION_M2.md` was written on 2026-10-21 before any classical forecast was scored, but it is **not committed** (the owner commits) | The owner can check the file's timestamp against `RUN_LOG.md` runs 11+ |
| D3 | New outputs go to `outputs/m2/tables/` and `outputs/m2/figures/`; the pre-registration's path `outputs/tables/m2_sample_items.csv` became `outputs/m2/tables/m2_sample_items.csv` | Hard rule 2 (new outputs in `outputs/m2/`) came after the pre-registration |
| D4 | Damped Holt-Winters is the **existing ETS** (notebook 02), relabelled, not refitted | The brief says to reuse ETS; refitting would duplicate it with a different window |
| D5 | Classical models use the **last 730 days** at each origin; the reused ETS, naive and LightGBM use all history | As pre-registered; the difference is stated wherever they are compared |
| D6 | SARIMAX gets a **constant** when d = D = 0 (none otherwise) | statsmodels' SARIMAX has no constant by default, so an undifferenced model would forecast towards zero |
| D7 | Holt-Winters multiplicative on items uses **sales + 1** when the window has a zero; it is fitted on items only for H4 | The brief says "multiplicative on aggregates only", but H4 compares additive vs multiplicative on items; both are honoured this way |
| D8 | Aggregates: 25 December replaced by the same weekday ±1 week **for decomposition only**; forecasting uses the real series | Stores close on Christmas, so the total is almost 0 and multiplicative decomposition breaks |
| D9 | Aggregate comparison: classical models are fitted on each total; naive, ETS and LightGBM are **bottom-up** (sum of the 726 item forecasts) | Those three were never fitted on totals, and the brief says not to retrain them |
| D10 | Shift-aware restart and placebo windows are **also capped at 730 days**. When the detected change week is older than the window, restart equals static; when it is before the window start, the level dummy is constant and also equals static. These item-origins count as a difference of 0 in the pre-registered test, and a second, descriptive test uses only item-origins where the variant differs | Keeps "restart" from ever having more data than "static"; reporting both views avoids hiding the dilution |
| D11 | The 8-section template is required for new notebooks (12+); existing notebooks get the **header cell only** (markdown) | Restructuring 01-11 into 8 sections would mean moving cells around in notebooks whose outputs must not change |
| D12 | Existing notebooks: "Phase N" → "Stage N" in markdown only; notebook 03's garbled characters (≤, ·, √ shown as "â‰¤" etc.) fixed in markdown | Clarity; code and outputs untouched (checked cell by cell) |
| D13 | Milestone 1 notebooks get task ids **M1-W1 … M1-W11** (work packages). The brief's M1-T1 … M1-T3 are the three safeguard tasks | The brief uses M1-T ids for safeguards, so another prefix was needed for the notebooks themselves |
| D14 | `00_data_scoping_eda_ORIGINAL.ipynb` is left untouched and excluded from the header test | It is the owner's backup copy, with the owner's uncommitted edits |
| D15 | `00a_data_audit.ipynb` had an uncommitted deletion of its title cell (not made by the assistant); the standard header now takes that place | The header test needs a first-cell header |
| D16 | `pmdarima==2.1.1` pinned again in `requirements.txt` | `ndiffs` / `nsdiffs`; it was installed but had been removed as unused |
| D17 | `run_all.py` and the Makefile now also run `update_readme.py` in the facts step | So the README's generated numbers never drift from the facts file |
| D18 | Milestone 1 notebooks 01-11 were **not** re-run | Their outputs must stay identical; only their markdown changed. 00, 00a and 00b were re-run (M1-T3) with every written file compared to a backup |
| D19 | Screenshots: only the Overview (it gained the Milestone 2 cards) and the three new pages are regenerated, light and dark | Other pages did not change |

## Mismatches between the brief and the repo
1. `make` is not available on this machine (see O3).
2. The brief asks for multiplicative Holt-Winters "on aggregates only" but H4 needs it on items (D7).
3. Notebook 00's saved files came from different runs (the first pass was saved before trimming); a straight re-run in cell order would have changed them (O4).
4. `nsdiffs` (OCSB) recommends D = 0 for most series, so weekly seasonality in SARIMA is carried by the seasonal AR/MA terms rather than by seasonal differencing (see notebook 12 for the shares).
5. The brief's optional Milestone 3 groundwork (XGBoost, Prophet, GRU) needs packages that are not installed (xgboost, prophet, torch; torch alone is 1-2 GB). It was **not started**: it is optional, and installing large packages during the mid-sem window risks the working environment.
6. Post-hoc additions, not in the pre-registration, clearly labelled where they appear: the degenerate-forecast count (notebook 13), the SARIMA-on-totals rescaling check (notebook 14), and the "only where the variant differs" and high-confidence views in notebook 15 (the pre-registered test is reported first).
