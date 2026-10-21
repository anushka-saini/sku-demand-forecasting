# Milestone 1 recap: what was promised and where it is

The Milestone 1 synopsis promised a dataset description, objectives, and an exploratory analysis (line plot,
missing/zero-value plot, seasonal plot, decomposition). This table maps each promise to the notebook and file
that delivers it. The numbers quoted in the synopsis are checked against the raw data by `tests/test_m1_facts.py`.

| Promise | Delivered by | Files |
|---|---|---|
| Dataset description (files, columns, size) | `00a_data_audit` | `outputs/figures/data_audit/snapshot_sales.png`, `snapshot_calendar.png`, `snapshot_prices.png`, `snapshot_all_files.png` |
| Objectives / research questions | `README.md` (first section), `docs/PREREGISTRATION.md` | – |
| Line plot of one product | `00a_data_audit` | `outputs/figures/data_audit/plot_line_one_product.png` (FOODS_3_586, weekly, 2011-2016) |
| Missing and zero values | `00a_data_audit` (plot), `00b_cross_category_scoping` (counts) | `outputs/figures/data_audit/plot_zero_values.png`; `data/interim/cross_category_summary.csv` |
| Seasonal (weekday) plot | `00a_data_audit` | `outputs/figures/data_audit/plot_seasonal_weekday.png` |
| Decomposition | `00_data_scoping_eda` | `outputs/figures/sample_decomposition.png`, `outputs/figures/foods_1_004_decomposition.png` |
| Sample series per category | `00b_cross_category_scoping` | `outputs/figures/cross_category_sample_series.png` |
| Data cleaning (pre-launch zeros) and item selection | `00_data_scoping_eda`, `00b_cross_category_scoping` | `data/interim/*_candidates.csv` |

## Synopsis numbers and their check
| Number in the synopsis | Test in `tests/test_m1_facts.py` | Result |
|---|---|---|
| 1,941 days per product, 29 Jan 2011 to 22 May 2016 | `test_1941_days_from_29_jan_2011_to_22_may_2016` | matches |
| 3,049 California products | `test_3049_california_products` | matches |
| 1,565 products with more than 30 leading zero days | `test_1565_products_start_with_more_than_30_zero_days` | matches |
| 726 selected items | `test_726_selected_items` | matches |
| Sales file 30,490 × 1,947; calendar 1,969 × 14; prices 6,841,121 × 4 | `test_file_shapes` | matches |
| No empty sales cells | `test_no_empty_sales_cells` | matches |
| Prices missing until a product goes on sale | `test_prices_missing_until_a_product_goes_on_sale` | matches (no store-item sells before its first price week) |
| Sales highest at the weekend in every category | `test_sales_are_highest_at_the_weekend_in_every_category` | matches |

## Re-run check (M1-T3)
Notebooks 00, 00a and 00b were re-run top to bottom on 2026-10-21 after the header and notebook 00 fixes. The
result of comparing every file they write with the copy taken before the re-run is in `docs/RUN_LOG.md` (run 12).
