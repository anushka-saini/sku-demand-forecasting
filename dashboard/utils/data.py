"""Loaders for the pre-computed result tables. Nothing is modelled in the app."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
APP = TABLES / "app"

# file -> (path, required columns, date columns)
SCHEMA = {
    "items": (APP / "items.csv",
              ["item_id", "category", "abc_class", "xyz_class", "segment", "mean_sales", "cv", "total_sales",
               "in_scope", "cusum_first_break", "first_break_week", "first_break_direction", "first_break_shift_sd",
               "break_at_last_origin", "last_break_week", "intermittent", "adf_p_value", "is_stationary",
               "break_group"],
              ["first_break_week", "last_break_week"]),
    "category_summary": (APP / "category_summary.csv",
                         ["category", "total_items", "items_with_leading_zeros", "pct_items_with_leading_zeros",
                          "ax_ay_candidates", "items_tested", "pct_stationary"], []),
    "leading_zero_example": (APP / "leading_zero_example.csv", ["item_id", "date", "sales", "kept_after_trim"], ["date"]),
    "forecast_daily": (APP / "forecast_daily.parquet",
                       ["item_id", "date", "origin", "model", "variant", "actual", "forecast", "lo_90", "hi_90", "horizon"],
                       ["date", "origin"]),
    "inventory_policy": (APP / "inventory_policy.csv",
                         ["item_id", "category", "policy", "model", "variant", "service_level", "lead_time", "fill_rate",
                          "stockout_days", "lost_units", "demand_units", "avg_inventory_value", "mean_safety_stock",
                          "final_rop", "holding_cost", "stockout_cost", "total_cost", "holding_value_days",
                          "lost_value", "break_item", "stockout_risk_flag"], []),
    "selection_bias": (APP / "selection_bias.csv",
                       ["kind", "variable", "break_items", "no_break_items", "smd", "pct_effect", "ci_low_pct",
                        "ci_high_pct", "p_value", "n_break", "n_no_break"], []),
    "sales_weekly": (TABLES / "pbi_fact_sales_weekly.csv", ["item_id", "week_start", "units"], ["week_start"]),
    "breaks_by_origin": (TABLES / "pbi_fact_breaks.csv",
                         ["item_id", "origin", "has_break", "n_breaks", "last_change_week", "last_alarm_week",
                          "adaptive_window_days", "is_eval_origin"], ["origin", "last_change_week", "last_alarm_week"]),
    "forecast_error": (TABLES / "pbi_fact_forecast_error.csv",
                       ["item_id", "origin", "model", "variant", "has_break", "mase", "is_eval_origin"], ["origin"]),
    "model_mase": (TABLES / "model_mase_by_category.csv", ["category", "ets", "best", "seasonal_naive"], []),
    "adaptive_tests": (TABLES / "adaptive_vs_static_tests.csv",
                       ["comparison", "model", "category", "n", "median_diff", "ci_low", "ci_high", "share_improved",
                        "rank_biserial", "p_value"], []),
    "power_check": (TABLES / "break_power_check.csv",
                    ["size_sd", "n_trials", "shift_pct_of_mean_median", "cusum_detection_rate",
                     "cusum_median_delay_wks", "pelt_hit_rate_8wk"], []),
    "event_alignment": (TABLES / "break_event_alignment.csv",
                        ["event", "window_wks", "n_breaks", "aligned_pct", "chance_pct", "p_value"], []),
    "calibration": (TABLES / "break_calibration.csv",
                    ["method", "parameter", "value", "heldout_false_alarm_rate", "n_heldout_surrogates"], []),
    "null_rates": (TABLES / "calibration_null_rates.csv",
                   ["group", "method", "n_null_series", "n_items", "observed", "far", "adjusted", "ci_low", "ci_high"], []),
    "injection": (TABLES / "calibration_injection.csv",
                  ["shape", "size", "n_trials", "cusum_power", "cusum_power_26wk", "cusum_median_delay_wks", "pelt_hit"], []),
    "hc_funnel": (TABLES / "hc_breaks_funnel.csv", ["items", "cusum_breaks", "1_pelt_agrees", "2_and_effect_20pct",
                                                    "3_and_assessable", "4_high_confidence"], []),
    "hc_items": (TABLES / "hc_breaks_items.csv", ["item_id", "cusum_break", "high_confidence"], []),
    "pairwise": (TABLES / "model_pairwise_tests.csv",
                 ["model_a", "model_b", "category", "n", "median_diff", "ci_low", "ci_high", "share_improved", "p_value"], []),
    "mase_unrounded": (TABLES / "model_mase_unrounded.csv", ["model", "category", "n_items", "median_item_mase"], []),
    "bias": (TABLES / "forecast_bias.csv", ["model", "variant", "category", "n_items", "pooled_bias", "median_item_bias",
                                           "ci_low", "ci_high"], []),
    "coverage": (TABLES / "conformal_coverage.csv", ["model", "variant", "level", "origins", "coverage_before",
                                                    "coverage_after", "n_days"], []),
    "item_bias": (TABLES / "item_bias.csv", ["item_id", "model", "variant", "bias"], []),
    "item_coverage": (TABLES / "item_band_coverage.csv", ["item_id", "model", "variant", "coverage90_before",
                                                         "coverage90_after"], []),
    "variants": (TABLES / "adaptation_variants_tests.csv", ["group", "variant", "n", "median_diff", "ci_low", "ci_high",
                                                           "share_improved", "p_value", "p_holm", "helps"], []),
    "variants_by_cat": (TABLES / "adaptation_variants_by_category.csv",
                        ["group", "variant", "category", "n", "median_diff", "ci_low", "ci_high", "p_value"], []),
    "calibrated_inv": (TABLES / "inventory_calibrated_vs_normal.csv",
                       ["method", "policy", "service_level", "fill_rate", "cycle_service_level", "avg_inventory_value",
                        "holding_cost", "stockout_cost", "total_cost", "n_items"], []),
    "calibrated_items": (TABLES / "inventory_calibrated_items.csv",
                         ["item_id", "model", "variant", "method", "service_level", "fill_rate", "lost_units",
                          "stockout_days", "total_cost", "mean_safety_stock"], []),
    "windows": (TABLES / "robustness_windows.csv", ["window", "seasonal_coverage", "n_items"], []),
    "window_summary": (TABLES / "robustness_inventory_summary.csv",
                       ["window", "policy", "service_level", "n_items", "fill_rate", "cycle_service_level",
                        "avg_inventory_value", "total_cost"], []),
    "window_diffs": (TABLES / "robustness_cost_differences.csv",
                     ["window", "service_level", "comparison", "n_items", "total_diff", "total_diff_pct", "ci_low",
                      "ci_high", "top10_share_of_abs_gap"], []),
    "newsvendor": (TABLES / "robustness_newsvendor.csv",
                   ["window", "policy", "holding_rate", "stockout_multiple", "critical_ratio", "simulated_cheapest_level"], []),
    "breakeven": (TABLES / "robustness_breakeven.csv", ["window", "policy", "breakeven_stockout_to_holding_ratio"], []),
    "rep_conclusions": (TABLES / "replication_conclusions.csv", ["state", "conclusion", "evidence", "holds"], []),
    "rep_scope": (TABLES / "replication_scope.csv", ["state", "category", "total_items", "ax_ay_candidates",
                                                    "unit_share_ax_ay"], []),
    "rep_breaks": (TABLES / "replication_breaks.csv", ["state", "category", "n_items", "cusum_break_rate",
                                                      "cusum_far_seasonal_null"], []),
    "rep_mase": (TABLES / "replication_mase.csv", ["state", "model", "n_items", "median_item_mase"], []),
    "rep_inventory": (TABLES / "replication_inventory.csv", ["state", "policy", "service_level", "n_items", "fill_rate",
                                                            "avg_inventory_value", "total_cost"], []),
    "kpi_all": (TABLES / "pbi_fact_inventory_kpi.csv",
                ["item_id", "category", "policy", "service_level", "lead_time", "fill_rate", "demand_units", "lost_units",
                 "stockout_days", "avg_inventory_value", "total_cost", "break_item"], []),
    "agreement": (TABLES / "break_method_agreement_summary.csv",
                  ["raw_agreement", "cohens_kappa", "within_4_wks", "within_8_wks", "within_13_wks", "n_items"], []),
    "policy_summary": (TABLES / "inventory_policy_summary.csv",
                       ["policy", "service_level", "fill_rate", "stockout_days_total", "avg_inventory_value",
                        "holding_cost", "stockout_cost", "total_cost"], []),
    "inventory_tests": (TABLES / "inventory_adaptive_vs_static_tests.csv",
                        ["lead_time", "service_level", "n_items", "median_cost_diff", "ci_low", "ci_high",
                         "total_cost_change_pct", "share_items_cheaper", "p_value"], []),
    "cost_sensitivity": (TABLES / "inventory_cost_sensitivity.csv",
                         ["lead_time", "holding_rate", "stockout_multiple", "policy", "service_level", "total_cost"], []),
}

# Exploration, classical models, stationarity and decomposition
SCHEMA.update({
    "weekday_index": (TABLES / "explore_weekday_index.csv", ["weekday", "FOODS", "HOBBIES", "HOUSEHOLD"], []),
    "month_index": (TABLES / "explore_month_index.csv", ["month", "FOODS", "HOBBIES", "HOUSEHOLD"], []),
    "calendar_effects": (TABLES / "explore_calendar_effects.csv", ["category", "snap_vs_other_days", "event_vs_normal_days",
                                                                   "event_vs_normal_no_christmas"], []),
    "price_correlation": (TABLES / "explore_price_correlation.csv", ["item_id", "spearman_units_vs_price"], []),
    "comparison_overview": (TABLES / "comparison_overview.csv", ["model", "n_items", "median_mase", "median_rmse", "median_mae",
                                                                 "median_smape", "median_wape", "median_mase_726",
                                                                 "beats_naive_share", "beats_best_share"], []),
    "classical_sample": (TABLES / "classical_sample_items.csv", ["item_id", "category", "break_item"], []),
    "stationarity_items": (TABLES / "stationarity_items.csv", ["item_id", "category", "adf_c", "kpss_c", "adf_ct", "kpss_ct",
                                                        "label", "d", "D"], []),
    "stationarity_summary": (TABLES / "stationarity_summary.csv", ["category", "n_items", "stationary", "trend-stationary",
                                                                  "difference-stationary", "conflicting", "d=0", "d=1", "D=1"], []),
    "stationarity_totals": (TABLES / "stationarity_aggregates.csv", ["series", "label", "d", "D", "adf_c", "kpss_c"], []),
    "data_quality": (TABLES / "data_quality.csv", ["category", "items", "raw_empty_cells", "leading_zero_days_trimmed",
                                             "missing_days_inside_series", "zero_days_kept_share", "outlier_days", "outlier_share"], []),
    "decomposition_choice": (TABLES / "decomposition_choice.csv", ["series", "level", "zero_handling", "spearman_level_vs_swing",
                                                           "rule_choice", "resid_cv_additive", "resid_cv_multiplicative",
                                                           "lower_residual"], []),
    "demo_series": (TABLES / "demo_series.parquet", ["item_id", "date", "sales_raw", "kept_after_trim"], ["date"]),
    "demo_decomposition": (TABLES / "demo_decomposition.parquet", ["series", "date", "observed", "trend", "seasonal", "resid"], ["date"]),
    "classical_orders": (TABLES / "classical_orders.csv", ["series", "level", "d", "D", "sarima_order", "sarima_sorder", "arima_order"], []),
    "classical_forecasts": (TABLES / "comparison_forecasts.parquet", ["item_id", "origin", "date", "model", "forecast", "actual", "category"],
                     ["origin", "date"]),
    "classical_item_metrics": (TABLES / "comparison_metrics_item_origin.csv", ["item_id", "origin", "model", "rmse", "mae", "mape",
                                                                   "smape", "wape", "mase", "category"], ["origin"]),
    "classical_summary": (TABLES / "comparison_metrics_summary.csv", ["category", "model", "n_items", "median_rmse", "median_mae",
                                                          "median_mape", "median_mase", "mape_excluded_share"], []),
    "classical_tests": (TABLES / "comparison_tests.csv", ["hypothesis", "model_a", "model_b", "n", "median_diff", "ci_low", "ci_high",
                                              "p_holm", "a_better"], []),
    "classical_h4": (TABLES / "classical_h4.csv", ["group", "n_items", "median_mase_hw_add", "median_mase_hw_mul"], []),
    "aggregate_metrics": (TABLES / "comparison_totals_metrics.csv", ["series", "model", "how", "rmse", "mae", "mape", "mase"], []),
    "classical_run_stats": (TABLES / "classical_run_stats.csv", ["model", "level", "fits", "fallback_rate", "not_converged_rate"], []),
    "residual_checks": (TABLES / "diagnostics_summary.csv", ["level", "model", "n", "ljung_box_pass", "normality_pass", "roots_ok"], []),
    "residual_checks_items": (TABLES / "diagnostics_items.csv", ["series", "model", "lb_p7", "lb_p14", "jb_p", "lb_pass", "jb_pass"], []),
    "classical_params": (TABLES / "parameters_items.csv", ["series", "model", "level"], []),
    "shift_aware_tests": (TABLES / "shift_aware_tests.csv", ["group", "variant", "n", "median_diff", "ci_low", "ci_high", "p_value"], []),
    "classical_stock_items": (TABLES / "stock_items.csv", ["item_id", "model", "method", "total_cost", "fill_rate", "avg_inventory_value"], []),
    "classical_stock": (TABLES / "stock_summary.csv", ["method", "model", "n_items", "fill_rate", "cycle_service_level",
                                            "avg_inventory_value", "total_cost"], []),
    "classical_stock_tests": (TABLES / "stock_tests.csv", ["method", "model_a", "model_b", "total_diff_pct", "total_ci_low",
                                                "total_ci_high"], []),
})

# Tables the planning pages read in full
SCHEMA.update({
    "hc_breaks_items": (TABLES / "hc_breaks_items.csv", ["item_id", "category", "cusum_break", "change_week", "alarm_week",
                                                        "direction", "high_confidence", "relative_change"],
                        ["change_week", "alarm_week"]),
    "shift_aware_items": (TABLES / "shift_aware_item_origin.csv", ["item_id", "origin", "variant", "mase_variant", "static"],
                          ["origin"]),
    "variants_items": (TABLES / "adaptation_variants_item_origin_mase.csv",
                       ["item_id", "origin", "static", "truncate", "blend", "has_break"], ["origin"]),
    "classical_726": (TABLES / "classical_726_scores.csv", ["item_id", "model", "mase", "category", "intermittent"], []),
    "funnel": (TABLES / "hc_breaks_funnel.csv", ["cusum_breaks", "4_high_confidence"], []),
})

# Time-series models (notebooks 07c to 07h)
SCHEMA.update({
    "ts_headline": (TABLES / "ts_headline.csv", ["model", "name", "kind", "rmse", "mae", "mape", "smape", "wape", "mase",
                                                 "val_rmse", "val_mae", "val_mape", "n_products"], []),
    "ts_ablation": (TABLES / "ts_sarima_ablation.csv", ["family", "step", "model", "kept", "val_rmse", "val_mae", "val_mape",
                                                        "scored_rmse", "scored_mae", "scored_mape"], []),
    "ts_tests": (TABLES / "ts_tests.csv", ["metric", "model", "median_diff", "p_holm", "better_than_sarima",
                                           "worse_than_sarima"], []),
    "varima_scores": (TABLES / "ts_varima_scores.csv", ["series", "model", "rmse", "mae", "mase"], []),
    "ts_forecasts": (TABLES / "ts_forecasts.parquet", ["item_id", "origin", "date", "model", "forecast", "actual", "category"],
                     ["origin", "date"]),
    "ts_combination_members": (TABLES / "ts_combination_members.csv", ["member", "validation_rmse"], []),
    "sarima_totals_rescaled": (TABLES / "sarima_totals_rescaled.csv", ["series", "origin", "scale", "mase"], []),
})

# Which model is best (notebook 07i)
SCHEMA.update({
    "verdict_metrics": (TABLES / "verdict_metrics.csv", ["group", "model", "n_products", "rmse", "mae", "mape", "smape",
                                                         "wape", "mase"], []),
    "verdict_ranks": (TABLES / "verdict_ranks.csv", ["model", "rank_rmse", "rank_mase", "avg_rank", "n_first", "n_top2",
                                                     "n_top3"], []),
    "verdict_tests": (TABLES / "verdict_tests.csv", ["metric", "reference", "model", "median_diff", "ci_low", "ci_high",
                                                     "gap_pct", "p_holm", "result"], []),
    "verdict_wins": (TABLES / "verdict_wins.csv", ["metric", "model", "wins"], []),
    "verdict_stock": (TABLES / "verdict_stock.csv", ["method", "model", "total_cost", "fill_rate", "avg_inventory_value"],
                      []),
})


class DataError(RuntimeError):
    """A required results file or column is missing."""


def read_table(path, required, dates=()):
    """Read one CSV/Parquet file and check its columns. Pure apart from file I/O, so it is unit-tested."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Missing results file: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}. "
            "Run the notebooks in order (python scripts/run_all.py) to regenerate it.")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing column(s): {', '.join(missing)}. Found: {', '.join(df.columns)}")
    for c in dates:
        df[c] = pd.to_datetime(df[c])
    return df


# Boolean flags that are empty for out-of-scope items (not modelled -> treated as "no")
BOOL_FILL = {"items": ["cusum_first_break", "break_at_last_origin", "intermittent"]}


@st.cache_data(show_spinner=False)
def load(name):
    path, required, dates = SCHEMA[name]
    df = read_table(path, required, dates)
    for c in BOOL_FILL.get(name, []):
        df[c] = df[c].fillna(False).astype(bool)
    if name == "items":
        # shift known at the forecast dates (the definition every forecasting test uses)
        df["has_shift"] = df["break_group"].eq("break detected")
    return df


def get(name):
    """Load a table, or stop the page with a clear error message."""
    try:
        return load(name)
    except DataError as e:
        st.error(f"**Data problem:** {e}")
        st.stop()
