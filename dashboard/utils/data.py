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
               "break_at_last_origin", "last_break_week", "intermittent", "adf_p_value", "is_stationary"],
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
    "model_mase": (TABLES / "model_mase_by_category.csv", ["category", "ets", "lightgbm", "seasonal_naive"], []),
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

# Milestone 2 (notebooks 12-16)
M2 = ROOT / "outputs" / "m2" / "tables"
SCHEMA.update({
    "m2_sample": (M2 / "m2_sample_items.csv", ["item_id", "category", "break_item"], []),
    "m2_stationarity": (M2 / "stationarity_items.csv", ["item_id", "category", "adf_c", "kpss_c", "adf_ct", "kpss_ct",
                                                        "label", "d", "D"], []),
    "m2_stationarity_summary": (M2 / "stationarity_summary.csv", ["category", "n_items", "stationary", "trend-stationary",
                                                                  "difference-stationary", "conflicting", "d=0", "d=1", "D=1"], []),
    "m2_stationarity_totals": (M2 / "stationarity_aggregates.csv", ["series", "label", "d", "D", "adf_c", "kpss_c"], []),
    "m2_quality": (M2 / "data_quality.csv", ["category", "items", "raw_empty_cells", "leading_zero_days_trimmed",
                                             "missing_days_inside_series", "zero_days_kept_share", "outlier_days", "outlier_share"], []),
    "m2_decomp_choice": (M2 / "decomposition_choice.csv", ["series", "level", "zero_handling", "spearman_level_vs_swing",
                                                           "rule_choice", "resid_cv_additive", "resid_cv_multiplicative",
                                                           "lower_residual"], []),
    "m2_demo_series": (M2 / "demo_series.parquet", ["item_id", "date", "sales_raw", "kept_after_trim"], ["date"]),
    "m2_demo_decomp": (M2 / "demo_decomposition.parquet", ["series", "date", "observed", "trend", "seasonal", "resid"], ["date"]),
    "m2_orders": (M2 / "classical_orders.csv", ["series", "level", "d", "D", "sarima_order", "sarima_sorder", "arima_order"], []),
    "m2_forecasts": (M2 / "classical_forecasts.parquet", ["item_id", "origin", "date", "model", "forecast", "actual", "category"],
                     ["origin", "date"]),
    "m2_item_metrics": (M2 / "classical_metrics_item_origin.csv", ["item_id", "origin", "model", "rmse", "mae", "mape",
                                                                   "smape", "wape", "mase", "category"], ["origin"]),
    "m2_summary": (M2 / "classical_metrics_summary.csv", ["category", "model", "n_items", "median_rmse", "median_mae",
                                                          "median_mape", "median_mase", "mape_excluded_share"], []),
    "m2_tests": (M2 / "classical_tests.csv", ["hypothesis", "model_a", "model_b", "n", "median_diff", "ci_low", "ci_high",
                                              "p_holm", "a_better"], []),
    "m2_h4": (M2 / "classical_h4.csv", ["group", "n_items", "median_mase_hw_add", "median_mase_hw_mul"], []),
    "m2_aggregate": (M2 / "aggregate_metrics.csv", ["series", "model", "how", "rmse", "mae", "mape", "mase"], []),
    "m2_run_stats": (M2 / "classical_run_stats.csv", ["model", "level", "fits", "fallback_rate", "not_converged_rate"], []),
    "m2_diagnostics": (M2 / "diagnostics_summary.csv", ["level", "model", "n", "ljung_box_pass", "normality_pass", "roots_ok"], []),
    "m2_diag_items": (M2 / "diagnostics_items.csv", ["series", "model", "lb_p7", "lb_p14", "jb_p", "lb_pass", "jb_pass"], []),
    "m2_params": (M2 / "parameters_items.csv", ["series", "model", "level"], []),
    "m2_shift": (M2 / "shift_aware_tests.csv", ["group", "variant", "n", "median_diff", "ci_low", "ci_high", "p_value"], []),
    "m2_stock_items": (M2 / "stock_items.csv", ["item_id", "model", "method", "total_cost", "fill_rate", "avg_inventory_value"], []),
    "m2_stock": (M2 / "stock_summary.csv", ["method", "model", "n_items", "fill_rate", "cycle_service_level",
                                            "avg_inventory_value", "total_cost"], []),
    "m2_stock_tests": (M2 / "stock_tests.csv", ["method", "model_a", "model_b", "total_diff_pct", "total_ci_low",
                                                "total_ci_high"], []),
})


class DataError(RuntimeError):
    """A required results file or column is missing."""


def read_table(path, required, dates=()):
    """Read one CSV/Parquet file and check its columns. Pure apart from file I/O, so it is unit-tested."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Missing results file: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}. "
            "Run the notebooks in order (01 to 05) to regenerate it.")
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
    return df


def get(name):
    """Load a table, or stop the page with a clear error message."""
    try:
        return load(name)
    except DataError as e:
        st.error(f"**Data problem:** {e}")
        st.stop()
