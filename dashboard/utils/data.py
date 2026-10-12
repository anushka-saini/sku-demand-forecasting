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
