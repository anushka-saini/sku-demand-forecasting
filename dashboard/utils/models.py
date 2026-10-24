"""Model names, colours and order, shared by every page so each model looks the same everywhere."""
NAMES = {"seasonal_naive": "Seasonal naive", "ets": "Damped Holt-Winters (ETS)", "hw_add": "Holt-Winters additive",
         "hw_mul": "Holt-Winters multiplicative", "arima": "ARIMA", "sarima": "SARIMA", "tsb": "TSB",
         "lightgbm": "LightGBM",
         # versions that react to a detected shift
         "ets_retrained": "ETS, retrained after a shift", "tsb_retrained": "TSB, retrained after a shift",
         "sarima_restart": "SARIMA, retrained after a shift", "sarima_dummy": "SARIMA with a shift marker",
         "lightgbm_restart": "LightGBM, retrained after a shift", "lightgbm_downweight": "LightGBM, old data down-weighted",
         "lightgbm_shift_feature": "LightGBM with a shift feature", "lightgbm_blend": "LightGBM, blend of static and retrained"}
SHORT = {"seasonal_naive": "Seasonal naive", "ets": "ETS", "hw_add": "HW additive", "hw_mul": "HW multiplicative",
         "arima": "ARIMA", "sarima": "SARIMA", "tsb": "TSB", "lightgbm": "LightGBM"}

# Okabe-Ito colours: readable for colour-blind viewers and in both themes. A shift-aware version keeps its model's colour.
BASE_COLORS = {"seasonal_naive": "#8C8C8C", "ets": "#009E73", "hw_add": "#0072B2", "hw_mul": "#56B4E9",
               "arima": "#CC79A7", "sarima": "#D55E00", "tsb": "#B8A100", "lightgbm": "#E69F00"}
COLORS = {**BASE_COLORS,
          **{k: BASE_COLORS[k.split("_")[0] if not k.startswith("seasonal") else "seasonal_naive"]
             for k in NAMES if k not in BASE_COLORS}}

ORDER = ["seasonal_naive", "ets", "hw_add", "hw_mul", "arima", "sarima", "lightgbm"]
SHIFT_AWARE = ["ets_retrained", "tsb_retrained", "sarima_restart", "sarima_dummy", "lightgbm_restart",
               "lightgbm_downweight", "lightgbm_shift_feature", "lightgbm_blend"]
FAMILY = {"seasonal_naive": "Baseline", "lightgbm": "Machine learning"}
TOTALS = ["FOODS", "HOBBIES", "HOUSEHOLD", "CA_TOTAL"]

POLICY_NAMES = {"lightgbm_static": "LightGBM forecast", "lightgbm_adaptive": "LightGBM, retrained after shifts",
                "seasonal_naive_static": "Seasonal naive forecast"}
POLICY_COLORS = {"lightgbm_static": BASE_COLORS["lightgbm"], "lightgbm_adaptive": "#A06A00",
                 "seasonal_naive_static": BASE_COLORS["seasonal_naive"]}


def name(model):
    return NAMES.get(model, model)


def family(model):
    if model in SHIFT_AWARE:
        return "Reacts to shifts"
    return FAMILY.get(model, "Classical")
