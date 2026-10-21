"""Small shared pieces for the Milestone 2 pages."""
NAMES = {"seasonal_naive": "Seasonal naive", "ets": "Damped Holt-Winters (ETS)", "hw_add": "Holt-Winters additive",
         "hw_mul": "Holt-Winters multiplicative", "arima": "ARIMA", "sarima": "SARIMA", "lightgbm": "LightGBM"}
COLORS = {"seasonal_naive": "#8C8C8C", "ets": "#009E73", "hw_add": "#0072B2", "hw_mul": "#56B4E9",
          "arima": "#CC79A7", "sarima": "#D55E00", "lightgbm": "#E69F00"}
ORDER = ["seasonal_naive", "ets", "hw_add", "hw_mul", "arima", "sarima", "lightgbm"]
TOTALS = ["FOODS", "HOBBIES", "HOUSEHOLD", "CA_TOTAL"]


def name(model):
    return NAMES.get(model, model)
