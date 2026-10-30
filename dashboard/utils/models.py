"""Model names, colours and order, shared by every page so each model looks the same everywhere."""
import json
from pathlib import Path

_BEST_FILE = Path(__file__).resolve().parents[2] / "outputs" / "tables" / "ts_best_model.json"
_BEST = json.loads(_BEST_FILE.read_text()) if _BEST_FILE.exists() else {"model": "best", "name": "Best time-series model"}
BEST = "best"                              # the key the winning time-series model is stored under in 09 to 11
BEST_NAME = f'{_BEST["name"]} (best time-series model)'   # what it is (from 07h); never reads like the 150-product SARIMA
BEST_SHORT = BEST_NAME.split(" (")[0]
# the eight time-series models compared head to head in 07h, all on the 150-item sample
CONTENDERS = ["seasonal_naive", "ets_boxcox", "sarima", "sarimax", "theta", "prophet", "autots", "combination"]
# On the 150-item sample the best model is the 07h contender itself (09 checks the forecasts are equal), so its sample
# rows repeat that contender's and the sample views show the contender only. 07h's 'ets' is called 'ets_boxcox' here.
_key = {"ets": "ets_boxcox"}.get(_BEST.get("model"), _BEST.get("model"))
SAME_ON_SAMPLE = _key if _key in CONTENDERS else None
SAMPLE_BEST = SAME_ON_SAMPLE or BEST
SAMPLE_DUPLICATES = {BEST: SAME_ON_SAMPLE} if SAME_ON_SAMPLE else {}
if SAME_ON_SAMPLE == "sarima":           # 08d retrained the same SARIMA, so its retrained version repeats too
    SAMPLE_DUPLICATES["best_restart"] = "sarima_restart"

NAMES = {"seasonal_naive": "Seasonal naive", "ets": "Damped Holt-Winters (ETS)", "hw_add": "Holt-Winters additive",
         "hw_mul": "Holt-Winters multiplicative", "arima": "ARIMA", "sarima": "SARIMA", "tsb": "TSB",
         BEST: BEST_NAME,
         # versions that react to a detected shift
         "ets_retrained": "ETS, retrained after a shift", "tsb_retrained": "TSB, retrained after a shift",
         "sarima_restart": "SARIMA, retrained after a shift", "sarima_dummy": "SARIMA with a shift marker",
         "best_restart": f"{BEST_SHORT}, retrained after a shift",
         "best_blend": f"{BEST_SHORT}, blend of static and retrained",
         # the time-series models of 07c to 07h (07h's ETS is the Box-Cox version, kept apart from the 07a ETS)
         "ets_boxcox": "Holt-Winters (damped ETS, Box-Cox)", "sarimax": "SARIMAX", "theta": "Theta", "prophet": "Prophet",
         "autots": "AutoTS", "combination": "Combination of SARIMAX, Holt-Winters and Theta",
         "combination_weighted": "Combination, inverse-error weights",
         "ar": "AR(p)", "ma": "MA(q)", "arma": "ARMA(p, q)", "arimax": "ARIMAX"}
SHORT = {"seasonal_naive": "Seasonal naive", "ets": "ETS", "hw_add": "HW additive", "hw_mul": "HW multiplicative",
         "arima": "ARIMA", "sarima": "SARIMA", "tsb": "TSB", BEST: BEST_SHORT,
         "ets_boxcox": "Holt-Winters (damped ETS, Box-Cox)", "sarimax": "SARIMAX", "theta": "Theta", "prophet": "Prophet", "autots": "AutoTS",
         "combination": "Combination", "combination_weighted": "Weighted combination",
         "ar": "AR", "ma": "MA", "arma": "ARMA", "arimax": "ARIMAX"}
# if the best model shares a name with a classical one, mark the classical model's versions, which only ran on the
# 150-item sample, so the two retrained versions never read the same
_classical = next((k for k, v in SHORT.items() if v == BEST_SHORT and k != BEST), None)
if _classical:
    for k in [k for k in ("sarima_restart", "sarima_dummy", "ets_retrained", "tsb_retrained") if k.startswith(f"{_classical}_")]:
        NAMES[k] += " (150-item sample)"

# Okabe-Ito colours: readable for colour-blind viewers and in both themes. A shift-aware version keeps its model's colour.
BASE_COLORS = {"seasonal_naive": "#8C8C8C", "ets": "#009E73", "hw_add": "#0072B2", "hw_mul": "#56B4E9",
               "arima": "#CC79A7", "sarima": "#D55E00", "tsb": "#B8A100", BEST: "#E69F00",
               "ets_boxcox": "#5FB49C", "sarimax": "#A0410D", "theta": "#AA4499", "prophet": "#3A68C4",
               "autots": "#88CCEE", "combination": "#7F3C8D", "combination_weighted": "#B07AA1",
               "ar": "#C9B07A", "ma": "#C9B07A", "arma": "#C9B07A", "arimax": "#C9B07A"}
COLORS = {**BASE_COLORS,
          **{k: BASE_COLORS[k.split("_")[0] if not k.startswith("seasonal") else "seasonal_naive"]
             for k in NAMES if k not in BASE_COLORS}}

ORDER = ["seasonal_naive", "ets", "hw_add", "hw_mul", "arima", "sarima", BEST]
SAMPLE_ORDER = [m for m in ORDER if m not in SAMPLE_DUPLICATES]
SHIFT_AWARE = ["ets_retrained", "tsb_retrained", "sarima_restart", "sarima_dummy", "best_restart", "best_blend"]
BLOCKS = ["ar", "ma", "arma", "arima", "arimax"]   # 07c: the parts SARIMA is built from (07a's ARIMA is the same)
PICKED = SAMPLE_BEST                              # which of the eight 07h picked for the stock work
FAMILY = {"seasonal_naive": "Baseline", BEST: "Best time-series model",
          **{m: "Time-series contender" for m in CONTENDERS[1:]}, **{m: "Building block" for m in BLOCKS},
          "combination_weighted": "Combination, second version"}
ASSUMES = {
    "seasonal_naive": "Next week repeats last week. No parameters; the floor every model has to clear.",
    "ets": "A slowly moving level, a trend that flattens out, and a fixed weekly pattern on top (07a, raw units).",
    "ets_boxcox": "Like ETS, fitted after a Box-Cox transform that evens out big and small days.",
    "hw_add": "Level, trend and a weekly pattern that adds the same number of units each week.",
    "hw_mul": "Level, trend and a weekly pattern that scales with the level.",
    "arima": "After differencing, today depends on the last few days; no weekly term.",
    "sarima": "After differencing, today depends on the last few days and the same weekday a week ago.",
    "sarimax": "SARIMA errors around a yearly wave (one Fourier pair) that repeats every year.",
    "theta": "A long-run straight line plus short-run smoothing, after the weekly pattern is divided out.",
    "prophet": "A smooth trend plus weekly and yearly curves plus a Christmas effect; days independent given those.",
    "autots": "Whatever single model won AutoTS's own backtest on the 150 products at that forecast date.",
    "combination": "Errors of different models partly cancel, so their plain average misses by less.",
    "combination_weighted": "Like the combination, but members that did better on a product at validation count more.",
    "ar": "Sales move around a fixed mean, and today depends only on the last p days.",
    "ma": "Sales move around a fixed mean, and today depends on the last q days' surprises.",
    "arma": "Both of the above together, still around a fixed mean.",
    "arimax": "ARIMA errors around the same yearly wave as SARIMAX.",
    "tsb": "Built for items that often sell nothing: forecasts how often a sale happens and how big it is.",
}
TOTALS = ["FOODS", "HOBBIES", "HOUSEHOLD", "CA_TOTAL"]

POLICY_NAMES = {f"{BEST}_static": f"{BEST_SHORT} forecast", f"{BEST}_adaptive": f"{BEST_SHORT}, retrained after shifts",
                "seasonal_naive_static": "Seasonal naive forecast"}
POLICY_COLORS = {f"{BEST}_static": BASE_COLORS[BEST], f"{BEST}_adaptive": "#A06A00",
                 "seasonal_naive_static": BASE_COLORS["seasonal_naive"]}


def name(model):
    return NAMES.get(model, model)


def family(model):
    if model in SHIFT_AWARE:
        return "Reacts to shifts"
    return FAMILY.get(model, "Classical")
