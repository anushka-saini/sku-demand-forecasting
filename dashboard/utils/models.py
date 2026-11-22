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

# the 12 models the project reports (VARIMA runs on the three category totals only)
MODELS = ["seasonal_naive", "hw_add", "ets_boxcox", "theta", "arma", "arima", "sarima", "sarimax", "varima", "prophet",
          "autots", "combination"]

NAMES = {"seasonal_naive": "Seasonal naive", "hw_add": "Holt-Winters additive", "arima": "ARIMA", "sarima": "SARIMA",
         BEST: BEST_NAME,
         # versions that react to a detected shift
         "sarima_restart": "SARIMA, retrained after a shift", "sarima_dummy": "SARIMA with a shift marker",
         "best_restart": f"{BEST_SHORT}, retrained after a shift",
         "best_blend": f"{BEST_SHORT}, blend of static and retrained",
         # the time-series models of 07c to 07h (07h stores the Box-Cox Holt-Winters as 'ets')
         "ets_boxcox": "Holt-Winters (damped, Box-Cox)", "sarimax": "SARIMAX", "theta": "Theta", "prophet": "Prophet",
         "autots": "AutoTS", "combination": "Combination of SARIMAX, Holt-Winters and Theta", "arma": "ARMA(p, q)",
         "varima": "VARIMA"}
SHORT = {"seasonal_naive": "Seasonal naive", "hw_add": "Holt-Winters additive", "arima": "ARIMA", "sarima": "SARIMA",
         BEST: BEST_SHORT, "ets_boxcox": "Holt-Winters (damped, Box-Cox)", "sarimax": "SARIMAX", "theta": "Theta",
         "prophet": "Prophet", "autots": "AutoTS", "combination": "Combination", "arma": "ARMA", "varima": "VARIMA"}
# if the best model shares a name with a classical one, mark the classical model's versions, which only ran on the
# 150-item sample, so the two retrained versions never read the same
_classical = next((k for k, v in SHORT.items() if v == BEST_SHORT and k != BEST), None)
if _classical:
    for k in [k for k in ("sarima_restart", "sarima_dummy") if k.startswith(f"{_classical}_")]:
        NAMES[k] += " (150-item sample)"

# Okabe-Ito colours: readable for colour-blind viewers and in both themes. A shift-aware version keeps its model's colour.
BASE_COLORS = {"seasonal_naive": "#8C8C8C", "hw_add": "#0072B2", "arima": "#CC79A7", "sarima": "#D55E00", BEST: "#E69F00",
               "ets_boxcox": "#5FB49C", "sarimax": "#A0410D", "theta": "#AA4499", "prophet": "#3A68C4",
               "autots": "#88CCEE", "combination": "#7F3C8D", "arma": "#C9B07A", "varima": "#56B4E9"}
COLORS = {**BASE_COLORS,
          **{k: BASE_COLORS[k.split("_")[0] if not k.startswith("seasonal") else "seasonal_naive"]
             for k in NAMES if k not in BASE_COLORS}}

ORDER = ["seasonal_naive", "hw_add", "arima", "sarima", BEST]
SAMPLE_ORDER = [m for m in ORDER if m not in SAMPLE_DUPLICATES]
SHIFT_AWARE = ["sarima_restart", "sarima_dummy", "best_restart", "best_blend"]
BLOCKS = ["arma", "arima"]                        # 07c: the steps below SARIMA
PICKED = SAMPLE_BEST                              # which of the eight 07h picked for the stock work
FAMILY = {"seasonal_naive": "Baseline", BEST: "Best time-series model",
          **{m: "Time-series contender" for m in CONTENDERS[1:]}, **{m: "Building block" for m in BLOCKS}}
ASSUMES = {
    "seasonal_naive": "Next week repeats last week. No parameters; the floor every model has to clear.",
    "ets_boxcox": "A slowly moving level, a trend that flattens out and a fixed weekly pattern, fitted after a Box-Cox "
                  "transform that evens out big and small days.",
    "hw_add": "Level, trend and a weekly pattern that adds the same number of units each week.",
    "arima": "After differencing, today depends on the last few days; no weekly term.",
    "sarima": "After differencing, today depends on the last few days and the same weekday a week ago.",
    "sarimax": "SARIMA errors around a yearly wave (one Fourier pair) that repeats every year.",
    "theta": "A long-run straight line plus short-run smoothing, after the weekly pattern is divided out.",
    "prophet": "A smooth trend plus weekly and yearly curves plus a Christmas effect; days independent given those.",
    "autots": "Whatever single model won AutoTS's own backtest on the 150 products at that forecast date.",
    "combination": "Errors of different models partly cancel, so their plain average misses by less.",
    "arma": "Sales move around a fixed mean. Its AR half uses the last p days, its MA half the last q surprises.",
    "varima": "The three category totals move together, so each helps forecast the others.",
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
