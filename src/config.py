"""Paths and fixed settings shared by the notebooks."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"          # every result table
FIGS = ROOT / "outputs" / "figures"          # every figure
CHECKPOINTS = PROCESSED / "classical_fits"     # per-item fits, so a rerun can pick up where it stopped

SEED = 42
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]

# Backtest: the same origins and horizon for every model (07a, 09a, 10)
HORIZON = 28
N_ORIGINS = 7
LAST_DAY = pd.Timestamp("2016-05-22")
ORIGINS = [LAST_DAY - pd.Timedelta(days=HORIZON * (N_ORIGINS - i)) for i in range(N_ORIGINS)]
EVAL_ORIGINS = ORIGINS[1:]                   # origin 0 is the warm-up

# Classical models
PERIOD = 7                                   # weekly season in daily data
FIT_DAYS = 730                               # two years: the yearly cycle twice, and fits stay fast
SAMPLE_PER_GROUP = 25                        # per category: 25 items with a break, 25 without

# Inventory base case
LEAD_TIME = 7
HOLDING_RATE = 0.25
STOCKOUT_MULTIPLE = 0.5

ALPHA = 0.05
PRACTICAL_SHARE = 0.01                       # H3: a variant must improve MASE by at least 1% of static SARIMA


# Slide-ready figures: white background, large text, 150 dpi
FIG_STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
             "savefig.dpi": 150, "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
             "axes.labelsize": 13, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False}
COLORS = {"seasonal_naive": "#8C8C8C", "ets": "#009E73", "hw_add": "#0072B2", "hw_mul": "#56B4E9",
          "arima": "#CC79A7", "sarima": "#D55E00", "actual": "#222222"}
NAMES = {"seasonal_naive": "Seasonal naive", "ets": "Damped Holt-Winters (ETS)", "hw_add": "Holt-Winters additive",
         "hw_mul": "Holt-Winters multiplicative", "arima": "ARIMA", "sarima": "SARIMA",
         # the time-series models of 07c to 07h (07h's Box-Cox ETS is renamed, as 07a already has an 'ets')
         "ets_boxcox": "Damped ETS, Box-Cox (tuned)", "sarimax": "SARIMAX", "theta": "Theta", "prophet": "Prophet",
         "autots": "AutoTS", "combination": "Combination", "combination_weighted": "Combination, inverse-error weights",
         "ar": "AR(p)", "ma": "MA(q)", "arma": "ARMA(p, q)", "arimax": "ARIMAX"}


def _best():
    """The best time-series model from 07h, which 09, 10 and 11 use for every product: (07h key, display name)."""
    import json
    p = TABLES / "ts_best_model.json"
    if not p.exists():
        return None, None
    b = json.loads(p.read_text())
    return b["model"], f'{b["name"]} (best time-series model)'   # so it never reads like the 150-product SARIMA


BEST = "best"                                # its forecasts are stored under this key, so they never clash with ets or sarima
BEST_MODEL, BEST_NAME = _best()
BEST_NAME = BEST_NAME or "Best time-series model"
NAMES[BEST] = BEST_NAME
COLORS[BEST] = "#E69F00"


def make_dirs():
    """Create the output folders used by the new notebooks."""
    for p in [TABLES, FIGS, CHECKPOINTS]:
        p.mkdir(parents=True, exist_ok=True)
