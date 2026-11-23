"""Loaders for the final comparison (notebook 13). Everything here reads files from outputs/tables/final_*; nothing is
modelled in the app."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

TABLES = Path(__file__).resolve().parents[2] / "outputs" / "tables"
FILES = {"leaderboard": "final_leaderboard.csv", "forecasts": "final_forecasts.parquet", "horizon": "final_horizon.csv",
         "weights": "final_stack_weights.csv", "dm": "final_dm_tests.csv", "segments": "final_segments.csv",
         "garch": "final_garch_intervals.csv", "daytype": "final_garch_daytype.csv", "archlm": "final_garch_archlm.csv",
         "intervals": "final_intervals.parquet", "stock": "final_stock_summary.csv", "stock_cmp": "final_stock_compare.csv",
         "totals": "final_totals_level.csv", "split": "final_split.csv", "var": "final_var_settings.csv",
         "validation": "final_validation.csv", "stock_items": "final_stock_items.csv"}

# colours for the models, the same ones the figures use (Okabe-Ito based)
COL = {"seasonal_naive": "#8C8C8C", "hw_add": "#0072B2", "ets_boxcox": "#5FB49C", "theta": "#AA4499", "arma": "#C9B07A",
       "arima": "#CC79A7", "sarima": "#D55E00", "sarimax": "#A0410D", "prophet": "#88CCEE", "autots": "#999933",
       "combination": "#7F3C8D", "prophet_tuned": "#3A68C4", "xgboost": "#009E73", "lightgbm": "#56B4E9", "lstm": "#E69F00",
       "gru": "#F0C070", "patchtst": "#117733", "var": "#332288", "avg_top3": "#882255", "stack": "#000000", "hybrid": "#44AA99"}


# the medium set shown by default: a baseline, the classical models that matter, the tree models, one neural net and the
# best combination. The full 21-model table is one click away on the leaderboard page.
MAIN = ["seasonal_naive", "sarima", "prophet", "combination", "xgboost", "lightgbm", "gru", "stack"]


def ready():
    return all((TABLES / f).exists() for f in FILES.values()) and (TABLES / "final_best_model.json").exists()


@st.cache_data(show_spinner=False)
def load(key):
    p = TABLES / FILES[key]
    d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    for c in ("origin", "date"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c])
    return d


@st.cache_data(show_spinner=False)
def best_info():
    return json.loads((TABLES / "final_best_model.json").read_text())


def names(lb):
    return dict(zip(lb["model"], lb["name"]))


def need_results():
    """Stop the page with a plain message if notebook 13 has not been run."""
    if not ready():
        st.info("The final comparison has not been run yet. Run src/run_models.py for each model and then notebook 13 to create "
                "the files this page reads from outputs/tables/.")
        st.stop()


@st.cache_data(show_spinner=False)
def text():
    """The written findings (src/final_text.py), built from the same tables the pages show."""
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
    import final_text
    return final_text.Final().all()


# ---------------------------------------------------------------- the Forecaster page
# Written so that sales after a window's start date are only read when you press Reveal: the forecast and band
# loaders never ask for the actual column, and the history loader stops at the forecast origin.
FC_FILES = {"bands": "final_forecaster_bands.parquet", "sales": "final_forecaster_sales.parquet",
            "summary": "final_forecaster_summary.csv"}
FC_MODELS = [m for m in MAIN if m != "gru"]       # GRU is out: its leaderboard row is a mean over seeds, not one forecast


def forecaster_ready():
    return ready() and all((TABLES / f).exists() for f in FC_FILES.values())


def _read(name, cols, flt):
    return pd.read_parquet(TABLES / name, columns=cols, filters=flt)


@st.cache_data(show_spinner=False)
def fc_index():
    """Every item and test window, read without any sales numbers."""
    d = _read(FILES["forecasts"], ["item_id", "origin"], [("split", "==", "test"), ("model", "==", "seasonal_naive")])
    return d.drop_duplicates().sort_values(["item_id", "origin"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fc_window(item, origin, model):
    """The 28-day forecast and its 80% band. No actuals."""
    o = pd.Timestamp(origin)
    f = _read(FILES["forecasts"], ["date", "forecast"], [("split", "==", "test"), ("model", "==", model),
                                                         ("item_id", "==", item), ("origin", "==", o)])
    b = _read(FC_FILES["bands"], ["date", "lo", "hi"], [("model", "==", model), ("item_id", "==", item), ("origin", "==", o)])
    return f.merge(b, on="date").sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fc_history(item, origin, days=90):
    """The last `days` days of sales up to and including the origin, never later."""
    o = pd.Timestamp(origin)
    d = _read(FC_FILES["sales"], ["date", "sales"], [("item_id", "==", item), ("date", "<=", o)])
    return d.sort_values("date").tail(days).reset_index(drop=True)


def fc_actuals(item, origin, models):
    """Called only after Reveal: forecast and actual for the given models over one window, not cached."""
    o = pd.Timestamp(origin)
    return _read(FILES["forecasts"], ["date", "model", "forecast", "actual"],
                 [("split", "==", "test"), ("item_id", "==", item), ("origin", "==", o), ("model", "in", list(models))])


@st.cache_data(show_spinner=False)
def fc_summary():
    """One row per model and horizon length (7, 14 or 28 days)."""
    return pd.read_csv(TABLES / FC_FILES["summary"]).set_index(["model", "h"])


def fc_past_errors(item, origin, model):
    """Forecast minus actual for every earlier window of this item (validation and earlier test windows). Strictly before
    the origin, so it holds nothing from the window being forecast."""
    o = pd.Timestamp(origin)
    d = _read(FILES["forecasts"], ["origin", "date", "forecast", "actual"],
              [("model", "==", model), ("item_id", "==", item), ("origin", "<", o)])
    return [g.sort_values("date").pipe(lambda x: (x["actual"] - x["forecast"]).to_numpy()) for _, g in d.groupby("origin")]


def fc_flags(item, origin):
    """Event and SNAP flags for one window. Only read after Reveal."""
    o = pd.Timestamp(origin)
    return _read(FILES["intervals"], ["date", "snap", "event"], [("item_id", "==", item), ("origin", "==", o)])
