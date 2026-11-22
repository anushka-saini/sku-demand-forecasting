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
