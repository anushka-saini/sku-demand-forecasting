"""What the Model Lab needs: a plain-language line per model and error tables cut by category and weekday."""
import pandas as pd
import streamlit as st

from utils.final import FILES, TABLES

# one line each, in plain words
ABOUT = {
    "seasonal_naive": "Repeats what sold on the same weekday last week. The simplest honest baseline.",
    "hw_add": "Holt-Winters: keeps a running level, trend and weekly pattern, and nudges them as new days come in.",
    "ets_boxcox": "Holt-Winters with a damped trend on a squared-down scale, which keeps big spikes from dominating.",
    "theta": "Splits the series into a long-run line and a short-run wiggle, forecasts each, and joins them.",
    "arma": "Predicts today from the last few days and the last few errors. No trend and no weekly pattern built in.",
    "arima": "Like ARMA but works on day-to-day changes, so it can follow a trend.",
    "sarima": "ARIMA with a weekly pattern added. The best classical model on this data.",
    "sarimax": "SARIMA that also reads outside signals: SNAP days, events and price.",
    "prophet": "Facebook's model: a smooth trend plus weekly and yearly curves and holiday bumps.",
    "prophet_tuned": "Prophet with its settings chosen on the validation windows and event days added.",
    "autots": "An automatic search that tries many simple models and keeps what fits.",
    "var": "Forecasts the three category totals together, then splits the result down to the items.",
    "combination": "The plain average of SARIMAX, Holt-Winters and Theta. Averaging cancels part of each model's mistakes.",
    "lightgbm": "A tree model that learns from rolling averages, same-weekday history, price and the calendar.",
    "xgboost": "A tree model like LightGBM with its settings searched on the validation windows. The leaderboard winner.",
    "lstm": "A neural network that reads the last weeks of sales as a sequence.",
    "gru": "A lighter neural network of the same kind as LSTM.",
    "patchtst": "A newer neural network that reads the history in short patches.",
    "hybrid": "A classical forecast plus an XGBoost model that learns the classical model's leftover errors.",
    "avg_top3": "The plain average of the three models that ranked best on the validation windows.",
    "stack": "A weighted mix of several models, with the weights fitted on the validation windows.",
}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(show_spinner=False)
def rows():
    """Every test item-day: forecast and actual for every model, with the item's category."""
    d = pd.read_parquet(TABLES / FILES["forecasts"], columns=["item_id", "origin", "date", "model", "forecast", "actual", "split"],
                        filters=[("split", "==", "test")]).drop(columns="split")
    d["cat"] = d["item_id"].str.split("_").str[0]
    return d


@st.cache_data(show_spinner=False)
def errors(models, level):
    """Error per row at the chosen level. 'item' scores each item-day; 'category' and 'total' first add the item forecasts and
    actuals up, so the totals are always the sum of the item-level forecasts."""
    d = rows()
    d = d[d["model"].isin(models)]
    if level == "item":
        g = d.assign(grp=d["cat"])
    elif level == "category":
        g = d.groupby(["model", "cat", "origin", "date"], as_index=False)[["forecast", "actual"]].sum().assign(grp=lambda x: x["cat"])
    else:
        g = d.groupby(["model", "origin", "date"], as_index=False)[["forecast", "actual"]].sum().assign(grp="All items")
    e = g["forecast"] - g["actual"]
    g = g.assign(ae=e.abs(), se=e ** 2, wd=g["date"].dt.dayofweek, h=(g["date"] - g["origin"]).dt.days)
    return g[["model", "grp", "wd", "h", "ae", "se"]]


def cut(g, by):
    """MAE and RMSE per model and group, pooled over the rows (every item-day, category-day or total-day counts once)."""
    t = g.groupby(["model", by]).agg(mae=("ae", "mean"), mse=("se", "mean")).reset_index()
    t["rmse"] = t["mse"] ** 0.5
    return t.drop(columns="mse")
