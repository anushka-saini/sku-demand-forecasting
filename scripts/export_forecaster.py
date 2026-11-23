"""Small files for the Forecaster page. Nothing is refitted: it only reads the saved final forecasts.

Writes to outputs/tables:
  final_forecaster_bands.parquet    80% band per item, day and model on the test windows
  final_forecaster_sales.parquet    daily sales of the 150 items from Aug 2015 on, for the history chart
  final_forecaster_summary.csv      one row per model: how it did over all item-windows

Then checks that the numbers match the leaderboard before it finishes.
Run from the project folder: python scripts/export_forecaster.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
sys.path.insert(0, str(ROOT / "dashboard"))
from utils.final import MAIN as SHOWN  # the same list of models the dashboard shows

# GRU is left out: its leaderboard row is the mean of three seeds, but the saved forecast is the three seeds averaged,
# which scores differently, so the page could not agree with the leaderboard for it
MAIN = [m for m in SHOWN if m != "gru"]

BEST = "xgboost"          # the leaderboard winner; its band is the GARCH band from the volatility page
Z80 = 1.2816


def rmse(a, f):
    return float(np.sqrt(np.mean((a - f) ** 2)))


fc = pd.read_parquet(T / "final_forecasts.parquet")
fc = fc[fc["model"].isin(MAIN)]
test, val = fc[fc["split"] == "test"].copy(), fc[fc["split"] == "validation"]

# bands: for XGBoost the GARCH band already saved; for the rest the 80th percentile of that item's
# own errors on the validation windows (so it only uses days before the test starts)
q = (val.assign(e=(val["actual"] - val["forecast"]).abs()).groupby(["item_id", "model"])["e"].quantile(0.8).rename("q").reset_index())
test = test.merge(q, on=["item_id", "model"], how="left")
test["lo"] = (test["forecast"] - test["q"]).clip(lower=0)
test["hi"] = test["forecast"] + test["q"]
iv = pd.read_parquet(T / "final_intervals.parquet")[["item_id", "date", "sig_garch"]]
test = test.merge(iv, on=["item_id", "date"], how="left")
g = (test["model"] == BEST)
test.loc[g, "lo"] = (test.loc[g, "forecast"] - Z80 * test.loc[g, "sig_garch"]).clip(lower=0)
test.loc[g, "hi"] = test.loc[g, "forecast"] + Z80 * test.loc[g, "sig_garch"]
assert test[["lo", "hi"]].notna().all().all()
test[["item_id", "origin", "date", "model", "lo", "hi"]].to_parquet(T / "final_forecaster_bands.parquet", index=False)

# daily sales for the history chart (the same series the models were scored on)
parts = [pd.read_csv(ROOT / "data" / "interim" / f"{c}_candidates.csv", parse_dates=["date"]) for c in ("foods", "hobbies", "household")]
sales = pd.concat(parts)
sales = sales[sales["item_id"].isin(fc["item_id"].unique()) & (sales["date"] >= "2015-08-01")][["item_id", "date", "sales"]]
chk = sales.merge(test[test["model"] == "seasonal_naive"][["item_id", "date", "actual"]], on=["item_id", "date"])
assert np.allclose(chk["sales"], chk["actual"]), "history file disagrees with the saved actuals"
sales.to_parquet(T / "final_forecaster_sales.parquet", index=False)

# one row per item, window and model, for the first 7, 14 and all 28 days of each window
test["ae"] = (test["actual"] - test["forecast"]).abs()
test["se"] = (test["actual"] - test["forecast"]) ** 2
test["inside"] = (test["actual"] >= test["lo"]) & (test["actual"] <= test["hi"])
test["day"] = (test["date"] - test["origin"]).dt.days
rows = []
for h in (7, 14, 28):
    w = test[test["day"] <= h].groupby(["item_id", "origin", "model"]).agg(
        mae=("ae", "mean"), mse=("se", "mean"), cover=("inside", "mean")).reset_index()
    w["rmse"] = np.sqrt(w["mse"])
    mae_w = w.pivot_table(index=["item_id", "origin"], columns="model", values="mae")
    for m in MAIN:
        x = w[w["model"] == m]
        # leaderboard way: each item's mean over its six windows, then the median over items
        r = {"model": m, "h": h, "item_windows": len(x), "median_mae": x["mae"].median(), "median_rmse": x["rmse"].median(),
             "coverage": x["cover"].mean(), "lb_mae": x.groupby("item_id")["mae"].mean().median(),
             "lb_rmse": x.groupby("item_id")["rmse"].mean().median()}
        for base in ("seasonal_naive", "sarima"):
            r[f"beat_{base}"] = np.nan if m == base else float((mae_w[m] < mae_w[base]).mean())
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(T / "final_forecaster_summary.csv", index=False)

# check against the leaderboard
lb = pd.read_csv(T / "final_leaderboard.csv").set_index("model")
for _, r in summ[summ["h"] == 28].iterrows():
    assert abs(r["lb_mae"] - lb.loc[r["model"], "mae"]) < 1e-9, (r["model"], "mae")
    assert abs(r["lb_rmse"] - lb.loc[r["model"], "rmse"]) < 1e-9, (r["model"], "rmse")
gi = pd.read_csv(T / "final_garch_intervals.csv")
want = float(gi.loc[gi["interval"].str.startswith("GARCH"), "coverage"].iloc[0])
got = float(test[test["model"] == BEST]["inside"].mean())
assert abs(got - want) < 1e-9, ("band coverage", got, want)
print(summ[summ["h"] == 28].round(3).to_string(index=False))
print(f"matches the leaderboard MAE and RMSE for all {len(MAIN)} models; {BEST} band coverage {got:.4f} = volatility page")
