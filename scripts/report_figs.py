"""Figures for the IEEE report, drawn from the saved result files at the size they are printed (single column 3.45 in,
double column 7.1 in), so the text is readable at 100%. Run by scripts/build_report.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import EVAL_ORIGINS, FIGS, INTERIM, TABLES  # noqa: E402
import ts_models as tm  # noqa: E402

OUT = ROOT / "outputs" / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.linewidth": 0.6, "lines.linewidth": 1.0})
COL = {"seasonal_naive": "#8C8C8C", "hw_add": "#0072B2", "ets_boxcox": "#5FB49C", "theta": "#AA4499", "arma": "#C9B07A", "arima": "#CC79A7",
       "sarima": "#D55E00", "sarimax": "#A0410D", "prophet": "#88CCEE", "autots": "#999933", "combination": "#7F3C8D", "prophet_tuned": "#3A68C4",
       "xgboost": "#009E73", "lightgbm": "#56B4E9", "lstm": "#E69F00", "gru": "#F0C070", "patchtst": "#117733", "var": "#332288",
       "avg_top3": "#882255", "stack": "#000000", "hybrid": "#44AA99", "actual": "#222222"}
SHORT = {"seasonal_naive": "Seasonal naive", "hw_add": "Holt-Winters", "ets_boxcox": "HW damped Box-Cox", "theta": "Theta", "arma": "ARMA", "arima": "ARIMA",
         "sarima": "SARIMA", "sarimax": "SARIMAX", "prophet": "Prophet", "autots": "AutoTS", "combination": "Combination (SARIMAX, HW, Theta)",
         "prophet_tuned": "Prophet (tuned)", "xgboost": "XGBoost", "lightgbm": "LightGBM", "lstm": "LSTM", "gru": "GRU", "patchtst": "PatchTST",
         "var": "VAR (top-down)", "avg_top3": "Average of top 3", "stack": "Stacked", "hybrid": "Hybrid"}


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return p


def lb_items():
    lb = pd.read_csv(TABLES / "final_leaderboard.csv")
    return lb[lb["level"] == "item"].sort_values("rank").reset_index(drop=True)


def dataset(sample_ids):
    long_df = pd.concat([pd.read_csv(INTERIM / f"{c}_candidates.csv", parse_dates=["date"]) for c in ["foods", "hobbies", "household"]])
    d = long_df[long_df["item_id"].isin(sample_ids)]
    tot = d.pivot_table(index="date", columns=d["item_id"].str.split("_").str[0], values="sales", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(7.1, 2.0))
    for c, col in zip(tot.columns, ["#0072B2", "#D55E00", "#009E73"]):
        ax.plot(tot.index, tot[c].rolling(7).mean(), color=col, lw=0.8, label=c.title())
    ax.axvspan(tm.VAL_ORIGINS[0] + pd.Timedelta(days=1), tm.VAL_ORIGINS[-1] + pd.Timedelta(days=28), color="#E69F00", alpha=0.3, lw=0, label="validation windows")
    ax.axvspan(EVAL_ORIGINS[0] + pd.Timedelta(days=1), EVAL_ORIGINS[-1] + pd.Timedelta(days=28), color="#999999", alpha=0.3, lw=0, label="test windows")
    ax.set_ylabel("units a day, 50 items per group\n(7-day mean)")
    ax.legend(ncol=5, frameon=False, loc="upper left")
    return save(fig, "dataset.png"), float((d["sales"] == 0).mean()), d["date"].min(), d["date"].max(), d["item_id"].nunique()


def workflow():
    fig, ax = plt.subplots(figsize=(7.1, 1.25))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = ["M5 sales,\ncalendar, prices", "Scope: 150 items,\nCalifornia, daily", "Shared features\n(past only +\nknown ahead)", "Models: classical,\nML, deep, VAR", "Tune on 3\nvalidation windows", "Test once; ensembles,\nDM test, stock"]
    w, gap = 0.145, 0.026
    for k, t in enumerate(boxes):
        x = k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.15), w, 0.7, boxstyle="round,pad=0.005,rounding_size=0.02", fc="#EAF2F8", ec="#1F6F8B", lw=0.8))
        ax.text(x + w / 2, 0.5, t, ha="center", va="center", fontsize=6.8)
        if k < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap, 0.5), xytext=(x + w, 0.5), arrowprops=dict(arrowstyle="->", color="#1F6F8B", lw=0.9))
    return save(fig, "workflow.png")


def bars(lb):
    d = lb.sort_values("rank")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), sharey=True)
    for ax, k, t in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"]):
        y = np.arange(len(d))
        ax.barh(y, d[k], color=[COL[m] for m in d["model"]], height=0.68, xerr=d[f"{k}_std"].fillna(0), error_kw={"lw": 0.6, "capsize": 1.2})
        ax.set_yticks(y, [SHORT[m] for m in d["model"]])
        ax.set_xlabel(f"median {t} per item (units a day)")
        ax.set_xlim(d[k].min() * 0.8, d[k].max() * 1.12)
        for i, v in enumerate(d[k]):
            ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=6)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.5)
    return save(fig, "bars.png")


def overlay(items):
    f = pd.read_parquet(TABLES / "final_forecasts.parquet")
    f = f[f["split"] == "test"]
    last = f["date"].max()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))
    for ax, it in zip(axes, items):
        g = f[(f["item_id"] == it) & (f["date"] > last - pd.Timedelta(days=56))]
        a = g[g["model"] == "xgboost"].sort_values("date")
        ax.plot(a["date"], a["actual"], color="#222222", lw=1.1, label="actual")
        for m in ["seasonal_naive", "combination", "xgboost"]:
            x = g[g["model"] == m].sort_values("date")
            ax.plot(x["date"], x["forecast"], color=COL[m], lw=0.8, label=SHORT[m].split(" (")[0])
        ax.set_title(it, loc="left")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel("units a day")
    axes[0].legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout(w_pad=0.6)
    return save(fig, "overlay.png")


def horizon(models):
    h = pd.read_csv(TABLES / "final_horizon.csv")
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True)
    for a, k, t in zip(ax, ["mae", "rmse"], ["MAE", "RMSE"]):
        for m in models:
            g = h[h["model"] == m]
            a.plot(g["h"], g[k], color=COL[m], lw=0.9, label=SHORT[m].split(" (")[0])
        a.set_ylabel(f"{t}, units a day")
    ax[1].set_xlabel("days ahead")
    ax[0].legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(h_pad=0.4)
    return save(fig, "horizon.png")


def weights(w):
    w = w[w["weight"] > 1e-6].sort_values("weight")
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    key = w.columns[0]
    ax.barh([SHORT[m] for m in w[key]], w["weight"], color=[COL[m] for m in w[key]], height=0.6)
    for i, v in enumerate(w["weight"]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=6.3)
    ax.set_xlabel("stacking weight (validation windows only)")
    ax.set_xlim(0, w["weight"].max() * 1.18)
    return save(fig, "weights.png")


def resid(model="xgboost"):
    f = pd.read_parquet(TABLES / "final_forecasts.parquet")
    g = f[(f["model"] == model) & (f["split"] == "test")].copy()
    g["e"] = g["actual"] - g["forecast"]
    from statsmodels.tsa.stattools import acf
    from scipy import stats
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    lim = np.quantile(np.abs(g["e"]), 0.995)
    ax[0].hist(g["e"].clip(-lim, lim), bins=70, color=COL[model])
    ax[0].axvline(0, color="#222222", lw=0.6)
    ax[0].set_xlabel("error: actual - forecast (units a day)")
    ax[0].set_ylabel("item-days")
    q = stats.probplot(g["e"] / g["e"].std(), dist="norm", fit=False)
    ax[1].plot(q[0], q[1], ".", ms=1.2, color=COL[model])
    lo, hi = -4, 4
    ax[1].plot([lo, hi], [lo, hi], color="#B0463C", lw=0.7)
    ax[1].set_xlabel("normal quantiles")
    ax[1].set_ylabel("standardised errors")
    day = g.groupby("date")["e"].sum().sort_index().to_numpy()
    r = acf(day, nlags=28)
    ax[2].bar(range(1, 29), r[1:], color=COL[model], width=0.7)
    b = 1.96 / np.sqrt(len(day))
    ax[2].axhline(b, color="#8C8C8C", ls="--", lw=0.6)
    ax[2].axhline(-b, color="#8C8C8C", ls="--", lw=0.6)
    ax[2].set_xlabel("lag in days (daily total error)")
    ax[2].set_ylabel("autocorrelation")
    fig.tight_layout(w_pad=0.8)
    return save(fig, "resid.png"), float(r[1]), float(b)


def failures(models=("xgboost", "combination")):
    s = pd.read_csv(TABLES / "final_segments.csv")
    keep = ["weekend", "event days", "SNAP days", "days with zero sales", "zero week before origin", "items with a detected shift", "all days"]
    p = s[s["segment"].isin(keep) & s["model"].isin(list(models) + ["seasonal_naive"])].pivot(index="segment", columns="model", values="mae").reindex(keep)
    rel = p[list(models)].div(p["seasonal_naive"], axis=0)
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    y = np.arange(len(keep))
    for k, m in enumerate(models):
        ax.barh(y + (k - 0.5) * 0.38, rel[m], 0.36, color=COL[m], label=SHORT[m].split(" (")[0])
    ax.axvline(1, color="#222222", lw=0.7)
    ax.set_yticks(y, [t.replace("zero week before origin", "no sales in week before") for t in keep])
    ax.invert_yaxis()
    ax.set_xlabel("MAE / seasonal-naive MAE on the same days")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return save(fig, "failures.png")


def stock(models):
    s = pd.read_csv(TABLES / "final_stock_summary.csv")
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    x = np.arange(len(models))
    for k, (meth, lab, hatch) in enumerate([("conformal", "safety stock from past errors", ""), ("normal formula", "textbook formula", "///")]):
        v = [float(s[(s["method"] == meth) & (s["model"] == m)]["total_cost"].iloc[0]) for m in models]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=[COL[m] for m in models], hatch=hatch, edgecolor="white", lw=0.3)
    ax.set_xticks(x, [SHORT[m].split(" (")[0].replace("Seasonal naive", "Naive") for m in models], fontsize=6.5)
    ax.set_ylabel("total cost, $ (assumed costs)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#999999", label="safety stock from past errors"), Patch(fc="#999999", hatch="///", ec="white", label="textbook formula")], frameon=False, loc="upper left", fontsize=6.3)
    ax.set_ylim(0, s["total_cost"].max() * 1.3)
    return save(fig, "stock.png")
