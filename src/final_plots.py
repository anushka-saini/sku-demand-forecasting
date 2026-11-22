"""Figures for the final comparison. White background, one fixed colour per model, thin lines, no dual axes."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import EVAL_ORIGINS, FIGS
from final_eval import NAMES

COL = {"seasonal_naive": "#8C8C8C", "hw_add": "#0072B2", "ets_boxcox": "#5FB49C", "theta": "#AA4499", "arma": "#C9B07A",
       "arima": "#CC79A7", "sarima": "#D55E00", "sarimax": "#A0410D", "prophet": "#88CCEE", "autots": "#999933",
       "combination": "#7F3C8D", "prophet_tuned": "#3A68C4", "xgboost": "#009E73", "lightgbm": "#56B4E9",
       "lstm": "#E69F00", "gru": "#F0C070", "patchtst": "#117733", "var": "#332288", "avg_top3": "#882255",
       "stack": "#000000", "hybrid": "#44AA99", "actual": "#222222"}
FAM_MARK = {"Baseline": "o", "Classical": "o", "Machine learning": "s", "Deep learning": "^", "Combination": "D"}


def save(fig, name, written):
    p = FIGS / name
    fig.savefig(p, bbox_inches="tight")
    written.append(p)
    plt.close(fig)


def bars(lb, written):
    """Median MAE and RMSE of every model, best at the top. Neural nets carry their seed spread."""
    d = lb[lb["level"] == "item"].sort_values("rank")
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.34 * len(d) + 1.6), sharey=True)
    for ax, k in zip(axes, ["mae", "rmse"]):
        y = np.arange(len(d))
        err = d[f"{k}_std"].fillna(0).to_numpy()
        ax.barh(y, d[k], color=[COL[m] for m in d["model"]], height=0.65, xerr=err, error_kw={"lw": 1, "capsize": 2})
        ax.set_yticks(y, [NAMES[m] for m in d["model"]])
        ax.set_xlabel(f"median {k.upper()} per item, units a day (lower is better)")
        ax.set_xlim(d[k].min() * 0.8, d[k].max() * 1.03)
        for i, v in enumerate(d[k]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    axes[0].invert_yaxis()                                  # best at the top; the axes share y
    fig.tight_layout()
    save(fig, "final_mae_rmse_bars.png", written)


def overlay(fc, models, items, written, days=56):
    """Actual sales and the chosen models for three items over the last `days` scored days."""
    last = fc["date"].max()
    fig, axes = plt.subplots(len(items), 1, figsize=(13, 3.1 * len(items)), sharex=True)
    for ax, it in zip(np.atleast_1d(axes), items):
        a = fc[(fc["item_id"] == it) & (fc["date"] > last - pd.Timedelta(days=days)) & (fc["model"] == models[0])].sort_values("date")
        ax.plot(a["date"], a["actual"], color=COL["actual"], lw=1.8, label="actual")
        for m in models:
            g = fc[(fc["item_id"] == it) & (fc["date"] > last - pd.Timedelta(days=days)) & (fc["model"] == m)].sort_values("date")
            ax.plot(g["date"], g["forecast"], color=COL[m], lw=1.3, label=NAMES[m])
        ax.set_title(it, loc="left")
        ax.set_ylabel("units a day")
    np.atleast_1d(axes)[0].legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.35))
    fig.tight_layout()
    save(fig, "final_forecast_overlay.png", written)


def horizon(h, models, written):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, k in zip(axes, ["mae", "rmse"]):
        for m in models:
            g = h[h["model"] == m]
            ax.plot(g["h"], g[k], color=COL[m], lw=1.6, label=NAMES[m])
        ax.set_xlabel("days ahead")
        ax.set_ylabel(f"{k.upper()}, units a day")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "final_error_by_horizon.png", written)


def residuals(fc, model, written):
    g = fc[(fc["model"] == model) & fc["origin"].isin(EVAL_ORIGINS)].copy()
    g["e"] = g["actual"] - g["forecast"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    lim = np.quantile(np.abs(g["e"]), 0.995)
    ax[0, 0].hist(g["e"].clip(-lim, lim), bins=80, color=COL[model])
    ax[0, 0].axvline(0, color="#222222", lw=1)
    ax[0, 0].set_title("Errors (actual - forecast)", loc="left")
    ax[0, 0].set_xlabel("units a day (clipped at the 99.5th percentile)")
    s = g.sample(min(len(g), 6000), random_state=1)
    ax[0, 1].scatter(s["forecast"], s["e"], s=4, alpha=0.25, color=COL[model])
    ax[0, 1].axhline(0, color="#222222", lw=1)
    ax[0, 1].set_title("Errors against the forecast", loc="left")
    ax[0, 1].set_xlabel("forecast, units a day")
    ax[0, 1].set_ylabel("error")
    stats.probplot(g["e"] / g["e"].std(), dist="norm", plot=ax[1, 0])
    ax[1, 0].get_lines()[0].set_markersize(2)
    ax[1, 0].get_lines()[0].set_color(COL[model])
    ax[1, 0].set_title("")
    ax[1, 0].set_title("Normal Q-Q plot of the errors", loc="left")
    ax[1, 0].set_xlabel("theoretical quantiles")
    ax[1, 0].set_ylabel("ordered errors (standardised)")
    day = g.groupby("date")["e"].sum().sort_index()
    from statsmodels.tsa.stattools import acf
    r = acf(day.to_numpy(), nlags=28)
    ax[1, 1].bar(range(1, 29), r[1:], color=COL[model], width=0.7)
    bound = 1.96 / np.sqrt(len(day))
    ax[1, 1].axhline(bound, color="#8C8C8C", ls="--", lw=1)
    ax[1, 1].axhline(-bound, color="#8C8C8C", ls="--", lw=1)
    ax[1, 1].set_title("Autocorrelation of the daily total error", loc="left")
    ax[1, 1].set_xlabel("lag in days")
    fig.suptitle(f"Residual checks, {NAMES[model]}, scored days", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "final_residuals.png", written)


def weights(w, avg_members, written):
    w = w[w > 1e-6].sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(w) + 1.4))
    ax.barh([NAMES[m] for m in w.index], w.to_numpy(), color=[COL[m] for m in w.index], height=0.6)
    for i, v in enumerate(w):
        ax.text(v, i, f" {v:.2f}", va="center")
    ax.set_xlabel("weight in the stacked forecast (fitted on the validation windows only)")
    ax.set_title(f"Weights sum to {w.sum():.2f}; the average of the top three used: " + ", ".join(NAMES[m] for m in avg_members),
                 loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, "final_ensemble_weights.png", written)


def segments(seg, models, written):
    """MAE of each model on each kind of day, divided by seasonal naive on the same days (below 1 = better than naive)."""
    keep = ["event days", "SNAP days", "weekend", "days with zero sales", "zero week before origin", "items with a detected shift", "all days"]
    d = seg[seg["segment"].isin(keep) & seg["model"].isin(models + ["seasonal_naive"])]
    p = d.pivot(index="segment", columns="model", values="mae").reindex(keep)
    rel = p[models].div(p["seasonal_naive"], axis=0)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(keep))
    wd = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.bar(x + (k - (len(models) - 1) / 2) * wd, rel[m], wd, color=COL[m], label=NAMES[m])
    ax.axhline(1, color="#222222", lw=1)
    ax.set_xticks(x, keep, rotation=15, ha="right")
    ax.set_ylabel("MAE / seasonal-naive MAE on the same days")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    save(fig, "final_failure_segments.png", written)


def volatility(daytype, iv, item, written):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(daytype))
    ax[0].bar(x - 0.2, daytype["mean_sigma_over_scale"], 0.4, color="#0072B2", label="GARCH spread (sigma)")
    ax[0].bar(x + 0.2, daytype["mean_abs_error_over_scale"], 0.4, color="#D55E00", label="actual absolute error")
    ax[0].set_xticks(x, daytype["day_type"], rotation=20, ha="right")
    ax[0].set_ylabel("relative to the item's usual sales")
    ax[0].set_ylim(0, max(daytype["mean_sigma_over_scale"].max(), daytype["mean_abs_error_over_scale"].max()) * 1.35)
    ax[0].legend(frameon=False, ncol=2, loc="upper center", fontsize=9)
    ax[0].set_title("Fitted spread and real error by type of day", loc="left")
    g = iv[(iv["item_id"] == item)].sort_values("date")
    o = g["origin"].iloc[len(g) // 2]
    g = g[g["origin"] == o]
    z = 1.2816
    ax[1].fill_between(g["date"], np.maximum(g["forecast"] - z * g["sig_garch"], 0), g["forecast"] + z * g["sig_garch"],
                       color="#0072B2", alpha=0.2, label="80% band, GARCH")
    ax[1].plot(g["date"], np.maximum(g["forecast"] - z * g["sig_const"], 0), color="#8C8C8C", ls="--", lw=1)
    ax[1].plot(g["date"], g["forecast"] + z * g["sig_const"], color="#8C8C8C", ls="--", lw=1, label="80% band, constant")
    ax[1].plot(g["date"], g["forecast"], color="#0072B2", lw=1.5, label="forecast")
    ax[1].plot(g["date"], g["actual"], color="#222222", lw=1.6, label="actual")
    ax[1].set_title(f"{item}, one 28-day window", loc="left")
    ax[1].legend(frameon=False, fontsize=9)
    import matplotlib.dates as mdates
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax[0].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "final_garch_volatility.png", written)


def stock(summ, models, written):
    s = summ[summ["method"] == "conformal"].set_index("model").loc[models]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar([NAMES[m] for m in models], s["total_cost"], color=[COL[m] for m in models], width=0.6)
    ax[0].set_ylabel("total cost over 168 days ($, assumed costs)")
    ax[1].bar([NAMES[m] for m in models], s["avg_inventory_value"], color=[COL[m] for m in models], width=0.6)
    ax[1].set_ylabel("average stock value ($)")
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "final_stock_impact.png", written)
