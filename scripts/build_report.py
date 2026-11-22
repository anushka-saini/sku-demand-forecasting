"""Builds the final report (IEEE style, two columns, at most 10 pages) as a PDF and an editable DOCX from the same content.

Every number comes from the saved result files (outputs/tables/final_*.csv, results_facts.json and the earlier tables); the
figures are drawn from them by scripts/report_figs.py at the size they are printed. Figures and tables are numbered
automatically and every one is cited in the text through a token such as [[F:overlay]] or [[T:lb]]. Run after notebook 13:

    python scripts/build_report.py
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import report_figs as RF  # noqa: E402
from config import TABLES  # noqa: E402

OUT = ROOT / "outputs" / "report"
OUT.mkdir(parents=True, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# ------------------------------------------------------------------ numbers

lb = RF.lb_items()
row = lambda m: lb[lb["model"] == m].iloc[0]
nm = {"seasonal_naive": "Seasonal naive", "hw_add": "Holt-Winters additive", "ets_boxcox": "Holt-Winters (damped, Box-Cox)", "theta": "Theta", "arma": "ARMA(p, q)",
      "arima": "ARIMA", "sarima": "SARIMA", "sarimax": "SARIMAX", "prophet": "Prophet", "autots": "AutoTS", "combination": "Combination (SARIMAX, HW, Theta)",
      "prophet_tuned": "Prophet (tuned, events)", "xgboost": "XGBoost (Optuna)", "lightgbm": "LightGBM (reference)", "lstm": "LSTM", "gru": "GRU", "patchtst": "PatchTST",
      "var": "VAR (top-down to items)", "avg_top3": "Average of top 3", "stack": "Stacked (non-negative weights)", "hybrid": "Hybrid (classical + XGBoost residuals)"}
info = json.loads((TABLES / "final_best_model.json").read_text())
facts = json.loads((ROOT / "outputs" / "results_facts.json").read_text())
B_ = lb.iloc[0]
assert B_["model"] == info["model"] == "xgboost", "the report text assumes the best model is XGBoost; check final_leaderboard.csv"
N_ = row("seasonal_naive")
X, CB, SA, AV, ST, HY, LG = (row(m) for m in ["xgboost", "combination", "sarima", "avg_top3", "stack", "hybrid", "lightgbm"])
dm = pd.read_csv(TABLES / "final_dm_tests.csv")
seg = pd.read_csv(TABLES / "final_segments.csv")
gar = pd.read_csv(TABLES / "final_garch_intervals.csv").set_index(["interval", "nominal"])
archlm = pd.read_csv(TABLES / "final_garch_archlm.csv")
daytype = pd.read_csv(TABLES / "final_garch_daytype.csv").set_index("day_type")
stock = pd.read_csv(TABLES / "final_stock_summary.csv").set_index(["method", "model"])
scmp = pd.read_csv(TABLES / "final_stock_compare.csv")
tot = pd.read_csv(TABLES / "final_totals_level.csv").set_index("model")
split = pd.read_csv(TABLES / "final_split.csv").set_index("part")
wts = pd.read_csv(TABLES / "final_stack_weights.csv")
hz = pd.read_csv(TABLES / "final_horizon.csv")
var_set = pd.read_csv(TABLES / "final_var_settings.csv")
trials = pd.read_csv(TABLES / "final_xgb_trials.csv")
PROC = ROOT / "data" / "processed" / "final"
xp = json.loads((PROC / "xgb_params.json").read_text())
lp = json.loads((PROC / "lgbm_params.json").read_text())
prop = json.loads((PROC / "prophet_config.json").read_text())
nn = {k: json.loads((PROC / f"{k}_config.json").read_text())["config"] for k in ["lstm", "gru", "patchtst"]}
scope = facts["scope"]
ax_ay, ca_items = int(scope["ax_ay_items"]["value"]), int(scope["ca_items"]["value"])
pct = lambda a, b: abs(a / b - 1) * 100
gv = lambda s, m="xgboost", k="mae": float(seg[(seg["segment"] == s) & (seg["model"] == m)][k].iloc[0])
dmx = lambda other, loss: dm[(dm["model_b"] == other) & (dm["loss"] == loss)].iloc[0]
money = lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
fp = lambda p: "< 0.0001" if p < 0.0001 else f"{p:.3f}"
mape_excl = float(pd.read_csv(TABLES / "comparison_metrics_summary.csv").query("category == 'ALL' and model == 'seasonal_naive'")["mape_excluded_share"].iloc[0])

# horizon pattern, XGBoost against the Combination, by block of forecast days
a_ = hz[hz["model"] == "xgboost"].set_index("h")["mae"]
c_ = hz[hz["model"] == "combination"].set_index("h")["mae"]
wins = (a_ < c_)
h_early, h_mid, h_late = wins.loc[:7].mean(), wins.loc[8:21].mean(), wins.loc[22:].mean()

# figures
sample_ids = pd.read_csv(TABLES / "classical_sample_items.csv")["item_id"]
f_ds, zero_share, d0, d1, n_items = RF.dataset(sample_ids)
f_wf = RF.workflow()
f_bars = RF.bars(lb)
items_show = []
daily = pd.read_csv(TABLES / "classical_sample_items.csv")
import ts_models as tm  # noqa: E402
inp = tm.load_inputs()
for c in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
    ids = [i for i in sample_ids if i.startswith(c)]
    m = {i: inp["daily"][i]["sales"].iloc[-365:].mean() for i in ids}
    items_show.append(sorted(ids, key=m.get)[len(ids) // 2])
f_ov = RF.overlay(items_show)
f_hz = RF.horizon(["xgboost", "combination", "sarima", "seasonal_naive"])
f_w = RF.weights(wts)
f_res, acf1, acf_band = RF.resid("xgboost")
f_fail = RF.failures()
f_stock = RF.stock(["seasonal_naive", "sarima", "combination", "xgboost"])

# ------------------------------------------------------------------ text

abstract = (f"A shop that orders from last week's sales runs short on some days and holds dead stock on others. We test whether better forecasts help, using the daily sales of {n_items} "
            f"high-volume California items from the Walmart M5 data, forecast 28 days ahead. Twenty-one models compete on the same six untouched test windows: classical time-series "
            f"models, XGBoost, LightGBM, a tuned Prophet, LSTM, GRU, PatchTST, a VAR and three ensembles. Every setting was chosen on three earlier validation windows. XGBoost ranks first, "
            f"with a median MAE of {X['mae']:.2f} and RMSE of {X['rmse']:.2f} units a day, {pct(X['mae'], N_['mae']):.0f}% below seasonal naive. No ensemble beats it on both measures, "
            f"which goes against our expectation. Diebold-Mariano tests separate it clearly from seasonal naive but not from the best classical models. Neural networks fall short of the "
            f"classical models, and GARCH does not improve prediction intervals. In a stock simulation, sizing safety stock from past errors cut cost far more than the choice among the top models.")
wc = len(abstract.split())
assert 150 <= wc <= 200, wc

INTRO = [
    "A shop that orders from last week's sales gets two kinds of wrong. On some days the shelf is empty and a customer leaves without buying; on others, units sit unsold and tie up money. "
    "Both losses start with the forecast of the next few weeks, so we study how well daily sales can be forecast for stock planning.",
    "We work with the public M5 data [1] from Walmart: five years of daily unit sales, plus a calendar with events and food-assistance (SNAP) days and weekly prices. "
    f"The task is to forecast {n_items} high-volume California items 28 days ahead. An earlier stage of our project had already produced a strong set of classical time-series models, so the question here is narrower: "
    "does a more advanced model family beat them, and does combining models beat the best single model? We expected a combination to win. We built the test so that this expectation could fail.",
    "The rest of the report covers: a leakage-checked feature set shared by all learning models ([[S:method]]); one protocol in which every choice is made on validation windows and the test windows are scored once; "
    "a twenty-one-model leaderboard ranked by MAE and RMSE ([[T:lb]]); significance tests ([[T:dm]]); an analysis of where the models fail; and a stock-cost simulation that shows what the forecasts are worth ([[T:stock]]).",
]

LIT = [
    "The M5 competition [1] is the closest precedent. Its results paper reports that the strongest entries were machine-learning methods, mostly gradient-boosted trees, that used calendar and price information. "
    "The competition scored a weighted hierarchy of 42,840 series and did not look at stock cost. Our setting is smaller and ends in an inventory decision.",
    "Makridakis et al. [2] reached the opposite conclusion on different data: on 1,045 monthly series, simple statistical methods were more accurate and cheaper than the machine-learning methods they tested. "
    "That paper is why we kept strong classical baselines in the comparison and tuned the learners only on validation data.",
    "Two newer tools appear in our comparison. Prophet [4] fits trend, seasonality and holidays as one additive curve and is popular because it needs little configuration. PatchTST [3] cuts a series into patches and reads "
    "them with a transformer; its published gains are on long-horizon benchmarks, mostly smooth series rather than daily counts with many zeros. We test both on count data.",
    "Bates and Granger [5] showed that combining forecasts can beat the individual forecasts, and later work often finds that a plain average is hard to beat. We therefore test three kinds of combination "
    "instead of assuming that they help.",
    "Gap. These studies use different data, score a different level, or stop at accuracy. We compare the families on one fixed set of items and windows, test whether the differences are statistically real, "
    "check whether combining helps, and follow the winner into a stock simulation.",
]

DATA = [
    f"The M5 files hold daily unit sales of {ca_items:,} California products across four stores (and two other states we use only for checks), a calendar with events and SNAP flags, and weekly prices. We add the four "
    f"stores into one series per product and cut each series at its first real sale, because the zeros before a launch are not demand. We keep the {ax_ay} products that sell a lot and fairly steadily (classes AX and AY of an "
    f"ABC-XYZ split by volume and variability), which carry {scope['ax_ay_share_of_units']['value'] * 100:.0f}% of the units. All models here use a fixed sample of {n_items} of them, 50 per category, "
    f"the same sample as our classical stage, so old and new results are comparable. The series run from {d0:%d %B %Y} to {d1:%d %B %Y} ([[F:data]]); {zero_share * 100:.1f}% of all item-days in the sample have no sales, "
    f"and in the six test windows {mape_excl * 100:.1f}% do. Prices are used as a relative price (today's price over the item's own average of the past 365 days). Calendar and prices are known ahead of time, as in the competition.",
    f"Split. Training uses everything before {split.loc['validation fold 1', 'origins']}. Three validation windows of 28 days follow, each starting at a forecast origin 28 days after the last, which makes a "
    f"rolling-origin backtest with three folds. All tuning uses these windows. The test is six more windows, from {split.loc['test (6 windows, scored once)', 'first_day']} to {split.loc['test (6 windows, scored once)', 'last_day']} "
    f"(168 days, {n_items * 168:,} item-days). A model forecasting from an origin sees only data up to that origin.",
]

METH = [
    ("A. Features and leakage",
     "XGBoost, LightGBM and the hybrid read one feature table with a row per item, origin and forecast step (1 to 28). Features about the past use sales up to the origin only: rolling means over 7 to 112 days, spread, "
     "share of zero days, and the same weekday one to four weeks back. Features about the target day are known ahead: weekday, month, yearly sine and cosine, event flag and type, SNAP, Christmas, relative price and its change "
     "since the origin. The neural networks get the same information as windows, with sales divided by the item's own 56-day mean and the other inputs standardised with training numbers. "
     "A unit test replaces every sale after an origin with random numbers, rebuilds the features, and fails if any feature changes."),
    ("B. Classical models",
     "From our earlier stage: seasonal naive, additive and damped Box-Cox Holt-Winters, Theta, ARMA, ARIMA, SARIMA, SARIMAX with one yearly Fourier pair, Prophet, AutoTS and a plain average of SARIMAX, Holt-Winters and Theta. "
     "Each is fitted per item on the last 730 days at every origin. SARIMA orders come from an AICc search at the warm-up origin (d and D from KPSS and OCSB tests, p, q from 0 to 2, P, Q from 0 to 1) and are then frozen. "
     f"The tuned Prophet adds events as holidays and SNAP and relative price as regressors; its grid of changepoint prior, seasonality mode and seasonality prior chose {prop['changepoint_prior_scale']}, "
     f"{prop['seasonality_mode']} and {prop['seasonality_prior_scale']}."),
    ("C. Machine-learning and neural models",
     f"XGBoost [6] is tuned with Optuna: {len(trials)} trials, three expanding-window folds on the validation origins, early stopping on the 28 days before each fold. The search covers the objective, learning rate, depth, "
     f"child weight, subsampling and regularisation. It chose a {xp['params']['objective'].replace('reg:', '')} objective, depth {xp['params']['max_depth']} and {xp['n_est']} trees, and is refitted at every test origin. "
     f"LightGBM is a reference with fixed settings and {lp['n_est']} trees. LSTM [10], GRU and a compact PatchTST [3], written in PyTorch for this project, read a window of past sales and covariates and output all 28 days; "
     f"known-ahead covariates join at the head, which the original PatchTST does not have. Each net has a small grid (hidden size, layers, dropout, window, learning rate; for PatchTST, patch length and learning rate), "
     f"early stopping and three seeds. To fit a CPU laptop, each net is trained once per stage and rolled forward, not refitted at every origin. The VAR is fitted to the three category totals "
     f"(ADF check, lag by AIC, {int(var_set['lag_aic'].min())} to {int(var_set['lag_aic'].max())} chosen) and shared to items by recent sales share. "
     "GARCH(1,1) [9] is fitted to the best model's residuals, scaled by each item's usual level, using residuals from earlier windows only."),
    ("D. Combinations and tests",
     f"Three combinations are built from validation forecasts only: the plain average of the three best single models ({', '.join(nm[m].split(' (')[0] for m in info['avg_top3_members'])}), "
     "a stack with non-negative least-squares weights [5], and a hybrid in which XGBoost learns the residuals of SARIMA. "
     "Models are ranked by the mean of their MAE rank and RMSE rank. Differences are tested with the Diebold-Mariano test [7] on the 168 daily loss differences, with a Newey-West variance (27 lags) and the "
     "small-sample correction of Harvey et al. [8]. MAPE counts only days with sales, and WAPE is reported as the zero-safe check."),
    ("E. Stock simulation and tools",
     "Each forecast drives a daily reorder-point simulation over the 168 test days with a 7-day lead time, in which unmet demand is lost. Safety stock is sized either by the textbook formula or from the model's own past "
     "error quantiles. Holding costs 25% of price a year and a lost sale costs half the price; both are assumptions. The code is Python with statsmodels, Prophet, XGBoost, LightGBM, Optuna, PyTorch, arch and Streamlit, "
     "and all seeds are fixed."),
]


def res_text():
    p = []
    p.append(("A. Overall ranking",
              f"[[T:lb]] and [[F:bars]] show all twenty-one models. XGBoost is first on both ranking measures, with a median MAE of {X['mae']:.3f} and RMSE of {X['rmse']:.3f} units a day against {N_['mae']:.2f} and {N_['rmse']:.2f} for "
              f"seasonal naive. The best earlier model, the Combination, scores {CB['mae']:.2f} and {CB['rmse']:.2f}. The average of the top three ({AV['mae']:.2f}, {AV['rmse']:.2f}), the stack ({ST['mae']:.2f}, {ST['rmse']:.2f}) "
              f"and the hybrid ({HY['mae']:.2f}, {HY['rmse']:.2f}) all come out slightly behind XGBoost on both measures, so our expectation that a combination would win is not supported. "
              f"The summary matters, though. Pooling all item-days instead of taking each item's median, the average of the top three has the lowest RMSE ({AV['pooled_rmse']:.2f} against {X['pooled_rmse']:.2f}) and the stack the lowest MAE by a hair "
              f"({ST['pooled_mae']:.3f} against {X['pooled_mae']:.3f}); on the median MASE the stack leads ({ST['mase']:.3f} against {X['mase']:.3f}). The honest statement is that no combination beat the best single model on the "
              f"ranking measures, and that the top of the table is close. The neural networks beat seasonal naive on the median item but sit behind every classical model: GRU {row('gru')['mae']:.2f} ± {row('gru')['mae_std']:.2f} MAE, "
              f"PatchTST {row('patchtst')['mae']:.2f} ± {row('patchtst')['mae_std']:.2f}, LSTM {row('lstm')['mae']:.2f} ± {row('lstm')['mae_std']:.2f} (mean ± standard deviation over three seeds). "
              f"[[F:overlay]] shows what this looks like on single items: the forecasts follow the weekly rhythm and miss the spikes. "
              f"The tuned Prophet ({row('prophet_tuned')['mae']:.2f}) is worse than the plain one ({row('prophet')['mae']:.2f}), and LightGBM with fixed settings ({LG['mae']:.2f}) trails XGBoost. "
              f"For cost, XGBoost's six refits take {X['fit_seconds']:.0f} s, against about {SA['fit_seconds']:.0f} s for SARIMA and {CB['fit_seconds']:.0f} s for the Combination (both estimated from a short timing run)."))
    p.append(("B. Is the lead real?",
              f"[[T:dm]] answers it. Against seasonal naive the difference is clear (p {fp(dmx('seasonal_naive', 'absolute error')['p_value'])}), and XGBoost has the lower MAE on "
              f"{dmx('seasonal_naive', 'absolute error')['share_items_a_better'] * 100:.0f}% of items. Against SARIMA the absolute-error p-value is {dmx('sarima', 'absolute error')['p_value']:.3f}, just above 5%, and the squared-error value "
              f"({dmx('sarima', 'squared error')['p_value']:.2f}) is not close; against the Combination, p = {dmx('combination', 'absolute error')['p_value']:.2f}. So XGBoost ranks first, but its lead over the best classical "
              f"models is within what six test windows can show. Across the forecast horizon ([[F:hz]]) there is no clear pattern: XGBoost beats the Combination on {h_early * 100:.0f}% of days 1 to 7, {h_mid * 100:.0f}% of days 8 to 21 "
              f"and {h_late * 100:.0f}% of days 22 to 28. One reading, which we did not test separately, is that the tree models see all 150 items and read the calendar and price columns directly, "
              f"while each classical model sees one item; LightGBM, with the same features, is the second-best single model, which fits that reading. The stack weights ([[F:weights]]) lean on the same two tree models."))
    p.append(("C. Where the models fail",
              f"[[T:fail]] and [[F:fail]] split the test days. Weekends are the hardest group for every model (XGBoost MAE {gv('weekend'):.2f} against {gv('weekday'):.2f} on weekdays). Event days are a little worse than ordinary days "
              f"({gv('event days'):.2f} against {gv('no event'):.2f}); SNAP days are no worse ({gv('SNAP days'):.2f} against {gv('non-SNAP days'):.2f}). On days with no sales XGBoost forecasts "
              f"{gv('days with zero sales', k='bias'):.1f} units too many on average, and after a week with no sales before the forecast date it is no better than seasonal naive ({gv('zero week before origin'):.2f} against "
              f"{gv('zero week before origin', 'seasonal_naive'):.2f}), where SARIMA is clearly worse ({gv('zero week before origin', 'sarima'):.2f}). On items with a detected shift in level it has no edge over the Combination "
              f"({gv('items with a detected shift'):.2f} against {gv('items with a detected shift', 'combination'):.2f}). Overall the model leans low by {abs(gv('all days', k='bias')):.2f} units a day. "
              f"[[F:resid]] shows why: the error histogram has a long right tail and the Q-Q plot bends upward, so the spikes are under-forecast, and the daily total error has an autocorrelation of {acf1:.2f} at lag 1 "
              f"(95% band ±{acf_band:.2f}), so neighbouring days' misses move together."))
    p.append(("D. Volatility and the category totals",
              f"GARCH models how large the errors are, not the forecast, so it cannot change MAE or RMSE, and it did not help as intervals either ([[T:garch]]): the 80% band covers "
              f"{gar.loc[('GARCH(1,1)', 0.8), 'coverage'] * 100:.1f}% of days against {gar.loc[('constant std', 0.8), 'coverage'] * 100:.1f}% for a constant spread, with a worse interval score "
              f"({gar.loc[('GARCH(1,1)', 0.8), 'interval_score']:.2f} against {gar.loc[('constant std', 0.8), 'interval_score']:.2f}). The Engle test finds changing variance in "
              f"{(archlm['arch_lm_p'] < 0.05).mean() * 100:.0f}% of items. The fitted spread barely differs between weekends and weekdays ({daytype.loc['weekend', 'mean_sigma_over_scale']:.2f} against "
              f"{daytype.loc['weekday', 'mean_sigma_over_scale']:.2f}), while the real error does ({daytype.loc['weekend', 'mean_abs_error_over_scale']:.2f} against {daytype.loc['weekday', 'mean_abs_error_over_scale']:.2f} of an item's "
              f"usual level): the volatility that matters follows the calendar, which a GARCH fitted to 28-day-ahead errors does not see. The VAR on the three category totals gives an item-level MAE of {row('var')['mae']:.2f} "
              f"and, at the totals it models, MAE {tot.loc['var', 'mae']:.1f} and RMSE {tot.loc['var', 'rmse']:.1f} units a day, against {tot.loc['xgboost', 'mae']:.1f} and {tot.loc['xgboost', 'rmse']:.1f} for the "
              f"summed XGBoost forecasts; linking the categories did not help."))
    cx = scmp[(scmp["method"] == "conformal") & (scmp["versus"] == "combination")].iloc[0]
    tx = scmp[(scmp["method"] == "normal formula") & (scmp["versus"] == "combination")].iloc[0]
    p.append(("E. What the forecasts are worth in stock",
              f"[[T:stock]] and [[F:stock]] replay the 168 test days. With safety stock sized from past errors, XGBoost costs ${stock.loc[('conformal', 'xgboost'), 'total_cost']:,.0f} against "
              f"${stock.loc[('conformal', 'combination'), 'total_cost']:,.0f} for the Combination and ${stock.loc[('conformal', 'seasonal_naive'), 'total_cost']:,.0f} for seasonal naive, at a fill rate of "
              f"{stock.loc[('conformal', 'xgboost'), 'fill_rate'] * 100:.1f}%. The 95% bootstrap interval of the saving over the Combination ({money(cx['ci_low'])} to {money(cx['ci_high'])}) includes zero, so it is not reliable. "
              f"With the textbook formula the same model costs {tx['total_cost_diff_pct'] * 100:.0f}% more than the Combination: it holds less stock but loses more sales. "
              f"In our earlier stage, on all 726 products, calibrated safety stock cut the Combination's cost by about a quarter. The safety-stock rule, not the choice among the top few models, is what moves the cost."))
    return p


def tbl_lb():
    rows = []
    for _, x in lb.iterrows():
        mae_s = f"{x['mae']:.2f}" + (f" ± {x['mae_std']:.2f}" if not pd.isna(x["mae_std"]) else "")
        rmse_s = f"{x['rmse']:.2f}" + (f" ± {x['rmse_std']:.2f}" if not pd.isna(x["rmse_std"]) else "")
        sec = "" if pd.isna(x["fit_seconds"]) else f"{x['fit_seconds']:.0f}" + ("*" if x["fit_seconds_kind"] == "estimated" else "")
        rows.append([int(x["rank"]), x["name"], x["family"], mae_s, rmse_s, f"{x['mape_nonzero']:.3f}", f"{x['wape']:.3f}", f"{x['mase']:.3f}", f"{x['pooled_mae']:.2f}", f"{x['pooled_rmse']:.2f}", sec])
    return ["#", "Model", "Group", "MAE", "RMSE", "MAPE", "WAPE", "MASE", "Pooled MAE", "Pooled RMSE", "Fit s"], rows


def tbl_dm():
    rows = []
    for _, x in dm.iterrows():
        rows.append([nm[x["model_b"]], "absolute" if x["loss"].startswith("abs") else "squared", fp(x["p_value"]), "yes" if x["significant_5pct"] else "no", f"{x['share_items_a_better'] * 100:.0f}%"])
    return ["XGBoost against", "Loss", "p", "5%", "Items won"], rows


def tbl_fail():
    sel = ["all days", "event days", "no event", "SNAP days", "non-SNAP days", "weekend", "weekday", "days with zero sales", "zero week before origin", "items with a detected shift", "items without a shift"]
    rows = [[s, f"{int(seg[(seg['segment'] == s) & (seg['model'] == 'xgboost')]['item_days'].iloc[0]):,}", f"{gv(s):.2f}", f"{gv(s, 'combination'):.2f}", f"{gv(s, 'seasonal_naive'):.2f}"] for s in sel]
    return ["Group of item-days", "Item-days", "XGBoost", "Combination", "Naive"], rows


def tbl_garch():
    rows = [[f"{int(lev * 100)}%", k, f"{gar.loc[(k, lev), 'coverage'] * 100:.1f}%", f"{gar.loc[(k, lev), 'mean_width']:.2f}", f"{gar.loc[(k, lev), 'interval_score']:.2f}"] for lev in (0.8, 0.95) for k in ("constant std", "GARCH(1,1)")]
    return ["Level", "Spread", "Coverage", "Width", "Score"], rows


def tbl_stock():
    rows = [[nm[m].split(" (")[0], f"{stock.loc[('conformal', m), 'total_cost']:,.0f}", f"{stock.loc[('conformal', m), 'fill_rate'] * 100:.1f}%", f"{stock.loc[('normal formula', m), 'total_cost']:,.0f}", f"{stock.loc[('normal formula', m), 'fill_rate'] * 100:.1f}%"] for m in ["seasonal_naive", "sarima", "combination", "xgboost"]]
    return ["Model", "Past-error cost $", "Fill", "Textbook cost $", "Fill"], rows


def tbl_team():
    return ["Member", "Roll no.", "Easy models", "Medium models", "Hard models and other work"], [
        ["[Team member 1 name]", "[roll no.]", "Seasonal naive, Holt-Winters, Theta", "SARIMA, SARIMAX", "GRU; report writing"],
        ["[Team member 2 name]", "[roll no.]", "ARMA, ARIMA", "Tuned Prophet, XGBoost, LightGBM, VAR", "Stacking and hybrid; dashboard"],
        ["[Team member 3 name]", "[roll no.]", "Data scoping, features", "Combination of SARIMAX, HW and Theta", "LSTM, PatchTST, GARCH; slides"]]


def tbl_app():
    rows = [["XGBoost", ", ".join(f"{k.replace('_', ' ')} {round(v, 4) if isinstance(v, float) else v}" for k, v in xp["params"].items()) + f", trees {xp['n_est']}"],
            ["Prophet (tuned)", f"changepoint prior {prop['changepoint_prior_scale']}, {prop['seasonality_mode']}, seasonality prior {prop['seasonality_prior_scale']}"],
            ["LSTM", ", ".join(f"{k} {v}" for k, v in nn["lstm"].items())], ["GRU", ", ".join(f"{k} {v}" for k, v in nn["gru"].items())],
            ["PatchTST", ", ".join(f"{k} {v}" for k, v in nn["patchtst"].items())],
            ["Average of top 3", ", ".join(nm[m] for m in info["avg_top3_members"])],
            ["Stack weights", ", ".join(f"{nm[m].split(' (')[0]} {v:.2f}" for m, v in info["stack_weights"].items())],
            ["Hybrid", f"base {nm[info['hybrid_base']]}; XGBoost depth 3, 150 trees, learning rate 0.05, min child weight 50, L2 20 (fixed)"]]
    return ["Model", "Setting chosen on the validation windows"], rows


CONCL = [
    f"XGBoost is the best model on the six test windows by both ranking measures and is clearly better than seasonal naive. It is not significantly better than the best classical models, and none of the three "
    f"combinations beats it on both MAE and RMSE. The neural networks, trained once and rolled forward on a laptop, did not reach the classical models. GARCH did not improve prediction intervals, and a VAR on the category "
    f"totals did not help. For stock, how the safety stock is sized mattered more than which of the top models made the forecast.",
]
LIMITS = [
    f"Scope: {n_items} high-volume items from one state with four stores added up; slow movers and store-level series are untested, and they are where deep models are often said to help.",
    "Evidence: six test windows. The top models are close and only some gaps are significant.",
    "Neural networks: one training run per stage, small random grids and a CPU laptop. They are probably under-tuned, so these results say little about deep learning in general.",
    "Inputs and costs: prices and the calendar are treated as known ahead, and the stock costs and lead time are assumptions, so the simulated saving is not a measured one.",
    "Combinations: the hybrid had only 3 to 8 earlier windows to learn from, and the stack and the top-three average were chosen on three validation windows.",
    "Fit times of the earlier models were not recorded; they are estimated from a short timing run.",
]
CHALLENGES = [
    "Keeping every choice on the validation windows, which each model needed its own way to do.",
    "Showing that the features hold no future information; the scrambling test was the practical answer.",
    f"Zeros: about one item-day in {round(1 / mape_excl)} has no sales, so MAPE is misleading and is reported on non-zero days only, next to WAPE.",
    "Run time: the networks took many minutes per fit on a CPU, which forced the one-training-per-stage design.",
]
FUTURE = [
    "Refit the neural networks at every origin and widen their search on a GPU.",
    f"Model the weekend and zero-day misses directly, for example with a two-part loss (zero or not, then how many); the lag-1 autocorrelation of the daily error ({acf1:.2f}) suggests that correcting yesterday's miss could also help.",
    "Use calendar-aware safety stock, larger on weekends and event days, instead of one buffer per item.",
    "Repeat the study on other states, store-level series and slow-moving items, with real costs and lead times.",
]

REFS = [
    'S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions," Int. J. Forecasting, vol. 38, no. 4, pp. 1346-1364, 2022.',
    'S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "Statistical and machine learning forecasting methods: Concerns and ways forward," PLoS ONE, vol. 13, no. 3, e0194889, 2018.',
    'Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in Proc. ICLR, 2023.',
    'S. J. Taylor and B. Letham, "Forecasting at scale," The American Statistician, vol. 72, no. 1, pp. 37-45, 2018.',
    'J. M. Bates and C. W. J. Granger, "The combination of forecasts," Operational Research Quarterly, vol. 20, no. 4, pp. 451-468, 1969.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785-794.',
    'F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," J. Business & Economic Statistics, vol. 13, no. 3, pp. 253-263, 1995.',
    'D. Harvey, S. Leybourne, and P. Newbold, "Testing the equality of prediction mean squared errors," Int. J. Forecasting, vol. 13, no. 2, pp. 281-291, 1997.',
    'T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," J. Econometrics, vol. 31, no. 3, pp. 307-327, 1986.',
    'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.',
]

# ------------------------------------------------------------------ blocks (numbered and cross-referenced at render time)

B = []
H1 = lambda t: B.append(("h1", t))
H2 = lambda t: B.append(("h2", t))
P = lambda t: B.append(("p", t))
H1("Abstract")
P(abstract)
H1("I. Introduction")
for t in INTRO:
    P(t)
H1("II. Literature review")
for t in LIT:
    P(t)
H1("III. Dataset description and preprocessing")
B.append(("fig", "data", f_ds, f"Daily units for the {n_items} items by category (7-day mean), with the validation and test windows shaded.", 2))
for t in DATA:
    P(t)
H1("IV. Methodology")
B.append(("fig", "wf", f_wf, "Workflow of the study.", 2))
P("[[F:wf]] shows the order of work. [[S:method]] gives the details, and Table A1 in the appendix lists the settings that the validation windows chose.")
B.append(("anchor", "method"))
for h, t in METH:
    H2(h)
    P(t)
H1("V. Results and discussion")
txt = res_text()
B.append(("table", "lb", "All models on the six test windows (median over items of each item's mean error; MAE and RMSE in units a day; MAPE on non-zero days only; WAPE and MASE as fractions). ± is the standard deviation over three seeds. Pooled = all item-days together. Fit s = seconds for the six test windows, 150 items; * = estimate from a short timing run; blank = not timed.", *tbl_lb(), 2))
B.append(("fig", "bars", f_bars, "Median MAE and RMSE of every model, best at the top; whiskers on the networks show the standard deviation over three seeds.", 2))
H2(txt[0][0]); P(txt[0][1])
B.append(("fig", "overlay", f_ov, "Actual sales and three forecasts for one median-volume item per category, over the last 56 test days.", 2))
H2(txt[1][0])
B.append(("table", "dm", "Diebold-Mariano tests of XGBoost against other models (168 test days). 'Items won' is the share of items where XGBoost has the lower MAE.", *tbl_dm(), 1))
P(txt[1][1])
B.append(("fig", "hz", f_hz, "Error by forecast day for the top models and seasonal naive.", 1))
B.append(("fig", "weights", f_w, "Stacking weights, fitted on the validation windows only.", 1))
H2(txt[2][0])
B.append(("table", "fail", "MAE on groups of item-days (all item-days in a group pooled), six test windows.", *tbl_fail(), 1))
P(txt[2][1])
B.append(("fig", "fail", f_fail, "MAE relative to seasonal naive on the same days (below 1 is better).", 1))
B.append(("fig", "resid", f_res, "Residual checks for XGBoost: error histogram, normal Q-Q plot, and autocorrelation of the daily total error.", 2))
H2(txt[3][0])
B.append(("table", "garch", "Prediction intervals around the XGBoost forecast: coverage, mean width and interval score (lower is better).", *tbl_garch(), 1))
P(txt[3][1])
H2(txt[4][0])
B.append(("table", "stock", "Stock replay, 168 days, 7-day lead time, 95% target, assumed costs: total cost and fill rate with safety stock sized from past errors and with the textbook formula.", *tbl_stock(), 1))
P(txt[4][1])
B.append(("fig", "stock", f_stock, "Total simulated cost with the two safety-stock rules.", 1))
H1("VI. Conclusion and future work")
for t in CONCL:
    P(t)
H2("Limitations")
B.append(("bullets", LIMITS))
H2("Challenges")
B.append(("bullets", CHALLENGES))
H2("Improvements")
B.append(("bullets", FUTURE))
B.append(("table", "team", "Contribution of team members. Names and roll numbers are placeholders; the split of models by difficulty is a proposal to confirm.", *tbl_team(), 1))
P("[[T:team]] lists who did what.")
H1("References")
B.append(("refs", REFS))
H1("Appendix")
B.append(("table", "app", "Settings chosen on the validation windows (Table A1).", *tbl_app(), 1))
P("Table A1 gives the final settings. The features, the leakage test, all tuning tables and every saved result are in notebook 13 and in outputs/tables/final_*.csv.")


# ------------------------------------------------------------------ numbering and cross references

def number(blocks):
    figs, tabs, sects = {}, {}, {"method": "Section IV"}
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    for b in blocks:
        if b[0] == "fig":
            figs[b[1]] = len(figs) + 1
        elif b[0] == "table" and b[1] != "app":
            tabs[b[1]] = roman[len(tabs)]
    tabs["app"] = "A1"
    return figs, tabs, sects


FIGS_N, TABS_N, SECTS_N = number(B)


def resolve(t):
    t = re.sub(r"\[\[F:(\w+)\]\]", lambda m: f"Fig. {FIGS_N[m.group(1)]}", t)
    t = re.sub(r"\[\[T:(\w+)\]\]", lambda m: f"Table {TABS_N[m.group(1)]}", t)
    t = re.sub(r"\[\[S:(\w+)\]\]", lambda m: SECTS_N[m.group(1)], t)
    return t


def check_cited():
    """Every figure and table must be cited in the running text."""
    body = " ".join(resolve(b[1]) for b in B if b[0] == "p") + " ".join(resolve(x) for b in B if b[0] == "bullets" for x in b[1])
    for k, n in FIGS_N.items():
        assert f"Fig. {n}" in body, f"Fig. {n} ({k}) is never cited"
    for k, n in TABS_N.items():
        if k != "app":
            assert f"Table {n}" in body, f"Table {n} ({k}) is never cited"


check_cited()


# ------------------------------------------------------------------ HTML / PDF

def to_html():
    css = """
    @page { size: A4; margin: 13mm 12mm 13mm 12mm; }
    body { font-family: 'Times New Roman', serif; font-size: 10pt; line-height: 1.25; color: #000; }
    .title { text-align: center; margin-bottom: 3mm; }
    .title h1 { font-size: 17pt; margin: 0 0 2mm 0; font-weight: bold; }
    .authors { display: flex; justify-content: space-around; font-size: 9pt; margin-bottom: 2mm; }
    .authors div { text-align: center; }
    .cols { column-count: 2; column-gap: 7mm; text-align: justify; }
    h2.s { font-size: 9.3pt; text-transform: uppercase; text-align: center; margin: 2.6mm 0 1mm 0; break-after: avoid; }
    h3.s { font-size: 9.4pt; font-style: italic; font-weight: normal; margin: 1.6mm 0 0.6mm 0; break-after: avoid; }
    p { margin: 0 0 1.5mm 0; }
    figure { margin: 1.2mm 0; break-inside: avoid; text-align: center; }
    figure img { max-width: 100%; }
    figcaption, .cap { font-size: 8.3pt; text-align: justify; margin: 0.4mm 0 1.2mm 0; }
    figure.wide { column-span: all; }
    table { border-collapse: collapse; width: 100%; font-size: 7.7pt; margin-bottom: 1.2mm; }
    th, td { border-top: 0.4pt solid #000; border-bottom: 0.4pt solid #000; padding: 0.3mm 0.9mm; text-align: left; }
    th { font-weight: bold; }
    .tbl { break-inside: avoid; margin: 1.2mm 0; }
    ul { margin: 0 0 1.4mm 4mm; padding-left: 3mm; } li { margin-bottom: 0.6mm; }
    ol.refs { font-size: 8.3pt; padding-left: 0; margin: 0; list-style: none; counter-reset: r; } ol.refs li { margin-bottom: 0.5mm; padding-left: 7mm; text-indent: -7mm; counter-increment: r; } ol.refs li:before { content: '[' counter(r) '] '; }
    .ph { background: #FFF2A8; }
    """
    out = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"]
    out.append("<div class='title'><h1>Forecasting Retail SKU Demand: Classical, Machine-Learning and Deep Models Compared on M5 Data</h1>"
               "<div class='authors'>" + "".join(f"<div><b><span class='ph'>[Team member {i} name]</span></b><br><span class='ph'>[Roll no. {i}]</span><br>CSET369, <span class='ph'>[University / department]</span></div>" for i in (1, 2, 3)) + "</div></div>")
    out.append("<div class='cols'>")
    first_p = True
    for blk in B:
        kind = blk[0]
        if kind == "h1":
            out.append(f"<h2 class='s'>{html.escape(blk[1])}</h2>")
        elif kind == "h2":
            out.append(f"<h3 class='s'>{html.escape(blk[1])}</h3>")
        elif kind == "p":
            t = html.escape(resolve(blk[1]))
            out.append(f"<p><b><i>Abstract</i>—</b>{t}</p>" if first_p else f"<p>{t}</p>")
            first_p = False
        elif kind == "fig":
            _, key, path, cap, span = blk
            c = f"<figcaption><b>Fig. {FIGS_N[key]}.</b> {html.escape(cap)}</figcaption>"
            if span == 2:
                out.append(f"</div><figure class='wide'><img src='{path.as_uri()}'>{c}</figure><div class='cols'>")
            else:
                out.append(f"<figure><img src='{path.as_uri()}'>{c}</figure>")
        elif kind == "table":
            _, key, cap, head, rows, span = blk
            t = f"<div class='tbl'><div class='cap'><b>Table {TABS_N[key]}.</b> {html.escape(cap)}</div><table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in head) + "</tr>"
            t += "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>"
            out.append(f"</div><div style='margin:1mm 0'>{t}</div><div class='cols'>" if span == 2 else t)
        elif kind == "bullets":
            out.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in blk[1]) + "</ul>")
        elif kind == "refs":
            out.append("<ol class='refs'>" + "".join(f"<li>{html.escape(x)}</li>" for x in blk[1]) + "</ol>")
    out.append("</div></body></html>")
    return "\n".join(out)


def make_pdf():
    h = OUT / "report.html"
    h.write_text(to_html(), encoding="utf-8")
    pdf = OUT / "final_report.pdf"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", h.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    h.unlink()
    return pdf


# ------------------------------------------------------------------ DOCX

def make_docx():
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(9.5)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.5)
    sec.top_margin = sec.bottom_margin = Cm(1.4)

    def set_cols(section, n):
        pr = section._sectPr
        for c in pr.findall(qn("w:cols")):
            pr.remove(c)
        c = OxmlElement("w:cols")
        c.set(qn("w:num"), str(n))
        c.set(qn("w:space"), "400")
        pr.append(c)

    def new_section(n):
        s = doc.add_section(WD_SECTION.CONTINUOUS)
        set_cols(s, n)
        return s

    def para(text, bold=False, size=None, align=None, italic=False, highlight=False):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold, run.italic = bold, italic
        if size:
            run.font.size = Pt(size)
        if highlight:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if align == "c":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "j":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        return p

    def table(cap, head, rows):
        para(cap, bold=True, size=8, align="j")
        t = doc.add_table(rows=1, cols=len(head))
        t.style = "Table Grid"
        for i, h in enumerate(head):
            t.rows[0].cells[i].text = ""
            r = t.rows[0].cells[i].paragraphs[0].add_run(str(h))
            r.bold, r.font.size = True, Pt(7)
        for row_ in rows:
            cells = t.add_row().cells
            for i, c in enumerate(row_):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(c)).font.size = Pt(7)

    para("Forecasting Retail SKU Demand: Classical, Machine-Learning and Deep Models Compared on M5 Data", bold=True, size=16, align="c")
    for i in (1, 2, 3):
        para(f"[Team member {i} name]  |  [Roll no. {i}]  |  CSET369, [University / department]", size=9, align="c", highlight=True)
    cols = 2
    set_cols(new_section(2), 2)
    first_p = True
    for blk in B:
        kind = blk[0]
        wide = (kind in ("fig", "table") and blk[-1] == 2)
        if wide:
            new_section(1)
        if kind == "h1":
            para(blk[1].upper(), bold=True, size=9.5, align="c")
        elif kind == "h2":
            para(blk[1], italic=True, size=9.5)
        elif kind == "p":
            if first_p:
                p = doc.add_paragraph()
                p.add_run("Abstract—").bold = True
                p.add_run(resolve(blk[1]))
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                first_p = False
            else:
                para(resolve(blk[1]), align="j")
        elif kind == "fig":
            _, key, path, cap, span = blk
            doc.add_picture(str(path), width=Cm(17.4 if span == 2 else 8.5))
            para(f"Fig. {FIGS_N[key]}. {cap}", size=8, align="j")
        elif kind == "table":
            _, key, cap, head, rows, span = blk
            table(f"Table {TABS_N[key]}. {cap}", head, rows)
        elif kind == "bullets":
            for x in blk[1]:
                doc.add_paragraph(x, style="List Bullet").paragraph_format.space_after = Pt(1)
        elif kind == "refs":
            for i, x in enumerate(blk[1], 1):
                para(f"[{i}] {x}", size=8)
        if wide:
            new_section(2)
    p = OUT / "final_report.docx"
    doc.save(p)
    return p


def pages(pdf):
    import pymupdf
    return len(pymupdf.open(pdf))


if __name__ == "__main__":
    pdf = make_pdf()
    docx = make_docx()
    print("abstract words:", wc)
    print("pdf pages:", pages(pdf), "->", pdf.relative_to(ROOT))
    print("docx ->", docx.relative_to(ROOT))
    print("figures:", FIGS_N, "tables:", TABS_N)
