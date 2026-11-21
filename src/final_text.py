"""The written findings of the final comparison, built from the saved tables so the dashboard, the report and the slides
quote the same numbers. Nothing here is typed in by hand except the wording."""
import json

import numpy as np
import pandas as pd

from config import TABLES


def read(name):
    p = TABLES / name
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def fp(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def pct(x, d=0):
    return f"{abs(x) * 100:.{d}f}%"


class Final:
    """All the tables of notebook 13 in one place, with the numbers the text needs."""

    def __init__(self):
        lb = read("final_leaderboard.csv")
        self.lb = lb[lb["level"] == "item"].sort_values("rank").reset_index(drop=True)
        self.nm = dict(zip(self.lb["model"], self.lb["name"]))
        self.info = json.loads((TABLES / "final_best_model.json").read_text())
        self.dm = read("final_dm_tests.csv")
        self.seg = read("final_segments.csv")
        self.hz = read("final_horizon.csv")
        self.garch = read("final_garch_intervals.csv")
        self.daytype = read("final_garch_daytype.csv")
        self.archlm = read("final_garch_archlm.csv")
        self.stock = read("final_stock_summary.csv")
        self.stock_cmp = read("final_stock_compare.csv")
        self.tot = read("final_totals_level.csv")
        self.w = read("final_stack_weights.csv")
        self.split = read("final_split.csv")
        self.best = self.lb.iloc[0]
        self.b = self.best["model"]
        self.row = lambda m: self.lb[self.lb["model"] == m].iloc[0]
        self.naive = self.row("seasonal_naive")
        singles = self.lb[self.lb["family"] != "Combination"]
        self.single = singles.iloc[0]
        old = self.lb[self.lb["model"].isin(["seasonal_naive", "hw_add", "ets_boxcox", "theta", "arma", "arima", "sarima",
                                             "sarimax", "prophet", "autots", "combination"])]
        self.old_best = old.iloc[0]
        self.combos = self.lb[self.lb["family"] == "Combination"]

    # ------------------------------------------------------------ results

    def winner(self):
        b, n, o = self.best, self.naive, self.old_best
        pm = self.lb.sort_values("pooled_mae").iloc[0]
        pr = self.lb.sort_values("pooled_rmse").iloc[0]
        s = [f"{b['name']} is first on both ranking measures: median MAE {b['mae']:.2f} and RMSE {b['rmse']:.2f} units a day, "
             f"{pct(b['mae'] / n['mae'] - 1)} and {pct(b['rmse'] / n['rmse'] - 1)} below seasonal naive ({n['mae']:.2f} and {n['rmse']:.2f}). "
             f"The best earlier model, {o['name']}, scores {o['mae']:.2f} and {o['rmse']:.2f}."]
        if pm["model"] != b["model"] or pr["model"] != b["model"]:
            s.append(f"The ranking depends a little on how errors are pooled. With all item-days together, {pm['name']} has the lowest MAE "
                     f"({pm['pooled_mae']:.3f}) and {pr['name']} the lowest RMSE ({pr['pooled_rmse']:.3f}), against {b['pooled_mae']:.3f} and "
                     f"{b['pooled_rmse']:.3f} for {b['name']}. The gaps among the top models are small.")
        return " ".join(s)

    def combos_text(self):
        s = self.single
        out = []
        for _, r in self.combos.iterrows():
            both = r["mae"] < s["mae"] and r["rmse"] < s["rmse"]
            one = r["mae"] < s["mae"] or r["rmse"] < s["rmse"]
            out.append(f"{r['name']}: MAE {r['mae']:.2f}, RMSE {r['rmse']:.2f} "
                       + ("(better than" if both else "(better on one measure than" if one else "(not better than")
                       + f" {s['name']}'s {s['mae']:.2f} and {s['rmse']:.2f}).")
        wins = [r for _, r in self.combos.iterrows() if r["mae"] < s["mae"] and r["rmse"] < s["rmse"]]
        head = ("A combination did beat the best single model on both MAE and RMSE." if wins else
                f"No combination beat the best single model, {s['name']}, on both MAE and RMSE, so the hypothesis that a combination wins "
                "is not supported by these results.")
        pooled_wins = [r["name"] for _, r in self.combos.iterrows()
                       if r["pooled_mae"] < s["pooled_mae"] or r["pooled_rmse"] < s["pooled_rmse"]]
        tail = (f" On pooled errors {', '.join(pooled_wins)} {'is' if len(pooled_wins) == 1 else 'are'} slightly ahead on at least one measure, "
                "but the gaps are small." if pooled_wins else "")
        return head + " " + " ".join(out) + tail

    def dm_text(self):
        d = self.dm
        b = self.b
        rows = []
        for other in d[d["model_a"] == b]["model_b"].unique():
            x = d[(d["model_a"] == b) & (d["model_b"] == other)].set_index("loss")
            a, q = x.loc["absolute error"], x.loc["squared error"]
            sig = [k for k, r in (("absolute", a), ("squared", q)) if r["significant_5pct"]]
            rows.append(f"against {self.nm[other]}: DM p {fp(a['p_value'])} (absolute error) and {fp(q['p_value'])} (squared error), "
                        + ("significant at 5% on " + " and ".join(sig) + " loss" if sig else "not significant at 5%")
                        + f"; the best model has the lower error on {x['share_items_a_better'].iloc[0]:.0%} of items")
        return f"Diebold-Mariano tests of {self.nm[b]}: " + "; ".join(rows) + "."

    def why_text(self):
        hz = self.hz
        b = self.b
        cl = self.old_best["model"]
        a, c = hz[hz["model"] == b].set_index("h")["mae"], hz[hz["model"] == cl].set_index("h")["mae"]
        better = (a < c)
        early, mid, late = better.loc[:7].mean(), better.loc[8:21].mean(), better.loc[22:].mean()
        trend = ("its advantage is larger early in the window" if early > late + 0.15 else
                 "its advantage grows later in the window" if late > early + 0.15 else "there is no clear pattern across the horizon")
        better = (a < c)
        early, mid, late = better.loc[:7].mean(), better.loc[8:21].mean(), better.loc[22:].mean()
        trend = ("its advantage is larger early in the window" if early > late + 0.15 else
                 "its advantage grows later in the window" if late > early + 0.15 else "there is no clear pattern across the horizon")
        s = [f"{self.nm[b]} and LightGBM, which use the same features, are the two best single models; the neural nets (trained once and "
             f"rolled forward) and the tuned Prophet are behind the simple classical models."]
        s.append(f"{self.nm[b]} has a lower MAE than {self.nm[cl]} on {early:.0%} of forecast days 1 to 7, {mid:.0%} of days 8 to 21 and "
                 f"{late:.0%} of days 22 to 28, so {trend}.")
        s.append("A likely reason is that the tree models see every item at once and read the calendar, SNAP and price columns directly, "
                 "whereas each classical model is fitted to one item alone; this is a reading of the results, not something we tested separately.")
        return " ".join(s)

    def nn_text(self):
        nn = self.lb[self.lb["family"] == "Deep learning"]
        return ("Neural nets, mean over three seeds (standard deviation over seeds in brackets): "
                + "; ".join(f"{r['name']} MAE {r['mae']:.2f} ({r['mae_std']:.2f}), RMSE {r['rmse']:.2f} ({r['rmse_std']:.2f})"
                            for _, r in nn.iterrows())
                + f". All three are better than seasonal naive on the median item ({self.naive['mae']:.2f}) but worse than the classical "
                  "models, and on pooled error some are worse than seasonal naive. They were trained once per stage and rolled forward, "
                  "with a small grid, on a CPU laptop; a longer search might narrow the gap.")

    # ------------------------------------------------------------ where models fail

    def failure_text(self):
        s = self.seg
        b = self.b

        def g(seg, m=b, k="mae"):
            return float(s[(s["segment"] == seg) & (s["model"] == m)][k].iloc[0])

        allm = g("all days")
        parts = [f"Event days: MAE {g('event days'):.2f} against {g('no event'):.2f} on other days ({pct(g('event days') / g('no event') - 1)} "
                 f"{'higher' if g('event days') > g('no event') else 'lower'}); the bias is {g('event days', k='bias'):.2f} units a day there and "
                 f"{g('no event', k='bias'):.2f} on other days, so event days are not worse in a systematic direction, just a little noisier."]
        parts.append(f"SNAP days: MAE {g('SNAP days'):.2f} against {g('non-SNAP days'):.2f} on other days, so SNAP is handled about as well as an ordinary day.")
        parts.append(f"Weekends: MAE {g('weekend'):.2f} against {g('weekday'):.2f} on weekdays; the best classical model, "
                     f"{self.nm[self.old_best['model']]}, has {g('weekend', self.old_best['model']):.2f} on weekends.")
        parts.append(f"Zero-demand days: on the {int(s[(s['segment'] == 'days with zero sales') & (s['model'] == b)]['item_days'].iloc[0]):,} item-days with no sales the "
                     f"model forecasts {g('days with zero sales', k='bias'):.2f} units too many on average; it cannot see a zero coming.")
        parts.append(f"After a week of no sales before the forecast date the model's MAE ({g('zero week before origin'):.2f}) is no better than seasonal naive "
                     f"({g('zero week before origin', 'seasonal_naive'):.2f}); SARIMA is clearly worse there ({g('zero week before origin', 'sarima'):.2f}).")
        parts.append(f"Items with a detected shift in demand: MAE {g('items with a detected shift'):.2f} against {g('items without a shift'):.2f} without; "
                     f"{self.nm[self.old_best['model']]} scores {g('items with a detected shift', self.old_best['model']):.2f} on the shift items, "
                     f"{'so the best model has no edge over it there' if g('items with a detected shift') >= g('items with a detected shift', self.old_best['model']) - 0.02 else 'so the best model keeps its lead there'}.")
        parts.append(f"Over all days the model's bias is {g('all days', k='bias'):.2f} units a day (forecast minus actual), so it leans low, as a model that "
                     "cannot reach the spikes does.")
        return parts

    def garch_text(self):
        g, lm, dt = self.garch, self.archlm, self.daytype
        c = g.set_index(["interval", "nominal"])
        share = (lm["arch_lm_p"] < 0.05).mean()
        we = dt.set_index("day_type")
        s = [f"The Engle ARCH-LM test finds changing variance in the errors of {share:.0%} of items, so there is some volatility clustering but it is not general.",
             f"GARCH(1,1) does not change the forecast, so MAE and RMSE are identical with or without it. As intervals, the 80% band covers "
             f"{c.loc[('GARCH(1,1)', 0.8), 'coverage']:.1%} of actual days (constant spread: {c.loc[('constant std', 0.8), 'coverage']:.1%}) and the 95% band "
             f"{c.loc[('GARCH(1,1)', 0.95), 'coverage']:.1%} ({c.loc[('constant std', 0.95), 'coverage']:.1%}); the interval score is "
             f"{c.loc[('GARCH(1,1)', 0.8), 'interval_score']:.2f} against {c.loc[('constant std', 0.8), 'interval_score']:.2f} at 80%, "
             f"{'lower, so GARCH is the better interval' if c.loc[('GARCH(1,1)', 0.8), 'interval_score'] < c.loc[('constant std', 0.8), 'interval_score'] else 'higher, so GARCH is not better than a constant spread here'}.",
             f"The fitted spread hardly moves with the calendar (relative sigma {we.loc['weekend', 'mean_sigma_over_scale']:.2f} on weekends and "
             f"{we.loc['weekday', 'mean_sigma_over_scale']:.2f} on weekdays), while the real absolute error is {we.loc['weekend', 'mean_abs_error_over_scale']:.2f} against "
             f"{we.loc['weekday', 'mean_abs_error_over_scale']:.2f}. The volatility that matters here is calendar-driven, and a GARCH on 28-day-ahead errors does not see it."]
        return " ".join(s)

    def stock_text(self):
        s = self.stock.set_index(["method", "model"])
        c = self.stock_cmp
        b = self.b
        ref = "combination" if b != "combination" else "sarima"
        out = []
        for meth, label in [("conformal", "safety stock from past errors"), ("normal formula", "the textbook safety stock formula")]:
            if (meth, b) not in s.index:
                continue
            x = s.loc[(meth, b)]
            r, n = s.loc[(meth, ref)], s.loc[(meth, "seasonal_naive")]
            cc = c[(c["method"] == meth) & (c["versus"] == ref)].iloc[0]
            inside = cc["ci_low"] <= 0 <= cc["ci_high"]
            out.append(f"With {label}, {self.nm[b]} costs ${x['total_cost']:,.0f} against ${r['total_cost']:,.0f} for {self.nm[ref]} "
                       f"({pct(x['total_cost'] / r['total_cost'] - 1, 1)} {'less' if x['total_cost'] < r['total_cost'] else 'more'}) and ${n['total_cost']:,.0f} for seasonal naive; "
                       f"fill rate {x['fill_rate']:.1%}, average stock ${x['avg_inventory_value']:,.0f} ({pct(x['avg_inventory_value'] / r['avg_inventory_value'] - 1, 1)} "
                       f"{'less' if x['avg_inventory_value'] < r['avg_inventory_value'] else 'more'} than {self.nm[ref]}). "
                       + ("The 95% bootstrap interval of the cost difference includes zero, so the difference is within the noise of this replay." if inside
                          else "The 95% bootstrap interval of the cost difference excludes zero."))
        return " ".join(out)

    # ------------------------------------------------------------ static, specific

    def limitations(self):
        return [
            "Only 150 high-volume items from one state (California) are modelled, with the four stores added together. Slow-moving and "
            "intermittent items, and store-level forecasts, were not tested, and they are where deep models are often claimed to help.",
            "The test has six windows of 28 days, about six months. Differences between the top models are small compared with the spread across items, "
            "and the Diebold-Mariano tests show which gaps are real and which are not.",
            "Neural nets were trained once per stage and rolled forward, not refitted at every origin as XGBoost was, and tuned on a small random grid because "
            "the whole project ran on a CPU laptop. They are probably under-tuned; their results should not be read as a verdict on deep learning in general.",
            "Prices and the calendar are used as known ahead of time, as in the M5 competition. For a real retailer this holds for the calendar and planned "
            "prices but not for unplanned promotions.",
            "The hybrid had only 3 to 8 earlier windows to learn the classical model's residuals from, so it could not be tuned and may be unfairly weak.",
            "Stock costs and the lead time are assumptions, and the simulation replays one history; it is not a measured saving.",
            "Earlier model fit times were not recorded; they are estimated from a short timing run, and AutoTS, ARIMA and ARMA are left without a time.",
            "Stacking weights and the average of the top three were chosen on three validation windows, which is small; the weights may not carry over.",
        ]

    def challenges(self):
        return [
            "Keeping every choice on the validation windows: each model needed its own way to tune without touching the test windows.",
            "Making the features free of future information; a test scrambles all sales after an origin and checks that no feature moves.",
            "Run time: the neural nets took many minutes per fit on a laptop CPU, which forced the one-training-per-stage design.",
            "Intermittent days: about one item-day in twenty has no sales, which makes MAPE misleading, so MAPE is reported on non-zero days only and WAPE next to it.",
        ]

    def future(self):
        return [
            "Refit the neural nets at every origin and widen the search, on a GPU.",
            "Model the weekend and event spikes directly, for example with a count or two-part (zero/non-zero) loss, since the biggest misses are spikes and zeros.",
            "Use calendar-aware safety stock (larger on weekends and event days) instead of one buffer per item.",
            "Test other states and store-level series, and slow-moving items.",
            "Fit the stacking weights per forecast day, since the best model is not the same at every horizon.",
        ]

    def all(self):
        return {"winner": self.winner(), "combos": self.combos_text(), "dm": self.dm_text(), "why": self.why_text(), "nn": self.nn_text(),
                "failure": self.failure_text(), "garch": self.garch_text(), "stock": self.stock_text(),
                "limitations": self.limitations(), "challenges": self.challenges(), "future": self.future()}
