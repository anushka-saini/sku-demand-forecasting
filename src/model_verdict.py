"""The 'which model is best' verdict, written from the 07i tables. Pure pandas, so the notebook and the dashboard
print the same sentences from the same numbers.

The rule: RMSE is the measure fixed before the run for choosing the stock model. The model with the lowest RMSE leads;
every model whose gap to it is a tie (95% interval of the paired difference includes zero, or not significant after
Holm) shares the lead. A shared lead is broken on the average rank over the six measures, then on stock cost.
"""
LABEL = {"rmse": "RMSE", "mae": "MAE", "mape": "MAPE", "smape": "sMAPE", "wape": "WAPE", "mase": "MASE"}
TESTED = ["rmse", "mae", "mase"]


def _list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def against(tests, ref, metric):
    """The other models against ref on one measure: model -> row with result ('better', 'tie', 'worse') and gap_pct."""
    t = tests[(tests["reference"] == ref) & (tests["metric"] == metric)]
    return t.set_index("model")


def recommend(metrics, ranks, tests, stock, key="rmse"):
    """(recommended model, the models sharing the lead on key)."""
    lead = metrics[key].idxmin()
    t = against(tests, lead, key)
    group = [lead] + [m for m in t.index if t.loc[m, "result"] in ("tie", "better")]
    if len(group) == 1:
        return lead, group
    g = ranks.loc[group, ["avg_rank"]].assign(cost=stock.reindex(group)["total_cost"].to_numpy())
    return g.sort_values(["avg_rank", "cost"]).index[0], group


def verdict(metrics, ranks, tests, wins, stock, names, origins, n_products):
    """
    metrics: median per model (index) for the six measures, all products. ranks: avg_rank, n_first, n_top3 per model.
    tests: metric, reference, model, gap_pct, result. wins: metric, model, wins. stock: total_cost, fill_rate per model
    (safety stock from past errors). Returns (recommended model, list of sentences).
    """
    nm = lambda m: names.get(m, m)
    out = []

    # the lowest typical error on each measure
    by = {}
    for k in LABEL:
        by.setdefault(metrics[k].idxmin(), []).append(k)
    out.append("Lowest typical error by measure: " + "; ".join(
        f"{nm(m)} on " + _list(f"{LABEL[k]} ({metrics.loc[m, k]:.3f})" for k in ks) for m, ks in by.items()) + ".")

    # which gaps are real and which are ties, against the leader of each tested measure
    for k in TESTED:
        lead = metrics[k].idxmin()
        t = against(tests, lead, k)
        tie = t[t["result"] == "tie"]
        worse = t[t["result"] == "worse"]
        better = t[t["result"] == "better"]
        s = f"On {LABEL[k]}, {nm(lead)} has the lowest median, but product by product " + _list(
            nm(m) for m in better.index) + " beat it. It " if len(better) else f"On {LABEL[k]}, {nm(lead)} "
        if len(tie):
            s += ("is tied with " + _list(f"{nm(m)} at {r.gap_pct:+.1%}" for m, r in tie.iterrows())
                  + " (median difference per product; its interval includes zero), so that part of its lead is not real")
            s += (f"; it is really ahead of the other {len(worse)}, by {worse['gap_pct'].min():.1%} to "
                  f"{worse['gap_pct'].max():.1%}." if len(worse) else ".")
        else:
            s += (f"is really ahead of all {len(worse)} others, by {worse['gap_pct'].min():.1%} to "
                  f"{worse['gap_pct'].max():.1%}.")
        # a tie can hide a large gap in the typical error: close on most products, far off on a few
        for m, r in tie.iterrows():
            typical = metrics.loc[m, k] / metrics.loc[lead, k] - 1
            if typical > 0.03 and typical > 2 * abs(r.gap_pct):
                s += (f" {nm(m)}'s typical {LABEL[k]} is {typical:.1%} higher, but it is close on most products and far "
                      f"off on a few, so the paired test cannot separate the two.")
        out.append(s)

    # the most consistent model
    best_rank = ranks["avg_rank"].min()
    top = ranks[ranks["avg_rank"] == best_rank]
    r = top.iloc[0]
    out.append((f"The most consistent model is {nm(top.index[0])}, with the best average rank, {best_rank:.2f} over the six "
                f"measures (first on {int(r['n_first'])}, in the top three on {int(r['n_top3'])})." if len(top) == 1 else
                f"{_list(nm(m) for m in top.index)} share the best average rank, {best_rank:.2f} over the six measures."))

    # the recommendation for stock planning
    rec, group = recommend(metrics, ranks, tests, stock)
    st = stock.reindex(group)
    if len(group) == 1:
        s = (f"For stock planning: {nm(rec)}. It has the lowest RMSE, the measure fixed before the run for choosing the "
             f"stock model, and its gap to every other model is real.")
    else:
        s = (f"For stock planning: {nm(rec)}. The top models on RMSE, the measure fixed before the run, are tied ("
             + _list(nm(m) for m in group) + "), so the tie is broken on average rank ("
             + ", ".join(f"{nm(m)} {ranks.loc[m, 'avg_rank']:.2f}" for m in group) + ") and then on stock cost.")
    if st["total_cost"].notna().all():
        s += (f" With safety stock sized from past errors it costs ${st.loc[rec, 'total_cost']:,.0f} over the simulation, "
              f"with a fill rate of {st.loc[rec, 'fill_rate']:.1%}")
        others = [m for m in group if m != rec]
        s += ("; " + _list(f"{nm(m)} ${st.loc[m, 'total_cost']:,.0f}" for m in others) if others else "") + "."
        if others and st["total_cost"].idxmin() != rec:
            s += (f" {nm(st['total_cost'].idxmin())} is cheaper in the simulation, but the costs are assumptions and come "
                  f"from one 168-day run, so they only break a tie in average rank.")
    out.append(s)

    # accuracy and money side by side, over all the models
    cost = stock["total_cost"].dropna().sort_values()
    if rec in cost.index:
        cheaper = cost[cost < cost[rec]]
        out.append(f"In the stock simulation {nm(rec)} is the cheapest of the {len(cost)} models." if cheaper.empty else
                   f"In the stock simulation {nm(rec)} ranks {list(cost.index).index(rec) + 1} of {len(cost)} on cost: "
                   + _list(f"{nm(m)} (${c:,.0f})" for m, c in cheaper.items())
                   + f" cost less than its ${cost[rec]:,.0f}, so a lower forecast error does not turn one for one into "
                     f"a lower stock cost. The costs rest on assumed prices and one 168-day run.")

    # product by product
    w = wins[wins["metric"] == "rmse"].set_index("model")["wins"]
    m = w.idxmax()
    out.append((f"Product by product no model wins most of the {n_products}: on RMSE the most wins go to {nm(m)} "
                f"({int(w[m])} products)." if w[m] < n_products / 2 else
                f"Product by product {nm(m)} wins {int(w[m])} of the {n_products} products on RMSE."))

    out.append(f"Limits: one evaluation window of {len(origins)} forecast dates ({min(origins):%d %b %Y} to "
               f"{max(origins):%d %b %Y}), {n_products} products, and every setting tuned on earlier validation dates "
               f"only, so another window or another set of products could reorder models this close.")
    return rec, out
