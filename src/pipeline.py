"""
Reusable end-to-end pieces so the same frozen pipeline can be run on other windows and states
(docs/PREREGISTRATION.md, Tasks 4-6). Every setting defaults to the frozen CA value.
"""
import numpy as np
import pandas as pd

from break_detection import to_weekly, surrogate_max_stats, weekly_ca_prices
from forecasting import HORIZON, is_intermittent, local_forecasts, training_windows
from inventory import (cycle_service_level, lead_time_error_sums, policy_kpis, safety_stock_empirical,
                       safety_stock_normal, simulate_policy)
from metrics import mase_scale
from segmentation import abc_xyz_segment, scope_and_aggregate, trim_leading_zeros

CATEGORIES = ['FOODS', 'HOBBIES', 'HOUSEHOLD']


# ---------------------------------------------------------------- scoping

def count_leading_zero_items(item_level, day_cols, min_gap=30):
    vals = item_level[day_cols].to_numpy()
    first = np.argmax(vals > 0, axis=1)
    has_sale = (vals > 0).any(axis=1)
    return int(((first > min_gap) & has_sale).sum())


def scope_state(sales, calendar, state):
    """
    Same scoping as 02_scope for one state: sum the state's stores per item, trim leading
    zeros, ABC-XYZ within each category, keep AX/AY. Returns (long_df of candidates with a
    category column, per-category summary DataFrame, segmentation DataFrame).
    """
    parts, summary, segs = [], [], []
    for cat in CATEGORIES:
        item_level, day_cols = scope_and_aggregate(sales, state, cat)
        long_df = item_level.melt(id_vars='item_id', var_name='d', value_name='sales')
        long_df = long_df.merge(calendar[['d', 'date']], on='d', how='left')
        long_df = trim_leading_zeros(long_df)
        stats = abc_xyz_segment(long_df)
        cand = stats.loc[stats['segment'].isin(['AX', 'AY']), 'item_id']
        parts.append(long_df[long_df['item_id'].isin(cand)].assign(category=cat))
        segs.append(stats.assign(category=cat))
        summary.append({'state': state, 'category': cat, 'total_items': len(item_level),
                        'items_with_leading_zeros': count_leading_zero_items(item_level, day_cols),
                        'ax_ay_candidates': len(cand),
                        'unit_share_ax_ay': stats.loc[stats['item_id'].isin(cand), 'total_sales'].sum() / stats['total_sales'].sum()})
    long_df = pd.concat(parts, ignore_index=True).sort_values(['item_id', 'date']).reset_index(drop=True)
    return long_df[['item_id', 'date', 'sales', 'category']], pd.DataFrame(summary), pd.concat(segs, ignore_index=True)


# ---------------------------------------------------------------- backtest

def make_origins(end, n_origins, horizon=HORIZON):
    """n_origins origins spaced `horizon` days apart; the last window ends on `end`."""
    end = pd.Timestamp(end)
    return [end - pd.Timedelta(days=horizon * (n_origins - i)) for i in range(n_origins)]


def calibrate_h(daily, before, k=1.0, burn_in=8, ref_len=52, seed=42, q=0.95):
    """CUSUM threshold by the frozen procedure, using only weeks up to `before`."""
    weekly = {i: to_weekly(d[d['date'] <= before]).to_numpy() for i, d in daily.items()}
    return float(np.quantile(surrogate_max_stats(weekly, burn_in, ref_len, k=k, seed=seed), q))


def run_backtest(long_df, calendar, price_wk, origins, h, snap_col='snap_CA', run_ets=False,
                 n_jobs=-1, log=print, folder=None, best=None):
    """
    Seasonal naive (+ static ETS if run_ets) and the best time-series model from 07h, static and retrained after a
    shift, at each origin, with break detection at each origin using data up to that origin only. Items without a
    break keep the static forecast as their retrained one. folder holds the per-item checkpoints of the best model.
    Returns (forecasts, windows, scales): forecasts has item_id, origin, date, model, variant,
    forecast, actual; scales has item_id, origin, mase_scale.
    """
    import best_model as bm
    import ts_models as tm
    from joblib import Parallel, delayed
    daily = {i: g[['date', 'sales']].reset_index(drop=True) for i, g in long_df.groupby('item_id')}
    windows = training_windows(daily, origins, h)
    starts = {i: {o: {'static': None} for o in origins} for i in daily}
    res = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(local_forecasts)(i, d['date'], d['sales'], origins, starts[i], run_ets=run_ets) for i, d in daily.items())
    local = pd.DataFrame([r for rows in res for r in rows])
    log(f"local models: {len(local)} rows")

    best = best or bm.load_best()
    jobs = 10 if n_jobs == -1 else n_jobs
    orders = bm.orders_for(daily, origins[0], folder, n_jobs=jobs) if bm.needs_orders(best) else None
    flags = tm.calendar_flags(calendar, snap_col)
    prices = {i: tm.relative_price(price_wk[i], calendar) for i in daily if i in price_wk}
    st = bm.starts_from(windows, variants=('adaptive',))
    top = bm.run_best(daily, origins, st, folder, best, orders, flags, prices, n_jobs=jobs)
    log(f"{best['model']}: {len(top)} rows")
    fc = pd.concat([local, top.drop(columns=['converged', 'fallback'])], ignore_index=True)
    fc = fc.merge(long_df[['item_id', 'date', 'sales']].rename(columns={'sales': 'actual'}), on=['item_id', 'date'])

    scales = pd.DataFrame([{'item_id': i, 'origin': o, 'mase_scale': mase_scale(d.loc[d['date'] <= o, 'sales'].to_numpy())}
                           for i, d in daily.items() for o in origins])
    return fc, windows, scales


def item_mase(fc, scales, origins):
    """Per item mean MASE over the given origins, per model and variant."""
    e = fc[fc['origin'].isin(origins)].assign(ae=lambda x: (x['actual'] - x['forecast']).abs())
    e = e.groupby(['item_id', 'origin', 'model', 'variant'], as_index=False)['ae'].mean().merge(scales, on=['item_id', 'origin'])
    e['mase'] = e['ae'] / e['mase_scale']
    return e.groupby(['item_id', 'model', 'variant'], as_index=False)['mase'].mean()


# ---------------------------------------------------------------- inventory

def price_daily_frame(price_wk, days):
    """Daily unit price per item (the week's price, forward-filled), indexed by day."""
    days = pd.DatetimeIndex(days)
    week_start = days - pd.to_timedelta((days.dayofweek - 5) % 7, unit='D')
    out = price_wk.ffill().reindex(week_start)
    out.index = days
    return out


def normal_or_empirical_ss(past_errors, lead_time, service_level, intermittent):
    """The 11b rule: empirical lead-time quantile for intermittent items, normal formula otherwise."""
    if intermittent:
        return safety_stock_empirical(lead_time_error_sums(past_errors, lead_time), service_level)
    return safety_stock_normal(np.std(np.concatenate(past_errors), ddof=1), lead_time, service_level)


def simulate_inventory(fc, origins, price_daily, policies, service_levels, lead_time, intermittent,
                       holding_rate=0.25, stockout_multiple=0.5, review_days=7, ss_fn=None):
    """
    Notebook 03's simulation as a function. origins[0] is the warm-up; origins[1:] are simulated.
    ss_fn(item, model, variant, origin, lead_time, service_level, past_errors) -> safety stock;
    default is the 11b rule. Returns one row per item x policy x service level, including the
    cost bases (holding_value_days, lost_value) and achieved cycle service level counts.
    """
    eval_origins = origins[1:]
    days = pd.date_range(eval_origins[0] + pd.Timedelta(days=1), periods=HORIZON * len(eval_origins))
    sub = fc[fc.set_index(['model', 'variant']).index.isin(policies) & fc['origin'].isin(origins)]
    rows = []
    for (item, model, variant), g in sub.groupby(['item_id', 'model', 'variant']):
        g = g.sort_values(['origin', 'date'])
        ev = g[g['origin'].isin(eval_origins)]
        if len(ev) != len(days):
            continue
        demand = ev['actual'].to_numpy()
        window_fcs = [w['forecast'].to_numpy() for _, w in ev.groupby('origin')]
        errors = {o: (w['actual'] - w['forecast']).to_numpy() for o, w in g.groupby('origin')}
        price = price_daily[item].to_numpy()
        for sl in service_levels:
            ss = []
            for o in eval_origins:
                past = [e for po, e in errors.items() if po < o]
                ss.append(ss_fn(item, model, variant, o, lead_time, sl, past) if ss_fn
                          else normal_or_empirical_ss(past, lead_time, sl, intermittent[item]))
            sim = simulate_policy(demand, window_fcs, ss, lead_time=lead_time, review_days=review_days)
            k = policy_kpis(sim, demand, price, holding_rate, stockout_multiple)
            ok, n = cycle_service_level(sim, lead_time)
            k.update(item_id=item, model=model, variant=variant, policy=f"{model}_{variant}", lead_time=lead_time,
                     service_level=sl, demand_units=demand.sum(), cycles_ok=ok, cycles=n,
                     holding_value_days=np.sum(price * sim['on_hand']) / 365, lost_value=np.sum(price * sim['lost']),
                     mean_safety_stock=np.mean(ss))
            rows.append(k)
    return pd.DataFrame(rows)


def summarise_inventory(kpis, by=('policy', 'service_level')):
    g = kpis.groupby(list(by))
    out = g.agg(n_items=('item_id', 'nunique'), demand=('demand_units', 'sum'), lost=('lost_units', 'sum'),
                stockout_days=('stockout_days', 'sum'), avg_inventory_value=('avg_inventory_value', 'sum'),
                holding_cost=('holding_cost', 'sum'), stockout_cost=('stockout_cost', 'sum'),
                total_cost=('total_cost', 'sum'), cycles_ok=('cycles_ok', 'sum'), cycles=('cycles', 'sum'),
                holding_value_days=('holding_value_days', 'sum'), lost_value=('lost_value', 'sum')).reset_index()
    out['fill_rate'] = 1 - out['lost'] / out['demand']
    out['cycle_service_level'] = out['cycles_ok'] / out['cycles']
    return out


def intermittency(long_df, before):
    return {i: is_intermittent(g.loc[g['date'] <= before, 'sales']) for i, g in long_df.groupby('item_id')}


def state_prices(prices, calendar, item_ids, state):
    return weekly_ca_prices(prices, calendar, item_ids, state=state)
