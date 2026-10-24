"""
Compute every headline number from the committed result tables in outputs/tables/ and write
outputs/results_facts.json. Each fact records its value, n, CI where relevant, and the source file.

Run: python scripts/build_facts.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs" / "results_facts.json"


def num(x):
    if isinstance(x, (list, tuple)):
        return [num(v) for v in x]
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return float(x)


def fact(value, source, n=None, ci=None, **extra):
    out = {'value': num(value), 'source': source}
    if n is not None:
        out['n'] = num(n)
    if ci is not None:
        out['ci95'] = [num(ci[0]), num(ci[1])]
    out.update({k: (num(v) if not isinstance(v, str) else v) for k, v in extra.items()})
    return out


def read(name):
    path = T / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the notebook that creates it (see Makefile).")
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def build():
    F = {}

    # ---------------- scope
    items = read("app/items.csv")
    scope = items[items['in_scope']]
    cat = read("app/category_summary.csv")
    F['scope'] = {
        'ca_items': fact(len(items), 'app/items.csv'),
        'ax_ay_items': fact(len(scope), 'app/items.csv'),
        'ax_ay_share_of_items': fact(len(scope) / len(items), 'app/items.csv', n=len(items)),
        'ax_ay_share_of_units': fact(scope['total_sales'].sum() / items['total_sales'].sum(), 'app/items.csv'),
        'items_with_over_30_leading_zero_days': fact(cat['items_with_leading_zeros'].sum(), 'app/category_summary.csv',
                                                     n=cat['total_items'].sum()),
    }

    # ---------------- breaks
    rates = {c: fact(g['cusum_first_break'].mean(), 'app/items.csv', n=len(g))
             for c, g in scope.groupby('category')}
    rates['ALL'] = fact(scope['cusum_first_break'].mean(), 'app/items.csv', n=len(scope))
    cal = read("break_calibration.csv").set_index('method')
    agree = read("break_method_agreement_summary.csv").iloc[0]
    null = read("calibration_null_rates.csv")
    nall = null[null['group'] == 'ALL (seasonal null)'].set_index('method')
    inj = read("calibration_injection.csv")
    power = read("break_power_check.csv")
    align = read("break_event_alignment.csv")
    a4 = align[(align['event'] == 'sustained price change') & (align['window_wks'] == 4)].iloc[0]
    fun = read("hc_breaks_funnel.csv").set_index(read("hc_breaks_funnel.csv").columns[0])
    F['breaks'] = {
        'cusum_break_rate_full_series': rates,
        'old_null_false_alarm_rate': {m: fact(cal.loc[m, 'heldout_false_alarm_rate'], 'break_calibration.csv',
                                              n=cal.loc[m, 'n_heldout_surrogates'],
                                              note='4-week block bootstrap of the reference year; threshold set on the same null')
                                      for m in ['CUSUM', 'PELT']},
        'seasonal_null_false_alarm_rate': {m: fact(nall.loc[m, 'far'], 'calibration_null_rates.csv', n=nall.loc[m, 'n_null_series'])
                                           for m in ['CUSUM', 'PELT']},
        'adjusted_share_with_real_shift': {m: fact(nall.loc[m, 'adjusted'], 'calibration_null_rates.csv', n=nall.loc[m, 'n_items'],
                                                   ci=(nall.loc[m, 'ci_low'], nall.loc[m, 'ci_high']),
                                                   observed=nall.loc[m, 'observed']) for m in ['CUSUM', 'PELT']},
        'cusum_pelt_agreement': fact(agree['raw_agreement'], 'break_method_agreement_summary.csv', n=agree['n_items'],
                                     cohens_kappa=agree['cohens_kappa'], within_8_weeks=agree['within_8_wks']),
        'real_series_alarm_after_random_date_no_injection': fact(
            power.loc[power['size_sd'] == 0, 'cusum_detection_rate'].iloc[0], 'break_power_check.csv',
            n=power.loc[power['size_sd'] == 0, 'n_trials'].iloc[0],
            note='not a false-alarm rate: real series, no ground truth'),
        'injection_power_cusum': {f"{r.shape}_{int(round(100 * r.size))}pct": fact(r.cusum_power, 'calibration_injection.csv',
                                  n=r.n_trials, median_delay_weeks=r.cusum_median_delay_wks, power_within_26_weeks=r.cusum_power_26wk)
                                  for r in inj.itertuples()},
        'price_change_alignment_4wk': fact(a4['aligned_pct'] / 100, 'break_event_alignment.csv', n=a4['n_breaks'],
                                           chance=a4['chance_pct'] / 100, p_value=a4['p_value']),
        'high_confidence_breaks': {c: fact(fun.loc[c, '4_high_confidence'], 'hc_breaks_funnel.csv', n=fun.loc[c, 'cusum_breaks'],
                                           share_of_breaks=fun.loc[c, 'hc_share_of_breaks']) for c in fun.index},
    }

    # ---------------- forecasts
    mase = read("model_mase_unrounded.csv")
    tests = read("adaptive_vs_static_tests.csv")
    var = read("adaptation_variants_tests.csv")
    lge = read("lgb_vs_ets_test.csv")
    bias = read("forecast_bias.csv")
    cov = read("conformal_coverage.csv")

    def tfact(r, src):
        return fact(r['median_diff'], src, n=r['n'], ci=(r['ci_low'], r['ci_high']), p_value=r['p_value'],
                    rank_biserial=r['rank_biserial'], share_improved=r['share_improved'])

    F['forecasts'] = {
        'median_item_mase': {f"{r.model}_{r.category}": fact(r.median_item_mase, 'model_mase_unrounded.csv', n=r.n_items)
                             for r in mase.itertuples()},
        'adaptive_minus_static_mase': {f"{r['model']}_{r['comparison'].split()[0]}_{r['category']}": tfact(r, 'adaptive_vs_static_tests.csv')
                                       for _, r in tests.iterrows()},
        'adaptation_variants': {f"{r['group']}|{r['variant']}": dict(tfact(r, 'adaptation_variants_tests.csv'),
                                                                      p_holm=num(r['p_holm']), helps=bool(r['helps']))
                                for _, r in var.iterrows()},
        'lightgbm_minus_ets_mase': {r['category']: tfact(r, 'lgb_vs_ets_test.csv') for _, r in lge.iterrows()},
        'bias': {f"{r.model}_{r.variant}_{r.category}": fact(r.pooled_bias, 'forecast_bias.csv', n=r.n_items,
                                                             median_item_bias=r.median_item_bias,
                                                             median_item_bias_ci95=[num(r.ci_low), num(r.ci_high)])
                 for r in bias.itertuples()},
        'band_coverage': {f"{r.model}_{r.variant}_{r.level}_{'origins2-6' if '2-6' in r.origins else 'origin1'}":
                          fact(r.coverage_after, 'conformal_coverage.csv', n=r.n_days, coverage_before=r.coverage_before)
                          for r in cov.itertuples()},
    }

    # ---------------- inventory (base window, lead time 7)
    kpi = read("pbi_fact_inventory_kpi.csv")
    k7 = kpi[kpi['lead_time'] == 7].groupby(['policy', 'service_level']).agg(
        demand=('demand_units', 'sum'), lost=('lost_units', 'sum'), inv=('avg_inventory_value', 'sum'),
        cost=('total_cost', 'sum'), stockout_days=('stockout_days', 'sum'), n=('item_id', 'nunique')).reset_index()
    cal_inv = read("inventory_calibrated_vs_normal.csv")
    rob = read("robustness_cost_differences.csv")
    nv = read("robustness_newsvendor.csv")
    be = read("robustness_breakeven.csv")
    F['inventory'] = {
        'base_window_lead7': {f"{r.policy}_{r.service_level}": fact(1 - r.lost / r.demand, 'pbi_fact_inventory_kpi.csv', n=r.n,
                                                                    metric='fill_rate', avg_inventory_value=r.inv,
                                                                    total_cost=r.cost, stockout_days=r.stockout_days)
                              for r in k7.itertuples()},
        'calibrated_vs_normal': {f"{r.method}|{r.policy}_{r.service_level}": fact(r.total_cost, 'inventory_calibrated_vs_normal.csv',
                                                                                  n=r.n_items, fill_rate=r.fill_rate,
                                                                                  cycle_service_level=r.cycle_service_level,
                                                                                  avg_inventory_value=r.avg_inventory_value)
                                 for r in cal_inv.itertuples()},
        'multi_window_cost_differences': {f"{r.window}|{r.comparison}|{r.service_level}": fact(
            r.total_diff, 'robustness_cost_differences.csv', n=r.n_items, ci=(r.ci_low, r.ci_high),
            total_diff_pct=r.total_diff_pct, median_item_diff=r.median_item_diff, top10_share_of_abs_gap=r.top10_share_of_abs_gap)
            for r in rob.itertuples()},
        'newsvendor_lightgbm_static': {f"{r.window}|h{r.holding_rate}|m{r.stockout_multiple}": fact(
            r.simulated_cheapest_level, 'robustness_newsvendor.csv', critical_ratio=r.critical_ratio)
            for r in nv[nv['policy'] == 'lightgbm_static'].itertuples()},
        'breakeven_stockout_to_holding_ratio_99_vs_95': {f"{r.window}|{r.policy}": fact(r.breakeven_stockout_to_holding_ratio,
                                                                                       'robustness_breakeven.csv')
                                                         for r in be.itertuples()},
    }

    pw = read("model_pairwise_tests.csv")
    F['forecasts']['pairwise'] = {f"{r['model_a']}-{r['model_b']}|{r['category']}": tfact(r, 'model_pairwise_tests.csv')
                                  for _, r in pw.iterrows()}

    s95 = k7[k7['service_level'] == 0.95].set_index('policy')
    F['inventory']['static_vs_naive_base_95'] = fact(
        s95.loc['lightgbm_static', 'cost'] / s95.loc['seasonal_naive_static', 'cost'] - 1, 'pbi_fact_inventory_kpi.csv',
        n=s95.loc['lightgbm_static', 'n'], metric='relative total cost change, lead time 7, base costs',
        inventory_change=s95.loc['lightgbm_static', 'inv'] / s95.loc['seasonal_naive_static', 'inv'] - 1)
    F['inventory']['adaptive_vs_static_base_95'] = fact(
        s95.loc['lightgbm_adaptive', 'cost'] / s95.loc['lightgbm_static', 'cost'] - 1, 'pbi_fact_inventory_kpi.csv',
        n=s95.loc['lightgbm_adaptive', 'n'], metric='relative total cost change, lead time 7, base costs')
    c = cal_inv[cal_inv['policy'] == 'lightgbm_static'].set_index(['method', 'service_level'])
    F['inventory']['conformal_vs_normal_cost_change'] = {
        str(sl): fact(c.loc[('conformal', sl), 'total_cost'] / c.loc[('normal formula', sl), 'total_cost'] - 1,
                      'inventory_calibrated_vs_normal.csv', n=c.loc[('conformal', sl), 'n_items'])
        for sl in sorted(cal_inv['service_level'].unique())}

    # ---------------- planner watchlist: LightGBM static, lead time 7, items below the target fill rate
    items_cal = read("inventory_calibrated_items.csv")
    w = items_cal[(items_cal['model'] == 'lightgbm') & (items_cal['variant'] == 'static')]
    F['planner'] = {f"{m}|{sl}": fact(int((g['fill_rate'] < sl).sum()), 'inventory_calibrated_items.csv', n=len(g),
                                      metric='items whose simulated fill rate is below the target')
                    for (m, sl), g in w.groupby(['method', 'service_level'])}

    win = read("robustness_windows.csv")
    F['scope']['windows'] = {r.window: {'first_eval_day': r.first_eval_day, 'last_eval_day': r.last_eval_day,
                                        'n_items': int(r.n_items), 'seasonal_coverage': r.seasonal_coverage,
                                        'source': 'robustness_windows.csv'} for r in win.itertuples()}

    # ---------------- replication
    rep = read("replication_conclusions.csv")
    F['replication'] = {f"{r.state}|{r.conclusion}": {'holds': bool(r.holds), 'evidence': r.evidence,
                                                      'source': 'replication_conclusions.csv'} for r in rep.itertuples()}
    rt = read("replication_tests.csv")
    F['replication_tests'] = {f"{r['state']}|{r['test']}": tfact(r, 'replication_tests.csv') for _, r in rt.iterrows()}
    rs = read("replication_scope.csv")
    F['replication_scope'] = {r.state: fact(int(rs[rs['state'] == r.state]['ax_ay_candidates'].sum()), 'replication_scope.csv')
                              for r in rs.drop_duplicates('state').itertuples()}
    c = build_classical()
    if c:
        F['classical'] = c
    return F


# ---------------------------------------------------------------- classical models, stationarity and decomposition

CLASSICAL_FILES = ["classical_sample_items.csv", "stationarity_summary.csv", "comparison_metrics_summary.csv", "comparison_tests.csv",
            "classical_h4.csv", "comparison_totals_metrics.csv", "classical_run_stats.csv", "diagnostics_summary.csv",
            "shift_aware_tests.csv", "stock_summary.csv", "stock_tests.csv"]


def build_classical():
    """Classical-model headline numbers. Returns {} if those notebooks have not been run yet."""
    if not all((T / f).exists() for f in CLASSICAL_FILES):
        return {}
    r = {f: pd.read_csv(T / f) for f in CLASSICAL_FILES}
    src = lambda f: f
    M = {}

    s = r["classical_sample_items.csv"]
    M['sample'] = fact(len(s), src("classical_sample_items.csv"), break_items=int(s['break_item'].sum()))

    st = r["stationarity_summary.csv"].set_index("category")
    M['stationarity'] = {f"{lab}|{cat}": fact(st.loc[cat, lab], src("stationarity_summary.csv"), n=st.loc[cat, 'n_items'])
                         for cat in st.index for lab in ['stationary', 'trend-stationary', 'difference-stationary',
                                                         'conflicting', 'd=0', 'd=1', 'D=1']}

    sm = r["comparison_metrics_summary.csv"]
    M['metrics'] = {f"{row.model}|{row.category}|{k}": fact(getattr(row, f"median_{k}"), src("comparison_metrics_summary.csv"),
                                                          n=row.n_items, mean=getattr(row, f"mean_{k}"))
                    for row in sm.itertuples() for k in ['rmse', 'mae', 'mape', 'smape', 'wape', 'mase']}
    M['mape_excluded_share'] = fact(sm.loc[sm['category'] == 'ALL', 'mape_excluded_share'].iloc[0], src("comparison_metrics_summary.csv"))

    M['tests'] = {f"{row['hypothesis']}|{row['model_a']}-{row['model_b']}":
                  fact(row['median_diff'], src("comparison_tests.csv"), n=row['n'], ci=(row['ci_low'], row['ci_high']),
                       p_value=row['p_value'], p_holm=row['p_holm'], a_better=bool(row['a_better']))
                  for _, row in r["comparison_tests.csv"].iterrows()}
    t = r["comparison_tests.csv"]
    M['h1_supported'] = {row['model_a']: bool(row['a_better']) for _, row in t[t['hypothesis'] == 'H1'].iterrows()}
    alls = sm[sm['category'] == 'ALL'].set_index('model')['median_mase']
    M['h2_lowest_median_mase'] = {'model': str(alls.idxmin()), 'value': num(alls.min()), 'source': src("comparison_metrics_summary.csv"),
                                  'significant_vs_all': bool(t[t['hypothesis'] == 'H2']['a_better'].all())}

    h4 = r["classical_h4.csv"]
    M['h4'] = {row['group']: fact(row['median_mase_hw_add'], src("classical_h4.csv"), n=row['n_items'],
                                  hw_mul=row['median_mase_hw_mul'],
                                  share_items_add_better=row.get('share_items_add_better'),
                                  mul_within_5pct=row.get('mul_within_5pct'))
               for _, row in h4.iterrows()}

    ag = r["comparison_totals_metrics.csv"]
    M['aggregate_mase'] = {f"{row.series}|{row.model}": fact(row.mase, src("comparison_totals_metrics.csv"), how=row.how)
                           for row in ag.itertuples()}

    rs = r["classical_run_stats.csv"]
    M['run_stats'] = {f"{row.model}|{row.level}": fact(row.fallback_rate, src("classical_run_stats.csv"), n=row.fits,
                                                      not_converged_rate=row.not_converged_rate)
                      for row in rs.itertuples()}

    dg = r["diagnostics_summary.csv"]
    M['diagnostics'] = {f"{row.model}|{row.level}": fact(row.ljung_box_pass, src("diagnostics_summary.csv"), n=row.n,
                                                        normality_pass=row.normality_pass, roots_ok=row.roots_ok)
                        for row in dg.itertuples()}

    sa = r["shift_aware_tests.csv"]
    M['shift_aware'] = {f"{row['group']}|{row['variant']}":
                        fact(row['median_diff'], src("shift_aware_tests.csv"), n=row['n'], ci=(row['ci_low'], row['ci_high']),
                             p_value=row['p_value'], p_holm=row.get('p_holm'), helps=row.get('helps'),
                             practical_threshold=row.get('practical_threshold'))
                        for _, row in sa.iterrows()}
    main = sa[sa['group'] == 'all break item-origins']
    M['h3_supported'] = not bool(main['helps'].astype(bool).any())

    ss = r["stock_summary.csv"]
    M['stock'] = {f"{row.method}|{row.model}": fact(row.total_cost, src("stock_summary.csv"), n=row.n_items,
                                                   fill_rate=row.fill_rate, cycle_service_level=row.cycle_service_level,
                                                   avg_inventory_value=row.avg_inventory_value)
                  for row in ss.itertuples()}
    stt = r["stock_tests.csv"]
    M['stock_tests'] = {f"{row.method}|{row.model_a}-{row.model_b}":
                        fact(row.total_diff_pct, src("stock_tests.csv"), n=row.n_items, total_cost_diff=row.total_cost_diff,
                             total_ci_dollars=[num(row.total_ci_low), num(row.total_ci_high)],
                             median_fill_rate_diff=row.median_fill_rate_diff, p_value_cost=row.p_value_cost)
                        for row in stt.itertuples()}
    return M


if __name__ == '__main__':
    facts = build()
    OUT.write_text(json.dumps(facts, indent=2))
    n = sum(len(v) for v in facts.values())
    print(f"wrote {OUT.relative_to(ROOT)} ({n} fact groups)")
