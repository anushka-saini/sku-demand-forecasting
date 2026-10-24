"""
The four headline verdicts, computed from results_facts.json (no Streamlit imports; unit-tested).
Rules were agreed before the dashboard was built and are shown on the Overview page.
"""
from utils.facts import lookup, value

POWER_FOR_YES = 0.80     # a 20% step must be detected at least this often for "Yes, detectable"
STATES = ('TX', 'WI')


def _replicates(f, conclusion_prefix):
    rows = [v for k, v in lookup(f, 'replication').items() if k.split('|')[1].startswith(conclusion_prefix)]
    return bool(rows) and all(r['holds'] for r in rows), len(rows)


def shifts(f):
    adj = lookup(f, 'breaks', 'adjusted_share_with_real_shift', 'CUSUM')
    power20 = value(f, 'breaks', 'injection_power_cusum', 'step_20pct')
    lo, hi = adj['ci95']
    exists = lo > 0
    rep, _ = _replicates(f, 'R1')
    if exists and power20 >= POWER_FOR_YES:
        verdict = 'Yes'
    elif exists:
        verdict = 'Partly'
    else:
        verdict = 'No'
    strength = 'strong' if exists and rep else ('moderate' if exists else 'weak')
    return {
        'question': 'Do demand shifts exist, and can we detect them?', 'verdict': verdict, 'strength': strength,
        'answer': f"An estimated {adj['value']:.1%} of items have a real shift (95% CI {lo:.1%} to {hi:.1%}), "
                  f"after removing chance alarms.",
        'caveat': f"The detector is coarse: a sudden 20% shift is caught only {power20:.0%} of the time.",
        'page': 'views/demand_shifts.py'}


def best_model(f):
    med = {m: value(f, 'forecasts', 'median_item_mase', f'{m}_ALL') for m in ('seasonal_naive', 'ets', 'lightgbm')}
    best = min(med, key=med.get)
    others = [m for m in med if m != best]
    ca_sig = all(value(f, 'forecasts', 'pairwise', f'{best}-{o}|ALL') < 0 and
                 lookup(f, 'forecasts', 'pairwise', f'{best}-{o}|ALL')['p_value'] < 0.05
                 for o in others if f'{best}-{o}|ALL' in lookup(f, 'forecasts', 'pairwise'))
    rep_sig = all(lookup(f, 'replication_tests', f'{s}|{best} - {o}')['p_value'] < 0.05
                  for s in STATES for o in others if f'{s}|{best} - {o}' in lookup(f, 'replication_tests'))
    strength = 'strong' if ca_sig and rep_sig else ('moderate' if ca_sig else 'weak')
    bias = value(f, 'forecasts', 'bias', f'{best}_static_ALL')
    names = {'lightgbm': 'LightGBM', 'ets': 'ETS', 'seasonal_naive': 'Seasonal naive'}
    return {
        'question': 'Which forecast is best?', 'verdict': names[best], 'strength': strength,
        'answer': f"{names[best]}: median error {med[best]:.3f} vs {med['seasonal_naive']:.3f} for the simple "
                  f"baseline (MASE; lower is better).",
        'caveat': (f"It under-forecasts by {abs(bias):.1%} in total." if bias < 0 else f"It over-forecasts by {bias:.1%} in total."),
        'page': 'views/accuracy.py'}


def retraining(f):
    var = lookup(f, 'forecasts', 'adaptation_variants')
    n_variants = len({k.split('|')[1] for k in var})
    any_helps = any(v['helps'] for v in var.values())
    helps_all = any(v['helps'] for k, v in var.items() if k.startswith('all detected'))
    rep, n_rep = _replicates(f, 'R4')
    trunc = lookup(f, 'forecasts', 'adaptation_variants', 'all detected breaks|truncate')
    if helps_all:
        verdict = 'Yes'
    elif any_helps:
        verdict = 'Partly'
    else:
        verdict = 'No'
    strength = 'strong' if verdict == 'No' and rep else 'moderate'
    return {
        'question': 'Does retraining after a shift help?', 'verdict': verdict, 'strength': strength,
        'answer': f"None of {n_variants} retraining methods improved forecasts. Simply restarting from the shift made "
                  f"error slightly worse ({trunc['value']:+.4f} MASE, n = {trunc['n']} items)"
                  + (f", and the same holds in {n_rep} other states." if rep else "."),
        'caveat': "Tested with one model (LightGBM) and four simple methods; small shifts are invisible to the detector.",
        'page': 'views/demand_shifts.py'}


def stock(f):
    md = lookup(f, 'inventory', 'multi_window_cost_differences')
    rows = {k: v for k, v in md.items() if k.endswith('|static - naive|0.95')}
    n_win = len(rows)
    n_sig = sum(v['ci95'][1] < 0 for v in rows.values())
    rep, _ = _replicates(f, 'R5')
    base = lookup(f, 'inventory', 'static_vs_naive_base_95')
    if n_sig >= 2 and rep:
        verdict = 'Yes'
    elif n_sig >= 1:
        verdict = 'Partly'
    else:
        verdict = 'No'
    strength = 'strong' if n_sig == n_win and rep else ('moderate' if verdict != 'No' else 'weak')
    return {
        'question': 'Does a better forecast cut stock cost?', 'verdict': verdict, 'strength': strength,
        'answer': f"At a 95% target, LightGBM-based reorder points hold {abs(base['inventory_change']):.1%} less stock at "
                  f"{abs(base['value']):.1%} lower cost than the simple baseline; clearly lower in {n_sig} of {n_win} time windows.",
        'caveat': "Lead time and costs are assumptions, not measured data.",
        'page': 'views/stock_impact.py'}


def all_verdicts(f):
    return [shifts(f), best_model(f), retraining(f), stock(f)]


# ---------------------------------------------------------------- classical models

def classical_vs_naive(f):
    """H1: do SARIMA and Holt-Winters additive beat seasonal naive? Yes if both, Partly if one."""
    c = f['classical']
    ok = c['h1_supported']
    n_ok = sum(bool(v) for v in ok.values())
    verdict = 'Yes' if n_ok == len(ok) else ('Partly' if n_ok else 'No')
    med = {m: c['metrics'][f'{m}|ALL|mase']['value'] for m in ('sarima', 'hw_add', 'seasonal_naive')}
    n = c['metrics']['sarima|ALL|mase']['n']
    return {
        'question': 'Do classical models beat the simple baseline?', 'verdict': verdict,
        'strength': 'moderate' if verdict != 'No' else 'weak',
        'answer': f"Median MASE: SARIMA {med['sarima']:.3f}, Holt-Winters {med['hw_add']:.3f}, seasonal naive "
                  f"{med['seasonal_naive']:.3f} (n = {n} items, Holm-corrected Wilcoxon).",
        'caveat': "150-item sample in California only; classical models fit on the last 730 days.",
        'page': 'views/accuracy.py'}


def classical_shift(f):
    """H3: does a shift-aware SARIMA (restart or level dummy) meet the pre-registered 'helps' rule?"""
    c = f['classical']
    sa = c['shift_aware']
    r, d = sa['all break item-origins|restart'], sa['all break item-origins|dummy']
    verdict = 'No' if c['h3_supported'] else 'Yes'
    return {
        'question': 'Does a shift-aware SARIMA help?', 'verdict': verdict, 'strength': 'moderate',
        'answer': f"Restart from the break: {r['value']:+.4f} MASE; level-shift dummy: {d['value']:+.4f} "
                  f"(median per item, n = {r['n']} items with a break).",
        'caveat': ("The same answer as for LightGBM. " if verdict == 'No' else "Unlike LightGBM. ")
                  + "Breaks older than the 730-day fit window leave the model unchanged.",
        'page': 'views/demand_shifts.py'}


def classical_verdicts(f):
    """Classical-model cards, or [] if those notebooks have not been run."""
    return [classical_vs_naive(f), classical_shift(f)] if 'classical' in f else []
