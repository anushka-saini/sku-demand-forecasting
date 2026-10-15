import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from pipeline import count_leading_zero_items, make_origins, price_daily_frame, summarise_inventory


def test_make_origins_matches_notebook_02():
    o = make_origins('2016-05-22', 7)
    assert o[0] == pd.Timestamp('2015-11-08') and o[-1] == pd.Timestamp('2016-04-24')
    assert all((b - a).days == 28 for a, b in zip(o, o[1:]))
    assert make_origins('2015-12-06', 7)[0] == pd.Timestamp('2015-05-24')


def test_count_leading_zero_items_uses_more_than_30_days():
    cols = [f'd_{i}' for i in range(1, 41)]
    rows = np.zeros((3, 40))
    rows[0, 31] = 1          # first sale on day 32 -> counted
    rows[1, 30] = 1          # first sale on day 31 (index 30) -> not counted
    df = pd.DataFrame(rows, columns=cols).assign(item_id=['a', 'b', 'c'])     # c never sells
    assert count_leading_zero_items(df, cols) == 1


def test_price_daily_frame_maps_days_to_walmart_week():
    weeks = pd.DatetimeIndex(['2016-01-02', '2016-01-09'])          # Saturdays
    pw = pd.DataFrame({'X': [1.0, np.nan]}, index=weeks)
    p = price_daily_frame(pw, pd.date_range('2016-01-02', '2016-01-10'))
    assert p.loc['2016-01-08', 'X'] == 1.0 and p.loc['2016-01-09', 'X'] == 1.0     # NaN week forward-filled


def test_summarise_inventory_rates():
    k = pd.DataFrame({'item_id': ['a', 'b'], 'policy': 'p', 'service_level': 0.95, 'demand_units': [100, 300],
                      'lost_units': [10, 10], 'stockout_days': [1, 2], 'avg_inventory_value': [5.0, 5.0],
                      'holding_cost': [1.0, 1.0], 'stockout_cost': [2.0, 2.0], 'total_cost': [3.0, 3.0],
                      'cycles_ok': [8, 9], 'cycles': [10, 10], 'holding_value_days': [4.0, 4.0], 'lost_value': [4.0, 4.0]})
    s = summarise_inventory(k).iloc[0]
    assert np.isclose(s['fill_rate'], 0.95) and np.isclose(s['cycle_service_level'], 0.85) and s['total_cost'] == 6


def test_build_facts_runs_on_committed_tables():
    if not (ROOT / 'outputs' / 'tables' / 'replication_conclusions.csv').exists():
        pytest.skip("result tables not generated")
    import build_facts
    f = build_facts.build()
    assert {'scope', 'breaks', 'forecasts', 'inventory', 'replication', 'planner'} <= set(f)
    assert f['scope']['ax_ay_items']['value'] == 726
    assert 0 < f['breaks']['seasonal_null_false_alarm_rate']['CUSUM']['value'] < 1
