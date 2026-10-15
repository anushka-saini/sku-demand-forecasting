import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from break_quality import assess_break, assess_break_at_origin, break_position


def step_series(n=200, c=100, pre=100.0, post=150.0, later=None):
    y = np.full(n, pre)
    y[c:] = post
    if later is not None:
        y[c + 27:] = later
    return y


def test_clean_persistent_step_is_high_confidence():
    r = assess_break(step_series(), 100, pelt_cps=[101])
    assert r['agreement'] and r['effect'] and r['assessable'] and r['persistence'] and r['high_confidence']
    assert np.isclose(r['relative_change'], 0.5) and np.isclose(r['persistence_ratio'], 1.0)


def test_each_criterion_can_fail_alone():
    assert not assess_break(step_series(), 100, pelt_cps=[120])['agreement']            # PELT 20 weeks away
    assert assess_break(step_series(), 100, pelt_cps=[108])['agreement']                # exactly 8 weeks: ok
    assert not assess_break(step_series(post=115.0), 100, pelt_cps=[100])['effect']     # 15% < 20%
    r = assess_break(step_series(later=110.0), 100, pelt_cps=[100])                     # shift mostly fades
    assert r['effect'] and not r['persistence'] and not r['high_confidence']
    assert np.isclose(r['persistence_ratio'], 0.2)


def test_break_too_close_to_end_is_not_assessable():
    r = assess_break(step_series(n=140), 100, pelt_cps=[100])
    assert not r['assessable'] and not r['high_confidence']
    assert assess_break(step_series(n=153), 100, pelt_cps=[100])['assessable']          # exactly 52 weeks after


def test_downward_break_and_zero_pre_mean():
    r = assess_break(step_series(pre=100.0, post=60.0), 100, pelt_cps=[99])
    assert r['high_confidence'] and np.isclose(r['relative_change'], -0.4)
    assert not assess_break(step_series(pre=0.0, post=10.0), 100, pelt_cps=[100])['effect']


def test_origin_version_uses_only_available_data():
    y = step_series()
    assert not assess_break_at_origin(y[:105], 100, [100])['high_confidence']           # 4 post weeks < 8
    r = assess_break_at_origin(y[:115], 100, [100])                                     # 14 post weeks, too recent for persistence
    assert r['high_confidence'] and r['weeks_since_change'] == 14
    faded = step_series(later=100.0)
    assert assess_break_at_origin(faded[:120], 100, [100])['high_confidence']           # fade not yet visible
    assert not assess_break_at_origin(faded[:160], 100, [100])['high_confidence']       # last 8 weeks back at pre level


def test_break_position_flags():
    y = step_series(n=200, c=70, pre=100.0, post=5.0)
    p = break_position(y, 70, direction=-1, monitor_start=60, pre_mean=100.0, post_mean=5.0)
    assert p['in_launch_ramp'] and not p['near_series_end'] and p['possible_delisting_artifact']
    y2 = step_series(n=200, c=180)
    p2 = break_position(y2, 180, direction=1, monitor_start=60, pre_mean=100.0, post_mean=150.0)
    assert p2['near_series_end'] and not p2['in_launch_ramp'] and not p2['possible_delisting_artifact']
