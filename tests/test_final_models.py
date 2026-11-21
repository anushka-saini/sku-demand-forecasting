"""Shared features, neural windows, Diebold-Mariano, stacking and GARCH: no lookahead, sane output."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import nn_models as N  # noqa: E402
import vol_models as vm  # noqa: E402


def toy_cal(n=1200):
    d = pd.date_range("2013-01-01", periods=n)
    cal = pd.DataFrame({"date": d, "snap_CA": (d.day <= 10).astype(int), "event_name_1": None, "event_type_1": None,
                        "event_name_2": None, "event_type_2": None})
    cal.loc[cal.index % 40 == 0, ["event_name_1", "event_type_1"]] = ["Superbowl", "Sporting"]
    return cal


def toy_series(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2013-01-01", periods=n)
    return pd.DataFrame({"date": d, "sales": (8 + 3 * (d.dayofweek >= 5) + rng.poisson(2, n)).astype(float)})


def test_features_ignore_sales_after_the_origin():
    cal, s = toy_cal(), toy_series()
    price = {"A_1_1": pd.Series(1.0, index=pd.DatetimeIndex(cal["date"]))}
    s = {"FOODS_1_001": s}
    price = {"FOODS_1_001": price["A_1_1"]}
    o = pd.Timestamp("2015-05-01")
    s2 = {"FOODS_1_001": s["FOODS_1_001"].copy()}
    m = s2["FOODS_1_001"]["date"] > o
    s2["FOODS_1_001"].loc[m, "sales"] = np.random.default_rng(1).integers(0, 999, m.sum())
    a = F.build_rows(s, cal, price, [o])
    b = F.build_rows(s2, cal, price, [o])
    a, b = a[a["origin"] == o], b[b["origin"] == o]
    assert len(a) == F.H
    assert np.allclose(a[F.FEATURES].to_numpy(), b[F.FEATURES].to_numpy(), equal_nan=True)
    assert not np.allclose(a["y"].to_numpy(), b["y"].to_numpy())          # the target does change, as it should


def test_training_rows_end_before_the_origin():
    import ml_models as M
    rows = pd.DataFrame({"origin": pd.date_range("2015-01-04", periods=40, freq="7D")})
    o = pd.Timestamp("2015-09-13")
    tr = M.train_part(rows, o)
    assert (tr["origin"] + pd.Timedelta(days=F.H) <= o).all()


def test_neural_windows_use_only_the_past():
    cal = toy_cal()
    ct = F.calendar_table(cal)
    s = toy_series()
    rp = pd.Series(1.0, index=pd.DatetimeIndex(cal["date"]))
    o = pd.Timestamp("2015-05-01")
    s2 = s.copy()
    s2.loc[s2["date"] > o, "sales"] = 12345.0
    w1 = N.windows({"FOODS_1_001": N.prep_item(s, ct, rp)}, [o], 56)
    w2 = N.windows({"FOODS_1_001": N.prep_item(s2, ct, rp)}, [o], 56)
    assert np.allclose(w1["seq"], w2["seq"]) and np.allclose(w1["fut"], w2["fut"]) and np.allclose(w1["scale"], w2["scale"])
    assert w1["seq"].shape == (1, 56, 5) and w1["fut"].shape == (1, F.H, 6)


def test_standardizer_uses_training_numbers_only():
    tr = np.random.default_rng(0).normal(5, 2, (500, 3))
    sc = F.Standardizer().fit(tr)
    z = sc.transform(tr)
    assert np.allclose(z.mean(0), 0, atol=1e-4) and np.allclose(z.std(0), 1, atol=1e-3)
    new = sc.transform(np.full((2, 3), 100.0))                              # not re-centred on the new data
    assert new.mean() > 10


def test_dm_equal_losses_not_significant_and_better_model_is_negative():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.3, 168)
    s, p, d = vm.dm_test(a, a + rng.normal(0, 0.3, 168))
    assert p > 0.05
    s, p, d = vm.dm_test(a, a + 0.5 + rng.normal(0, 0.1, 168))
    assert s < 0 and p < 0.05 and d < 0


def test_garch_gives_positive_spread_and_falls_back_on_junk():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 150) * (1 + 0.5 * (np.arange(150) % 7 == 0))
    sig, cond, ok = vm.garch_vol(e)
    assert sig.shape == (vm.H,) and (sig > 0).all() and len(cond) == 150
    sig2, _, ok2 = vm.garch_vol(np.zeros(60))
    assert np.isfinite(sig2).all()


def test_stack_weights_are_non_negative():
    from scipy.optimize import nnls
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 400)
    X = np.c_[y + rng.normal(0, 1, 400), y + rng.normal(0, 3, 400), rng.normal(10, 2, 400)]
    w, _ = nnls(X, y)
    assert (w >= 0).all() and w[0] > w[2]
