"""One feature set for every ML model (XGBoost, LightGBM, LSTM, GRU, PatchTST and the hybrid).

The setup is direct: one row per item, forecast origin and horizon step h (1..28). Everything that describes the
past is read from sales up to the origin only. Everything that describes the target day (weekday, event, SNAP,
price) is known ahead of time, because the calendar and the retailer's prices are fixed in advance, as in M5.
test_features.py checks this by scrambling all sales after an origin and confirming the rows do not change.
"""
import numpy as np
import pandas as pd

from config import HORIZON

H = HORIZON
LAGS = (7, 14, 28, 56, 112)               # window lengths for the rolling mean at the origin
EV_TYPES = ["Sporting", "Cultural", "National", "Religious"]

PAST = [f"mean{w}" for w in LAGS] + ["std28", "zero28", "max28", "wd_mean8", "wd1", "wd2", "wd3", "wd4", "trend"]
CAL = ["h", "dow", "dom", "month", "sin1", "cos1", "sin2", "cos2", "snap", "christmas", "event",
       "ev_sporting", "ev_cultural", "ev_national", "ev_religious", "event_soon", "rel_price", "price_chg"]
STATIC = ["cat", "dept"]
FEATURES = PAST + CAL + STATIC


def calendar_table(cal, snap_col="snap_CA"):
    """Calendar features for every date, from the M5 calendar. All of it is known in advance."""
    c = cal.copy()
    c["date"] = pd.to_datetime(c["date"])
    c = c.set_index("date")
    out = pd.DataFrame(index=c.index)
    t = (c.index - pd.Timestamp("2011-01-29")).days.to_numpy()
    out["dow"] = c.index.dayofweek
    out["dom"] = c.index.day
    out["month"] = c.index.month
    for k in (1, 2):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    out["snap"] = c[snap_col].astype(float)
    out["christmas"] = ((c.index.month == 12) & (c.index.day == 25)).astype(float)
    name = c["event_name_1"]
    out["event"] = (name.notna() & (name != "Christmas")).astype(float)
    for typ in EV_TYPES:
        has = (c["event_type_1"] == typ) | (c["event_type_2"] == typ)
        out[f"ev_{typ.lower()}"] = (has & (name != "Christmas")).astype(float)
    # an event in the next 3 days (the build-up), still read from the calendar only
    ev = out["event"].to_numpy()
    soon = np.zeros(len(ev))
    for k in (1, 2, 3):
        soon[:-k] = np.maximum(soon[:-k], ev[k:])
    out["event_soon"] = soon
    return out


def _roll(cs, idx, w):
    """Mean of the last w values ending at each position in idx, from the cumulative sum cs."""
    lo = np.maximum(0, idx - w + 1)
    return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)


def item_rows(item, y, dates, origins_idx, cal_t, rel_price, cat, dept, with_target=True):
    """All rows for one item. origins_idx are positions in y; y beyond an origin is only used for the target
    column, never for a feature."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    O = np.asarray(origins_idx, dtype=int)
    hh = np.arange(1, H + 1)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    past = {f"mean{w}": _roll(cs, O, w) for w in LAGS}
    m28 = past["mean28"]
    ex2 = _roll(cs2, O, 28)
    past["std28"] = np.sqrt(np.maximum(ex2 - m28 ** 2, 0.0))
    zc = np.concatenate([[0.0], np.cumsum(y == 0)])
    past["zero28"] = _roll(zc, O, 28)
    past["max28"] = np.array([y[max(0, o - 27):o + 1].max() for o in O])
    past["trend"] = past["mean28"] / np.maximum(past["mean112"], 0.1)
    # the same weekday as the target day, 1..4 weeks back from the last one before the origin
    k0 = np.ceil(hh / 7).astype(int)
    lagged = []
    for j in range(8):
        idx = O[:, None] + hh[None, :] - 7 * (k0[None, :] + j)
        ok = idx >= 0
        lagged.append(np.where(ok, y[np.clip(idx, 0, n - 1)], np.nan))
    lagged = np.stack(lagged)                                   # 8 x origins x 28
    rows = {k: np.repeat(v[:, None], H, axis=1) for k, v in past.items()}
    for j in range(4):
        rows[f"wd{j + 1}"] = lagged[j]
    with np.errstate(all="ignore"):
        rows["wd_mean8"] = np.nanmean(lagged, axis=0)
    rows["wd_mean8"] = np.where(np.isnan(rows["wd_mean8"]), rows["mean28"], rows["wd_mean8"])
    for j in range(4):
        rows[f"wd{j + 1}"] = np.where(np.isnan(rows[f"wd{j + 1}"]), rows["wd_mean8"], rows[f"wd{j + 1}"])
    tgt = O[:, None] + hh[None, :]
    tdate = pd.DatetimeIndex(dates[0] + pd.to_timedelta(tgt.ravel(), unit="D"))
    c = cal_t.reindex(tdate)
    for k in ["dow", "dom", "month", "sin1", "cos1", "sin2", "cos2", "snap", "christmas", "event", "ev_sporting",
              "ev_cultural", "ev_national", "ev_religious", "event_soon"]:
        rows[k] = c[k].to_numpy().reshape(tgt.shape)
    rp = rel_price.reindex(tdate).fillna(1.0).to_numpy().reshape(tgt.shape)
    rp0 = rel_price.reindex(pd.DatetimeIndex(dates[0] + pd.to_timedelta(O, unit="D"))).fillna(1.0).to_numpy()
    rows["rel_price"] = rp
    rows["price_chg"] = rp / rp0[:, None]                      # price on the target day vs the price at the origin
    rows["h"] = np.tile(hh, (len(O), 1))
    df = pd.DataFrame({k: v.ravel() for k, v in rows.items()})
    df["cat"], df["dept"] = cat, dept
    df.insert(0, "item_id", item)
    df.insert(1, "origin", np.repeat(dates[0] + pd.to_timedelta(O, unit="D"), H))
    df.insert(2, "date", tdate)
    if with_target:
        df["y"] = np.where(tgt.ravel() < n, y[np.clip(tgt.ravel(), 0, n - 1)], np.nan)
    return df


def build_rows(series, cal, rel_prices, origins, step=7, anchor=None):
    """Rows for every item at a weekly grid of origins (anchored on `anchor`) plus the given origins.
    series: item -> DataFrame(date, sales). Only origins whose full 28-day target window lies inside the data
    get a row."""
    cal_t = calendar_table(cal)
    anchor = pd.Timestamp(anchor if anchor is not None else origins[0])
    out = []
    for item, d in series.items():
        dates = pd.DatetimeIndex(d["date"])
        y = d["sales"].to_numpy(dtype=float)
        first = 112                                              # need some history for the longest window
        grid = np.arange(first, len(y) - H)
        keep = ((dates[grid] - anchor).days % step == 0)
        extra = [dates.get_loc(o) for o in origins if o in dates]
        idx = np.unique(np.concatenate([grid[keep], extra])).astype(int)
        cat = {"FOODS": 0, "HOBBIES": 1, "HOUSEHOLD": 2}[item.split("_")[0]]
        dept = int(item.split("_")[1])
        out.append(item_rows(item, y, dates, idx, cal_t, rel_prices[item], cat, dept))
    df = pd.concat(out, ignore_index=True)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df


class Standardizer:
    """Mean and std from training rows only; applied to anything else with the same numbers."""
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return np.nan_to_num((np.asarray(X, dtype=float) - self.mu) / self.sd).astype("float32")
