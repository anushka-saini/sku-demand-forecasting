"""LSTM, GRU and a small PatchTST for the 28-day forecast of every item (one global model per origin).

Input: the last W days of sales (divided by the item's mean of the last 56 days up to the origin) together with
snap, event and weekday for those days. The next 28 days of snap, event, weekday, Christmas and relative price are
known ahead and go in at the head, together with the category. Output: 28 values, in units of the item's scale.

PatchTST here is written directly in torch (neuralforecast would pull in ray and lightning): the sales window is cut
into patches, each patch is embedded, a transformer encoder reads them, and the head adds the known-ahead covariates.
That last part is our change; the original PatchTST has no future covariates.

Each net is trained once per stage and then rolled forward: a fit with cutoff c uses training windows with origins up
to c - 56 days and an early-stopping holdout with origins in (c - 56, c - 28], so every target it sees is over by c.
It then forecasts any number of later origins from their own past windows. Nothing after c is ever seen in training,
and the inputs of a window stop at its own origin. (A refit at every origin, as XGBoost does, was too slow on the
laptop; the report says so.)
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import features as F
from config import SEED
from features import Standardizer

H = F.H
DAY = pd.Timedelta(days=1)
SEQ_COLS = ["snap", "event", "dsin", "dcos"]
FUT_COLS = ["snap", "event", "dsin", "dcos", "christmas", "price_ratio"]


def prep_item(d, cal_t, rel_price):
    """Per-day arrays for one item over the whole calendar range of its sales: sales, seq covariates, future
    covariates. Covariates come from the calendar and prices, which are known ahead."""
    dates = pd.DatetimeIndex(d["date"])
    ext = pd.date_range(dates[0], cal_t.index.max())
    c = cal_t.reindex(ext)
    dow = c["dow"].to_numpy()
    cov = np.stack([c["snap"], c["event"], np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], 1)
    fut = np.concatenate([cov, c[["christmas"]].to_numpy(), rel_price.reindex(ext).fillna(1.0).to_numpy()[:, None]], 1)
    return d["sales"].to_numpy(dtype=float), cov.astype("float32"), fut.astype("float32"), dates[0]


def windows(items, origin_dates, W, ids=None):
    """Stack windows for the given origin dates (one array of dates per item is not needed: the same dates for all).
    Returns dict of arrays and the item ids and origins that were kept."""
    X, Z, Fu, Y, S, C, who = [], [], [], [], [], [], []
    for item, (y, cov, fut, t0) in items.items():
        o = np.asarray([(pd.Timestamp(x) - t0).days for x in origin_dates])
        ok = (o >= W - 1) & (o + H < len(y))
        # future targets are only needed for training rows; at prediction time y may stop at the origin
        ok = ok & (o >= 56)
        o = o[ok]
        if len(o) == 0:
            continue
        scale = np.maximum(np.array([y[max(0, k - 55):k + 1].mean() for k in o]), 0.1)
        idx = o[:, None] + np.arange(-W + 1, 1)[None, :]
        seq = np.concatenate([(y[idx] / scale[:, None])[:, :, None], cov[idx]], 2)
        fidx = o[:, None] + np.arange(1, H + 1)[None, :]
        fu = fut[fidx].copy()
        base = fut[o, 5][:, None]                              # price on the origin day, to make a ratio
        fu[:, :, 5] = fu[:, :, 5] / base
        X.append(seq.astype("float32"))
        Fu.append(fu.astype("float32"))
        Y.append((y[fidx] / scale[:, None]).astype("float32"))
        S.append(scale)
        cat = {"FOODS": 0, "HOBBIES": 1, "HOUSEHOLD": 2}[item.split("_")[0]]
        C.append(np.tile(np.eye(3, dtype="float32")[cat], (len(o), 1)))
        who += [(item, t0 + pd.Timedelta(days=int(k))) for k in o]
    return {"seq": np.concatenate(X), "fut": np.concatenate(Fu), "y": np.concatenate(Y), "scale": np.concatenate(S),
            "cat": np.concatenate(C), "who": who}


class Net(nn.Module):
    def __init__(self, kind, W, hidden=64, layers=1, dropout=0.0, patch=16, n_seq=5, n_fut=6):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.enc = rnn(n_seq, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            out = hidden
        else:                                                   # patchtst
            d = 64
            self.patch, self.stride = patch, patch // 2
            n_p = (W - patch) // self.stride + 1
            self.emb = nn.Linear(patch * n_seq, d)
            self.pos = nn.Parameter(torch.zeros(1, n_p, d))
            layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.pool = nn.Linear(n_p * d, hidden)
            out = hidden
        self.head = nn.Sequential(nn.Linear(out + H * n_fut + 3, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, H))

    def forward(self, seq, fut, cat):
        if self.kind in ("lstm", "gru"):
            o, _ = self.enc(seq)
            z = o[:, -1]
        else:
            p = seq.unfold(1, self.patch, self.stride)           # batch, n_p, features, patch
            p = p.permute(0, 1, 3, 2).reshape(seq.shape[0], p.shape[1], -1)
            z = torch.relu(self.pool(self.enc(self.emb(p) + self.pos).flatten(1)))
        return self.head(torch.cat([z, fut.flatten(1), cat], 1))


def standardize(tr, other):
    """Scalers from the training windows only. The sales channel is already in item-scale units."""
    s_seq = Standardizer().fit(tr["seq"].reshape(-1, tr["seq"].shape[2])[:, 1:])
    s_fut = Standardizer().fit(tr["fut"].reshape(-1, tr["fut"].shape[2])[:, 5:])
    out = []
    for w in [tr] + other:
        seq = w["seq"].copy()
        seq[:, :, 1:] = s_seq.transform(seq[:, :, 1:].reshape(-1, seq.shape[2] - 1)).reshape(seq.shape[0], seq.shape[1], -1)
        fut = w["fut"].copy()
        fut[:, :, 5:] = s_fut.transform(fut[:, :, 5:].reshape(-1, 1)).reshape(fut.shape[0], fut.shape[1], 1)
        out.append({**w, "seq": seq, "fut": fut})
    return out


def fit_predict(items, cutoff, origins, kind, cfg, seed, threads=3, max_epochs=12, patience=3, stride=5, span=700):
    """Train once with this cutoff, forecast every origin in `origins`. Returns (forecast frame, seconds, epochs)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    W = cfg["W"]
    t = time.time()
    last_train = cutoff - 56 * DAY
    tr_dates = pd.date_range(last_train - span * DAY, last_train, freq=f"{stride}D")
    ho_dates = pd.date_range(cutoff - 55 * DAY, cutoff - H * DAY, freq="4D")
    tr, ho, ev = windows(items, tr_dates, W), windows(items, ho_dates, W), windows(items, list(origins), W)
    tr, ho, ev = standardize(tr, [ho, ev])
    net = Net(kind, W, cfg.get("hidden", 64), cfg.get("layers", 1), cfg.get("dropout", 0.0), cfg.get("patch", 16))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    T = lambda w: [torch.from_numpy(w[k]) for k in ("seq", "fut", "cat", "y")]
    trs, trf, trc, try_ = T(tr)
    hos, hof, hoc, hoy = T(ho)
    n, best, bad, best_state, used = len(try_), 1e18, 0, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = ((net(trs[b], trf[b], trc[b]) - try_[b]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(hos, hof, hoc) - hoy) ** 2).mean())
        used = ep + 1
        if v < best - 1e-5:
            best, bad, best_state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(*[torch.from_numpy(ev[k]) for k in ("seq", "fut", "cat")]).numpy()
    p = np.clip(p * ev["scale"][:, None], 0, None)
    rows = []
    for (item, o), row in zip(ev["who"], p):
        d = pd.date_range(o + DAY, periods=H)
        rows.append(pd.DataFrame({"item_id": item, "origin": o, "date": d, "forecast": row}))
    return pd.concat(rows, ignore_index=True), time.time() - t, used


GRIDS = {
    "lstm": [dict(hidden=h, layers=l, dropout=d, W=w, lr=lr) for h in (32, 64) for l in (1, 2) for d in (0.0, 0.2)
             for w in (56, 112) for lr in (1e-3, 3e-3)],
    "patchtst": [dict(hidden=64, layers=2, dropout=0.1, W=112, patch=p, lr=lr) for p in (8, 16) for lr in (5e-4, 1e-3, 2e-3)],
}
GRIDS["gru"] = GRIDS["lstm"]
N_CONFIGS = {"lstm": 5, "gru": 5, "patchtst": 6}


def sample_configs(kind, seed=SEED):
    g = GRIDS[kind]
    if len(g) <= N_CONFIGS[kind]:
        return g
    rng = np.random.default_rng(seed)
    return [g[i] for i in sorted(rng.choice(len(g), N_CONFIGS[kind], replace=False))]
