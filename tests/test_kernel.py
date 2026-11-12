"""The fused int8 numba kernel (connectome/numba_flat.py) must reproduce the PyTorch ensemble.

Synthetic data only (no competition files needed): random GRU and LSTM members on all five input views
(raw, standardised, change-only, rank-Gaussian, whitened), random view tables, and a random-walk input where
only a few columns change per step (like the real order-book features). int8 weights give ~1% differences.

Run: python -m pytest tests/  (or python tests/test_kernel.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "connectome"))
from model import RK_K, GRUModel  # noqa: E402
from numba_flat import FlatGRUQ  # noqa: E402
from numba_gru import NumbaGRUQ  # noqa: E402

N = 112
SLOT = {0: 0, 1: 1, 5: 2, 3: 3, 4: 4}  # model norm id -> kernel view slot


def tables(rng):
    mu = rng.normal(0, 0.5, N).astype(np.float32)
    sd = rng.uniform(0.5, 2.0, N).astype(np.float32)
    drms = rng.uniform(0.1, 1.0, N).astype(np.float32)
    knots = np.sort(rng.normal(0, 1.5, (N, RK_K)), axis=1) + np.arange(RK_K) * 1e-3  # strictly increasing
    vals = np.sort(rng.normal(0, 1.0, (N, RK_K)), axis=1)
    wx = (np.eye(N) + rng.normal(0, 0.05, (N, N))).astype(np.float32)
    wd = (np.eye(N) + rng.normal(0, 0.05, (N, N))).astype(np.float32)
    return mu, sd, drms, knots.astype(np.float32), vals.astype(np.float32), wx, wd


def inputs(rng, T=400):
    x = np.zeros((T, N), np.float32)
    x[0] = rng.normal(0, 1, N)
    for t in range(1, T):
        x[t] = x[t - 1]
        ch = rng.random(N) < 0.15
        x[t, ch] += rng.normal(0, 0.5, ch.sum())
    return x.astype(np.float16).astype(np.float32)  # solution.py rounds inputs to float16


def build(specs, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu, sd, drms, knots, vals, wx, wd = tables(rng)
    members = []
    for cell, layers, norm in specs:
        m = GRUModel(hidden=32, layers=layers, norm=norm, cell=cell).eval()
        if norm:
            m.in_mu.copy_(torch.from_numpy(mu))
            m.in_sd.copy_(torch.from_numpy(sd))
            m.d_rms.copy_(torch.from_numpy(drms))
        if norm == 3:
            m.rk_knots.copy_(torch.from_numpy(knots))
            m.rk_vals.copy_(torch.from_numpy(vals))
        if norm == 4:
            m.zca_x.copy_(torch.from_numpy(wx))
            m.zca_d.copy_(torch.from_numpy(wd))
        members.append(m)
    weights = rng.uniform(0.5, 1.5, len(specs)).astype(np.float32)
    q = [NumbaGRUQ({k: v.numpy() for k, v in m.state_dict().items()}, 32, m.layers) for m in members]
    ens = FlatGRUQ(q, weights, norm=(mu, sd, drms), views=[SLOT[n] for _, _, n in specs],
                   rank=(knots, vals), zca=(wx, wd))
    return members, weights, ens


def run(specs, seed=0):
    members, weights, ens = build(specs, seed)
    x = inputs(np.random.default_rng(seed + 1))
    with torch.no_grad():
        ref = sum(w * m(torch.from_numpy(x)[None])[0][0].numpy() for m, w in zip(members, weights))
    out = np.zeros_like(ref)
    ens.reset()
    prev = x[0]
    for t in range(len(x)):
        out[t] = ens.step(x[t], prev)
        prev = x[t]
    return ref, out


def check(ref, out):
    scale = ref.std()
    err = np.abs(ref - out).max() / scale
    corr = min(np.corrcoef(ref[:, k], out[:, k])[0, 1] for k in range(2))
    assert err < 0.05 and corr > 0.999, f"max err {err:.4f} of output std, corr {corr:.5f}"
    return err, corr


def test_gru_all_views():
    check(*run([("gru", 2, 0), ("gru", 3, 1), ("gru", 3, 5), ("gru", 2, 3), ("gru", 3, 4)]))


def test_lstm_members():
    check(*run([("gru", 2, 0), ("lstm", 2, 3), ("lstm", 2, 4), ("lstm", 2, 0)], seed=1))


def test_final_ensemble_layout():
    # final_v9e structure: raw 2L GRU + rank LSTM + zca LSTM + change-only 3L GRU
    check(*run([("gru", 2, 0), ("lstm", 2, 3), ("lstm", 2, 4), ("gru", 3, 5)], seed=2))


def test_state_reset():
    members, weights, ens = build([("gru", 2, 0), ("lstm", 2, 4)], seed=3)
    x = inputs(np.random.default_rng(9), T=50)
    outs = []
    for _ in range(2):
        ens.reset()
        prev = x[0]
        o = []
        for t in range(len(x)):
            o.append(ens.step(x[t], prev).copy())
            prev = x[t]
        outs.append(np.array(o))
    assert np.array_equal(outs[0], outs[1]), "reset() must clear hidden and cell states"


if __name__ == "__main__":
    for f in (test_gru_all_views, test_lstm_members, test_final_ensemble_layout, test_state_reset):
        f()
        print("ok", f.__name__)
