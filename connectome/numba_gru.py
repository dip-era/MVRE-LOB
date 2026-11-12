"""Numba single-step GRU inference, numerically equivalent to model.GRUModel (PyTorch GRU semantics).

Weights are stored input-major, shape (In, 3H), so each step is a sequence of axpy updates over the
outputs; that vectorizes well and lets zero inputs (most of the diff features) be skipped outright.
Rows are applied 4 at a time, so the accumulator is loaded/stored once per 4 rows instead of once per row.
Gate order (r, z, n) and n = tanh(W_in x + b_in + r * (W_hn h + b_hn)) follow torch.nn.GRU.
"""
import numpy as np
from numba import njit

N_RAW = 112
BLOCKS = np.array([(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104)], np.int64)


@njit(fastmath=True, inline="always")
def _matvec_acc(acc, W, v, idx, n, G):
    """acc[:G] += sum_t v[idx[t]] * W[idx[t], :G] for t < n, 4 rows at a time."""
    t = 0
    while t + 4 <= n:
        k0, k1, k2, k3 = idx[t], idx[t + 1], idx[t + 2], idx[t + 3]
        a0, a1, a2, a3 = v[k0], v[k1], v[k2], v[k3]
        r0, r1, r2, r3 = W[k0], W[k1], W[k2], W[k3]
        for j in range(G):
            acc[j] += a0 * r0[j] + a1 * r1[j] + a2 * r2[j] + a3 * r3[j]
        t += 4
    while t < n:
        k = idx[t]
        a = v[k]
        r = W[k]
        for j in range(G):
            acc[j] += a * r[j]
        t += 1


@njit(fastmath=True, inline="always")
def _dense_acc(acc, W, v, G):
    K = v.shape[0]
    k = 0
    while k + 4 <= K:
        a0, a1, a2, a3 = v[k], v[k + 1], v[k + 2], v[k + 3]
        r0, r1, r2, r3 = W[k], W[k + 1], W[k + 2], W[k + 3]
        for j in range(G):
            acc[j] += a0 * r0[j] + a1 * r1[j] + a2 * r2[j] + a3 * r3[j]
        k += 4
    while k < K:
        a = v[k]
        r = W[k]
        for j in range(G):
            acc[j] += a * r[j]
        k += 1


@njit(fastmath=True, inline="always")
def _gates(gi, gh, h, H):
    for j in range(H):
        r = np.float32(1.0) / (np.float32(1.0) + np.exp(-(gi[j] + gh[j])))
        z = np.float32(1.0) / (np.float32(1.0) + np.exp(-(gi[H + j] + gh[H + j])))
        n = np.tanh(gi[2 * H + j] + r * gh[2 * H + j])
        h[j] = n + z * (h[j] - n)


@njit(fastmath=True)
def gru_step(x, prev, inp, idx, hs, Wi0, Wis, Whs, bis, bhs, Wo, bo, gi, gh, out):
    H = hs.shape[1]
    L = hs.shape[0]
    G = 3 * H
    n = 0
    for k in range(N_RAW):
        v = x[k]
        inp[k] = v
        if v != 0.0:
            idx[n] = k
            n += 1
    for k in range(N_RAW):
        d = x[k] - prev[k]
        inp[N_RAW + k] = d
        if d != 0.0:
            idx[n] = N_RAW + k
            n += 1
    for b in range(BLOCKS.shape[0]):
        f = 0.0
        for k in range(BLOCKS[b, 0], BLOCKS[b, 1]):
            if inp[N_RAW + k] != 0.0:
                f = 1.0
                break
        inp[2 * N_RAW + b] = f
        if f != 0.0:
            idx[n] = 2 * N_RAW + b
            n += 1
    for j in range(G):
        gi[j] = bis[0, j]
        gh[j] = bhs[0, j]
    _matvec_acc(gi, Wi0, inp, idx, n, G)
    _dense_acc(gh, Whs[0], hs[0], G)
    _gates(gi, gh, hs[0], H)
    for l in range(1, L):
        for j in range(G):
            gi[j] = bis[l, j]
            gh[j] = bhs[l, j]
        _dense_acc(gi, Wis[l - 1], hs[l - 1], G)
        _dense_acc(gh, Whs[l], hs[l], G)
        _gates(gi, gh, hs[l], H)
    out[0] = bo[0]
    out[1] = bo[1]
    for k in range(H):
        out[0] += hs[L - 1, k] * Wo[k, 0]
        out[1] += hs[L - 1, k] * Wo[k, 1]


class NumbaGRU:
    """Holds weights + per-sequence state. Construct from a GRUModel state_dict (numpy arrays)."""

    def __init__(self, sd, hidden, layers):
        f = lambda k: np.ascontiguousarray(np.asarray(sd[k], dtype=np.float32))
        H, L = hidden, layers
        self.Wi0 = np.ascontiguousarray(f("gru.weight_ih_l0").T)                       # (230, 3H)
        self.Wis = np.stack([f(f"gru.weight_ih_l{l}").T for l in range(1, L)]) if L > 1 \
            else np.zeros((1, H, 3 * H), np.float32)                                     # (L-1, H, 3H)
        self.Whs = np.stack([f(f"gru.weight_hh_l{l}").T for l in range(L)])               # (L, H, 3H)
        self.bis = np.stack([f(f"gru.bias_ih_l{l}") for l in range(L)])                   # (L, 3H)
        self.bhs = np.stack([f(f"gru.bias_hh_l{l}") for l in range(L)])
        self.Wo = np.ascontiguousarray(f("head.weight").T)                                 # (H, 2)
        self.bo = f("head.bias")
        self.Wis, self.Whs = np.ascontiguousarray(self.Wis), np.ascontiguousarray(self.Whs)
        self.hs = np.zeros((L, H), np.float32)
        self.inp = np.zeros(self.Wi0.shape[0], np.float32)
        self.idx = np.zeros(self.Wi0.shape[0], np.int64)
        self.gi = np.zeros(3 * H, np.float32)
        self.gh = np.zeros(3 * H, np.float32)
        self.out = np.zeros(2, np.float32)

    def reset(self):
        self.hs[:] = 0.0

    def step(self, x, prev):
        gru_step(x, prev, self.inp, self.idx, self.hs, self.Wi0, self.Wis, self.Whs, self.bis, self.bhs,
                 self.Wo, self.bo, self.gi, self.gh, self.out)
        return self.out


# ---- int8 weights (per-output-column scale): 4x less memory traffic, so several models stay in L2 ----

@njit(fastmath=True, inline="always")
def _matvec_acc_q(acc, Wq, v, idx, n, G):
    """acc[:G] += sum_t v[idx[t]] * Wq[idx[t], :G] (int8 rows, unscaled), 4 rows at a time."""
    t = 0
    while t + 4 <= n:
        k0, k1, k2, k3 = idx[t], idx[t + 1], idx[t + 2], idx[t + 3]
        a0, a1, a2, a3 = v[k0], v[k1], v[k2], v[k3]
        r0, r1, r2, r3 = Wq[k0], Wq[k1], Wq[k2], Wq[k3]
        for j in range(G):
            acc[j] += (a0 * np.float32(r0[j]) + a1 * np.float32(r1[j])
                       + a2 * np.float32(r2[j]) + a3 * np.float32(r3[j]))
        t += 4
    while t < n:
        k = idx[t]
        a = v[k]
        r = Wq[k]
        for j in range(G):
            acc[j] += a * np.float32(r[j])
        t += 1


@njit(fastmath=True, inline="always")
def _dense_acc_q(acc, Wq, v, G):
    K = v.shape[0]
    k = 0
    while k + 4 <= K:
        a0, a1, a2, a3 = v[k], v[k + 1], v[k + 2], v[k + 3]
        r0, r1, r2, r3 = Wq[k], Wq[k + 1], Wq[k + 2], Wq[k + 3]
        for j in range(G):
            acc[j] += (a0 * np.float32(r0[j]) + a1 * np.float32(r1[j])
                       + a2 * np.float32(r2[j]) + a3 * np.float32(r3[j]))
        k += 4
    while k < K:
        a = v[k]
        r = Wq[k]
        for j in range(G):
            acc[j] += a * np.float32(r[j])
        k += 1


@njit(fastmath=True)
def gru_step_q(x, prev, inp, idx, hs, Wi0, Wis, Whs, si0, sis, shs, bis, bhs, Wo, bo, gi, gh, out):
    """Same as gru_step with int8 weights; s* are per-output-column scales (W = s * q)."""
    H = hs.shape[1]
    L = hs.shape[0]
    G = 3 * H
    n = 0
    for k in range(N_RAW):
        v = x[k]
        inp[k] = v
        if v != 0.0:
            idx[n] = k
            n += 1
    for k in range(N_RAW):
        d = x[k] - prev[k]
        inp[N_RAW + k] = d
        if d != 0.0:
            idx[n] = N_RAW + k
            n += 1
    for b in range(BLOCKS.shape[0]):
        f = 0.0
        for k in range(BLOCKS[b, 0], BLOCKS[b, 1]):
            if inp[N_RAW + k] != 0.0:
                f = 1.0
                break
        inp[2 * N_RAW + b] = f
        if f != 0.0:
            idx[n] = 2 * N_RAW + b
            n += 1
    for j in range(G):
        gi[j] = 0.0
        gh[j] = 0.0
    _matvec_acc_q(gi, Wi0, inp, idx, n, G)
    _dense_acc_q(gh, Whs[0], hs[0], G)
    for j in range(G):
        gi[j] = gi[j] * si0[j] + bis[0, j]
        gh[j] = gh[j] * shs[0, j] + bhs[0, j]
    _gates(gi, gh, hs[0], H)
    for l in range(1, L):
        for j in range(G):
            gi[j] = 0.0
            gh[j] = 0.0
        _dense_acc_q(gi, Wis[l - 1], hs[l - 1], G)
        _dense_acc_q(gh, Whs[l], hs[l], G)
        for j in range(G):
            gi[j] = gi[j] * sis[l - 1, j] + bis[l, j]
            gh[j] = gh[j] * shs[l, j] + bhs[l, j]
        _gates(gi, gh, hs[l], H)
    out[0] = bo[0]
    out[1] = bo[1]
    for k in range(H):
        out[0] += hs[L - 1, k] * Wo[k, 0]
        out[1] += hs[L - 1, k] * Wo[k, 1]


def _quantize(W):
    """W: (In, G) float32 -> (int8 (In, G), scale (G,)) with symmetric per-output-column scales."""
    s = np.abs(W).max(0) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), s.astype(np.float32)


class NumbaGRUQ(NumbaGRU):
    """NumbaGRU with int8 recurrent/input weights (biases and head stay float32)."""

    def __init__(self, sd, hidden, layers):
        super().__init__(sd, hidden, layers)
        self.Wi0, self.si0 = _quantize(self.Wi0)
        q = [_quantize(w) for w in self.Wis]
        self.Wis = np.ascontiguousarray(np.stack([a for a, _ in q]))
        self.sis = np.ascontiguousarray(np.stack([s for _, s in q]))
        q = [_quantize(w) for w in self.Whs]
        self.Whs = np.ascontiguousarray(np.stack([a for a, _ in q]))
        self.shs = np.ascontiguousarray(np.stack([s for _, s in q]))

    def step(self, x, prev):
        gru_step_q(x, prev, self.inp, self.idx, self.hs, self.Wi0, self.Wis, self.Whs, self.si0, self.sis, self.shs,
                   self.bis, self.bhs, self.Wo, self.bo, self.gi, self.gh, self.out)
        return self.out
