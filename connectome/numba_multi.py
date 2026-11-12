"""Fused multi-model int8 GRU step: one numba call per row for the whole ensemble.

Same math as numba_gru.gru_step_q, but the input features (x, dx, change flags and the nonzero-index list) are built
once per row instead of once per model, the weighted average is accumulated inside the kernel, and Python makes one
call per row instead of one per model. Models are passed as homogeneous tuples of arrays, so 2- and 3-layer GRUs mix.
"""
import numpy as np
from numba import njit

from numba_gru import BLOCKS, N_RAW, _dense_acc_q, _gates, _matvec_acc_q


@njit(fastmath=True, inline="always")
def _build_inp(x, prev, inp, idx, mu, isd, idr):
    """Layer-0 input row [x, dx, change flags] with scaling x -> (x - mu) * isd, dx -> dx * idr (identity 0 / 1 / 1)
    and the list of its nonzero positions. Returns the number of nonzeros."""
    n = 0
    for k in range(N_RAW):
        v = (x[k] - mu[k]) * isd[k]
        inp[k] = v
        if v != 0.0:
            idx[n] = k
            n += 1
    for k in range(N_RAW):
        d = (x[k] - prev[k]) * idr[k]
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
    return n


@njit(fastmath=True)
def multi_step_q(x, prev, inp2, idx2, ns, need, Wi0s, Wiss, Whss, si0s, siss, shss, biss, bhss, Wos, bos, hss, ws,
                 gi, gh, out, nmu2, nisd2, nidr2, usen):
    # two input variants: row 0 raw inputs, row 1 standardised (model.GRUModel norm=True); usen[i] picks the row
    for r in range(2):
        if need[r]:
            ns[r] = _build_inp(x, prev, inp2[r], idx2[r], nmu2[r], nisd2[r], nidr2[r])
    out[0] = 0.0
    out[1] = 0.0
    for i in range(len(hss)):
        hs = hss[i]
        Wis, Whs, sis, shs, bis, bhs = Wiss[i], Whss[i], siss[i], shss[i], biss[i], bhss[i]
        si0 = si0s[i]
        H = hs.shape[1]
        L = hs.shape[0]
        G = 3 * H
        for j in range(G):
            gi[j] = 0.0
            gh[j] = 0.0
        r = usen[i]
        _matvec_acc_q(gi, Wi0s[i], inp2[r], idx2[r], ns[r], G)
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
        Wo, bo = Wos[i], bos[i]
        o0 = bo[0]
        o1 = bo[1]
        for k in range(H):
            o0 += hs[L - 1, k] * Wo[k, 0]
            o1 += hs[L - 1, k] * Wo[k, 1]
        out[0] += ws[i] * o0
        out[1] += ws[i] * o1


class MultiGRUQ:
    """Ensemble of NumbaGRUQ models (already quantised) stepped by one fused kernel call."""

    def __init__(self, models, weights, norm=None, usen=None):
        """norm: (mu, sd, drms) input scaling for the standardised members, or None.
        usen: per member 1 = standardised inputs, 0 = raw inputs (default: all raw)."""
        t = lambda name: tuple(getattr(m, name) for m in models)
        self.Wi0s, self.Wiss, self.Whss = t("Wi0"), t("Wis"), t("Whs")
        self.si0s, self.siss, self.shss = t("si0"), t("sis"), t("shs")
        self.biss, self.bhss, self.Wos, self.bos = t("bis"), t("bhs"), t("Wo"), t("bo")
        self.hss = t("hs")
        self.ws = np.asarray(weights, np.float32)
        G = max(m.gi.shape[0] for m in models)
        n_in = models[0].inp.shape[0]
        self.inp2 = np.zeros((2, n_in), np.float32)
        self.idx2 = np.zeros((2, n_in), np.int64)
        self.ns = np.zeros(2, np.int64)
        self.gi = np.zeros(G, np.float32)
        self.gh = np.zeros(G, np.float32)
        self.out = np.zeros(2, np.float32)
        self.usen = np.zeros(len(models), np.int64) if usen is None else np.asarray(usen, np.int64)
        assert self.usen.shape == (len(models),)
        assert norm is not None or not self.usen.any(), "standardised members need norm stats"
        self.need = np.array([bool((self.usen == 0).any()), bool((self.usen == 1).any())])
        self.nmu2 = np.zeros((2, 112), np.float32)
        self.nisd2 = np.ones((2, 112), np.float32)
        self.nidr2 = np.ones((2, 112), np.float32)
        if norm is not None:
            mu, sd, drms = (np.asarray(a, np.float32) for a in norm)
            self.nmu2[1], self.nisd2[1], self.nidr2[1] = mu, 1.0 / sd, 1.0 / drms

    def reset(self):
        for h in self.hss:
            h[:] = 0.0

    def step(self, x, prev):
        multi_step_q(x, prev, self.inp2, self.idx2, self.ns, self.need, self.Wi0s, self.Wiss, self.Whss, self.si0s,
                     self.siss, self.shss, self.biss, self.bhss, self.Wos, self.bos, self.hss, self.ws, self.gi,
                     self.gh, self.out, self.nmu2, self.nisd2, self.nidr2, self.usen)
        return self.out
