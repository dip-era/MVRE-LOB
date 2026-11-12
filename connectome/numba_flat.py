"""Fused int8 GRU ensemble step, v2: same math as numba_multi.multi_step_q, faster, plus extra input views.
Members may also be LSTMs (G = 4H, cell state stored after h; detected from the weight shapes).

- All weights and state live in three flat arrays (int8 Wq, float32 F, int64 I), so the per-row numba call passes
  6 arguments instead of ~50 (dispatch cost ~3.5 us before).
- Gate nonlinearities use a vectorisable rational tanh (Eigen's fast tanh, abs err < 3e-7; sigmoid(x) = 0.5 + 0.5
  tanh(x / 2)) instead of scalar libm exp/tanh calls (numba has no SVML here): ~2.3x faster gates.
- Input views per member (model.GRUModel norm): 0 raw, 1 standardised, 2 change-only (model norm 5), 3 rank-Gaussian,
  4 whitened (ZCA). Each view's
  layer-0 input row is built once per row if any member needs it. Change flags always come from the raw diffs.
Quantisation is done by numba_gru.NumbaGRUQ, so int8 weights and scales are bit-identical to numba_multi.
"""
import numpy as np
from numba import njit

from numba_gru import BLOCKS, N_RAW, _dense_acc_q, _matvec_acc_q

N_IN = 2 * N_RAW + BLOCKS.shape[0]
NV = 5        # view slots 0..4 (slot 2 = change-only view, model norm 5; model norm 2 is not supported)
RK = 129      # rank-view knots per column (model.RK_K)
MSTRIDE = 8   # int64 slots per member in I
I0 = 8        # I[0] n_models, I[1 + v] need view v (v < 5), I[6] offset of idx scratch, I[7] Gmax
# F header offsets
O_MU = 0
O_ISD = O_MU + N_RAW
O_IDR = O_ISD + N_RAW
O_KN = O_IDR + N_RAW
O_KV = O_KN + N_RAW * RK
O_ZX = O_KV + N_RAW * RK
O_ZD = O_ZX + N_RAW * N_RAW       # stored transposed: row k = column k of Wd
O_INP = O_ZD + N_RAW * N_RAW
O_TMP112 = O_INP + NV * N_IN      # 2 x 112 scratch
O_RKX = O_TMP112 + 2 * N_RAW      # rank-view cache: last input value per column ...
O_RKZ = O_RKX + N_RAW             # ... and its transformed value (exact: keyed by value equality)
O_SCR = O_RKZ + N_RAW
RK_EMPTY = 3.0e38                 # cache sentinel (never a data value)


@njit(fastmath=True, inline="always")
def _tanh(x):
    x = min(max(x, np.float32(-7.90531110763549805)), np.float32(7.90531110763549805))
    x2 = x * x
    p = np.float32(-2.76076847742355e-16)
    p = p * x2 + np.float32(2.00018790482477e-13)
    p = p * x2 + np.float32(-8.60467152213735e-11)
    p = p * x2 + np.float32(5.12229709037114e-08)
    p = p * x2 + np.float32(1.48572235717979e-05)
    p = p * x2 + np.float32(6.37261928875436e-04)
    p = p * x2 + np.float32(4.89352455891786e-03)
    p = p * x
    q = np.float32(1.19825839466702e-06)
    q = q * x2 + np.float32(1.18534705686654e-04)
    q = q * x2 + np.float32(2.26843463243900e-03)
    q = q * x2 + np.float32(4.89352518554385e-03)
    return p / q


@njit(fastmath=True, inline="always")
def _gates_fast(gi, gh, si, sh, bi, bh, h, H, tmp):
    """gi/gh hold unscaled int8 accumulations; apply scale + bias, then torch GRU gates (r, z, n)."""
    G = 3 * H
    for j in range(G):
        gi[j] = gi[j] * si[j] + bi[j]
        gh[j] = gh[j] * sh[j] + bh[j]
    for j in range(2 * H):
        tmp[j] = _tanh(np.float32(0.5) * (gi[j] + gh[j]))
    for j in range(H):
        r = np.float32(0.5) + np.float32(0.5) * tmp[j]
        tmp[2 * H + j] = gi[2 * H + j] + r * gh[2 * H + j]
    for j in range(H):
        n = _tanh(tmp[2 * H + j])
        z = np.float32(0.5) + np.float32(0.5) * tmp[H + j]
        h[j] = n + z * (h[j] - n)


@njit(fastmath=True, inline="always")
def _gates_lstm(gi, gh, si, sh, bi, bh, h, c, H):
    """LSTM cell (torch gate order i, f, g, o): c = f c + i g, h = o tanh(c). gi/gh: unscaled int8 accumulations."""
    for j in range(4 * H):
        gi[j] = gi[j] * si[j] + bi[j] + gh[j] * sh[j] + bh[j]
    for j in range(H):
        ig = np.float32(0.5) + np.float32(0.5) * _tanh(np.float32(0.5) * gi[j])
        fg = np.float32(0.5) + np.float32(0.5) * _tanh(np.float32(0.5) * gi[H + j])
        gg = _tanh(gi[2 * H + j])
        og = np.float32(0.5) + np.float32(0.5) * _tanh(np.float32(0.5) * gi[3 * H + j])
        cj = fg * c[j] + ig * gg
        c[j] = cj
        h[j] = og * _tanh(cj)


@njit(fastmath=True, inline="always")
def _rank1(F, o, v):
    """np.interp(v, knots, vals) for one column (knots at F[O_KN + o:], RK of them, strictly increasing; vals at
    F[O_KV + o:]); flat outside, as model.GRUModel.rank_view. Branchless search: RK = 129 = 2^7 + 1."""
    kn = O_KN + o
    if v <= F[kn]:
        return F[O_KV + o]
    if v >= F[kn + RK - 1]:
        return F[O_KV + o + RK - 1]
    lo = 0  # largest lo with knot[lo] <= v (F[kn] < v here, so lo stays in [0, RK - 2])
    for step in (64, 32, 16, 8, 4, 2, 1):
        lo = lo + step if F[kn + lo + step] <= v else lo
    k0 = F[kn + lo]
    w = (v - k0) / (F[kn + lo + 1] - k0)
    a = F[O_KV + o + lo]
    return a + w * (F[O_KV + o + lo + 1] - a)


@njit(fastmath=True, inline="always")
def _flags(x, prev, fl):
    for b in range(BLOCKS.shape[0]):
        f = 0.0
        for k in range(BLOCKS[b, 0], BLOCKS[b, 1]):
            if x[k] != prev[k]:
                f = 1.0
                break
        fl[b] = f


@njit(fastmath=True, inline="always")
def _finish(inp, idx, fl):
    """Append the change flags and build the nonzero index list of a view row. Returns the nonzero count."""
    for b in range(BLOCKS.shape[0]):
        inp[2 * N_RAW + b] = fl[b]
    n = 0
    for k in range(N_IN):
        if inp[k] != 0.0:
            idx[n] = k
            n += 1
    return n


@njit(fastmath=True)
def flat_step(x, prev, Wq, F, I, out):
    nmod = I[0]
    oidx = I[6]
    Gmax = I[7]
    gi = F[O_SCR:O_SCR + Gmax]
    gh = F[O_SCR + Gmax:O_SCR + 2 * Gmax]
    tmp = F[O_SCR + 2 * Gmax:O_SCR + 3 * Gmax]
    fl = F[O_SCR + 3 * Gmax:O_SCR + 3 * Gmax + BLOCKS.shape[0]]
    ns = I[oidx + NV * N_IN:oidx + NV * N_IN + NV]
    _flags(x, prev, fl)
    mu = F[O_MU:O_MU + N_RAW]
    isd = F[O_ISD:O_ISD + N_RAW]
    idr = F[O_IDR:O_IDR + N_RAW]
    if I[1]:  # raw
        inp = F[O_INP:O_INP + N_IN]
        for k in range(N_RAW):
            inp[k] = x[k]
            inp[N_RAW + k] = x[k] - prev[k]
        ns[0] = _finish(inp, I[oidx:oidx + N_IN], fl)
    if I[2]:  # standardised
        inp = F[O_INP + N_IN:O_INP + 2 * N_IN]
        for k in range(N_RAW):
            inp[k] = (x[k] - mu[k]) * isd[k]
            inp[N_RAW + k] = (x[k] - prev[k]) * idr[k]
        ns[1] = _finish(inp, I[oidx + N_IN:oidx + 2 * N_IN], fl)
    if I[3]:  # change-only (model norm 5): levels zero, scaled diffs (sparse), flags
        inp = F[O_INP + 2 * N_IN:O_INP + 3 * N_IN]
        for k in range(N_RAW):
            inp[k] = 0.0
            inp[N_RAW + k] = (x[k] - prev[k]) * idr[k]
        ns[2] = _finish(inp, I[oidx + 2 * N_IN:oidx + 3 * N_IN], fl)
    if I[4]:  # rank-Gaussian
        inp = F[O_INP + 3 * N_IN:O_INP + 4 * N_IN]
        xl = F[O_RKX:O_RKX + N_RAW]
        zl = F[O_RKZ:O_RKZ + N_RAW]
        for k in range(N_RAW):
            xk = x[k]
            pk = prev[k]
            o = k * RK
            z = zl[k] if xk == xl[k] else _rank1(F, o, xk)
            if pk == xk:
                d = np.float32(0.0)
            elif pk == xl[k]:
                d = z - zl[k]
            else:
                d = z - _rank1(F, o, pk)
            inp[k] = z
            inp[N_RAW + k] = d
            xl[k] = xk
            zl[k] = z
        ns[3] = _finish(inp, I[oidx + 3 * N_IN:oidx + 4 * N_IN], fl)
    if I[5]:  # whitened
        inp = F[O_INP + 4 * N_IN:O_INP + 5 * N_IN]
        t = F[O_TMP112:O_TMP112 + N_RAW]
        for k in range(N_RAW):
            t[k] = (x[k] - mu[k]) * isd[k]
        for j in range(2 * N_RAW):
            inp[j] = 0.0
        for k in range(N_RAW):  # z = Wx t (row j of Wx at O_ZX + j * 112)
            a = np.float32(0.0)
            row = F[O_ZX + k * N_RAW:O_ZX + (k + 1) * N_RAW]
            for c in range(N_RAW):
                a += row[c] * t[c]
            inp[k] = a
        dz = inp[N_RAW:2 * N_RAW]
        for k in range(N_RAW):  # dz = Wd (dx / drms), only changed columns (WdT row k = Wd column k)
            if x[k] != prev[k]:
                d = (x[k] - prev[k]) * idr[k]
                col = F[O_ZD + k * N_RAW:O_ZD + (k + 1) * N_RAW]
                for j in range(N_RAW):
                    dz[j] += d * col[j]
        ns[4] = _finish(inp, I[oidx + 4 * N_IN:oidx + 5 * N_IN], fl)
    out[0] = 0.0
    out[1] = 0.0
    for i in range(nmod):
        m = I0 + MSTRIDE * i
        H = I[m]
        L = I[m + 1]
        v = I[m + 2]
        oW = I[m + 3]   # Wq: Wi0 (N_IN, G), then L-1 x Wi (H, G), then L x Wh (H, G)
        oP = I[m + 4]   # F: per layer [si, sh, bi, bh] (G each), Wo (H, 2), bo (2), w (1), h (L, H) [, c (L, H)]
        lstm = I[m + 5]
        G = 4 * H if lstm else 3 * H
        oWl = oW + N_IN * G
        oH = oP + 4 * G * L + 2 * H + 3
        for l in range(L):
            h = F[oH + l * H:oH + (l + 1) * H]
            for j in range(G):
                gi[j] = 0.0
                gh[j] = 0.0
            if l == 0:
                Wi0 = Wq[oW:oW + N_IN * G].reshape(N_IN, G)
                _matvec_acc_q(gi, Wi0, F[O_INP + v * N_IN:O_INP + (v + 1) * N_IN],
                              I[oidx + v * N_IN:oidx + (v + 1) * N_IN], ns[v], G)
            else:
                Wi = Wq[oWl + (l - 1) * H * G:oWl + l * H * G].reshape(H, G)
                _dense_acc_q(gi, Wi, F[oH + (l - 1) * H:oH + l * H], G)
            oWh = oWl + (L - 1) * H * G + l * H * G
            _dense_acc_q(gh, Wq[oWh:oWh + H * G].reshape(H, G), h, G)
            p = oP + 4 * G * l
            if lstm:
                oc = oH + L * H + l * H
                _gates_lstm(gi, gh, F[p:p + G], F[p + G:p + 2 * G], F[p + 2 * G:p + 3 * G], F[p + 3 * G:p + 4 * G],
                            h, F[oc:oc + H], H)
            else:
                _gates_fast(gi, gh, F[p:p + G], F[p + G:p + 2 * G], F[p + 2 * G:p + 3 * G], F[p + 3 * G:p + 4 * G],
                            h, H, tmp)
        oWo = oP + 4 * G * L
        hl = F[oH + (L - 1) * H:oH + L * H]
        o0 = F[oWo + 2 * H]
        o1 = F[oWo + 2 * H + 1]
        for k in range(H):
            o0 += hl[k] * F[oWo + 2 * k]
            o1 += hl[k] * F[oWo + 2 * k + 1]
        w = F[oWo + 2 * H + 2]
        out[0] += w * o0
        out[1] += w * o1


class FlatGRUQ:
    """Drop-in replacement for numba_multi.MultiGRUQ, built from the same NumbaGRUQ members.

    views: per member view slot (0 raw, 1 standardised, 2 change-only = model norm 5, 3 rank, 4 whitened); default raw.
    norm: (mu, sd, drms) for views 1 and 4; rank: (knots, vals) (112, RK) for view 3; zca: (Wx, Wd) for view 4."""

    def __init__(self, models, weights, norm=None, views=None, rank=None, zca=None):
        nmod = len(models)
        views = np.zeros(nmod, np.int64) if views is None else np.asarray(views, np.int64)
        assert views.shape == (nmod,) and set(views.tolist()) <= {0, 1, 2, 3, 4}, views
        assert norm is not None or not np.isin(views, (1, 2, 4)).any(), "views 1/2/4 need norm stats"
        assert rank is not None or not (views == 3).any(), "view 3 needs rank knots"
        assert zca is not None or not (views == 4).any(), "view 4 needs zca matrices"
        Gmax = max(m.Wi0.shape[1] for m in models)  # 3H (GRU) or 4H (LSTM)
        head = np.zeros(O_SCR + 3 * Gmax + BLOCKS.shape[0], np.float32)
        head[O_ISD:O_IDR + N_RAW] = 1.0
        head[O_RKX:O_RKX + N_RAW] = RK_EMPTY
        if norm is not None:
            mu, sd, drms = (np.asarray(a, np.float32) for a in norm)
            head[O_MU:O_MU + N_RAW] = mu
            head[O_ISD:O_ISD + N_RAW] = 1.0 / sd
            head[O_IDR:O_IDR + N_RAW] = 1.0 / drms
        if rank is not None:
            kn, kv = (np.asarray(a, np.float32) for a in rank)
            assert kn.shape == (N_RAW, RK) and kv.shape == (N_RAW, RK)
            assert (np.diff(kn, axis=1) > 0).all(), "rank knots must be strictly increasing"
            head[O_KN:O_KN + N_RAW * RK] = kn.ravel()
            head[O_KV:O_KV + N_RAW * RK] = kv.ravel()
        if zca is not None:
            wx, wd = (np.asarray(a, np.float32) for a in zca)
            head[O_ZX:O_ZX + N_RAW * N_RAW] = wx.ravel()
            head[O_ZD:O_ZD + N_RAW * N_RAW] = np.ascontiguousarray(wd.T).ravel()
        I = np.zeros(I0 + MSTRIDE * nmod + NV * N_IN + NV, np.int64)
        I[0] = nmod
        for v in range(NV):
            I[1 + v] = int((views == v).any())
        I[6] = I0 + MSTRIDE * nmod
        I[7] = Gmax
        wq_parts, f_parts = [], [head]
        oW, oP = 0, head.size
        self._h_views = []
        for i, mdl in enumerate(models):
            L, H = mdl.hs.shape
            lstm = int(mdl.Wi0.shape[1] == 4 * H)
            G = 4 * H if lstm else 3 * H
            assert mdl.Wi0.shape == (N_IN, G) and mdl.Wi0.dtype == np.int8
            wq = np.concatenate([mdl.Wi0.ravel()] + [mdl.Wis[l].ravel() for l in range(L - 1)]
                                + [mdl.Whs[l].ravel() for l in range(L)])
            par = []
            for l in range(L):
                par += [mdl.si0 if l == 0 else mdl.sis[l - 1], mdl.shs[l], mdl.bis[l], mdl.bhs[l]]
            par += [mdl.Wo.ravel(), mdl.bo, np.array([weights[i]]), np.zeros((1 + lstm) * L * H)]  # h [, c]
            fp = np.concatenate([np.asarray(a, np.float32).ravel() for a in par])
            m = I0 + MSTRIDE * i
            I[m:m + 6] = (H, L, views[i], oW, oP, lstm)
            self._h_views.append((oP + 4 * G * L + 2 * H + 3, (1 + lstm) * L * H))
            wq_parts.append(wq)
            f_parts.append(fp)
            oW += wq.size
            oP += fp.size
        self.Wq = np.ascontiguousarray(np.concatenate(wq_parts).astype(np.int8))
        self.F = np.ascontiguousarray(np.concatenate(f_parts).astype(np.float32))
        self.I = I
        self.out = np.zeros(2, np.float32)

    def reset(self):
        for o, n in self._h_views:
            self.F[o:o + n] = 0.0

    def step(self, x, prev):
        flat_step(x, prev, self.Wq, self.F, self.I, self.out)
        return self.out
