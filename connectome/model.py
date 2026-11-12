"""Stateful GRU model. Feature engineering lives inside the module so the exported ONNX step graph
computes exactly what training saw: input = (x_t, x_{t-1}) -> raw, diff and per-block change flags."""
import torch
import torch.nn.functional as F
from torch import nn

BLOCKS = [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104)]  # i0 p, i0 v, i0 trade, i1 p, i1 v, i1 trade
N_IN = 112 * 2 + len(BLOCKS)


def make_features(x, x_prev):
    """x, x_prev: (..., 112) -> (..., N_IN). Works for (B, T, 112) and (1, 1, 112)."""
    dx = x - x_prev
    adx = dx.abs()
    flags = [(adx[..., a:b].sum(-1, keepdim=True) > 0).to(x.dtype) for a, b in BLOCKS]
    return torch.cat([x, dx] + flags, dim=-1)


# multi-timescale inputs: per-block level signals relative to their EMAs, and EMA rates of the block change flags
MS_TAUS = (10, 100, 1000)
MS_LEVEL = [(0, 22), (22, 44), (44, 48), (48, 52), (52, 74), (74, 96), (96, 100), (100, 104)]
N_MS = len(MS_TAUS) * (len(MS_LEVEL) + len(BLOCKS))


def ema(s, tau):
    """Causal EMA along time, e_t = e_{t-1} + (s_t - e_{t-1}) / tau with e_{-1} = s_0. s: (B, T, K).

    Computed as an FFT convolution with the kernel a(1-a)^k (float64), which equals the step-by-step recursion
    used at inference."""
    B, T, K = s.shape
    a = 1.0 / tau
    k = torch.arange(T, device=s.device, dtype=torch.float64)
    n = 2 * T
    S = torch.fft.rfft(s.double().transpose(1, 2), n=n)
    e = torch.fft.irfft(S * torch.fft.rfft(a * (1 - a) ** k, n=n), n=n)[..., :T]
    e = e + (1 - a) ** (k + 1) * s[:, 0, :].double()[..., None]
    return e.transpose(1, 2).to(s.dtype)


def ms_features(x):
    """x: (B, T, 112) -> (B, T, N_MS) unnormalised multi-timescale features."""
    lev = torch.stack([x[..., a:b].mean(-1) for a, b in MS_LEVEL], -1)
    adx = (x - shift_prev(x)).abs()
    fl = torch.stack([(adx[..., a:b].sum(-1) > 0).to(x.dtype) for a, b in BLOCKS], -1)
    out = []
    for tau in MS_TAUS:
        out += [lev - ema(lev, tau), ema(fl, tau)]
    return torch.cat(out, -1)


def shift_prev(x):
    """x: (B, T, F) -> previous row, with row 0 repeated (diff = 0 at step 0, as at inference)."""
    return torch.cat([x[:, :1], x[:, :-1]], dim=1)


RK_K = 129  # knots per column of the rank-Gaussian view (tools/view_stats.py)


class GRUModel(nn.Module):
    RTAU = 1000  # norm=2: EMA time constant (rows) of the per-sequence running mean/variance

    def __init__(self, hidden=128, layers=2, dropout=0.0, n_aux=0, ms=False, wdrop=0.0, norm=0, cell="gru"):
        super().__init__()
        self.hidden, self.layers, self.ms, self.norm, self.cell = hidden, layers, ms, int(norm), cell
        # norm=5: change-only view = norm 1 with the level inputs zeroed (scaled diffs + change flags only);
        # runs in view slot 2 of numba_flat.py (sparse: only changed columns are nonzero)
        # norm=1: global standardisation; norm=2: per-sequence running z-score of the levels;
        # norm=3: rank-Gaussian view; norm=4: whitened (ZCA) view
        # z_t = (x_t - m_t) / sqrt(v_t + (sd/2)^2), m/v = causal EMAs (tau RTAU) of x and x^2; dx scaled as norm=1
        if norm:
            # input scaling (data/norm_stats.npz, set by train.py --norm 1, saved in the checkpoint):
            # x -> (x - in_mu) / in_sd, dx -> dx / d_rms (zeros stay zeros), change flags unchanged
            self.register_buffer("in_mu", torch.zeros(112))
            self.register_buffer("in_sd", torch.ones(112))
            self.register_buffer("d_rms", torch.ones(112))
        if norm == 3:
            # rank-Gaussian view (data/view_stats.npz): per-column piecewise-linear map through quantile knots
            self.register_buffer("rk_knots", torch.zeros(112, RK_K))
            self.register_buffer("rk_vals", torch.zeros(112, RK_K))
        if norm == 4:
            # whitened view (data/view_stats.npz): regularised ZCA of the standardised levels and the scaled diffs
            self.register_buffer("zca_x", torch.eye(112))
            self.register_buffer("zca_d", torch.eye(112))
        self.wdrop = wdrop  # DropConnect on the recurrent weights (AWD-LSTM weight drop), training only
        if ms:
            # normalisation of the multi-timescale inputs; set from data before training, saved in the checkpoint
            self.register_buffer("ms_mu", torch.zeros(N_MS))
            self.register_buffer("ms_sd", torch.ones(N_MS))
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]  # attribute keeps the name gru for both cells
        self.gru = rnn(N_IN + (N_MS if ms else 0), hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 2)
        # training-only head for auxiliary targets (shifted copies of y); never exported
        self.aux = nn.Linear(hidden, 2 * n_aux) if n_aux else None

    def forward(self, x, h=None, with_aux=False, x0=None):
        """Full-sequence forward for training/eval. x: (B, T, 112).
        x0: (B, 112) row before x[:, 0] when continuing a sequence in chunks (default: diff 0 at the first row)."""
        prev = shift_prev(x) if x0 is None else torch.cat([x0[:, None], x[:, :-1]], 1)
        f = make_features(x, prev)
        if self.norm == 1:
            f = torch.cat([(f[..., :112] - self.in_mu) / self.in_sd, f[..., 112:224] / self.d_rms, f[..., 224:]], -1)
        elif self.norm == 5:
            f = torch.cat([torch.zeros_like(f[..., :112]), f[..., 112:224] / self.d_rms, f[..., 224:]], -1)
        elif self.norm == 2:
            xs = x.float()
            m = ema(xs, self.RTAU)
            v = (ema(xs * xs, self.RTAU) - m * m).clamp_min(0)
            z = ((xs - m) / torch.sqrt(v + 0.25 * self.in_sd ** 2)).to(f.dtype)
            f = torch.cat([z, f[..., 112:224] / self.d_rms, f[..., 224:]], -1)
        elif self.norm == 3:
            z = self.rank_view(x.float())
            zp = shift_prev(z) if x0 is None else torch.cat([self.rank_view(x0.float()[:, None]), z[:, :-1]], 1)
            f = torch.cat([z, z - zp, f[..., 224:].float()], -1)
        elif self.norm == 4:
            zs = (f[..., :112].float() - self.in_mu) / self.in_sd
            f = torch.cat([zs @ self.zca_x.T, (f[..., 112:224].float() / self.d_rms) @ self.zca_d.T, f[..., 224:].float()],
                          -1)
        if self.ms:
            f = torch.cat([f, (ms_features(x) - self.ms_mu) / self.ms_sd], -1)
        if self.training and self.wdrop > 0:
            # one dropout mask on each weight_hh per forward (shared by the batch and all time steps)
            params = dict(self.gru.named_parameters())
            for k in range(self.layers):
                n = f"weight_hh_l{k}"
                params[n] = F.dropout(params[n], self.wdrop, True)
            out, h = torch.func.functional_call(self.gru, params, (f, h))
        else:
            out, h = self.gru(f, h)
        if with_aux:
            return self.head(out), h, self.aux(out)
        return self.head(out), h

    def rank_view(self, x):
        """(B, T, 112) float32 -> rank-Gaussian values: linear interpolation between knots, flat outside."""
        B, T, _ = x.shape
        xt = x.reshape(-1, 112).T.contiguous()
        i = torch.searchsorted(self.rk_knots, xt).clamp(1, RK_K - 1)
        k0, k1 = self.rk_knots.gather(1, i - 1), self.rk_knots.gather(1, i)
        v0, v1 = self.rk_vals.gather(1, i - 1), self.rk_vals.gather(1, i)
        w = ((xt - k0) / (k1 - k0)).clamp(0, 1)
        return (v0 + w * (v1 - v0)).T.reshape(B, T, 112)


def aux_horizons(args):
    """Parse the --aux string ("-1000,-300,100,300") from a run's args dict."""
    s = args.get("aux", "")
    return [int(v) for v in s.split(",")] if s else []


def load_model(path, device="cpu"):
    """Build a GRUModel matching a checkpoint's args and load it. Returns (model in eval mode, checkpoint)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    m = GRUModel(a["hidden"], a["layers"], n_aux=len(aux_horizons(a)), ms=bool(a.get("ms", 0)),
                  norm=int(a.get("norm", 0)), cell=a.get("cell", "gru")).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, ck


class StepModel(nn.Module):
    """One-step wrapper for ONNX export: (x, x_prev, h) -> (pred, h_new).

    Mathematically identical to GRUModel but cheaper per call: the diff term is folded into the first-layer
    weights (W·[x, x-xp] = (Wx+Wd)·x - Wd·xp) and the change flags are one Abs->MatMul->Sign chain
    instead of a ReduceSum/Greater/Cast per block. Cuts the graph from 77 to 25 nodes."""

    def __init__(self, m: GRUModel):
        super().__init__()
        self.gru = nn.GRU(N_IN, m.hidden, num_layers=m.layers, batch_first=True)
        self.gru.load_state_dict(m.gru.state_dict())
        with torch.no_grad():
            w = m.gru.weight_ih_l0
            wx, wd = w[:, :112], w[:, 112:224]
            self.gru.weight_ih_l0[:, :112] = wx + wd
            self.gru.weight_ih_l0[:, 112:224] = -wd
        self.head = m.head
        bm = torch.zeros(112, len(BLOCKS))
        for k, (a, b) in enumerate(BLOCKS):
            bm[a:b, k] = 1
        self.register_buffer("bm", bm)

    def forward(self, x, x_prev, h):
        flags = torch.sign(torch.abs(x - x_prev) @ self.bm)
        out, h_new = self.gru(torch.cat([x, x_prev, flags], -1), h)
        return self.head(out), h_new
