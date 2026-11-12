"""Fit a tiny logistic model P(is_scored | causal features) on cached validation sequences.

The same feature function runs in torch during training (to weight the loss), so it is written in torch.
Usage: python mask_model.py <valid_sample.npz>  -> writes data/mask_model.pt
"""
import sys

import numpy as np
import torch

from common import DATA, WARMUP


def time_since(ev):
    """ev: (B, T) bool -> steps since last True (capped at 5000; 5000 if none yet)."""
    B, T = ev.shape
    t = torch.arange(T, device=ev.device).expand(B, T)
    last = torch.where(ev, t, torch.full_like(t, -5000))
    last = torch.cummax(last, dim=1).values
    return (t - last).clamp(max=5000)


def mask_features(x):
    """x: (B, T, 112) -> (B, T, K) causal features for the scoring-probability model."""
    def chg(a, b):
        c = (x[:, 1:, a:b] != x[:, :-1, a:b]).any(-1)
        return torch.cat([torch.zeros_like(c[:, :1]), c], 1)

    feats = []
    for a, b in [(52, 74), (0, 22), (74, 96), (22, 44), (96, 104), (44, 52)]:
        ts = time_since(chg(a, b)).float()
        feats += [torch.log1p(ts), (ts == 0).float()]
    a = x[..., 104:112]
    feats += [a[..., j] for j in range(8)] + [a[..., 2] ** 2, a[..., 3] ** 2, a[..., 2] * a[..., 3]]
    return torch.stack(feats, -1)


def main():
    v = np.load(sys.argv[1])
    X = torch.from_numpy(v["X"])
    M = torch.from_numpy(v["is_scored"])
    n = X.shape[0]
    tr, te = np.arange(0, n, 2), np.arange(1, n, 2)  # split by sequence
    feats = torch.cat([mask_features(X[i:i + 10]) for i in range(0, n, 10)])[:, WARMUP:]
    M = M[:, WARMUP:].float()
    mu = feats[tr].reshape(-1, feats.shape[-1]).mean(0)
    sd = feats[tr].reshape(-1, feats.shape[-1]).std(0) + 1e-6
    Z = (feats - mu) / sd
    lin = torch.nn.Linear(Z.shape[-1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200, line_search_fn="strong_wolfe")
    ztr, mtr = Z[tr].reshape(-1, Z.shape[-1]), M[tr].reshape(-1)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(ztr).squeeze(-1), mtr)
        loss.backward()
        return loss

    opt.step(closure)

    def auc(s, y):
        o = torch.argsort(s)
        y = y[o]
        r = torch.arange(1, len(y) + 1, dtype=torch.float64)
        npos = y.sum()
        return float(((r * y).sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))

    with torch.no_grad():
        s = lin(Z[te].reshape(-1, Z.shape[-1])).squeeze(-1)
        p = torch.sigmoid(s)
        y = M[te].reshape(-1)
        print(f"held-out AUC {auc(s, y):.4f}; mean p {p.mean():.4f} vs rate {y.mean():.4f}")
        for lo, hi in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1)]:
            sel = (p >= lo) & (p < hi)
            print(f"   p in [{lo:.2f},{hi:.2f}): rows {int(sel.sum()):8d}  actual rate {float(y[sel].mean()) if sel.any() else float('nan'):.4f}")
    torch.save({"mu": mu, "sd": sd, "w": lin.weight.detach().squeeze(0), "b": lin.bias.detach()}, DATA / "mask_model.pt")
    print("saved", DATA / "mask_model.pt")


if __name__ == "__main__":
    main()
