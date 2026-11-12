"""Statistics for two extra input views (saved to data/view_stats.npz), from 100 random train sequences (every 4th row), like
tools/norm_stats.py (float16-rounded, as in training).

norm=3, rank-Gaussian view: x_k -> Phi^-1(F_k(x_k)), piecewise linear through K quantile knots per column (ties merged,
  flat beyond the outer knots); dx view = difference of consecutive transformed rows (unchanged column -> exactly 0).
norm=4, whitened view: regularised ZCA of the standardised levels and of the scaled diffs,
  z = Wx (x - mu) / sd,  dz = Wd dx / drms,  W = U diag((lam + EPS)^-1/2) U^T (lam = eigenvalues of the correlation
  matrix, mean 1). EPS caps the gain of tiny-variance directions (e.g. spreads between co-moving price levels).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "connectome"))
from cache import open_cache  # noqa: E402
from common import DATA  # noqa: E402

K = 129
EPS = 0.05
X = open_cache("train")[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(X.shape[0], 100, replace=False))
xs = np.concatenate([np.asarray(X[g][::4], np.float32) for g in idx])   # (100 T / 4, 112)
ds = np.concatenate([np.diff(np.asarray(X[g], np.float32), axis=0)[::4] for g in idx]).astype(np.float64)
ns = np.load(DATA / "norm_stats.npz")
mu, sd, drms = ns["mu"].astype(np.float64), ns["sd"].astype(np.float64), ns["drms"].astype(np.float64)

# ---- rank-Gaussian knots ----
probs = np.linspace(0.0, 1.0, K)
targ = ndtri(np.clip(probs, 0.001, 0.999))   # +-3.09 at the ends
knots = np.zeros((112, K), np.float64)
vals = np.zeros((112, K), np.float64)
nk = np.zeros(112, np.int64)
for c in range(112):
    q = np.quantile(xs[:, c].astype(np.float64), probs)
    u, inv = np.unique(q, return_inverse=True)
    t = np.array([targ[inv == i].mean() for i in range(len(u))])
    n = len(u)
    nk[c] = n
    knots[c, :n], vals[c, :n] = u, t
    # pad: strictly increasing far knots with the last value (flat extension)
    knots[c, n:] = u[-1] + 1e4 * np.arange(1, K - n + 1)
    vals[c, n:] = t[-1]
print("rank view: knots per column min/median/max", nk.min(), int(np.median(nk)), nk.max())


def rank_tf(x):
    out = np.empty_like(x, dtype=np.float64)
    for c in range(112):
        out[:, c] = np.interp(x[:, c], knots[c, :nk[c]], vals[c, :nk[c]])
    return out


z = rank_tf(xs[:200000].astype(np.float64))
print("rank view: z std median", np.median(z.std(0)).round(3), " min", z.std(0).min().round(3),
      " corr with standardised (median over cols)",
      np.median([np.corrcoef(z[:, c], xs[:200000, c])[0, 1] for c in range(112) if z[:, c].std() > 0]).round(3))

# ---- regularised ZCA ----
def zca(m):
    c = np.corrcoef(m, rowvar=False)
    lam, U = np.linalg.eigh(c)
    lam = np.clip(lam, 0, None)
    return (U * (1.0 / np.sqrt(lam + EPS))) @ U.T, lam


Wx, lx = zca((xs.astype(np.float64) - mu) / sd)
Wd, ld = zca(ds / drms)
print("ZCA x: eigenvalues smallest", np.sort(lx)[:5].round(4), " #<0.05:", (lx < 0.05).sum(), " max gain",
      (1 / np.sqrt(lx.min() + EPS)).round(2))
print("ZCA dx: eigenvalues smallest", np.sort(ld)[:5].round(4), " #<0.05:", (ld < 0.05).sum())
zz = ((xs[:100000].astype(np.float64) - mu) / sd) @ Wx.T
print("whitened x: std per col median", np.median(zz.std(0)).round(3), " 99.9% |z|", np.quantile(np.abs(zz), 0.999).round(2))
dz = (ds[:100000] / drms) @ Wd.T
print("whitened dx: nonzero frac (raw dx nonzero frac)", (np.abs(dz) > 1e-9).mean().round(3), (ds[:100000] != 0).mean().round(3))
np.savez(DATA / "view_stats.npz", rk_knots=knots.astype(np.float32), rk_vals=vals.astype(np.float32),
         zca_x=Wx.astype(np.float32), zca_d=Wd.astype(np.float32))
print("saved", DATA / "view_stats.npz")
