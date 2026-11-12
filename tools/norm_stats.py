"""Per-column input statistics for train.py --norm 1 (saved to data/norm_stats.npz).

x is standardised: (x - mu) / sd. dx is only scaled: dx / rms(dx), so unchanged columns stay exactly 0 (sparse kernel).
Computed on float16-rounded train sequences (same rounding as training and solution.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "connectome"))
from cache import open_cache  # noqa: E402
from common import DATA  # noqa: E402

X = open_cache("train")[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(X.shape[0], 400, replace=False))
s1 = np.zeros(112); s2 = np.zeros(112); d2 = np.zeros(112); n = 0; nd = 0
for g in idx:
    x = np.asarray(X[g], np.float64)
    d = np.diff(x, axis=0)
    s1 += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
    d2 += (d ** 2).sum(0); nd += len(d)
mu = s1 / n
sd = np.sqrt(s2 / n - mu ** 2)
drms = np.sqrt(d2 / nd)
np.set_printoptions(precision=3, suppress=True, linewidth=140)
print("x sd  min/median/max", sd.min(), np.median(sd), sd.max())
print("dx rms min/median/max", drms.min(), np.median(drms), drms.max())
# how large do normalised values get on a few valid-like train seqs?
z = (np.asarray(X[idx[0]], np.float64) - mu) / sd
print("max |z| on one seq", np.abs(z).max().round(1), " 99.9% |z|", np.quantile(np.abs(z), 0.999).round(1))
np.savez(DATA / "norm_stats.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32), drms=drms.astype(np.float32))
print("saved", DATA / "norm_stats.npz")
