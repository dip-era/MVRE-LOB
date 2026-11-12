"""Leak-free group check for train-only members on the FULL validation set (no member saw any validation rows).

Usage: python group_eval.py run:ckpt [run:ckpt ...] -- "a+b+c" "a+b+d" ...
Specs and group members use the same run:ckpt strings; a spec repeated in a group acts as a weight.
Each member's output is divided by its std (rows past warmup), groups average them, x0.25, clip +-2 (as package.py
--stdw --scale 0.25). Predictions on the scored rows are cached in tools/preds/ so later calls are fast.
Prints the full-valid WP and the even/odd halves (a rough noise gauge: a gain should show on both halves)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "connectome"))
from cache import open_cache  # noqa: E402
from common import RUNS, WARMUP  # noqa: E402
from model import load_model  # noqa: E402

PRED = HERE / "preds"
PRED.mkdir(exist_ok=True)
X, Y, M = open_cache("valid")
n = len(X)
argv = sys.argv[1:]
specs = argv[:argv.index("--")] if "--" in argv else argv
groups = argv[argv.index("--") + 1:] if "--" in argv else []

# scored rows, their targets and which half (even/odd sequence index) they belong to
mask = np.asarray(M, bool)
ys = np.clip(np.asarray(Y)[mask].astype(np.float64), -2, 2)
half = np.broadcast_to((np.arange(n) % 2)[:, None], mask.shape)[mask]


def preds(spec):
    f = PRED / (spec.replace(":", "__").replace(".pt", "") + ".npy")
    if f.exists():
        return np.load(f)
    run, ck = spec.split(":")
    m, _ = load_model(RUNS / run / ck, "cuda")
    out = np.zeros((n, X.shape[1], 2), np.float32)
    with torch.no_grad():
        for i in range(0, n, 16):
            out[i:i + 16] = m(torch.from_numpy(np.array(X[i:i + 16])).cuda().float())[0].cpu().numpy()
    p = (out[mask] / out[:, WARMUP:].std()).astype(np.float32)
    np.save(f, p)
    return p


def wp(p, sel=None):
    y, q = (ys, p) if sel is None else (ys[sel], p[sel])
    q = np.clip(0.25 * q.astype(np.float64), -2, 2)
    w = np.abs(y)
    r = []
    for t in range(2):
        ww, a, b = w[:, t], y[:, t], q[:, t]
        sw = ww.sum()
        ma, mb = (ww * a).sum() / sw, (ww * b).sum() / sw
        cov = (ww * a * b).sum() / sw - ma * mb
        va, vb = (ww * a * a).sum() / sw - ma * ma, (ww * b * b).sum() / sw - mb * mb
        r.append(cov / np.sqrt(va * vb))
    return float(np.mean(r))


def line(p):
    return f"{wp(p):.5f}  (even {wp(p, half == 0):.5f} odd {wp(p, half == 1):.5f})"


P = {}
for s in specs:
    P[s] = preds(s)
    print(f"single {line(P[s])}  {s}", flush=True)
for g in groups:
    ss = g.split("+")
    print(f"group  {line(sum(P[s] for s in ss) / len(ss))}  {g}", flush=True)
