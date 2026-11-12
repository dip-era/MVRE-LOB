"""Diversity check on the held-out odd valid half: base model + one partner, std-weighted average, no-clip
(x0.25 before the metric's clip, as in package.py --stdw --scale 0.25). All partners share the base recipe/seed,
so differences in the pair score measure how much new information each partner adds."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "connectome"))
from cache import open_cache  # noqa: E402
from common import RUNS, WARMUP, WPAccumulator  # noqa: E402
from model import load_model  # noqa: E402

X, Y, M = open_cache("valid")
idx = np.arange(1, len(X), 2)
argv = sys.argv[1:]
specs = argv[:argv.index("--")] if "--" in argv else argv
EXTRA = argv[argv.index("--") + 1:] if "--" in argv else []
P = {}
for spec in specs:
    run, ck = spec.split(":")
    m, _ = load_model(RUNS / run / ck, "cuda")
    out = np.zeros((len(idx), X.shape[1], 2), np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            b = idx[i:i + 16]
            out[i:i + len(b)] = m(torch.from_numpy(np.array(X[b])).cuda().float())[0].cpu().numpy()
    sd = out[:, WARMUP:].std()
    P[spec] = out / sd
    print(f"loaded {spec} (output std {sd:.3f})", flush=True)


def score(p):
    acc = WPAccumulator()
    for i, s in enumerate(idx):
        acc.add(np.array(Y[s]), 0.25 * p[i], np.array(M[s]))
    return acc.result()[0]


base = specs[0]
print(f"single {score(P[base]):.5f}  {base}")
for s in specs[1:]:
    print(f"single {score(P[s]):.5f}  {s}   pair with base {score((P[base] + P[s]) / 2):.5f}", flush=True)
# extra groups after "--": '+'-joined specs that were all loaded above
for g in EXTRA:
    ss = g.split("+")
    print(f"group  {score(sum(P[s] for s in ss) / len(ss)):.5f}  {g}", flush=True)
