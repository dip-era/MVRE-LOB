"""Cache a random sample of sequences as numpy arrays for fast exploration.

Usage: python make_sample.py <parquet> <out.npz> <n_sequences> [seed]
Output: X (n, 20000, 112) float32, Y (n, 20000, 2) float32, seq_ix (n,), row_group (n,)
        plus is_scored (n, 20000) bool when the file has it (validation).
"""
import sys
import time

import numpy as np
import pyarrow.parquet as pq

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

pf = pq.ParquetFile(src)
names = pf.schema_arrow.names
feat_cols = names[names.index("need_prediction") + 1:]
if "is_scored" in feat_cols:
    feat_cols = feat_cols[1:]
feat_cols = feat_cols[:-2]
assert len(feat_cols) == 112, len(feat_cols)
has_mask = "is_scored" in names

rng = np.random.default_rng(seed)
groups = np.sort(rng.choice(pf.num_row_groups, size=n, replace=False))
X = np.empty((n, 20_000, 112), np.float32)
Y = np.empty((n, 20_000, 2), np.float32)
M = np.empty((n, 20_000), bool) if has_mask else None
seq = np.empty(n, np.int64)
t = time.time()
for i, g in enumerate(groups):
    tb = pf.read_row_group(int(g))
    X[i] = np.column_stack([tb[c].to_numpy() for c in feat_cols])
    Y[i] = np.column_stack([tb["t0"].to_numpy(), tb["t1"].to_numpy()])
    seq[i] = tb["seq_ix"][0].as_py()
    if has_mask:
        M[i] = tb["is_scored"].to_numpy()
    if i % 50 == 0:
        print(f"{i}/{n}  {time.time() - t:.0f}s", flush=True)
extra = {"is_scored": M} if has_mask else {}
np.savez(out, X=X, Y=Y, seq_ix=seq, row_group=groups, feature_names=np.array(feat_cols), **extra)
print("saved", out, X.shape, f"{time.time() - t:.0f}s")
