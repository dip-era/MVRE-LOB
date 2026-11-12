"""Convert train/valid parquet to raw memmaps: x float16 (N, T, 112), y float32 (N, T, 2), valid mask bool (N, T).

Inputs are stored as float16; solution.py rounds test inputs to float16 too so train and inference match exactly.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import DATA, TRAIN, VALID, T, N_FEATURES, SeqReader

CACHE = DATA / "cache"


def paths(split):
    return {k: CACHE / f"{split}_{k}.bin" for k in ("x", "y", "m")}


def open_cache(split, mode="r"):
    n = int((CACHE / f"{split}_n.txt").read_text())
    p = paths(split)
    return (np.memmap(p["x"], np.float16, mode, shape=(n, T, N_FEATURES)),
            np.memmap(p["y"], np.float32, mode, shape=(n, T, 2)),
            np.memmap(p["m"], np.bool_, mode, shape=(n, T)))


_job = {}


def _init(split, src):
    _job["reader"] = SeqReader(src)
    _job["mm"] = open_cache(split, "r+")


def _do(g):
    x, y, m = _job["reader"].read(g)
    X, Y, M = _job["mm"]
    X[g] = x.astype(np.float16)
    Y[g] = y
    M[g] = m
    return g


def build(split, src, workers=8):
    n = SeqReader(src).num_seqs
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{split}_n.txt").write_text(str(n))
    for k, (dt, shape) in {"x": (np.float16, (n, T, N_FEATURES)), "y": (np.float32, (n, T, 2)),
                           "m": (np.bool_, (n, T))}.items():
        np.memmap(paths(split)[k], dt, "w+", shape=shape).flush()
    t = time.time()
    with Pool(workers, _init, (split, src)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_do, range(n), chunksize=4)):
            if (i + 1) % 500 == 0:
                print(f"{split} {i + 1}/{n} {time.time() - t:.0f}s", flush=True)
    print(f"{split} done: {n} seqs in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    for split in sys.argv[1:] or ["valid", "train"]:
        build(split, {"train": TRAIN, "valid": VALID}[split])
