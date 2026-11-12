"""Shared data access and metric helpers."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "wnn_connectome_starterpack"
TRAIN = PACK / "datasets" / "train.parquet"
VALID = PACK / "datasets" / "valid.parquet"
DATA = ROOT / "data"
RUNS = ROOT / "runs"

T = 20_000
WARMUP = 99
N_FEATURES = 112
FEATURE_COLUMNS = [f"i{i}_{g}{j}" for i in range(2)
                   for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)] for j in range(n)]
FEATURE_COLUMNS += [f"a{i}" for i in range(8)]


class SeqReader:
    """Reads one sequence (= one parquet row group) as numpy arrays. Opens the file lazily so it is fork/spawn safe."""

    def __init__(self, path):
        self.path = str(path)
        self._pf = None

    @property
    def pf(self):
        if self._pf is None:
            self._pf = pq.ParquetFile(self.path)
        return self._pf

    @property
    def num_seqs(self):
        return self.pf.num_row_groups

    def read(self, g):
        has_mask = "is_scored" in self.pf.schema_arrow.names
        cols = FEATURE_COLUMNS + ["t0", "t1"] + (["is_scored"] if has_mask else [])
        tb = self.pf.read_row_group(int(g), columns=cols, use_threads=False)
        x = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32, copy=False)
        y = np.column_stack([tb["t0"].to_numpy(), tb["t1"].to_numpy()]).astype(np.float32, copy=False)
        if has_mask:
            m = tb["is_scored"].to_numpy().astype(bool)
        else:
            m = np.ones(T, bool)
        m[:WARMUP] = False
        return x, y, m


class WPAccumulator:
    """Global weighted Pearson over many sequences, same math as the official scorer (pooled moments)."""

    def __init__(self):
        self.sw = np.zeros(2)
        self.swy = np.zeros(2)
        self.swp = np.zeros(2)
        self.swyy = np.zeros(2)
        self.swpp = np.zeros(2)
        self.swyp = np.zeros(2)

    def add(self, y, p, mask):
        y = np.clip(y[mask], -2, 2).astype(np.float64)
        p = np.clip(p[mask], -2, 2).astype(np.float64)
        w = np.abs(y)
        self.sw += w.sum(0)
        self.swy += (w * y).sum(0)
        self.swp += (w * p).sum(0)
        self.swyy += (w * y * y).sum(0)
        self.swpp += (w * p * p).sum(0)
        self.swyp += (w * y * p).sum(0)

    def result(self):
        my, mp = self.swy / self.sw, self.swp / self.sw
        cov = self.swyp / self.sw - my * mp
        vy = self.swyy / self.sw - my * my
        vp = self.swpp / self.sw - mp * mp
        r = cov / np.sqrt(np.maximum(vy * vp, 1e-30))
        return float(r.mean()), float(r[0]), float(r[1])
