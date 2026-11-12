"""Stateful GRU ensemble, single-step inference in numba (no ONNX). Weights in models.npz next to this file.

If models.npz has int8=1, weights are quantised to int8 at load time (NumbaGRUQ, ~1.35x faster)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from numba_gru import NumbaGRU, NumbaGRUQ  # noqa: E402


class PredictionModel:
    def __init__(self):
        z = np.load(HERE / "models.npz")
        n_models = int(z["n_models"])
        cls = NumbaGRUQ if "int8" in z.files and int(z["int8"]) else NumbaGRU
        self.models = []
        for i in range(n_models):
            pre = f"m{i}/"
            sd = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            self.models.append(cls(sd, int(z[f"hidden{i}"]), int(z[f"layers{i}"])))
        self.weights = np.asarray(z["weights"], dtype=np.float32)
        self.out = np.zeros(2, np.float32)
        self.seq_ix = None
        self.prev = None
        # compile now so the first sequence is not slowed down
        x0 = np.zeros(112, np.float32)
        for m in self.models:
            m.step(x0, x0)
            m.reset()

    def predict(self, data_point):
        # models are trained on float16-rounded inputs (cache.py), so round here too
        x = np.asarray(data_point.state).astype(np.float16).astype(np.float32)
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            for m in self.models:
                m.reset()
            self.prev = x
        if len(self.models) == 1:
            out = self.models[0].step(x, self.prev).copy()
        else:
            out = self.out
            out[:] = 0.0
            for m, w in zip(self.models, self.weights):
                out += w * m.step(x, self.prev)
            out = out.copy()
        self.prev = x
        if not data_point.need_prediction:
            return None
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
        return out
