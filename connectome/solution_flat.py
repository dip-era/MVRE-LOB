"""Stateful int8 GRU ensemble, one fused numba call per row (numba_flat.FlatGRUQ). Weights in models.npz next to
this file (same layout as solution_numba.py; `weights` already include the output scale)."""
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
from numba_gru import NumbaGRUQ  # noqa: E402
from numba_flat import FlatGRUQ  # noqa: E402


class PredictionModel:
    def __init__(self):
        z = np.load(HERE / "models.npz")
        models = []
        for i in range(int(z["n_models"])):
            pre = f"m{i}/"
            sd = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            models.append(NumbaGRUQ(sd, int(z[f"hidden{i}"]), int(z[f"layers{i}"])))
        norm = (z["norm_mu"], z["norm_sd"], z["norm_drms"]) if "norm_mu" in z.files else None
        # per member input view (model norm id): 0 raw, 1 standardised, 3 rank-Gaussian, 4 whitened
        views = z["view"] if "view" in z.files else (z["usen"] if "usen" in z.files else None)
        rank = (z["rk_knots"], z["rk_vals"]) if "rk_knots" in z.files else None
        zca = (z["zca_x"], z["zca_d"]) if "zca_x" in z.files else None
        self.ens = FlatGRUQ(models, z["weights"], norm, views, rank, zca)
        self.mix = z["mix"].astype(np.float32) if "mix" in z.files else np.zeros(2, np.float32)  # out_t += mix_t * out_other
        self.use_mix = bool(np.any(self.mix != 0))
        self.seq_ix = None
        self.prev = np.zeros(112, np.float32)
        x0 = np.zeros(112, np.float32)
        self.ens.step(x0, x0)  # compile now so the first sequence is not slowed down
        self.ens.reset()

    def predict(self, data_point):
        # models are trained on float16-rounded inputs (cache.py), so round here too
        x = np.asarray(data_point.state).astype(np.float16).astype(np.float32)
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.ens.reset()
            self.prev = x
        out = self.ens.step(x, self.prev)
        self.prev = x
        if not data_point.need_prediction:
            return None
        out = out.copy()
        if self.use_mix:
            out = out + self.mix * out[::-1]
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
        return out
