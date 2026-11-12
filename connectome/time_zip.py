"""Time packaged submissions side by side (interleaved, median and min over windows of rows).

Usage: python time_zip.py <submission_dir> [<submission_dir> ...]   (dirs under runs/submissions)
Calibrate against a zip with a known scorer runtime: scorer_min ~= known_min * (median_new / median_known).
"""
import importlib.util
import sys
import time

import numpy as np

from common import RUNS, VALID, WARMUP, SeqReader

sys.path.insert(0, str(RUNS.parent / "wnn_connectome_starterpack"))
from utils import DataPoint  # noqa: E402

names = sys.argv[1:]
pms = []
for i, n in enumerate(names):
    spec = importlib.util.spec_from_file_location(f"sol{i}", RUNS / "submissions" / n / "solution.py")
    for _m in ("numba_gru", "numba_multi", "numba_flat"):
        sys.modules.pop(_m, None)  # each package carries its own kernel
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pms.append(mod.PredictionModel())
xs, _, _ = SeqReader(VALID).read(5)
W = 500
times = {n: [] for n in names}
for rep in range(3):
    for n, pm in zip(names, pms):
        for c in range(0, len(xs), W):
            t = time.perf_counter()
            for s in range(c, c + W):
                pm.predict(DataPoint(5 + rep, s, s >= WARMUP, xs[s]))
            times[n].append((time.perf_counter() - t) / W * 1e6)
for n in names:
    a = np.array(times[n])
    print(f"{n:28s} median {np.median(a):6.1f} us  min {a.min():6.1f} us")
