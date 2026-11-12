"""Package one or more checkpoints as a numba submission, verify against torch, time it, zip it.

Usage: python package.py <out_name> <run1>[:ckpt][=weight] [<run2>...] [--check N]
Example: python package.py ens2 gru128_ema gru128_ema_i1w3 --check 3
"""
import argparse
import importlib.util
import os
import shutil
import sys
import time
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU on Windows
import numpy as np
import torch

from common import RUNS, VALID, WARMUP, SeqReader, WPAccumulator
from model import load_model

sys.path.insert(0, str(RUNS.parent / "wnn_connectome_starterpack"))
from utils import DataPoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("runs", nargs="+")
ap.add_argument("--check", type=int, default=2)
ap.add_argument("--int8", action="store_true", help="quantise GRU weights to int8 at load time (faster, ~1e-2 diff)")
ap.add_argument("--scale", type=float, default=1.0,
                help="multiply the ensemble output (the metric clips at +-2; Pearson ignores scale, the clip does not)")
ap.add_argument("--multi", action="store_true",
                help="use solution_multi.py (one fused numba call per row for the whole ensemble, ~6%% faster)")
ap.add_argument("--flat", action="store_true",
                help="with --multi: use solution_flat.py / numba_flat.py (flat arrays + fast gates, ~1.44x faster)")
ap.add_argument("--mix", default="0,0",
                help="cross-target mixing 'c0,c1': out_t += c_t * out_other (fitted on valid halves: c1 ~ -0.05)")
ap.add_argument("--stdw", action="store_true",
                help="divide each model's weight by its output std on the check sequences (equal contributions)")
args = ap.parse_args()

models, arrays, ws, norms, views, vstats = [], {}, [], [], [], {}
VIEW_KEYS = ("rk_knots", "rk_vals", "zca_x", "zca_d")  # extra-view buffers (model norm 3 / 4), shared per zip
for i, spec in enumerate(args.runs):
    w = 1.0
    if "=" in spec:
        spec, w = spec.split("=")
        w = float(w)
    run, ck_name = (spec.split(":") + ["best.pt"])[:2]
    cks = ck_name.split("+")  # "ep4.pt+ep5.pt+ep6.pt" = average of those checkpoints' weights
    m, ck = load_model(RUNS / run / cks[0])
    if len(cks) > 1:
        sds = [torch.load(RUNS / run / c, map_location="cpu", weights_only=False)["model"] for c in cks]
        m.load_state_dict({k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]})
        m.eval()
    a = ck["args"]
    models.append(m)
    ws.append(w)
    print(f"model {i}: {run}/{ck_name} epoch {ck['epoch']} valid WP {ck['wp']:.5f} weight {w}")
    sdm = m.state_dict()
    norms.append(tuple(sdm[k].numpy() for k in ("in_mu", "in_sd", "d_rms")) if "in_mu" in sdm else None)
    views.append(int(a.get("norm", 0)))
    for k in VIEW_KEYS:
        if k in sdm:
            vstats.setdefault(k, []).append(sdm[k].numpy())
    for k, v in sdm.items():
        if k.startswith("aux.") or k in ("in_mu", "in_sd", "d_rms") or k in VIEW_KEYS:
            continue
        arrays[f"m{i}/{k}"] = v.numpy().astype(np.float32)
    arrays[f"hidden{i}"] = np.array(a["hidden"])
    arrays[f"layers{i}"] = np.array(a["layers"])
ws = np.array(ws, np.float32)
if args.stdw:
    rd = SeqReader(VALID)
    sds = np.zeros(len(models))
    for g in range(args.check):
        xs = rd.read(g)[0].astype(np.float16).astype(np.float32)
        with torch.no_grad():
            for i, m in enumerate(models):
                sds[i] += m(torch.from_numpy(xs)[None])[0][0].numpy()[WARMUP:].std() / args.check
    print("model output std:", sds.round(3))
    ws = ws / sds.astype(np.float32)
ws = ws / ws.sum() * args.scale
print("final weights:", ws.round(4))
arrays["weights"] = ws
mix = np.array([float(v) for v in args.mix.split(",")], np.float32)
assert mix.shape == (2,), "--mix needs two values 'c0,c1'"
print("cross-target mix:", mix)
arrays["mix"] = mix
# raw-input and standardised-input members can be mixed; the standardised ones must share one set of stats
usen = np.array([n is not None for n in norms], np.int64)
if usen.any():
    assert args.multi, "standardised-input models need --multi (solution_multi.py)"
    ref = norms[int(np.argmax(usen))]
    assert all(n is None or all(np.array_equal(a, b) for a, b in zip(n, ref)) for n in norms), "different norm stats"
    arrays["norm_mu"], arrays["norm_sd"], arrays["norm_drms"] = (a.astype(np.float32) for a in ref)
    arrays["usen"] = usen
    print("input scaling per member (1 = standardised):", usen)
views = np.array(views, np.int64)
if (views >= 2).any():
    assert args.flat and not (views == 2).any(), "views 3/4/5 need --flat (numba_flat.py); norm=2 is not supported"
    for k, vs in vstats.items():
        assert all(np.array_equal(v, vs[0]) for v in vs), f"different {k} across members"
        arrays[k] = vs[0].astype(np.float32)
    if (views == 4).any() and "norm_mu" not in arrays:
        ref = norms[int(np.argmax(views == 4))]
        arrays["norm_mu"], arrays["norm_sd"], arrays["norm_drms"] = (a.astype(np.float32) for a in ref)
    arrays.pop("usen", None)
    views[views == 5] = 2  # change-only view runs in kernel slot 2
    print("input view slot per member (0 raw, 1 std, 2 change-only, 3 rank, 4 zca):", views)
arrays["view"] = views
arrays["n_models"] = np.array(len(models))
arrays["int8"] = np.array(int(args.int8))

out = RUNS / "submissions" / args.name
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
here = RUNS.parent / "connectome"
files = ["solution.py", "numba_gru.py", "models.npz"]
if args.multi:
    assert args.int8 and len(models) > 1, "--multi needs --int8 and an ensemble"
    kern = "flat" if args.flat else "multi"
    shutil.copy(here / f"solution_{kern}.py", out / "solution.py")
    shutil.copy(here / f"numba_{kern}.py", out / f"numba_{kern}.py")
    files.append(f"numba_{kern}.py")
else:
    shutil.copy(here / "solution_numba.py", out / "solution.py")
shutil.copy(here / "numba_gru.py", out / "numba_gru.py")
np.savez(out / "models.npz", **arrays)
for mod_name in ("numba_gru", "numba_multi", "numba_flat"):
    sys.modules.pop(mod_name, None)

spec = importlib.util.spec_from_file_location("pkg_solution", out / "solution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
t = time.perf_counter()
pm = mod.PredictionModel()
print(f"init (incl. numba compile): {time.perf_counter() - t:.1f}s")

reader = SeqReader(VALID)
acc_t, acc_s = WPAccumulator(), WPAccumulator()
max_diff, t_pred, rows = 0.0, 0.0, 0
for g in range(args.check):
    xs, ys, ms = reader.read(g)
    xs = xs.astype(np.float16).astype(np.float32)  # same rounding as solution.py
    with torch.no_grad():
        xt = torch.from_numpy(xs)[None]
        p_t = sum(w * m(xt)[0][0].numpy() for m, w in zip(models, ws))
        p_t = p_t + mix * p_t[:, ::-1]
    p_s = np.zeros_like(p_t)
    t0 = time.perf_counter()
    for s in range(xs.shape[0]):
        o = pm.predict(DataPoint(g, s, s >= WARMUP, xs[s]))
        if o is not None:
            p_s[s] = o
    t_pred += time.perf_counter() - t0
    rows += xs.shape[0]
    max_diff = max(max_diff, float(np.abs(p_t[WARMUP:] - p_s[WARMUP:]).max()))
    acc_t.add(ys, p_t, ms)
    acc_s.add(ys, p_s, ms)
us = t_pred / rows * 1e6
print(f"max |torch - submission| = {max_diff:.2e}")
print(f"WP on {args.check} seqs: torch {acc_t.result()[0]:.5f}  submission {acc_s.result()[0]:.5f}")
print(f"latency (predict incl. DataPoint): {us:.1f} us/row -> {us * 39.4e6 / 1e6 / 60:.1f} min for 39.4M test rows")

zp = RUNS / "submissions" / f"{args.name}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(out / f, f)
print(f"wrote {zp}  ({zp.stat().st_size / 1e6:.2f} MB)")
