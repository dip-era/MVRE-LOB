"""Build a self-contained private Kaggle script job that trains models with our code on Kaggle GPUs.

The job embeds the project modules (and data/mask_model.pt), downloads the dataset from the organisers' link inside
Kaggle, runs a short smoke test, then trains one run per GPU in parallel and leaves runs/<name>/ in /kaggle/working.

Usage: python build_job.py [--user <kaggle user>] <job_name> "<train.py args for run 1>" ["<args for run 2>" ...]
Then:  kaggle kernels push -p kaggle_jobs/<job_name> -t <seconds>
"""
import base64
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
CODE = ["common.py", "model.py", "train.py", "mask_model.py", "cache.py"]

TEMPLATE = r'''
import base64, os, shutil, subprocess, sys, time
from pathlib import Path

FILES = __FILES__
RUNS = __RUNS__
URLS = ["https://files.wundernn.io/wnn_connectome_starterpack.tar.gz",
        "https://files-alternative.wundernn.io/wnn_connectome_starterpack.zip",
        "https://storage.yandexcloud.net/files-alternative.wundernn.io/wnn_connectome_starterpack.zip"]


def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


sh("nvidia-smi -L; df -h / /kaggle/working /kaggle/tmp /tmp 2>/dev/null; nproc; free -g")
cands = [p for p in ("/kaggle/tmp", "/tmp", "/root") if os.path.isdir(p) or p == "/kaggle/tmp"]
os.makedirs("/kaggle/tmp", exist_ok=True)
base = max(cands, key=lambda p: shutil.disk_usage(p).free)
free_gb = shutil.disk_usage(base).free / 1e9
print(f"work dir {base}, free {free_gb:.0f} GB", flush=True)
if free_gb < 40:
    sys.exit("not enough disk for the dataset")
W = Path(base) / "wunder"
for rel, b64 in FILES.items():
    p = W / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
os.makedirs("/kaggle/working/runs", exist_ok=True)
if not (W / "runs").exists():
    os.symlink("/kaggle/working/runs", W / "runs")
try:
    import pyarrow  # noqa: F401
except ImportError:
    sh("pip install -q pyarrow")

t = time.time()
ok = False
for url in URLS:
    # download to disk with resume (a reset mid-stream used to kill a piped tar), then unpack
    arc = Path(base) / url.rsplit("/", 1)[1]
    for attempt in range(8):
        if sh(f"curl -sSL -C - --retry 10 --retry-delay 5 --retry-all-errors -o {arc} {url}") == 0:
            break
        time.sleep(10)
    unpack = f"tar -xzf {arc} -C {W}" if arc.suffix == ".gz" else f"{sys.executable} -m zipfile -e {arc} {W}"
    if arc.exists() and sh(unpack) == 0 and (W / "wnn_connectome_starterpack/datasets/train.parquet").exists():
        arc.unlink()
        ok = True
        break
if not ok:
    sys.exit("dataset download failed")
print(f"dataset ready in {time.time() - t:.0f}s", flush=True)
sh(f"du -sh {W}/wnn_connectome_starterpack/datasets/*")

os.chdir(W / "connectome")
first = RUNS[0].replace("--name ", "--name smoke_")
if sh(f"{sys.executable} train.py {first} --n_train 64 --n_valid 16 --epochs 1 --workers 2") != 0:
    sys.exit("smoke test failed")
shutil.rmtree("/kaggle/working/runs/" + first.split("--name ")[1].split()[0], ignore_errors=True)

import torch  # noqa: E402
n_gpu = max(1, torch.cuda.device_count())
print(f"{n_gpu} GPU(s); {len(RUNS)} run(s)", flush=True)
pending = list(enumerate(RUNS))
active = {}
while pending or active:
    while pending and len(active) < n_gpu:
        i, args = pending.pop(0)
        gpu = [g for g in range(n_gpu) if g not in active.values()][0]
        name = args.split("--name ")[1].split()[0]
        out = open(f"/kaggle/working/{name}.out", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.Popen(f"{sys.executable} train.py {args}", shell=True, env=env, stdout=out, stderr=subprocess.STDOUT)
        active[p] = gpu
        print(f"started {name} on GPU {gpu}", flush=True)
    time.sleep(30)
    for p in [p for p in active if p.poll() is not None]:
        print(f"run on GPU {active.pop(p)} exited with {p.returncode}", flush=True)
sh("grep -h -E 'EPOCH|DONE' /kaggle/working/runs/*/log.txt")
'''


def main():
    argv = sys.argv[1:]
    user = os.environ.get("KAGGLE_USERNAME", "")
    if argv[0] == "--user":
        user, argv = argv[1], argv[2:]
    assert user, "pass --user <kaggle user> or set KAGGLE_USERNAME"
    name, runs = argv[0], argv[1:]
    files = {f"connectome/{f}": (SRC / f).read_bytes() for f in CODE}
    for d in ("mask_model.pt", "norm_stats.npz", "view_stats.npz"):  # small fitted stats used by train.py
        if (SRC.parent / "data" / d).exists():
            files[f"data/{d}"] = (SRC.parent / "data" / d).read_bytes()
    enc = {k: base64.b64encode(v).decode() for k, v in files.items()}
    job = TEMPLATE.replace("__FILES__", repr(enc)).replace("__RUNS__", repr(runs))
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / "job.py").write_text(job)
    meta = {"id": f"{user}/wnn-{name}", "title": f"wnn-{name}", "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'job.py'} ({len(job) / 1e3:.0f} kB) for runs: {[r.split('--name ')[1].split()[0] for r in runs]}")


if __name__ == "__main__":
    main()
