"""Train a stateful GRU on full 20k-step sequences, streaming from parquet.

Example: python train.py --name gru128 --epochs 12
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from cache import CACHE, open_cache
from common import DATA, RUNS, T, TRAIN, VALID, WARMUP, SeqReader, WPAccumulator
from mask_model import mask_features
from model import GRUModel, aux_horizons, ms_features


class SeqDataset(Dataset):
    def __init__(self, path, idx):
        self.reader = SeqReader(path)
        self.idx = np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y, m = self.reader.read(self.idx[i])
        x = x.astype(np.float16)  # same rounding as the cache and solution.py
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


class CacheDataset(Dataset):
    """Same as SeqDataset but reads the float16 memmap cache built by cache.py (x stays float16 until the GPU)."""

    def __init__(self, split, idx, pscored="", teacher=""):
        self.split = split
        self.idx = np.asarray(idx)
        self.pscored = pscored
        self.teacher = teacher
        self.mm = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = open_cache(self.split)
            if self.pscored:
                self.ps = np.memmap(CACHE / f"{self.split}_pscored_{self.pscored}.bin", np.float16, "r",
                                    shape=self.mm[2].shape)
            if self.teacher:
                self.tp = np.memmap(CACHE / f"{self.split}_teacher_{self.teacher}.bin", np.float16, "r",
                                    shape=self.mm[1].shape)
        X, Y, M = self.mm
        g = self.idx[i]
        out = (torch.from_numpy(np.array(X[g])), torch.from_numpy(np.array(Y[g])), torch.from_numpy(np.array(M[g])))
        if self.pscored:
            out += (torch.from_numpy(np.array(self.ps[g])),)
        if self.teacher:
            out += (torch.from_numpy(np.array(self.tp[g])),)
        return out


class Tagged(Dataset):
    """Appends a constant tag to each item (1 = validation sequence, whose true is_scored mask is used as weights)."""

    def __init__(self, ds, tag):
        self.ds, self.tag = ds, tag

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return (*self.ds[i], torch.tensor(self.tag))


def wcorr_loss(y, p, w_extra=None):
    """1 - mean over targets of the batch-global weighted Pearson (weights |clip(y)|). y, p: (N, 2)."""
    y = y.clamp(-2, 2)
    w = y.abs()
    if w_extra is not None:
        w = w * w_extra[:, None]
    W = w.sum(0)
    my = (w * y).sum(0) / W
    mp = (w * p).sum(0) / W
    yc, pc = y - my, p - mp
    cov = (w * yc * pc).sum(0)
    r = cov / torch.sqrt((w * yc * yc).sum(0) * (w * pc * pc).sum(0) + 1e-12)
    return 1 - r.mean(), r


def wcorr_loss_w(t, p, w):
    """1 - mean over targets of the weighted Pearson between t and p with explicit weights w. t, p, w: (N, 2)."""
    W = w.sum(0)
    mt = (w * t).sum(0) / W
    mp = (w * p).sum(0) / W
    tc, pc = t - mt, p - mp
    r = (w * tc * pc).sum(0) / torch.sqrt((w * tc * tc).sum(0) * (w * pc * pc).sum(0) + 1e-12)
    return 1 - r.mean()


class RunningCorrLoss:
    """Weighted Pearson over a decaying pool of recent batches, closer to the metric's pooling over the whole test set.

    Keeps decayed sums (W, Wy, Wp, Wyy, Wpp, Wyp) from earlier batches (detached) and adds the current batch's sums
    (with gradient), so each step's correlation uses means/variances from ~1/(1-decay) batches."""

    def __init__(self, decay):
        self.decay = decay
        self.prev = None

    def __call__(self, y, p, w_extra=None):
        y = y.clamp(-2, 2).double()
        p = p.double()
        w = y.abs()
        if w_extra is not None:
            w = w * w_extra[:, None].double()
        s = torch.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0), (w * y * y).sum(0), (w * p * p).sum(0),
                         (w * y * p).sum(0)])
        if self.prev is not None:
            s = s + self.decay * self.prev
        self.prev = s.detach()
        W, Wy, Wp, Wyy, Wpp, Wyp = s
        my, mp = Wy / W, Wp / W
        r = (Wyp / W - my * mp) / torch.sqrt((Wyy / W - my * my) * (Wpp / W - mp * mp) + 1e-12)
        return 1 - r.mean().float(), r.float()


EVAL_SCALE = 0.1  # predictions are scaled before the metric's +-2 clip (Pearson ignores scale; the clip does not)
LAST_CLIPPED = [float("nan")]  # WP(mask) of the unscaled predictions from the last evaluate() call, for logging


@torch.no_grad()
def evaluate(model, loader, dev, amp, keep_preds=False):
    """Returns no-clip scores (predictions x EVAL_SCALE). The training loss is scale-free, so raw outputs can drift
    past +-2 and lose up to ~0.005 to the metric's clip; solutions apply the same scale, so this is the real score."""
    model.eval()
    acc_mask, acc_all, acc_raw = WPAccumulator(), WPAccumulator(), WPAccumulator()
    preds = []
    for x, y, m in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            p, _ = model(x.to(dev, non_blocking=True).float())
        p = p.float().cpu().numpy()
        y, m = y.numpy(), m.numpy()
        req = np.zeros_like(m)
        req[:, WARMUP:] = True
        for b in range(len(p)):
            acc_mask.add(y[b], p[b] * EVAL_SCALE, m[b])
            acc_all.add(y[b], p[b] * EVAL_SCALE, req[b])
            acc_raw.add(y[b], p[b], m[b])
        if keep_preds:
            preds.append(p.astype(np.float16))
    model.train()
    LAST_CLIPPED[0] = acc_raw.result()[0]
    return acc_mask.result(), acc_all.result(), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--norm", type=int, default=0, help="1: standardise inputs with data/norm_stats.npz; 2: per-sequence running z-score of the levels; 3: rank-Gaussian view; 4: whitened view (data/view_stats.npz)")
    ap.add_argument("--eval_raw", type=int, default=1, help="also score the non-EMA weights each epoch (0 = skip, faster)")
    ap.add_argument("--cell", default="gru", choices=["gru", "lstm"], help="recurrent cell")
    ap.add_argument("--view_file", default="view_stats.npz",
                    help="data/<file> with the norm 3/4 tables (default: tools/view_stats.py output)")
    ap.add_argument("--crop", type=int, default=0, help="train on a random window of T - crop rows per batch (augmentation)")
    ap.add_argument("--wdrop", type=float, default=0.0, help="DropConnect rate on the GRU recurrent weights (training only)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--mse", type=float, default=0.0, help="weight of |y|-weighted MSE added to the corr loss")
    ap.add_argument("--i1w", type=float, default=1.0, help="loss weight multiplier for rows right after an i1 price change")
    ap.add_argument("--n_train", type=int, default=0, help="use only this many train sequences (0 = all)")
    ap.add_argument("--n_valid", type=int, default=0, help="evaluate on this many valid sequences (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights per step (0 = off); eval/save use EMA")
    ap.add_argument("--maskw", type=float, default=0.0,
                    help="blend of loss row weights toward P(is_scored|x): w = (1-a) + a * p/mean(p)")
    ap.add_argument("--aux", default="", help="aux targets: comma-separated time shifts of y, e.g. -1000,-300,100,300")
    ap.add_argument("--aux_w", type=float, default=0.3, help="total weight of the aux losses")
    ap.add_argument("--cache", type=int, default=1, help="read the float16 memmap cache instead of parquet")
    ap.add_argument("--teacher", default="",
                    help="distillation: data/cache/train_teacher_<name>.bin from distill.py")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the teacher term in the loss")
    ap.add_argument("--mirror", type=float, default=0.0,
                    help="probability of mirroring a training sequence with data/mirror_map.npz (t0 <-> t1 swapped)")
    ap.add_argument("--chunk", type=int, default=0,
                    help="truncated BPTT: update every this many steps, carrying the GRU state (detached) across chunks")
    ap.add_argument("--with_valid", type=int, default=0,
                    help="also train on all validation sequences, weighting their rows by the true is_scored mask "
                         "(needs --maskw; per-epoch validation is skipped since it would be in-sample)")
    ap.add_argument("--valid_half", type=int, default=-1,
                    help="with --with_valid: train on the validation seqs with index %% 2 == h only and evaluate each "
                         "epoch on the other half (honest score for the train+valid recipe)")
    ap.add_argument("--va_w", type=float, default=1.0,
                    help="with --with_valid: extra loss weight on validation rows (multiplies their true mask)")
    ap.add_argument("--stop_ep", type=int, default=-1,
                    help="with --with_valid: stop after saving this epoch (the cosine schedule still spans --epochs)")
    ap.add_argument("--corr_decay", type=float, default=0.0,
                    help="pool the correlation loss over recent batches with this decay (0 = this batch only)")
    ap.add_argument("--ms", type=int, default=0, help="add multi-timescale EMA inputs (model.ms_features)")
    ap.add_argument("--maskfile", default="",
                    help="use per-row P(is_scored) from data/cache/train_pscored_<name>.bin (mask_gru.py) for --maskw")
    ap.add_argument("--maskg", type=float, default=1.0,
                    help="sharpen the mask weights: use P(is_scored|x)**maskg before normalising")
    ap.add_argument("--init", default="", help="checkpoint path to initialise weights from")
    ap.add_argument("--resume", type=int, default=0,
                    help="save full training state after every epoch (state.pt) and continue from it if present")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "log.txt", "a", buffering=1)

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    dev = "cuda"
    n_tr = SeqReader(TRAIN).num_seqs
    n_va = SeqReader(VALID).num_seqs
    rng = np.random.default_rng(args.seed)
    tr_idx = rng.permutation(n_tr)[: args.n_train or n_tr]
    va_idx = np.arange(n_va)[: args.n_valid or n_va] if not args.n_valid else np.sort(
        np.random.default_rng(123).choice(n_va, args.n_valid, replace=False))
    tr_ds = CacheDataset("train", tr_idx, args.maskfile, args.teacher) if args.cache else SeqDataset(TRAIN, tr_idx)
    assert not (args.teacher and (args.maskfile or args.with_valid or args.chunk or args.mirror)), "teacher: plain runs only"
    if args.with_valid:
        assert args.maskw and not args.maskfile
        va_tr = np.arange(n_va) if args.valid_half < 0 else np.arange(args.valid_half, n_va, 2)
        if args.n_valid:  # smoke tests: also limit the validation seqs used for training
            va_tr = va_tr[: args.n_valid]
        va_tr_ds = CacheDataset("valid", va_tr) if args.cache else SeqDataset(VALID, va_tr)
        tr_ds = ConcatDataset([Tagged(tr_ds, 0), Tagged(va_tr_ds, 1)])
        if args.valid_half >= 0:
            va_idx = np.arange(1 - args.valid_half, n_va, 2)[: args.n_valid or None]
    va_ds = CacheDataset("valid", va_idx) if args.cache else SeqDataset(VALID, va_idx)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                           num_workers=args.workers, persistent_workers=True, pin_memory=True, prefetch_factor=4)
    va_loader = DataLoader(va_ds, batch_size=16, shuffle=False,
                           num_workers=args.workers, persistent_workers=True, pin_memory=True)

    horizons = aux_horizons(vars(args))
    model = GRUModel(args.hidden, args.layers, args.dropout, n_aux=len(horizons), ms=bool(args.ms),
                     wdrop=args.wdrop, norm=args.norm, cell=args.cell).to(dev)
    if args.norm:
        ns = np.load(DATA / "norm_stats.npz")
        model.in_mu.copy_(torch.from_numpy(ns["mu"]))
        model.in_sd.copy_(torch.from_numpy(ns["sd"]))
        model.d_rms.copy_(torch.from_numpy(ns["drms"]))
    if args.norm in (3, 4):
        vs = np.load(DATA / args.view_file)
        if args.norm == 3:
            model.rk_knots.copy_(torch.from_numpy(vs["rk_knots"]))
            model.rk_vals.copy_(torch.from_numpy(vs["rk_vals"]))
        else:
            model.zca_x.copy_(torch.from_numpy(vs["zca_x"]))
            model.zca_d.copy_(torch.from_numpy(vs["zca_d"]))
    if args.ms:
        # fixed normalisation from the first 64 training sequences
        X = open_cache("train")[0]
        with torch.no_grad():
            f = torch.cat([ms_features(torch.from_numpy(np.array(X[i:i + 8])).to(dev).float())[:, WARMUP:]
                           for i in range(0, 64, 8)]).reshape(-1, model.ms_mu.numel())
        model.ms_mu.copy_(f.mean(0))
        model.ms_sd.copy_(f.std(0) + 1e-6)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["model"])
    ema = copy.deepcopy(model).eval() if args.ema else None
    mm = {k: v.to(dev) for k, v in torch.load(DATA / "mask_model.pt").items()} if args.maskw and not args.maskfile else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    corr_loss = RunningCorrLoss(args.corr_decay) if args.corr_decay else wcorr_loss
    steps_total = max(1, args.epochs * len(tr_loader) * (math.ceil(T / args.chunk) if args.chunk else 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps_total) / steps_total)))
    scaler = torch.amp.GradScaler(enabled=bool(args.amp))
    say(f"train seqs {len(tr_ds)}  valid seqs {len(va_idx)}  params {sum(p.numel() for p in model.parameters()):,}")

    if args.mirror:
        mz = np.load(DATA / "mirror_map.npz")
        mperm = torch.as_tensor(mz["perm"], device=dev)
        msign = torch.as_tensor(mz["sign"], device=dev, dtype=torch.float32)
    best = -1
    step = 0
    start_ep = 0
    if args.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        if ema is not None:
            ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best, start_ep = st["step"], st["best"], st["ep"] + 1
        say(f"RESUME from epoch {st['ep']} (step {step}, best {best:.5f})")

    def save_state(ep):
        if args.resume:
            st = {"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                  "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best, "ep": ep}
            torch.save(st, out / "state.tmp")
            os.replace(out / "state.tmp", out / "state.pt")
    run_loss, run_r = 0.0, np.zeros(2)

    def update(loss, r):
        nonlocal step, run_loss, run_r
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if ema is not None:
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
        run_loss += loss.item()
        run_r += r.detach().cpu().numpy()
        if step % 100 == 0:
            say(f"  ep {ep} step {step}/{steps_total} loss {run_loss / 100:.4f} r {run_r / 100} "
                f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
            run_loss, run_r = 0.0, np.zeros(2)

    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        run_loss, run_r = 0.0, np.zeros(2)
        for batch in tr_loader:
            x, y, m = batch[:3]
            x = x.to(dev, non_blocking=True).float()
            y = y.to(dev, non_blocking=True)
            if args.crop:
                # random window of T - crop rows (one offset per batch): the GRU starts from zero at a new place
                s0 = int(torch.randint(0, args.crop + 1, (1,)))
                L = x.shape[1] - args.crop
                x, y, m = x[:, s0:s0 + L], y[:, s0:s0 + L], m[:, s0:s0 + L]
            x_mask = x  # scored-row weights come from the unmirrored sequence
            if args.mirror:
                flip = torch.rand(len(x), device=dev) < args.mirror
                if flip.any():
                    x, y = x.clone(), y.clone()
                    x[flip] = x[flip][..., mperm] * msign
                    y[flip] = y[flip].flip(-1)
            if args.chunk:
                assert not (horizons or args.ms or args.maskfile or args.with_valid or args.mse or args.i1w != 1.0)
                pm_full = None
                if args.maskw:
                    with torch.no_grad():
                        z = (mask_features(x) - mm["mu"]) / mm["sd"]
                        pm_full = torch.sigmoid(z @ mm["w"] + mm["b"])
                h = None
                for c0 in range(0, x.shape[1], args.chunk):
                    c1 = min(c0 + args.chunk, x.shape[1])
                    p, h = model(x[:, c0:c1], h, x0=x[:, c0 - 1] if c0 else None)
                    h = h.detach()
                    lo = max(WARMUP, c0)
                    pc = p.float()[:, lo - c0:].reshape(-1, 2)
                    yc = y[:, lo:c1].reshape(-1, 2)
                    w_extra = None
                    if pm_full is not None:
                        pm = pm_full[:, lo:c1].reshape(-1)
                        w_extra = (1 - args.maskw) + args.maskw * pm / pm.mean()
                    loss, r = corr_loss(yc, pc, w_extra)
                    update(loss, r)
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
                if horizons:
                    p, _, pa = model(x, with_aux=True)
                else:
                    p, _ = model(x)
            aux_loss = 0.0
            for i, hz in enumerate(horizons):
                # target at row t is y[t + hz]; keep rows where that exists and t is past warmup
                lo, hi = max(WARMUP, -hz), y.shape[1] - max(hz, 0)
                ya = y[:, lo + hz:hi + hz].reshape(-1, 2)
                aux_loss = aux_loss + wcorr_loss(ya, pa.float()[:, lo:hi, 2 * i:2 * i + 2].reshape(-1, 2))[0]
            p = p.float()[:, WARMUP:].reshape(-1, 2)
            yy = y[:, WARMUP:].reshape(-1, 2)
            w_extra = None
            if args.maskw:
                with torch.no_grad():
                    if args.maskfile:
                        pm = batch[3].to(dev, non_blocking=True)[:, WARMUP:].float().reshape(-1)
                    else:
                        z = (mask_features(x_mask)[:, WARMUP:] - mm["mu"]) / mm["sd"]
                        pm = torch.sigmoid(z @ mm["w"] + mm["b"]).reshape(-1)
                    if args.with_valid:
                        # validation sequences: weight by the true mask instead of the predicted P(is_scored)
                        is_va = batch[-1].to(dev).bool()[:, None]
                        pm = torch.where(is_va, args.va_w * m.to(dev)[:, WARMUP:].float(),
                                         pm.view(len(x), -1)).reshape(-1)
                    if args.maskg != 1.0:
                        pm = pm ** args.maskg
                    w_extra = (1 - args.maskw) + args.maskw * pm / pm.mean()
            if args.i1w != 1.0:
                ch = (x[:, 1:, 52:74] != x[:, :-1, 52:74]).any(-1)
                ch = torch.cat([torch.zeros_like(ch[:, :1]), ch], 1)[:, WARMUP:].reshape(-1)
                w_extra = 1 + (args.i1w - 1) * ch.float()
            loss, r = corr_loss(yy, p, w_extra)
            if args.teacher:
                tp = batch[3].to(dev, non_blocking=True)[:, WARMUP:].float().reshape(-1, 2)
                wt = yy.clamp(-2, 2).abs()
                if w_extra is not None:
                    wt = wt * w_extra[:, None]
                loss = (1 - args.alpha) * loss + args.alpha * wcorr_loss_w(tp, p, wt)
            if horizons:
                loss = loss + args.aux_w / len(horizons) * aux_loss
            if args.mse:
                yc = yy.clamp(-2, 2)
                loss = loss + args.mse * (yc.abs() * (p - yc) ** 2).mean()
            update(loss, r)
        em = ema if ema is not None else model
        if args.with_valid and args.valid_half < 0:
            # validation is part of training: no honest score; keep every epoch, best.pt = latest
            say(f"EPOCH {ep}  (trained on validation, no eval)  {time.time() - t0:.0f}s")
            for f in ("last.pt", "best.pt", f"ep{ep}.pt"):
                torch.save({"model": em.state_dict(), "args": vars(args), "epoch": ep, "wp": float("nan")}, out / f)
            save_state(ep)
            if ep == args.stop_ep:
                say(f"STOP after epoch {ep} (--stop_ep; lr schedule still set by --epochs {args.epochs})")
                break
            continue
        (vm, vm0, vm1), (va, _, _), _ = evaluate(em, va_loader, dev, bool(args.amp))
        clipped = LAST_CLIPPED[0]
        if ema is not None:
            ema.eval()
        if ema is not None and args.eval_raw:
            (rm, _, _), (ra, _, _), _ = evaluate(model, va_loader, dev, bool(args.amp))
            say(f"         raw (non-EMA) weights: WP(mask) {rm:.5f}  WP(all rows) {ra:.5f}")
        say(f"EPOCH {ep}  valid WP(mask, no-clip) {vm:.5f} (t0 {vm0:.4f} t1 {vm1:.4f})  WP(all rows) {va:.5f}  "
            f"[as-is clipped {clipped:.5f}]  {time.time() - t0:.0f}s")
        torch.save({"model": em.state_dict(), "args": vars(args), "epoch": ep, "wp": vm}, out / "last.pt")
        if args.with_valid and args.valid_half >= 0:  # keep every epoch for checkpoint-averaging tests
            torch.save({"model": em.state_dict(), "args": vars(args), "epoch": ep, "wp": vm}, out / f"ep{ep}.pt")
        if vm > best:
            best = vm
            torch.save({"model": em.state_dict(), "args": vars(args), "epoch": ep, "wp": vm}, out / "best.pt")
        save_state(ep)
        if ep == args.stop_ep:
            say(f"STOP after epoch {ep} (--stop_ep; lr schedule still set by --epochs {args.epochs})")
            break
    say(f"DONE best valid WP(mask) {best:.5f}")


if __name__ == "__main__":
    main()
