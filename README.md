# MVRE-LOB

**Multi-View Recurrent Ensemble for streaming Limit Order Book forecasting.**

A solution to the Wunder Fund *Alfa Connectome* challenge (2026): predict two forward-looking targets for every step of
anonymised two-instrument order-book streams, one row at a time, on a **single CPU core**.

The final model reads the same order book through four different *input views* with four stateful recurrent
networks (GRU and LSTM), and runs them as one fused int8 numba kernel at ~70 µs per row.

| | |
|---|---|
| Final public score | **0.66216** weighted Pearson (organisers' GRU baseline: 0.617 on validation) |
| Inference | 39.4M rows in 46 min on 1 vCPU (limit 60 min), 4.6 MB submission |
| Members | raw-input GRU · rank-Gaussian LSTM · ZCA-whitened LSTM · change-only GRU |

---

## The problem

- **Data:** 10,607 training and 1,873 validation sequences of 20,000 steps × 112 anonymised features (prices,
  volumes and trades of two instruments, plus a few slow regime signals). Each row has two targets `t0`, `t1`.
- **Metric:** global weighted Pearson correlation, weights `|y|`, predictions clipped to ±2, averaged over the two
  targets, computed only on hidden *scored* rows.
- **Constraints:** the scorer calls `predict()` once per row, in order, with no look-ahead, on 1 vCPU, for 39.4M
  rows within 60 minutes (≈ 90 µs per row including harness overhead), in a ≤ 20 MB zip. PyTorch eager costs
  300 µs per step, so a custom kernel is essential.

## Approach

```
           ┌─ raw x, Δx, change flags ───────────► GRU 2×128 ──┐
 order     ├─ rank-Gaussian(x), Δ ───────────────► LSTM 2×128 ─┤   std-weighted
 book  ────┤                                                   ├─► average ──► (t0, t1)
 row t     ├─ ZCA-whitened(x), whitened Δx ──────► LSTM 2×128 ─┤   × 0.25
           └─ Δx only (levels dropped), flags ───► GRU 3×128 ──┘
```

1. **Stateful recurrent members** on `[x_t, x_t − x_{t−1}, 6 per-block change flags]`, trained on whole
   20k-step sequences with loss `1 − weighted Pearson` (batch-global, `|y|` weights). EMA weights (0.999),
   AdamW with weight decay 0.1, a 9-epoch cosine schedule stopped near the peak.
2. **Input views as the main source of diversity.** The members see the same data through different transforms:
   raw values, a per-column rank-Gaussian map, a regularised ZCA whitening, and a change-only view. Changing the
   view did more for the ensemble than any architecture or training tweak (+0.0014 public for the first
   normalised member alone).
3. **LSTM cells on the normalised views.** LSTMs on the rank and ZCA views beat the GRUs they replaced, as
   singles and especially as partners (+0.0005 public, +0.0017 on leak-free validation).
4. **Scored-row weighting.** A tiny causal logistic model estimates `P(is_scored | x)`. Training rows are
   weighted by it, and validation rows by the true mask at ×3 weight.
5. **Train on everything.** The final members are trained on train + validation. Stop epochs come from
   train-only runs: the train-only peak epoch plus one.
6. **A fused int8 kernel** (`connectome/numba_flat.py`): all members in one numba call per row, per-column int8
   weights, a rational tanh approximation instead of libm, sparse input products (most Δx are zero), cached rank
   lookups, and LSTM support. It is 1.44× faster than the previous numba kernel and roughly 3–4× faster per model
   than ONNX Runtime.

Details: [docs/approach.md](docs/approach.md) · experiment history: [docs/experiment-log.md](docs/experiment-log.md) ·
data analysis: [docs/eda-findings.md](docs/eda-findings.md)

## Validation lesson: watch for leaks in your held-out set

Halfway through, model selection used "train + even half of validation, score on the odd half". That split leaks:
both halves cover the same time period, and the even half is up-weighted ×3 in training. Under that split,
later epochs looked like *better* ensemble partners (+0.0018). On the public leaderboard they were worse.

Retraining the members **without any validation data** and scoring on the full validation set reversed the result:
later epochs cost −0.0020. From then on, every candidate was screened this way (`tools/group_eval.py`). It
matched the public direction of the later structural changes: LSTM views helped, and dropping the raw-input GRU
hurt (−0.0006 on both validation and public).

## Public leaderboard progression

| Step | Public |
|---|---|
| Single GRU-128 | 0.64743 |
| 2-model GRU ensemble, scored-row weighting | 0.65160 |
| Fine-tuning on the validation scoring mask | 0.65691 |
| Weight decay 0.1 + 9-epoch schedule | 0.65842 |
| Train on train + validation (validation ×3) | 0.65923 |
| + standardised-input member (first input view) | 0.66065 |
| Faster kernel, 4 members | 0.66124 |
| + rank-Gaussian and ZCA views | 0.66160 |
| LSTMs on the rank and ZCA views | 0.66207 |
| **+ change-only view (final)** | **0.66216** |

## Repository layout

```
connectome/            core code
  common.py            paths, sequence reader, metric accumulator
  cache.py             parquet -> float16 memmap cache
  make_sample.py       random sequence sample for analysis / the mask model
  mask_model.py        causal P(is_scored) model for loss weighting
  model.py             GRU/LSTM model with input views (norm 0 raw, 1 std, 3 rank, 4 ZCA, 5 change-only)
  train.py             training (whole-sequence, weighted Pearson loss, EMA, train+valid mode)
  numba_gru.py         single-model numba step and int8 quantisation
  numba_flat.py        fused int8 multi-model kernel (GRU + LSTM, all views)
  solution_flat.py     the submission's PredictionModel
  package.py           builds a submission zip and checks it against PyTorch
  time_zip.py          side-by-side latency of submission zips
  kaggle_jobs/         builder for self-contained Kaggle GPU training jobs
tools/
  norm_stats.py        per-column statistics for the standardised view
  view_stats.py        rank-Gaussian knots and ZCA matrices
  group_eval.py        leak-free ensemble scoring of train-only members on the full validation set
  pair_eval.py         ensemble scoring on a validation half (for the train + half-validation protocol)
tests/
  test_kernel.py       int8 kernel vs PyTorch on synthetic data, all views, GRU and LSTM
```

## Reproducing

Requirements: Python 3.11, a CUDA GPU (6 GB is enough), ~60 GB of disk for the float16 cache.

```bash
pip install -r requirements.txt
```

1. **Data.** Download the organisers' starter pack and unpack it as `wnn_connectome_starterpack/` in the repository
   root (it provides `datasets/{train,valid}.parquet` and `utils.py`). The data is not redistributed here.
2. **Cache and statistics.**
   ```bash
   cd connectome
   python cache.py                                   # float16 memmaps in data/cache
   python make_sample.py ../wnn_connectome_starterpack/datasets/valid.parquet ../data/valid_s100.npz 100 1
   python mask_model.py ../data/valid_s100.npz       # data/mask_model.pt
   python ../tools/norm_stats.py                     # data/norm_stats.npz
   python ../tools/view_stats.py                     # data/view_stats.npz
   ```
3. **Train the four members** on train + validation:
   ```bash
   C="--epochs 9 --ema 0.999 --maskw 1.0 --wd 0.1 --with_valid 1 --va_w 3"
   python train.py --name final_raw2L  --layers 2 --lr 1.5e-3 --seed 1  --stop_ep 6 $C
   python train.py --name final_lstmR  --layers 2 --lr 2e-3 --cell lstm --norm 3 --seed 23 --stop_ep 5 $C
   python train.py --name final_lstmZ  --layers 2 --lr 2e-3 --cell lstm --norm 4 --seed 24 --stop_ep 6 $C
   python train.py --name final_chg3L  --layers 3 --lr 2e-3 --norm 5 --seed 22 --stop_ep 5 $C
   ```
4. **Package and verify:**
   ```bash
   python package.py mvre_lob final_raw2L:ep6.pt final_lstmR:ep5.pt final_lstmZ:ep6.pt final_chg3L:ep5.pt \
       --int8 --stdw --scale 0.25 --multi --flat --check 3
   ```
   This writes `runs/submissions/mvre_lob.zip` and prints the difference from PyTorch and the latency.
5. **Tests:** `python -m pytest tests/` (synthetic data, CPU only).

The packaged final submission is attached to the GitHub release.

## License

MIT, see [LICENSE](LICENSE). The competition data belongs to its organisers and is not included.
