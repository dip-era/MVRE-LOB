# Approach

## 1. Inputs

Each step gives a 112-dimensional row `x_t`: 22 price-like, 22 volume-like and 8 trade columns for each of two
instruments, plus 8 auxiliary columns. Every member gets:

- the levels `x_t`,
- the one-step differences `Δx_t = x_t − x_{t−1}` (zero at the first step), and
- 6 binary *change flags*, one per block (i0 price / volume / trade, i1 price / volume / trade), set when any
  column of that block changed.

Inputs are rounded to float16 both in training (the cache stores float16) and at inference (`solution_flat.py`),
so the two see identical numbers.

## 2. Input views

The same row is shown to different members through different transforms (`model.GRUModel`, `norm` argument):

| norm | view | levels | differences |
|---|---|---|---|
| 0 | raw | `x` | `Δx` |
| 1 | standardised | `(x − μ) / σ` | `Δx / rms(Δx)` |
| 3 | rank-Gaussian | `Φ⁻¹(F(x))`, piecewise linear through 129 quantile knots per column | difference of transformed rows |
| 4 | ZCA-whitened | `W_x (x − μ)/σ`, `W = U (Λ + 0.05)^-½ Uᵀ` of the correlation matrix | `W_d Δx / rms(Δx)` |
| 5 | change-only | zero | `Δx / rms(Δx)` |

Raw, standardised and whitened views are linear transforms of each other. A network could in principle absorb
them into its first layer, but in practice weight decay, initialisation and Adam's per-coordinate steps make
the trained models differ. Those differences decorrelate the members' errors. The rank-Gaussian view is a
genuinely nonlinear transform, and the change-only view removes information (absolute levels) on purpose.

Views that did **not** help: a per-sequence running z-score, an `asinh` compression (a weak single and a negative
partner), and PCA whitening, which is just ZCA rotated and added nothing.

## 3. Members and training

- **Cells:** GRU or LSTM, 128 hidden units, 2–3 layers, linear head to the two targets.
- **Loss:** `1 − weighted Pearson` computed over all rows of the batch (8 sequences × 20k steps), with weights
  `|y|` times a scored-row weight. The first 99 steps are warm-up and excluded.
- **Scored-row weight:** `w = (1 − a) + a · p / mean(p)` with `a = 1` (`--maskw 1.0`), where `p` is
  `P(is_scored | x)` from a small causal logistic model (`mask_model.py`, AUC ≈ 0.78). On validation rows used
  for training, the true mask is used instead, at ×3 weight (`--va_w 3`).
- **Optimiser:** AdamW, lr 1.5e-3 (2-layer GRU) or 2e-3 (others), weight decay 0.1, 200-step warm-up then a
  9-epoch cosine schedule, gradient clipping 1.0, EMA of weights (0.999) used for evaluation and export.
- **Stopping:** the full-data runs keep the 9-epoch schedule but stop after the epoch that was best in an honest
  run. Train + validation members peak about one epoch later than train-only members.

Final members (`final_v9e`):

| member | view | cell | layers | stop epoch |
|---|---|---|---|---|
| raw GRU | raw | GRU | 2 | 6 |
| rank LSTM | rank-Gaussian | LSTM | 2 | 5 |
| ZCA LSTM | ZCA-whitened | LSTM | 2 | 6 |
| change-only GRU | change-only | GRU | 3 | 5 |

The ensemble divides each member's output by its standard deviation, averages them and multiplies by 0.25. Pearson
ignores scale, but the metric clips at ±2, so the output is kept well inside the clip.

## 4. Validation protocol

- **Leak-free screening:** members trained on the training set only and scored on all 1,873 validation sequences
  (`tools/group_eval.py`). It reports full, even-half and odd-half scores; a real gain should appear on both
  halves.
- **What not to do:** training on train + one half of validation and scoring the other half. The halves share a
  time period, and with the validation rows up-weighted, longer-trained members look like better partners on
  the other half (+0.0018) while being worse on the test (−0.0020 leak-free).
- **Transfer to public:** honest validation gains carried over to the public leaderboard at roughly 55–100%.
  Public ≈ validation − 0.028.

## 5. Inference kernel (`connectome/numba_flat.py`)

The scorer calls `predict()` once per row. Per-row work:

1. Build the change flags once, and each needed view row once (shared by all members on that view).
2. For each member and layer: `g_i = W_i · input` and `g_h = W_h · h`, with int8 weights and per-output-column
   float scales.
   - The input product only touches non-zero inputs. Most `Δx` columns are zero, and in the change-only view so
     are all levels.
   - Rows are processed four at a time to cut accumulator traffic.
3. Gates use a rational `tanh` (Eigen's approximation, |error| < 3e-7) with `σ(x) = ½ + ½ tanh(x/2)`. numba has no
   vectorised libm here, so this alone sped up the gates 2.3×. LSTM members keep their cell state next to `h`.
4. The rank view caches each column's last value and transformed value, so only columns that changed are
   re-interpolated (a branchless 7-step binary search over 129 knots).

All weights live in three flat arrays (int8, float32, int64), so each row is a single numba call with 6
arguments. This saved ~3.5 µs per row of dispatch overhead. The kernel matches PyTorch within int8 rounding
(`tests/test_kernel.py`).

Cost per member on the scorer: ~7 min (2-layer GRU) to ~10 min (3-layer GRU / 2-layer LSTM) of the 60-minute
budget. The scorer's runtime varied by up to ~13% for the same model cost, so the final zips leave ≥ 10 minutes
of margin.
