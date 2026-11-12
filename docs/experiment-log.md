# Experiment log (condensed)

Scores are weighted Pearson. "Valid" is the full validation set, "held-out" is a validation half or the
leak-free protocol (see `approach.md`), and "public" is the leaderboard. Public ≈ valid − 0.028. Differences
below ~0.0003 on public are noise.

## What worked

| Change | Effect |
|---|---|
| GRU-128 ×2 on `[x, Δx, change flags]`, whole-sequence training, Pearson loss | 0.6660 valid after 1 epoch (baseline 0.617) |
| EMA of weights (0.999) | +0.005 valid |
| Scored-row loss weighting (`mask_model.py`) | +0.003–0.004 valid |
| Numba GRU step instead of ONNX Runtime | ~60 → ~20 µs per model per row; made 2–3 member ensembles possible |
| 3-layer GRU as second member | public 0.65160 → 0.65321 |
| Fine-tuning on the true validation mask (later replaced by train + valid) | public +0.0037 |
| Scoring without the ±2 clip during training and scaling outputs ×0.25 | fixed up to −0.005 lost to clipping |
| Higher learning rates once clipping was fixed (1.5e-3 / 2e-3) | best singles +0.002 |
| int8 weights with per-column scales | 1.35× faster, accuracy unchanged |
| Weight decay 0.1 + 9-epoch cosine | singles +0.001–0.0016; public 0.65842 |
| Training on train + valid, valid rows ×3 with the true mask | public 0.65923 |
| **Input views:** standardised member next to raw ones | public +0.0014 (0.66065), the largest single step |
| Flat-array kernel with a rational tanh | 1.44× faster, same outputs; room for 4–5 members |
| Rank-Gaussian and ZCA-whitened views | public 0.66160 |
| Leak-free selection (train-only members, full valid) | fixed misleading epoch choices (see below) |
| **LSTM cells on the rank and ZCA views** | public 0.66207 (+0.0005; leak-free +0.0017) |
| Change-only view (levels removed) as the 4th member | public **0.66216** (final) |

## What did not work

| Idea | Result |
|---|---|
| Lower learning rate / longer schedules (12 epochs) | ≤ +0.0003, noise |
| Dropout 0.1, DropConnect on recurrent weights, crop augmentation | −0.0006 to −0.0007 or no gain |
| Weight decay 0.05 / 0.2 / 0.3 | 0.2 helped singles but made worse partners; 0.05 and 0.3 worse |
| Wider GRU (192) | +0.0007 single, loses to two 128s at equal inference cost |
| 4-layer GRU | overfits, worse partner |
| TCN, GRU + local attention, MLP front-end, two-clock (event-time) GRU | +0.0001–0.0003 in ensembles at best |
| Auxiliary shifted targets, multi-timescale EMA inputs, pooled-batch correlation loss | −0.0014 to −0.0037 |
| Truncated BPTT in chunks, bigger batches (16/24) | −0.001 to −0.003 |
| Buy/sell mirror augmentation | no clean symmetry in the anonymised features, −0.0055 |
| Distillation from an 8-model teacher | +0.0006 single, no ensemble gain |
| LightGBM stack on GRU hidden states | +0.0013 held-out, below simpler options |
| Checkpoint averaging (SWA), cross-target mixing, fitted ensemble weights | no gain (fitted weights ≈ equal) |
| Per-sequence running z-score view, asinh view, PCA-whitened view | +0.0001, −0.0004 and +0.0000 as partners |
| LSTM on the raw view as an extra member | leak-free +0.0007, public −0.00015 versus a 2-layer ZCA GRU |
| LSTM on the change-only or standardised view | −0.0007 / +0.0000 as partners |
| Dropping the raw-input GRU for a second ZCA-LSTM seed | −0.0006 both leak-free and public |

## Validation lessons

1. **Half-split leak.** Training on train + even validation half (×3) and scoring the odd half favoured longer
   training: past-peak epochs looked like better partners (+0.0018). Public reversed it (−0.0009). Train-only
   members on the full validation set showed the truth: −0.0020. Use peak epochs.
2. **Selection noise.** Picking the best of ~40 combinations on one held-out split inflates the winner. Prefer
   gains that appear on both validation halves.
3. **Runtime variance.** The scorer's time for the same zip varied by up to 13% against laptop-calibrated
   estimates. Keep ≥ 10 minutes of margin.
4. **Composition saturates.** Four different 5-member sets scored within 0.00014 on public. Past that point,
   only new member types (LSTM cells) moved the score.
