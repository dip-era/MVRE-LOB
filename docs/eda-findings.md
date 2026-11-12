# Data findings

Exploratory analysis on random samples of 200 training and 100 validation sequences (20,000 steps each).

## Targets

- **Quantised.** Within a sequence, `t0` and `t1` are integer multiples of a per-sequence unit `u` (e.g. 0.125,
  0.241, 0.469), with 12–19 distinct values per sequence.
- **Piecewise constant, in instrument-0 event time.** A target changes on ~8% of steps, and only on steps where
  instrument 0's prices change (P(change | no i0 price change) = 0; P(change | i0 price change) ≈ 0.57). ACF 0.97 at
  lag 1, 0.46 at lag 100.
- **Heavy tails.** Mean ≈ −0.27, std ≈ 1.0. About 3% of rows have |y| > 2. The top 20% of rows by |y| carry ~50% of
  the metric weight.
- **Mirror pair.** corr(t0, t1) = −0.73, and common (t0, t1) pairs occur in swapped versions. Split
  `S = (t0 + t1)/u` (a spread-like term, concentrated at −1 and −2) from `D = (t0 − t1)/u` (direction):
  - An oracle that predicts only `S/2` scores WP ≈ 0.41; one that predicts only `±D/2` scores ≈ 0.95. The score is
    mostly about direction.
  - `S` is the easier half: it correlates ~0.3 with single price features.
- **The unit is inferable.** log(u) correlates −0.81 with the number of distinct price values in a sequence. The
  metric is pooled over all sequences, so predictions must carry the right per-sequence scale. Stateful models
  learn this implicitly.

## Features

- All columns look **quantile-transformed to a normal distribution** (most within ±2.4, some clipped at ±5.1993 =
  Φ⁻¹(1 − 1e−7)), each with its own transform. Arithmetic across columns (e.g. ask − bid) is not meaningful.
- **Price-like columns** (22 per instrument) all change on exactly the same steps (~21% of steps for instrument 0),
  in a few highly correlated clusters, with only 44–74 distinct values per sequence (discrete ticks).
- **Volume-like columns** change on ~90% of steps and are nearly continuous.
- **Trade columns:** the signed volumes are symmetric. The price-like trade columns have 240–370 distinct values
  per sequence.
- **Auxiliary columns:** two are very smooth (ACF₁ ≈ 0.99, slow clocks or regimes), and one takes ~34 levels (a
  categorical regime).
- **Price levels carry the most signal.** Best single-feature WP: i0 price ≈ 0.19, i0 volume ≈ 0.15, i1 price ≈
  0.13. One-step differences carry little on their own but matter inside a recurrent model.

## Scored rows (`is_scored`)

- 8.6% of rows after the warm-up are scored (0.9–31% per sequence), mostly in short runs (median length 2).
- **Not random:** scored rows are twice as likely to follow an instrument-1 price change (0.197 vs 0.101), sit at
  higher values of the slow auxiliary signals, and vary 0.025–0.19 across regime levels. A causal logistic model
  predicts them with AUC ≈ 0.78. Weighting the training loss by it gives +0.003–0.004.

## Sequence structure

- About 17% of training sequences end within distance 1 of another sequence's first row, i.e. sequences chain.
  This happens within train and within validation, never across, and not in file order.
- It is not a useful lever. Early steps are not predicted worse than late ones (WP at steps 99–500 is 0.692),
  so warm-starting from a predecessor gains nothing. Using it would also be a data exploit.

## Inference budget

| Measurement (laptop, single thread) | Time per row |
|---|---|
| Organisers' baseline `predict()` (ONNX GRU) | 64.6 µs |
| ONNX Runtime fixed per-call overhead | ~10 µs |
| PyTorch eager GRU step | ~300 µs |
| numba 2-layer GRU-128 step (float32) | 16–25 µs |
| Final 4-member int8 fused kernel | ~65 µs (scorer: ~70 µs incl. harness → 46 min) |
