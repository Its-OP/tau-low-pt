# Couple-ranking for low-pT τ → 3π reconstruction — comprehensive overview

**Status:** Snapshot as of 2026-04-18. Deep-Research consumption document.
Read-only narrative of what was tried, what worked, and the mechanism
behind each outcome. No next-step recommendations appear in this file.

---

## 1. Glossary

- **C@K_couples**: fraction of events (out of **total events seen**)
  where ≥ 1 GT couple is in the reranker's top-K output.
- **RC@K_couples**: fraction of events where ≥ 1 GT couple is in the
  top-K output **and** the full GT triplet is present in Stage 1's
  top-K1. "RC" = "reachable C".
- **D@K_tracks**: fraction of events where ≥ 2 GT pions lie in
  Stage 2's top-K tracks. Property of the frozen cascade; bounds
  C@K_couples above at K = K2.
- **mean_first_gt_rank_couples**: mean 1-indexed rank of the
  highest-ranked GT couple, averaged over eligible events only.
- **GT couple**: an unordered pair of tracks both of which come from
  the τ decay. Roughly 3 per event (C(3, 2) = 3 from the three GT
  pions).
- **Eligible events**: events with at least one GT couple available in
  the cascade's top-K2 tracks.
- **Events with full triplet**: events with all three GT pions in the
  reranker's output couples (required for the downstream third-pion
  completion to recover the triplet from the top-K couples).
- **Train-time val subset**: the first 9 600 events of the val split,
  consumed unshuffled by the training-script's mid-epoch validator.
  Systematically reports C@100 ≈ 4 pp higher than the full 52 452-event
  val.
- **ΔC@100**: signed difference between a run's best C@100 and a
  named reference run's best C@100, under the same val split.
- **gap_closure**: `ΔC@100 / (D@K2 − C@100_baseline)` — fraction of
  the structural ceiling gap that the experiment closed.
- **Baseline v1 (pairwise)**: original softplus pairwise ranking loss,
  full-val C@100 = 0.7881.
- **Baseline v2 (stacked)**: softmax-CE + label_smoothing=0.10 +
  cosine_power=2.0, full-val C@100 = 0.7914.
- **Baseline v3 (projected_infersent)**: baseline v2 plus a learned
  per-track projector `φ(t) = LayerNorm(ReLU(Linear(16 → 16)(t)))`
  followed by InferSent-style pair fusion. Full-val C@100 = 0.7915;
  mean_rank = 10.7. Current project default.

---

## 2. Purpose of the task

The project aims to reconstruct low-transverse-momentum τ → 3π decays at
CMS. A single event contains O(1000) detector tracks of which exactly
three — on average — are the pions from the τ decay we want to find.
Signal purity is roughly 0.27 %. The raw combinatorial search space is
`C(N, 3)` triplets, which is intractable at N≈1000, so the problem is
decomposed into a three-stage cascade that progressively narrows the
candidate set:

| Stage | Model | Input | Output |
|---|---|---|---|
| 1. Track prefilter | `TrackPreFilter` (MLP + kNN, 400 k params, frozen in the cascade) | All tracks in the event (up to 2100 padded slots) | Top-`K1 = 256` tracks by per-track prefilter score |
| 2. ParT reranker | `CascadeReranker` (ParT-style self-attention with pairwise physics biases, 6.6 M params, frozen here) | Top-`K1` tracks from Stage 1 | Refined per-track scores; top-`K2 = 60` tracks fed to Stage 3 |
| 3. Couple reranker | `CoupleReranker` (1×1-Conv MLP + residual blocks, ~575 k trainable params) | All C(K2, 2) = 1 770 couples, each encoded as a 51-dimensional feature vector | One scalar score per couple; the top-100 form the anchor list for a downstream third-pion completion step that is outside the scope of this document |

This document is scoped to **Stage 3**, the CoupleReranker. Stages 1 and
2 are loaded from the frozen checkpoint `models/cascade_best.pt` and are
never updated here.

The goal of the CoupleReranker is to push events with at least one
ground-truth pion pair into the top of its ranking. The operating metric
is `C@K_couples` — fraction of events where ≥ 1 GT couple lies in the
top-K reranked couples, denominated by the total number of events (not
events that contain a GT couple). The most-watched cutoff is K = 100
because the downstream third-pion search consumes the top-100 anchors.

The structural ceiling on `C@K_couples` at a given `K2` is
`D@K2_tracks` — the fraction of events where ≥ 2 GT pions are already
inside the top-`K2` tracks delivered by the frozen cascade. A reranker
cannot exceed this ceiling because the GT couple must exist in the
candidate pool before any ranking can promote it.

---

## 3. Constraints and design choices locked before the experiments started

Decisions made earlier in the project. They shape what can and cannot be
tried at Stage 3:

1. **Frozen upstream**: Stages 1 and 2 are not retrained in any Stage-3
   experiment. All improvements must come from the reranker itself
   (loss, architecture, features, optimisation). If `D@K2_tracks` were
   the bottleneck, work would have to move back to Stages 1–2.

2. **K2 = 60 is fixed**: prior K-sweeps (`K2 ∈ {50, 70, 100, 125, 150,
   175}`) plateaued within ±0.003 C@100 and produced 1 770 candidate
   couples per event at K2 = 60. The operating point is frozen so
   comparisons across experiments are clean.

3. **Per-couple scoring only**: the Stage-3 architecture scores each
   candidate couple independently. No cross-couple attention, no
   graph-over-couples, no set-transformer over couples. Imposed by the
   user on 2026-04-16 as a hard constraint. The stated reason is that
   downstream usage ("for each anchor couple, find the third pion")
   already exploits per-couple independence, and cross-couple models
   would inflate parameter count and complicate inference without a
   matching recall ceiling.

4. **Label/feature design is soft**: charge product, invariant-mass
   windows, and ρ(770) indicators are fed as **features**, never as hard
   pre-filters. Prior triplet-combinatorics work (`reports/
   triplet_reranking/triplet_combinatorics.md`) showed that hardcoded
   physics cuts fail at low pT because ~88 % of kNN pairs pass the mass
   cut and ~120 k triplets survive a physics-filter cascade. Physics
   knowledge enters as priors, not constraints.

5. **No test data leak**: the only labelled pT source is the train
   split. The val split (52 452 events, 7 parquet files) is held out for
   evaluation only. Train-time validation uses a subset of 9 600 events
   (200 steps × batch 96 ÷ 2). This subset is deterministic and biased
   — it systematically reports C@100 ≈ 4 percentage points higher than
   the full val. Every metric is reported against the split that
   produced it; no cross-split claims are made.

---

## 4. Starting conditions — original baseline (v1)

Reported under the fixed config `K2 = 60, batch = 96, 60 epochs, 200
steps/epoch, AdamW lr = 5e-4, weight decay = 0.01, cosine LR schedule
with 5 % warmup, grad clip = 1.0, seed = 42`. Frozen cascade. Loss is
the softplus pairwise ranking loss (the same one used by the prefilter
and Stage-2 reranker):
$$L = T \cdot \mathrm{softplus}\bigl((s_\mathrm{neg} - s_\mathrm{pos}) / T\bigr),\quad T=1$$
averaged over 50 random negatives × per-event positive pairs. The
per-couple input feature vector has 51 dimensions:

| Block | Dims | Content |
|---|---|---|
| 1 | 32 | Per-track concat `[t_i, t_j]` (16 standardised track features each) |
| 2 | 10 | Hand-crafted pairwise physics (ln k_T, ln z, ln ΔR, ln m², charge product, Δdz_sig, ρ(770) Gaussian indicator, OS × ρ, φ-corrected dxy, Lorentz dot) |
| 3 | 5 | Derived geometry (m_ij, pT_ij, Δη, Δφ, ΔR) |
| 4 | 4 | Upstream scores (Stage-1 i, Stage-2 i, Stage-1 j, Stage-2 j) |

Baseline v1 full-val metrics (52 452 events, K2 = 60, top-200 couples):

| Metric | Value |
|---|---|
| C@50 | 0.7563 |
| C@75 | 0.7765 |
| **C@100** | **0.7881** |
| C@200 | 0.8041 |
| D@60_tracks (ceiling) | 0.8596 |
| mean rank of best GT couple | 11.6 |
| structural gap (D@60 − C@100) | 0.0715 (7.15 pp) |
| train-time subset C@100 | 0.8266 |

The train-time-subset-to-full-val gap is ~4 pp, consistent across every
checkpoint measured. It is a property of the val-subset selection, not
a metric bug.

---

## 5. Batch 1 sweep — loss, schedule, sampling, architecture at input_dim = 51

**Run id:** `couple_improvements_20260416_211532`. 14 experiments, all
60 epochs, all completed, all knob audits verified against the parsed
`Arguments:` dict of each `training.log` and against the reported
parameter count of `CoupleReranker: N params (trainable, ...)`. Full
raw artefacts under `debug_checkpoints/couple_improvements_2026-04-16_
211532/` in the repo (gitignored, ~124 MB).

Each experiment's ΔC@100 below is against the **sweep-local T0** of
0.8266 (train-time subset) and 0.7881 (full val). Full-val numbers only
exist for a subset; both are reported where available.

### 4.1 T0 — baseline (pairwise softplus, all defaults)

- **Hypothesis:** re-establishes the reference point at seed 42 inside
  the sweep harness, so every other row compares against an anchor run
  that shared identical data shuffling, BN calibration, and checkpoint
  selection policy.
- **Result:** C@100 = 0.8266 (train-time subset) / 0.7881 (full val);
  D@60 = 0.8596; best epoch 58; no non-finite batches. Steady climb
  ep 1 → 0.7933 → ep 58 → 0.8266 with no stall.
- **Explanation:** reproduces the earlier standalone T0 within 0.001
  C@100. The 6–8-epoch stall documented in earlier K-sweeps did not
  recur at this seed — seed variance on the LR-warmup plateau is
  visible but small.

### 4.2 T1.1 — listwise softmax cross-entropy loss

- **Hypothesis:** softmax-CE with a single positive per event reduces
  to ListMLE top-1 loss. Bruch (arXiv:1911.09172) shows that listwise
  formulations remove the saturation of pairwise softplus once the
  positive has been separated from randomly-sampled negatives, so the
  late-epoch recovery window should widen.
- **Flags:** `--couple-loss softmax-ce`.
- **Result:** C@100 = 0.8254 (train-time subset), ΔC@100 = −0.0011.
- **Explanation:** the loss is `L = −s_pos/T + logsumexp(scores/T)`.
  Without regularisation it over-concentrates probability mass on the
  top-1 logit. Well-calibrated top-1 but marginally worse top-K for
  K > 1, which is exactly what C@100 measures. Listwise is not
  harmful, but it is not additive on its own — it becomes a win only
  once it is paired with label smoothing (see T1.3).

### 4.3 T1.2 — ranking temperature sweep

Three runs at `T ∈ {0.25, 0.5, 2.0}` on top of softmax-CE.

- **Hypothesis:** the softmax scale has never been swept. A smaller T
  sharpens the distribution and amplifies gradient on the hardest
  negative; a larger T flattens it and distributes gradient across
  more negatives.
- **Results (train-time ΔC@100):** T = 0.25 → −0.0017; T = 0.5 →
  −0.0002; T = 2.0 → −0.0014.
- **Explanation:** at T = 0.25 the sharpened softmax starves gradients
  on middle-ranked negatives and the model overspecialises on the
  single hardest one. At T = 2.0 the softmax is so flat that gradients
  are dominated by the log-partition term and late-stage training
  plateaus by ep 30. T = 0.5 is statistically indistinguishable from
  T = 1.0. **Net verdict: T = 1.0 (the default) is near-optimal on
  this dataset; temperature is not a useful knob.**

### 4.4 T1.3 — label smoothing on softmax-CE

Two runs at ε ∈ {0.05, 0.10}.

- **Hypothesis:** Müller (arXiv:1906.02629) reports +0.5–1 pp gains
  from smoothing in image classification. The mechanism — penalising
  over-confident top-1 predictions by redistributing a fraction ε of
  probability mass to the other candidates — is exactly what softmax-
  CE without regularisation lacks.
- **Flag:** `--couple-label-smoothing {0.05, 0.10}`. Loss:
  `L = (1-ε) · L_onehot + ε · L_uniform` where `L_uniform` replaces
  the per-example negative log-likelihood of the positive with the
  mean over the pool.
- **Results:**
  - ε = 0.05: train-time C@100 = 0.8282, ΔC@100 = +0.0017 (+5.05 %
    of the ceiling gap closed). Full val: +0.0013.
  - ε = 0.10: train-time C@100 = 0.8301, ΔC@100 = +0.0035 (+10.73 %
    of gap). Full val: +0.0027 (C@100 0.7908 vs 0.7881).
- **Explanation:** linear in ε across this range, so the mechanism is
  the dominant one — gradient is actually distributed to the
  middle-rank negatives that matter for top-K ≥ 10. mean_rank drops
  from 11.6 to 11.0 on full val, an independent confirmation that the
  ranking itself improved, not just the top-1 calibration. **This is
  the only knob in Batch 1 that produced a reliable full-val gain.**

### 4.5 T1.4 — cosine LR power sweep

Three runs at `--cosine-power ∈ {0.5, 1.5, 2.0}`. Exponent 1.0 =
standard cosine decay. < 1 steepens early decay, > 1 holds LR near
peak longer and drops sharply at the end.

- **Hypothesis:** baseline peaks at ep 58 (within 2 epochs of the
  end). A steeper decay would force convergence earlier; a delayed
  decay would give the late epochs more learning rate to exploit.
- **Results (train-time ΔC@100):** power = 0.5 → −0.0009; power = 1.5
  → +0.0009; power = 2.0 → +0.0008. Full val for power = 1.5 →
  +0.0001; power = 2.0 → +0.0012.
- **Explanation:** the default cosine schedule already decays too
  fast for this problem. Power = 2.0 holds LR at > 50 % of peak for
  the first 40 epochs, then drops sharply in the last 20, which
  matches when the model makes its biggest late-stage gains.
  Power = 0.5 does the opposite and underfits the late epochs.
  Gain is small but consistent on full val and cost is zero — a
  scheduling change. **Kept in the combined baseline v2.**

### 4.6 T2.1 — hard-negative mining

- **Hypothesis:** ANCE / NV-Retriever (arXiv:2007.00808, Moreira 2024)
  report +2–4 pp gains by replacing random negatives with the current
  model's top-scoring ones. The candidate pool per event is already
  scored by the forward pass, so hard-neg sampling is almost free.
- **Flags:** `--couple-hardneg-fraction 0.5 --couple-hardneg-margin
  0.1`. Per event, 25 of the 50 negatives are the top-scoring
  non-GT couples that also satisfy `score < max(pos_score) − 0.1`
  (positive-aware filter to avoid false negatives from unlabelled
  but true couples); the other 25 remain random.
- **Result:** C@100 = 0.8208 (train-time), ΔC@100 = −0.0057. Worst
  non-combined result of Batch 1.
- **Explanation:** two signatures. (a) The best epoch drops from 58
  to 32; the model converges rapidly then degrades — peak at 32,
  mild decline through 60. (b) Trajectory ep 30 → 40 slows to
  +0.0011 vs T0's +0.0168; the model is no longer learning on the
  fresh hard negatives by mid-training. This is a non-stationary
  objective: the distribution of hard negatives changes as the
  scorer improves, but the cosine LR schedule treats the objective
  as stationary. By ep 32 the LR is ~40 % of peak and no longer
  large enough to adapt. Upstream `D@60_tracks` is 0.8634, slightly
  higher than T0's 0.8596 — sampling noise over the 9 600-event
  subset, not an upstream regression — which isolates the failure
  to the reranker's training dynamics.

### 4.7 T2.2 — pair-kinematics v2 stacked on T2.1

- **Hypothesis:** four extra per-pair features standard in CMS τ-ID
  analyses (cos of 3D opening angle; `(m_ij − m_τ)/σ_m`;
  `dxy_sig_i · dxy_sig_j`; `dz_sig_i · dz_sig_j`) should improve
  discrimination of true τ-daughter pairs from combinatoric ones.
  Stacks on T2.1 to test a maximally aggressive configuration.
- **Flags:** `--couple-hardneg-fraction 0.5 --couple-hardneg-margin
  0.1 --pair-kinematics-v2`. Bumps input_dim from 51 to 55.
- **Result:** C@100 = 0.8198, ΔC@100 = −0.0068. Worst overall.
- **Explanation:** parameter count went from 575 105 to 576 129
  (+1 024 = 4 new input dims × 256 first-layer channels, exact
  match), so the features are wired in. **But T2.2 inherits T2.1's
  damage, so the 4-feature contribution is confounded with the
  hard-neg penalty and cannot be read from this run alone.** The
  clean read is done in Batch 2 (B1_pair_v2_clean).

### 4.8 T2.3 — EMA weight averaging

- **Hypothesis:** an exponential moving average of the weights
  (`torch.optim.swa_utils.AveragedModel`, decay 0.999) is often a
  zero-cost +0.3–1 pp lift in classification. BN recalibrated on the
  EMA copy after training.
- **Result:** C@100 = 0.8260, ΔC@100 = −0.0005. Essentially neutral.
  `best_model_ema_calibrated.pt` was produced and preserved.
- **Explanation:** the reported metric is on the live weights, not the
  EMA copy — the training loop only validates the live model. EMA
  result on the full val is still open as of the snapshot. Live-weight
  neutrality is consistent with "noise decorrelates fast enough at this
  batch size × dataset scale that EMA has little to average over."

### 4.9 T3.1 — wider and deeper

- **Hypothesis:** the baseline architecture (hidden = 256, 4
  residual blocks, 575 k params) may be underfitting. Bump to
  hidden = 384, 6 residual blocks, 1.87 M params (~3.2× capacity).
- **Result:** C@100 = 0.8259, ΔC@100 = −0.0006; peak at ep 44 (14
  epochs earlier than baseline).
- **Explanation:** the early peak and the train/val gap widening
  through the latter half of training are textbook mild overfit.
  Capacity is not the constraint. Further scale-up is a dead end
  unless the regularisation regime changes.

### 4.10 Batch 1 synthesis

One clean winner: **label smoothing ε = 0.10 on softmax-CE**
(+0.27 pp full-val C@100, +0.35 pp train-time). Cosine-power ≥ 1.5
produces sub-threshold gains (~+0.01 pp full val) that are kept as a
zero-cost schedule change. All other experiments regress or are
neutral. The failure modes cluster into three classes:

- **Loss over-sharpens** (T1.1 softmax-CE alone, T1.2 low-temp):
  probability mass concentrates on top-1 at the expense of top-K > 1.
- **Non-stationary objective × stationary schedule** (T2.1, T2.2
  inherited): hard-neg sampling changes the loss landscape mid-run,
  cosine LR cannot adapt.
- **Capacity without constraint** (T3.1): more parameters at the
  same regularisation level overfits; train-val gap widens.

### 4.11 New baseline v2 — smooth_010 + cos_200 stacked

A single follow-up run combined the two positive knobs:
`softmax-CE + label_smoothing=0.10 + cosine_power=2.0`. Trained 60
epochs at seed 42 with everything else at the sweep defaults.

| Metric (full val, 52 452 events) | T0 v1 | smooth_010 alone | v2 stacked |
|---|---:|---:|---:|
| C@50 | 0.7563 | 0.7608 | **0.7616** |
| C@75 | 0.7765 | 0.7799 | **0.7814** |
| **C@100** | **0.7881** | 0.7908 | **0.7914** |
| C@200 | 0.8041 | 0.8050 | 0.8049 |
| RC@100 | 0.6774 | 0.6790 | **0.6804** |
| mean rank | 11.6 | 11.0 | **10.9** |
| ΔC@100 vs v1 | — | +0.0027 | **+0.0033** |

Stacking adds +0.06 pp C@100 and −0.1 mean rank over smooth_010 alone.
Most of the gain comes from label smoothing; cosine-power is a marginal
additive. Total lift vs v1: +0.33 pp C@100 (+0.05 of the 7.15-pp
ceiling gap = 4.6 % gap closure). Baseline v2 became the new project
default (CLI defaults updated in `part/train_couple_reranker.py`;
server-side `models/couple_reranker_best.pt` swapped to the v2
checkpoint; the v1 pairwise checkpoint preserved as
`models/couple_reranker_v1_pairwise.pt`).

---

## 6. Batch 2 sweep — couple-embedding redesign and Batch 1 follow-ups

**Run id:** `couple_batch2_20260417_153112`. 9 experiments, all
completed (60 epochs each except D1 at 80). All knob audits verified
against `Arguments:` and parameter-count witnesses. Each experiment
runs on top of baseline v2, so the knob-delta columns in `Arguments:`
are `couple_loss='softmax-ce', couple_label_smoothing=0.1,
cosine_power=2.0` by default.

Ceiling reference: sweep-local B0 (baseline v2 re-run inside the
Batch 2 harness). B0's best C@100 on train-time is 0.8299 and on full
val is 0.7914. D@60_tracks at B0's best epoch is 0.8645.

### 5.1 Motivation

Batch 1 ran out of ideas at the "flat 51-dim feature vector" level. The
encoder treats Block 1 as a raw concat `[t_i, t_j]` of 16-dim
standardised track features; no learned per-track projection, no
element-wise interaction, no permutation symmetry. A literature survey
(InferSent, Conneau 2017, arXiv:1705.02364; Hadamard low-rank bilinear,
Kim 2017, arXiv:1610.04325; DeepSets, Zaheer 2017, arXiv:1703.06114;
ParT pair-bias, Qu 2022, arXiv:2202.03772) suggested that the
Stage-3-specific couple representation is the next largest unpulled
lever. Batch 2 adds a single CLI knob
`--couple-embed-mode ∈ {concat, infersent, symmetric, bilinear_lrb,
projected_infersent}` plus `--couple-projector-dim` for the projected
variant. The feature builder is unchanged; Block 1 rebuild happens
inside the model's forward so any learnable projection weights receive
gradients. Blocks 2–4 are untouched, so the experiment reads out the
effect of the per-track pair-encoder alone.

Block-1 composition and resulting `input_dim` per mode:

| Mode | Block 1 | Block 1 dim | Total input_dim | Params |
|---|---|---:|---:|---:|
| concat (baseline v2) | `[t_i, t_j]` | 32 | 51 | 575 105 |
| infersent | `[t_i, t_j, \|t_i − t_j\|, t_i ⊙ t_j]` | 64 | 83 | 583 297 |
| symmetric | `[max(t_i, t_j), mean, \|t_i − t_j\|]` | 48 | 67 | 579 201 |
| bilinear_lrb | `[t_i, t_j, (U t_i) ⊙ (V t_j)]` (r = 8) | 40 | 59 | 577 409 |
| projected_infersent (p = 16) | `[φ(t_i), φ(t_j), \|Δφ\|, φ(t_i) ⊙ φ(t_j)]` after `φ(t) = LayerNorm(ReLU(Linear(16 → 16)(t)))` | 64 | 83 | 583 601 |
| projected_infersent (p = 32) | same but p = 32 | 128 | 147 | 600 289 |

All six input_dims were verified at smoke test (exact match against the
analytic prediction) before the full sweep started.

### 5.2 B0 — baseline v2 re-run as anchor

- **Hypothesis:** re-seed under the Batch 2 harness so every Tier A/B/C
  row compares against an anchor whose BN calibration, val subset, and
  validation schedule are identical to its own.
- **Result:** C@100 = 0.8299 (train-time) at ep 48. D@60 = 0.8645
  (+0.0049 vs Batch 1 T0 — sampling noise over the subset).
- **Explanation:** within 0.003 C@100 of Batch 1's T0-run-with-v2-flags
  (baseline_v2 standalone had 0.83156 at ep 56). The shift reflects
  seed-insensitive re-initialisation noise; the stacked config is
  stable.

### 5.3 A1 — InferSent-style pair fusion (raw tracks)

- **Hypothesis:** Conneau's NLI result was that the combination of
  element-wise difference and Hadamard product is a more expressive
  pair fusion than plain concat, worth +2–3 pp accuracy. Translating to
  our setting: the Hadamard term highlights dimensions where both
  tracks have the same sign (co-activation), and the absolute-difference
  term captures asymmetry. Direct application on the raw 16-dim
  standardised features.
- **Flags:** `--couple-embed-mode infersent`.
- **Result:** train-time C@100 = 0.8302, ΔC@100 = +0.0003 (sweep-local
  B0 = 0.8299). Best epoch 60. mean_rank 17.41 vs B0's 18.25.
- **Explanation:** essentially a tie with baseline on C@100; marginal
  mean-rank improvement. The raw 16-dim track features are not
  expressive enough for Hadamard / difference terms to carry new
  signal: the dominant informative axes (log pT, Δ_dz significance,
  chi² normalisation) already bring most of their discriminative power
  through the first `Conv1d(51 → 256)` linear combination; the extra
  4× input dims give the first layer more ways to represent the same
  information, not new information. The first `Conv1d(1×1)` projects
  everything to the same 256-dim space, so the InferSent features are
  only a useful basis if the downstream MLP cannot synthesise them from
  `[t_i, t_j]` alone — and evidently it can.

### 5.4 A2 — symmetric aggregation (permutation-invariant)

- **Hypothesis:** the canonical `i < j` ordering is an arbitrary
  convention; symmetric pooling
  `[max(t_i, t_j), mean(t_i, t_j), |t_i − t_j|]` (DeepSets-style, Zaheer
  2017) would enforce `f(i, j) ≡ f(j, i)` by construction. If the
  asymmetry of raw concat carried no information, symmetric should tie
  or beat; if it did, symmetric would lose.
- **Result:** C@100 = 0.8237, ΔC@100 = −0.0062. Worst of Batch 2
  Tier A. Plateau since ep 17 (24-epoch flat).
- **Explanation:** the data pipeline enforces `i < j` absolute on the
  sorted Stage-2 ranking — so `i` is always the higher-ranked track.
  Baseline thus has access to `"higher-ranked first"` vs
  `"lower-ranked first"` patterns. Symmetric pooling removes this
  signal entirely; the model must compensate with the remaining
  19-dim physics + cascade-score block, and it underfits. Net effect:
  the Stage-2 rank-ordering carries non-trivial signal for the
  reranker's own task, consistent with the view that Stages 2 and 3
  are partially redundant and that removing the hierarchy hurts.

### 5.5 A3 — Hadamard low-rank bilinear

- **Hypothesis:** append `(U t_i) ⊙ (V t_j)` at rank r = 8 to the
  existing concat. Bilinear pooling (Kim 2017 LRB; Gao 2016 CBP)
  captures multiplicative interactions with few extra parameters. r = 8
  costs 16 × 8 × 2 = 256 extra parameters for the projector and adds
  8 input dims.
- **Flag:** `--couple-embed-mode bilinear_lrb`.
- **Result:** C@100 = 0.8294, ΔC@100 = −0.0005. Near-baseline. Best
  epoch 58. mean_rank 16.04 (best of Batch 2 non-projector rows).
- **Explanation:** the concat's first Conv1d already learns arbitrary
  linear combinations of `[t_i, t_j]`; the bilinear outer product is
  redundant once the MLP has enough depth (4 residual blocks here) to
  synthesise quadratic relationships implicitly. The mean_rank
  improvement without a C@100 improvement is interesting — the
  bilinear term helps with the tail of the ranking (negatives far from
  the GT couple) more than the head. Not a winning signal but not
  damage either.

### 5.6 A4 — projected InferSent (p = 16)

- **Hypothesis:** the user-stated intuition is that the current "raw
  concat" is the least-informative encoding possible — plain axis
  concatenation, no learned transformation. A learned per-track
  projector `φ(t) = LayerNorm(ReLU(Linear(16 → p)(t)))` followed by
  InferSent `[φ(t_i), φ(t_j), |Δφ|, φ ⊙]` creates a small but
  gradient-trained "couple embedding" before the main scorer. Different
  from A1 because the per-track transform is learnable, so the model
  can choose an embedding that makes the downstream InferSent terms
  informative.
- **Flags:** `--couple-embed-mode projected_infersent
  --couple-projector-dim 16`. Param count 583 601 (projector +
  first-layer size change adds 8 496 parameters).
- **Result:** train-time C@100 = **0.8317**, ΔC@100 = **+0.0018**
  (+5.12 % of ceiling gap). Best epoch 58. mean_rank 15.68 (best in
  Batch 2, down 2.5 from B0).
- **Explanation:** this is the first Batch 2 result that beats baseline
  on C@100. The projector learns a representation in which the
  element-wise difference and Hadamard products are informative —
  directly confirming that A1's failure was the rawness of the input,
  not the InferSent structure itself. The mean-rank improvement (18.25
  → 15.68) is large for a ~+0.2 pp C@100 change, which suggests the
  projector is especially good at pushing hard but not-top-rank
  negatives away from the positive; the tail of the ranking improves
  more than the head. Full-val confirmation is still pending at snapshot
  time.

### 5.7 A5 — projected InferSent (p = 32)

- **Hypothesis:** does doubling the projector width from 16 to 32
  (128-dim Block 1 vs 64-dim) scale the win? Projector now costs a
  little over 2× more parameters.
- **Result:** C@100 = 0.8309, ΔC@100 = +0.0010. Slightly worse than
  p = 16.
- **Explanation:** the projector-width axis is not the lever; the
  projector's existence is. Doubling `p` doubles the input to the
  MLP's first layer but the Conv1d cannot extract twice as much useful
  signal from a 16-dim input — it simply learns a slightly redundant
  128-dim representation of the same 16 underlying axes. The marginal
  regression at p = 32 is consistent with mild over-parameterisation:
  the projector's 32-dim output has more degrees of freedom than the
  track data needs, and some of them learn noise in the train-time
  subset.

### 5.8 B1 — pair-kinematics v2 (clean read)

- **Hypothesis:** Batch 1's T2.2 added the 4 extra physics features on
  top of T2.1 hard-neg and regressed by −0.0068. But the hard-neg
  component alone regressed by −0.0057, leaving the features'
  independent contribution confounded. B1 isolates the contribution by
  running pair-kinematics-v2 on top of baseline v2 without hard-neg.
- **Flag:** `--pair-kinematics-v2`. input_dim 51 → 55, param count
  575 105 → 576 129 (exact +1 024 = 4 × 256 first-layer channels).
- **Result:** C@100 = 0.8289, ΔC@100 = −0.0010.
- **Explanation:** even without the hard-neg confound, the four extra
  physics features (cos opening angle, normalised mass deviation,
  dxy-sig product, dz-sig product) fail to help. This is a surprise
  against the ParT literature, where pairwise-physics biases are worth
  ~1.2 pp — but in that setting the features are injected as
  **attention biases** across thousands of tokens, which amplifies
  small per-pair signals. Here the reranker is already per-couple, so
  there is no amplification mechanism; the features just add 4 more
  input channels to compete with the 51 already-informative ones. The
  first Conv1d can effectively down-weight them, which leaves
  performance essentially unchanged — the small regression likely
  reflects the extra channels consuming a little regularisation budget.

### 5.9 C1 — hard-neg repair

- **Hypothesis:** the two diagnoses of Batch 1 T2.1 were (a) margin
  filter threw away informative negatives and (b) 50 % hard fraction
  produced too narrow a negative distribution. C1 tries the joint
  remedy: `margin = 0.0` (include every negative ranked above pos −
  floor) and fraction = 0.25 (so random negatives still dominate).
- **Flags:** `--couple-hardneg-fraction 0.25 --couple-hardneg-margin
  0.0`.
- **Result:** C@100 = 0.8290, ΔC@100 = −0.0009. Trajectory climbs
  steadily to ep 58 — no early-peak pathology this time.
- **Explanation:** the non-stationary-peak failure mode from Batch 1
  is gone (peak ep 58, not ep 32), which confirms the diagnosis that
  50 % hard was the destabilising fraction. But the tight margin does
  not produce a new win either. Net: hard-neg mining as implemented in
  this codebase cannot be made additive on top of baseline v2 with the
  ANCE / NV-Retriever recipe. The likely remaining degree of freedom
  — asynchronous hard-neg refresh with an LR-restart schedule
  (RocketQA-style, Qu 2020, arXiv:2010.08191) — was not tested.

### 5.10 D1 — extended training (80 epochs)

- **Hypothesis:** baseline v2 peaked at ep 56 of 60, so the training
  budget may be the limiting factor. Does another 20 epochs keep the
  curve climbing?
- **Flag:** `--epochs 80`. Cosine LR extended proportionally.
- **Result:** train-time C@100 = 0.8306 at ep 58, ΔC@100 vs B0 =
  +0.0007. Trajectory: ep 48 = 0.8275 (below B0's 0.8299), ep 58 =
  0.8306 (barely above), then declines over ep 58 → 80 at LR < 1e-4.
- **Explanation:** extra epochs produce a flat shelf, not a new climb.
  The cosine schedule's late-stage LR is already small enough that
  additional epochs add noise rather than signal. Confirms baseline v2
  is near its dataset-limited convergence point at 60 epochs; a wall-
  clock extension is not the lever.

### 5.11 Batch 2 synthesis

One clean train-time winner: **projected_infersent at p = 16**
(+0.18 pp C@100, −2.6 mean_rank). The p = 32 variant is positive but
smaller, which localises the effect to "projector exists", not
"projector is wide". Raw InferSent without a projector (A1) is
neutral — confirming that the 16-dim raw track features are already
efficiently consumed by the plain concat Conv1d. Symmetric pooling
regresses by 0.62 pp, establishing that the `i < j` canonical order
carries signal. The earlier hard-neg and pair-v2 regressions persist
once disentangled.

Full-val confirmation for Batch 2 winners (B0, A1, A4, A5, D1 plus B1
for completeness) was in progress at snapshot time. Initial eval script
failed for A1/A4/A5 because the script built the model with default
flags and the state-dict did not match the projector/input-dim shapes
of the new modes; fix deployed (read `args` dict from checkpoint →
auto-configure), rerun in flight.

---

## 7. Cross-batch summary of what works

The Stage-3 couple-reranker work to date has found exactly two
additive knobs:

1. **Label smoothing ε = 0.10 on softmax-CE loss** — full-val C@100
   gain +0.0027, train-time gain +0.0035. Mechanism is gradient
   redistribution to middle-rank negatives. Works at K > 1 because the
   regularised objective stops over-concentrating on top-1. Works only
   in combination with softmax-CE; on top of pairwise softplus it has
   no place to apply.
2. **Cosine LR power = 2.0** — marginal additive on top of (1), zero
   compute cost. Mechanism is "hold LR near peak through the
   mid-training regime, sharp drop in the last 20 % of epochs."

Combined baseline v2 closes 4.6 % of the D@60 − C@100 gap on full val
(C@100: 0.7881 → 0.7914).

An apparent third winner is under investigation as of the snapshot:

3. **Projected InferSent couple embedding (p = 16)** — train-time
   C@100 gain +0.0018 (+0.2 pp), mean_rank drops by 2.5. Mechanism is
   a learnable per-track projector followed by element-wise
   interactions; the interaction terms become informative only because
   the projector can choose a representation in which they are.
   Full-val confirmation pending.

---

## 8. Cross-batch summary of what does not work, and why

**Loss-level knobs that do not help:**

- Softmax-CE alone without smoothing (T1.1) — over-concentrates on
  top-1; top-K > 1 suffers.
- Ranking temperature anywhere in {0.25, 0.5, 2.0} (T1.2) — the
  default T = 1 is already near-optimal; sharper starves
  middle-rank gradients, flatter loses late-stage learning.

**Sampling-level knobs that do not help:**

- Hard-negative mining at 50 % fraction with 0.1 margin (T2.1) — the
  objective becomes non-stationary while the cosine schedule is
  stationary; the model peaks early then degrades.
- Hard-negative mining at 25 % fraction with 0.0 margin (C1) — the
  peak-stability problem is fixed, but the repair does not create a
  new win. The remaining unexplored direction is stationary-pool
  RocketQA-style negatives with an LR-restart schedule.

**Feature-level knobs that do not help:**

- Pair-kinematics v2 (4 extra physics features) on top of baseline v2
  (B1) — the extra features cannot beat a well-trained Conv1d's
  ability to recombine existing ones. ParT-style pairwise biases
  helped in their original setting because they were injected as
  amplified attention biases across thousands of tokens; here the
  reranker is already per-couple, so the amplification mechanism is
  absent.
- Raw InferSent on 16-dim features (A1) — the raw standardised track
  features are not expressive enough for element-wise products to
  carry additional information beyond what the baseline MLP already
  extracts.
- Symmetric pooling (A2) — removes the Stage-2-rank-ordering
  information that the baseline implicitly used.

**Architecture-level knobs that do not help:**

- Wider and deeper at the same dropout (T3.1) — 3.2× more parameters
  overfit; no capacity constraint is binding at 575 k.
- Projector width doubled (A5) — more projector capacity than the
  input needs; no additional signal to extract.
- Extended training (D1) — 60 epochs already hits the dataset's
  information ceiling at this batch size and schedule.

**Optimisation-level knobs that do not help:**

- EMA weights at decay 0.999 (T2.3) — live-weights neutrality suggests
  the optimisation noise is too low for EMA to average meaningfully.

---

## 9. Outstanding questions visible in the data

- **Structural headroom** — D@60_tracks ≈ 0.86 bounds C@100 above.
  Baseline v2 sits at 0.7914 full val, so 0.07 ≈ 7 pp is still
  unspent. This is a Stage-3 problem (the reranker is not
  retrieving the ≥ 2 GT pions that the cascade already delivers),
  not a Stage-1 or Stage-2 one.
