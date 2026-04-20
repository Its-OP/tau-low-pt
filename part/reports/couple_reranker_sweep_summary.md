# Couple-reranker sweep — summary across Batches 1–5

**Period:** 2026-04-16 → 2026-04-20
**Current baseline:** **v3** — `projected_infersent p=32 + softmax-CE + label_smoothing=0.10 + cosine_power=2.0`
**v3 full-val metrics (52,284 events, K2=60):** C@100 = **0.7925** | mean_rank **10.8**
**Ceiling (D@60 tracks entering Stage 3):** ~0.86

---

## TL;DR

Over 5 sweep batches and ~32 hypotheses tested against v3/v2, **no single candidate beats v3 on full-val by ≥ +0.002 pp C@100.** The baseline has advanced from v1 (0.7881) → v2 (0.7914) → v3 (0.7925), driven entirely by two changes: (i) Batch 1 label smoothing ε=0.10, (ii) Batch 2 projected_infersent p=32. Every subsequent batch (3/4/5) tested new losses, augmentations, tokenized encoders, feature sets, and optimizers; none delivered a full-val win large enough to promote. The main lever at this data scale is not encoder capacity, loss reformulation, or feature additions — those have all been ruled out. The next direction must be orthogonal to encoder design: representation learning (VICReg pretrain), Stage-2 signal reuse (distillation), sparse specialization (MoE), or curriculum on the full dataset.

---

## Baseline evolution (full-val C@100)

| Baseline | Config | Full-val C@100 | Δ vs prior | Source |
|---|---|---:|---:|---|
| v1 | pairwise softplus, 50 rand negs, no smoothing, cosine_power=1.0 | 0.7881 | — | pre-sweep |
| v2 | + softmax-CE + label_smoothing=0.10 + cosine_power=2.0 | 0.7914 | +0.0033 | Batch 1 winner `T1_3_smooth_010` combined with Batch 1 `T1_4_cos_200` |
| **v3** | + projected_infersent p=32 couple encoder | **0.7925** | +0.0011 | Batch 2 winner `A5_projected_infersent_p32` |

**Net gain since inception:** +0.0044 pp C@100 full-val. Both jumps came from Batches 1–2.

---

## Per-batch roll-up

### Batch 1 — softmax-CE loss family (14 runs, T-series)

**Winner (train-time & full-val):** `T1_3_smooth_010` — label smoothing ε=0.10 on softmax-CE, ΔC@100 full-val = **+0.0027**.
**Runner-up:** `T1_4_cos_200` — cosine_power=2.0 on target distribution, full-val +0.0012.
**Losers:** ListMLE (−0.0011), tempering T≠1 (−0.0014 to −0.0017), hardneg ratio sweep (−0.0057), pair_v2 features (−0.0068), EMA (−0.0005), wider hidden (T3_1, −0.0006).

**Verdict:** label smoothing + sharpened target is the only Batch-1 lever. Both promoted into v2.

### Batch 2 — couple-embedding geometry (9 runs, A/B/D-series)

**Winner (full-val):** `A5_projected_infersent_p32` — projected_infersent with projector width 32 → v3 baseline (+0.0011 full-val).
**Near-win:** `A4_projected_infersent_p16` — same idea at p=16, full-val +0.0004.
**Losers:** raw InferSent (A1), symmetric aggregation (A2, worst at −0.0062), Hadamard bilinear (A3), wider hidden (D1).

**Verdict:** projector width between baseline (0) and 32 is the lever; wider projector marginal. Promoted into v3.

### Batch 3 — S+A tier, SOTA-literature hypotheses (10 runs)

Train-time C@100c results (B0_v3_anchor = 0.83094, sweep-local reference):

| Run | Hypothesis | Train-time | Δ | Status |
|---|---|---:|---:|---|
| B0_v3_anchor | reference | 0.83094 | — | — |
| B3_6_tau_phys_v3 | pair_physics_v3 (5 extra features: Kalman-χ², DCA, helicity, BW ρ/a1) | **0.83240** | +0.00146 | **train-time win** |
| B3_2_multi_positive | uniform-target multi-positive loss | 0.83146 | +0.00052 | marginal |
| B3_3_event_ctx_film | DeepSet event-context + FiLM conditioning | 0.83146 | +0.00052 | marginal |
| B3_10_hardneg_delay | delayed hard-negative sampling | 0.82948 | −0.00146 | regression |
| B3_1_lambda_ndcg2pp | LambdaLoss NDCG-Loss2++, K=100 | 0.82958 | −0.00136 | regression |
| B3_9_vertex_aux | auxiliary vertex-compatibility head | 0.82937 | −0.00157 | regression |
| B3_4_ft_transformer (F5 mini) | tokenized encoder, d=16, 3 blocks | 0.82854 | −0.00240 | regression (led to Batch 5) |
| B3_5_per_track_tokens (F5 mini) | per-track tokens + role embed, d=16, 2 blocks | 0.82625 | −0.00469 | regression (led to Batch 5) |
| B3_8_asam | ASAM (adaptive SAM, 2-pass) | 0.82792 | −0.00302 | regression |
| B3_7_cov_smear | covariance-matrix feature smearing | — | — | **killed** (regressed to 0.76 early) |

**Verdict:** marginal positives in B3_2/B3_3/B3_6, clear regressions elsewhere. Tokenization regression at min-scale motivated Batch 5 (scale test). No full-val promotions.

### Batch 4 — B+C tier, cheap remaining hypotheses (3 runs)

| Run | Hypothesis | Train-time | Δ train | Full-val Δ | Status |
|---|---|---:|---:|---:|---|
| B0_v3_anchor | reference | 0.83094 | — | — | — |
| B4_4_full_list | use full negative list (not random subsample) | **0.83323** | +0.00229 | not eval'd | best train-time |
| B4_9_pair_signif | 3 significance features (mass pull, Δφ signif, pT-bal signif) | 0.83271 | +0.00177 | **+0.0010** | full-val winner |
| B4_3_soft_or | soft-OR (logsumexp) numerator in CE | 0.83062 | −0.00032 | not eval'd | regression |

**Verdict:** B4_9 is the closest-to-promotable candidate across all five batches, but +0.0010 full-val is still below the +0.002 pp bar. Not promoted. B4_4 full-list train-time looked strongest but was never full-val'd (higher priority went to Batch 5 launch).

### Batch 5 — tokenization scale & depth (5 runs, 1 killed)

| Run | Config | Train-time | Δ train | Full-val | Δ full-val | Status |
|---|---|---:|---:|---:|---:|---|
| B0_v3_anchor | projected_infersent p=32 | 0.83094 | — | 0.7925 (v3) | — | reference |
| **B5_1_d32_b4** | ft_transformer, d=32, blocks=4, batch=96 | **0.83229** | +0.00135 | **0.7917** | **−0.0008** | **reversal** — do not promote |
| B5_2_d32_b6 | d=32, blocks=6, batch=64 | 0.83167 | +0.00073 | not eval'd | — | marginal train-time |
| B5_3_d48_b4 | d=48, blocks=4, batch=48 | 0.83115 | +0.00021 | not eval'd | — | within noise |
| B5_4_d64_b4 | d=64, blocks=4, batch=32 | 0.8283 (ep 24) | −0.0026 | — | — | **killed** — plateau 14 ep, LR decayed |
| B5_5_d64_b6 | d=64, blocks=6 | — | — | — | — | **dropped** — VRAM > 100 GB |

**Verdict:** tokenization-capacity hypothesis fully ruled out. Δ monotonically shrinks with scale; train-time win at smallest config reverses at full-val. See `part/reports/couple_batch5_sweep.md` for full analysis.

---

## Overall scoreboard

**Full-val gains that survived (vs prior baseline):**

| Source | Lever | Full-val Δ C@100 | Promoted |
|---|---|---:|---|
| Batch 1 `T1_3_smooth_010` | label smoothing ε=0.10 | +0.0027 | ✓ (→ v2) |
| Batch 1 `T1_4_cos_200` | cosine_power=2.0 | +0.0012 | ✓ (→ v2) |
| Batch 2 `A5_projected_infersent_p32` | projector width 32 | +0.0011 | ✓ (→ v3) |
| Batch 4 `B4_9_pair_signif` | physics significance features | +0.0010 | ✗ (below +0.002 bar) |
| Batch 5 `B5_1_d32_b4` | tokenized encoder at min scale | **−0.0008** | ✗ (reversal) |

**Train-time wins that never got full-val confirmation:**
- B3_2 `multi_positive` (+0.00052)
- B3_3 `event_ctx_film` (+0.00052)
- B3_6 `tau_phys_v3` (+0.00146)
- B4_4 `full_list` (+0.00229, train-time only)

All within the train-time noise band (≤ 0.002 pp) that Batch 5 demonstrated does not reliably transfer to full-val. Not worth the GPU time unless combined.

---

## What was ruled out

1. **Loss reformulation:** ListMLE (B1), LambdaLoss NDCG-Loss2++ (B3_1), soft-OR numerator (B4_3), tempering (B1), ASAM optimizer (B3_8). No lever here at this dataset scale.
2. **Hard-negative mechanics:** hardneg ratio sweep (B1), delayed hardneg (B3_10), full negative list (B4_4, train-time only, unverified). Unclear — but not dominant.
3. **Encoder scale/tokenization:** B3_4/B3_5 (min scale regression) + B5_1/2/3 (scale up shrinks Δ) + B5_4 (deeper plateaus below v3). Entire tokenization direction ruled out.
4. **Auxiliary heads:** vertex auxiliary (B3_9). No lever.
5. **Augmentations:** covariance smearing (B3_7, killed). Not pursued further.
6. **Event-context conditioning:** DeepSet + FiLM (B3_3). Marginal train-time, deprioritized.
7. **EMA & wider/deeper:** B1 T2_3 EMA, B1 T3_1 wider. No lever.

---

## What remains (Phase-C deferred queue)

Ordered by expected ceiling (most-likely → least):

1. **H5 — Stage-2 distillation.** Use Stage-2 ParT's existing logits as soft targets on candidate couples. Reuses an information source none of Batches 1–5 touched. High ceiling if Stage-2 uncertainty is calibrated.
2. **H19 — VICReg pretrain.** Self-supervised representation learning on un-labeled couples. New inductive bias, cheap if we already have the data pipeline.
3. **H11 — curriculum.** Start with easier events (fewer tracks, cleaner topology), anneal to full distribution. Data-scale lever, orthogonal to arch.
4. **H18 — MoE encoder.** Sparse specialization per tau decay mode. Expensive but genuinely different model family.
5. **H17 — monotonic tower.** Structural prior that rank-order of input features should correlate with output. Low cost, narrow upside.
6. **H14 — manifold mixup.** Deferred because soft labels don't compose with softmax-CE cleanly; would need loss reformulation first.

Decisions to take **before** launching Phase-C:
- Are we satisfied with 0.7925 full-val as the Stage-3 endpoint, or is there a target gain that justifies another month of sweeps?
- Does the thesis narrative need "we explored all levers and found X works best" (in which case do ≥ 2 Phase-C directions), or "here's the best system we could build" (in which case pick one and commit)?

---

## Recommendations

1. **Keep v3 as the Stage-3 working baseline.** Full-val C@100 = 0.7925, mean_rank 10.8. Ship this configuration for downstream analysis.
2. **Do not promote B4_9 or B5_1.** Both gains are below the +0.002 pp bar; neither warrants the baseline churn.
3. **Do not run further loss / tokenization / feature-set sweeps.** Five batches of evidence say these levers are exhausted.
4. **Pick one Phase-C direction next** — recommendation: **H5 Stage-2 distillation** first, because it reuses existing Stage-2 signal without changing the Stage-3 architecture. Cheapest path to a novel lever.
5. **Change the sweep acceptance protocol:** run full-val eval on *every* train-time win ≥ +0.001 pp, not just the biggest. Batch 5 showed train-time ≤ 0.002 pp is within noise of the subset harness, so train-time-only ranking cannot decide promotion.
6. **Archive the B4_9 checkpoint** as a reference point in case Stage-2 distillation benefits from pair-significance features at the Stage-3 input.

---

## Cross-references

- `part/reports/couple_ranking_overview.md` — full Batch 1–2 analysis and v3 baseline derivation.
- `part/reports/couple_reranker_improvements_progress.md` — live Batch 1 progress tracker with per-epoch tables.
- `part/reports/couple_batch5_sweep.md` — detailed Batch 5 analysis (this session).
- Training script: `part/train_couple_reranker.py`
- Model: `weaver/weaver/nn/model/CoupleReranker.py`
- Features: `part/utils/couple_features.py`
