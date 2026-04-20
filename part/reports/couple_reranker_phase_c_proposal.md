# Couple-reranker Phase-C proposal — deferred hypotheses after Batch 5

**Date:** 2026-04-20
**Baseline:** v3 — `projected_infersent p=32 + softmax-CE + label_smoothing=0.10 + cosine_power=2.0`
**v3 full-val:** C@100 = 0.7925 | mean_rank 10.8 | ceiling (D@60 tracks) ≈ 0.86
**Prior evidence:** see `part/reports/couple_reranker_sweep_summary.md` for Batches 1–5 roll-up.

---

## Context

Batches 1–5 tested ~32 hypotheses across loss reformulation, encoder geometry, feature engineering, augmentation, and optimization. Only two levers moved full-val C@100: label smoothing (Batch 1) + projector width (Batch 2). Everything since Batch 2 — including Batch 3 SOTA-literature sweep, Batch 4 B+C tier, and Batch 5 tokenization-scale — delivered no promotable gain. Phase-C is the post-exhaustion queue: directions deliberately deferred during earlier batches because they are either high-cost, structurally different from anything tried so far, or blocked on another decision.

Phase-C hypotheses are **not** incremental encoder tweaks. Each one introduces a new information source, training protocol, or inductive bias. The expectation is either a ≥ +0.005 pp jump (novel lever) or a clean negative signal that retires the direction.

---

## Phase-C hypothesis catalog

Ordered by expected ceiling × feasibility, most-likely first.

### H5 — Stage-2 distillation (top priority)

**Lever:** teacher-student transfer from Stage-2 ParT to Stage-3 couple reranker.

**Why:** Stage-2 ParT produces track-level logits that Stage-3 currently reduces to a ranking (top-K2=60) and then discards. Those logits carry per-track signal that every other Batch 1–5 hypothesis ignored. If Stage-2 is better-calibrated on any slice than the aggregated pair features, distillation transfers that slice's accuracy into Stage-3 without changing Stage-3's architecture.

**Mechanism:** add a KL or MSE term between Stage-3's per-candidate score and a function of the two component tracks' Stage-2 logits. Two sub-variants:
- **H5a:** soft-target distillation — Stage-3's couple score regresses onto `sigmoid(s_i + s_j)` (Stage-2 logits sum).
- **H5b:** feature distillation — Stage-3 encoder's intermediate representation regresses onto Stage-2 ParT's penultimate embedding of each constituent track.

**Cost:** 1 sweep of 3 runs (v3 anchor, H5a, H5b), ~6 h total on one GPU. No new model code beyond the loss hook.

**Expected gain:** +0.003 to +0.010 pp full-val C@100 if Stage-2 uncertainty calibration is not degenerate. Strong negative signal possible if Stage-2 and Stage-3 see essentially the same information.

**Dependencies:** Stage-2 checkpoint with per-track logits exported (we already have this — `models/cascade_best.pt`).

**Risks:**
- Stage-2 and Stage-3 may be learning the same thing → distillation collapses.
- Distillation temperature + loss weight are two new hyperparameters; need a short sub-sweep.

---

### H19 — VICReg pretraining

**Lever:** self-supervised representation learning on couples before supervised ranking.

**Why:** v3's couple encoder is trained from scratch every run. VICReg (variance-invariance-covariance regularization) pretraining produces an encoder whose representations are structured before any label signal is seen. At our data scale (300k train events, 1770 couples each → ~5.3 × 10⁸ couples), self-supervision has more data than the supervised head sees per epoch.

**Mechanism:** two-stage training. Stage 1 pretrains the couple encoder with VICReg loss on unlabeled couples (anchor = couple, positive = same couple with feature-level augmentation — gaussian jitter on momentum / DCA / impact params). Stage 2 loads the pretrained encoder and trains the ranking head with supervised softmax-CE. Freeze vs. fine-tune encoder is a sub-axis.

**Cost:** 1 long pretrain run (~12 h) + fine-tune sweep of 3 runs (~4 h each). ~24 h total. New augmentation code (jitter), new loss module (VICReg components: variance, invariance, covariance).

**Expected gain:** +0.003 to +0.008 pp full-val. Stronger for higher-dimensional encoders; may compose with projected_infersent p=32.

**Dependencies:** jitter augmentation implementation + VICReg loss (straightforward, canonical implementation). No model changes.

**Risks:**
- Augmentation design is domain-dependent; wrong jitter scale destroys information.
- Pretrained encoder may not transfer cleanly if the supervised objective is too narrow.

---

### H11 — curriculum on event difficulty

**Lever:** train on easy events first, anneal to full distribution.

**Why:** current training is i.i.d. over all events. Hard events (many tracks, dense topology, close pions) dominate gradient noise. Curriculum pushes the optimization trajectory through a simpler landscape early before forcing convergence on the hard cases. This is orthogonal to every Batch 1–5 lever.

**Mechanism:** rank training events by difficulty (candidate — proxy: number of tracks, or D@60_tracks proxy, or Stage-2 confidence spread). Two variants:
- **H11a:** epoch-linear anneal — epoch e samples only events at or below difficulty quantile q(e) = e / E.
- **H11b:** SPL (self-paced learning) — weight each event's loss by `exp(−loss / T(e))` with annealed T.

**Cost:** 1 sweep of 3 runs (v3 anchor, H11a, H11b), ~4.5 h each. New dataloader sampler, trivial.

**Expected gain:** +0.002 to +0.005 pp full-val. Strongest for datasets with clear difficulty gradient, which ours has (track count ranges 200–1800).

**Dependencies:** difficulty estimator (pick one). Everything else reuses existing pipeline.

**Risks:**
- Wrong difficulty proxy can hurt more than help (seen in prefilter work).
- Training time increases because early epochs ignore hard events.

---

### H18 — Mixture-of-Experts encoder

**Lever:** sparse specialization per tau decay topology.

**Why:** tau → 3π events have two geometric modes (collimated vs. wider opening angle, determined by the tau boost). A single dense encoder averages these modes; MoE lets different experts handle different geometric regimes. The gating network routes each couple to the K most-relevant experts.

**Mechanism:** replace the projected_infersent encoder with a Switch-MoE layer (K=4 experts, top-1 routing). Gating network sees the same couple features as the encoder. Load-balancing auxiliary loss.

**Cost:** 1 sweep of 4 runs (v3 anchor, K=2, K=4, K=8 experts), ~5 h each. **Significant** model code — new MoE layer, gating network, auxiliary loss, expert capacity handling.

**Expected gain:** +0.002 to +0.006 pp full-val. Genuine ceiling only appears if there is actual modal structure in the data; if the distribution is unimodal, MoE collapses to dense and wastes params.

**Dependencies:** MoE layer implementation (~2 days of code), routing stability tests.

**Risks:**
- Expert collapse (one expert captures all tokens).
- Load balance instability at our batch sizes.
- Highest implementation cost in Phase-C.

---

### H17 — monotonic tower

**Lever:** structural prior that certain input features should monotonically increase the output score.

**Why:** physics features like pair-mass-consistency and vertex-χ² should have monotonic relationships with the couple score — smaller χ² ⇒ higher score. A monotonic tower (Lattice / partial-order constrained layer) embeds this prior as a hard constraint, reducing sample complexity.

**Mechanism:** add a monotonic branch (TensorFlow Lattice-style, or simpler PartiallyMonotonicMLP) alongside the current encoder. Combine via learned weight. Restrict monotonic inputs to a known-monotonic subset (pair_mass_consistency, delta_vtx_chi2, DCA-sig).

**Cost:** 1 sweep of 3 runs (v3 anchor, monotonic-branch-only, monotonic+encoder), ~5 h each. Moderate code — monotonic layer (no off-the-shelf PyTorch implementation; needs custom weight parametrization).

**Expected gain:** +0.001 to +0.003 pp full-val. Narrow because the prior only constrains part of the input space.

**Dependencies:** monotonic layer implementation.

**Risks:**
- Monotonic prior may be too strong; physics features are only approximately monotonic after detector effects.
- Custom layer = maintenance burden.

---

### H14 — manifold mixup (blocked)

**Lever:** interpolation in representation space as regularization.

**Why:** mixup in feature space is hard to make work with softmax-CE (soft labels require multi-positive target, which couples to loss reformulation). Batch 4 tried multi-positive via soft-OR (B4_3, −0.00032 train-time). Until a clean multi-positive loss is established, manifold mixup cannot compose without distorting the target distribution.

**Unblocking condition:** either (a) re-run B4_3 soft-OR with full-val eval and confirm it does not regress, then compose, or (b) switch to a different multi-positive target formulation that admits soft labels (e.g. continuous-label regression).

**Cost if unblocked:** 1 sweep of 3 runs, ~4 h each.

**Expected gain:** +0.001 to +0.004 pp full-val. Mixup historically helps more for classification than ranking.

**Deferred** until multi-positive path is resolved.

---

## Sequencing recommendation

**Phase-C.1 (start immediately):** **H5 Stage-2 distillation.** Lowest implementation cost, reuses existing Stage-2 checkpoint, orthogonal to everything tested, highest expected-gain / cost ratio.

**Phase-C.2 (conditional on C.1 outcome):**
- If H5 wins ≥ +0.003 pp → run **H19 VICReg** next. Two gains compose if VICReg pretrain improves representation quality independently of the distillation signal.
- If H5 is flat/negative → skip H19 and run **H11 curriculum** instead. Curriculum is the cheapest remaining orthogonal lever.

**Phase-C.3:** pick one of H18 / H17 based on whether the thesis narrative wants a "best system" (implement whichever is ahead) or an "explored-all-levers" story (do both). If budget-constrained, stop after C.2.

**Hard decision point** before Phase-C launches:
- **Gain target.** Is +0.005 pp full-val the target that justifies another month of sweeps, or is the thesis acceptable at v3 (0.7925)? If the latter, skip Phase-C entirely and spend time on third-pion completion + thesis writing instead.

---

## Acceptance protocol (all Phase-C runs)

Learned from Batch 5 reversal:
1. **Every** train-time win ≥ +0.001 pp gets full-val eval. No exceptions based on "it's small." Batch 5 showed ≤ 0.002 pp train-time is noise.
2. **Full-val C@100 Δ is the only promotion metric.** No rulings based on mean_rank, C@K at other K, or train-time.
3. **Promotion bar: ≥ +0.002 pp full-val C@100.** Anything below is within ensemble-swap-level noise (v2→v3 was +0.0011 and that was the clearest single-lever jump we've had; anything smaller is indistinguishable from seed noise).
4. **Sweep-local v3 anchor every batch.** Floats across GPU/driver versions + dataloader seed shifts; ~0.002 pp. Absolute train-time can drift, so the anchor re-establishes the reference.
5. **Full-val gate before combining levers.** Do not stack two train-time-only wins; one must have full-val confirmation first.

---

## Out-of-scope for Phase-C

These remain deferred indefinitely unless Phase-C reveals a new reason to revisit:

- **Further tokenization depth/width** — ruled out in Batch 5.
- **Further loss variants** — ruled out in Batches 1, 3, 4.
- **Further hard-negative schedules** — ruled out in Batches 1, 3.
- **Physics-feature expansion beyond v3** — B3_6 tau_phys_v3 and B4_9 pair_signif showed marginal wins; more features unlikely to move the needle without architectural support.
- **Wider / deeper MLP head** — T3_1 (B1) ruled this out.
- **EMA** — T2_3 (B1) ruled this out.

---

## Artifact locations

- Sweep summary: `part/reports/couple_reranker_sweep_summary.md`
- Batch 5 detail: `part/reports/couple_batch5_sweep.md`
- Batch 1–2 overview: `part/reports/couple_ranking_overview.md`
- Training entry point: `part/train_couple_reranker.py`
- Model: `weaver/weaver/nn/model/CoupleReranker.py`, `CoupleCascadeModel.py`
- v3 checkpoint (server): `models/couple_reranker_best.pt`
- Cascade (server): `models/cascade_best.pt`
