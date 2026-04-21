# CoupleReranker C@K Improvement Sweep (K2=60 fixed)

## Context

Stage 3 couple reranker currently sits at C@100 ≈ 0.791 at the best prior sweep point. Goal: improve C@K with the cascade (prefilter + ParT) **untouched**.

**Metrics, as implemented in `part/utils/training_utils.py`:**

| Metric | Numerator | Denominator |
|---|---|---|
| `C@K_couples` | events with ≥1 GT couple in top-K ranked couples | **total events** |
| `RC@K_couples` | events with GT couple in top-K couples **and** full triplet in top-K1 tracks | **total events** |
| `D@K_tracks` | events with ≥2 GT pions in top-K tracks after Stage 2 | **total events** |

All three use the same denominator. Structural ordering (code docstring `training_utils.py:541-546`):

```
RC@K_couples ≤ C@K_couples ≤ D@K2_tracks   (when K_tracks ≥ K2)
```

So the reranker's ceiling is **D@K2_tracks** — the fraction of events where Stage 1 + Stage 2 put ≥2 of the 3 GT pions into the top-K2 tracks. We cannot exceed it. The "conditional" C@Kc reported in older analyses (`C@K × total / eligible`) is a post-hoc statistic, not a structural ceiling — do not target it.

**Target framing:** measure `D@60_tracks` during Tier 0, then every experiment reports `ΔC@100` (absolute) and `gap_closure = (C@100_new − C@100_baseline) / (D@60_tracks − C@100_baseline)`. A real improvement halves the gap in one Tier.

**Operating point: K2 = 60 tracks** → C(60, 2) = 1 770 candidate couples / event. Fixed. No K-sweep.

**User hard constraint:** per-couple input already encodes pairwise interaction. No cross-couple attention, graph-over-couples, or set-transformer proposals.

**Wall-time policy:** every experiment below is expected to run regardless of training time. Rough wall-times from the `sweep_topk2_20260408_205250` log are listed per experiment for scheduling context only — they are never kill criteria.

Prior NaN-BatchNorm plan is deployed (code audit: `NanSafeBatchNorm1d` at `weaver/weaver/nn/model/CoupleReranker.py:34-64`, `calibrate_reranker_bn` at `part/train_couple_reranker.py:257-319`). This document supersedes it.

**Execution target:** vast.ai server (RTX PRO 6000 Blackwell Server Edition, 102 GB VRAM, branch `claude/server-setup`, workspace `/workspace/tau-low-pt/`). Reranker is tiny (~581 K params, ~7 MB checkpoints); disk is not a concern.

## Reference wall-times (informational; pulled from `sweep_topk2_20260408_205250`, 50 epochs, batch 96)

| K2 | C(K,2) | Total min (50 ep) | sec/epoch |
|---|---|---|---|
| 50 | 1 225 | 22 | 26.4 |
| **60** | **1 770** | **22** | **26.4** |
| 70 | 2 415 | 24 | 28.8 |
| 100 | 4 950 | 28 | 33.6 |

At K2=60, one full 60-epoch run ≈ 26 min.

Smoke-test budget on GPU: ≤ 30 s. Use `--epochs 1 --steps-per-epoch 3 --batch-size 16 --num-workers 0 --top-k2 60` purely as a wiring check.

## Architecture baseline (unchanged except where flagged)

- 51-dim couple input = 32 [track_i ‖ track_j] + 10 pairwise physics + 5 derived + 4 cascade scores.
- `Conv1d(51→256) + NanSafeBN + ReLU → 4 × ResidualBlock(256, dropout=0.1) → Conv1d(256→128) + BN + Dropout + Conv1d(128→1)`. ~581 K params.
- Loss: pairwise softplus `L = T · softplus((s_neg − s_pos)/T)`, T=1.0, 50 random negatives per positive per event (`CoupleReranker.compute_loss`).
- Training: AdamW lr=5e-4 wd=0.01, cosine+warmup (5 %), grad-clip 1.0, `--cosine-power` wired.

## BN / calibration compatibility

| Change class | Safe for `calibrate_reranker_bn`? |
|---|---|
| Add features (bump `COUPLE_FEATURE_DIM`) | Yes — update `input_dim` default in `lowpt_tau_CoupleReranker.get_model` and constructor. |
| Replace loss (ListMLE, label smoothing, hard-neg) | Yes — loss never touches BN; preserve `_scores / _couple_labels / _couple_mask` keys. |
| Tune T, cosine-power, negatives per positive | Yes — CLI flags already wired. |
| EMA weights | Yes **iff** BN recalibration runs on the EMA copy *after* EMA freeze. |

## Tier 0 — Baseline + ceiling measurement

Run once, lock-in.

- Branch: `exp/baseline-k60`.
- Seeds: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` at top of `main()` in `part/train_couple_reranker.py`; seed the `torch.randint` call inside `CoupleReranker.compute_loss` via a generator.
- Run: `--top-k2 60 --epochs 60 --batch-size 96`.
- Record: C@1, C@50, C@100, C@200, RC@100, RC@200, non-finite count, **D@60_tracks**. D@60_tracks becomes the ceiling every other experiment reports against.

Kill: `D@60_tracks − C@100_baseline < 0.02` → negligible ranking headroom, work must shift to prefilter / ParT layer. Otherwise proceed.

## Tier 1 — Loss + schedule knobs

Expected individual lifts from literature; assume ~60 % sub-additive stacking. Reported gains are **ΔC@100** vs Tier 0.

| # | Branch | Hypothesis | Code change | Success | ~Wall | Deps | Kill |
|---|---|---|---|---|---|---|---|
| T1.1 | `exp/listmle-loss` | Listwise softmax-CE / ListMLE beats pairwise softplus in "1 positive per event" settings (Bruch 2019; allRank). Also removes the epoch-26–48 stall (softplus saturates once pos beats random negs). | Add `--couple-loss {pairwise, softmax-ce}` flag. In `CoupleReranker.compute_loss` new branch: `loss_e = −s_pos + logsumexp([s_pos, s_negs])` per positive, mean over events with ≥1 positive. Keep dict contract. | ΔC@100 ≥ +1.4 pp | 26 min | T0 | Val NaN or C@100 < T0 @ epoch 30 → revert. |
| T1.2 | `exp/temp-tune` | T≠1 reshapes the softmax; never swept. | Sweep `--couple-ranking-temperature ∈ {0.25, 0.5, 1.0, 2.0}` against T1.1 winner. No code. | Best T ≥ +0.3 pp over T=1 | 4 × 26 min | T1.1 | All T within ±0.1 pp → pick 0.5, continue. |
| T1.3 | `exp/label-smoothing` | ε on softmax target adds +0.5–1 pp (Müller 2019); stacks with T1.1. | Blend target `(1-ε)·onehot + ε/N` in softmax-CE branch. Flag `--label-smoothing`. | +0.3 pp over T1.2 | 26 min | T1.1 | All ε ≥ 0 worse → skip. |
| T1.4 | `exp/cosine-power` | Already-deployed knob; tune to delay LR drop past the stall. | Sweep `--cosine-power ∈ {0.5, 1.5, 2.0}` against best so far. No code. | +0.3 pp | 3 × 26 min | T1.3 | No power beats 1.0 → drop. |

## Tier 2 — Sampling + features + EMA

| # | Branch | Hypothesis | Code change | Success | ~Wall | Deps | Kill |
|---|---|---|---|---|---|---|---|
| T2.1 | `exp/hard-neg-mining` | ANCE / NV-Retriever: online top-M negatives beat random by +2–4 pp; positive-aware `score < s_pos − margin` filter controls false-negative risk. Forward already scores every couple, so mining is near-zero cost. | In `compute_loss`: sort valid negatives desc, take top-M (default 25), drop any with `score ≥ s_pos − margin`, fill with random to 50 total. Flags `--hardneg-fraction --hardneg-margin`. | +1.0 pp over Tier 1 | 26 min | T1 winner | Margin filter drops >50 % of top-M → margin too tight. C@100 < T1 winner @ epoch 20 → revert. |
| T2.2 | `exp/pair-kinematics-v2` | Four extra per-pair features standard in CMS tau-ID: opening-angle `cos(θ)`, `(m_ij − m_τ)/σ_m`, `dxy_sig_i · dxy_sig_j`, `dz_sig_i · dz_sig_j`. | Extend `build_couple_features_batched` in `part/utils/couple_features.py` (keep non-batched in sync). Bump `COUPLE_FEATURE_DIM` 51→55. Update `input_dim=55` default in `lowpt_tau_CoupleReranker.get_model` and constructor. | +0.8 pp | 30 min | T2.1 | Added feature val std < 1e-6 → drop it. Overall lift < T2.1 alone → revert. |
| T2.3 | `exp/ema-weights` | +0.3–1 pp, orthogonal. | Wrap `model.couple_reranker` with `torch.optim.swa_utils.AveragedModel`, decay 0.999, update every step. Run `calibrate_reranker_bn` on the EMA copy **after** training. Save `best_model_ema_calibrated.pt`. | +0.3 pp | 26 min | Tier 1 | EMA worse than live by > 0.002 → drop. |

## Tier 3 — Exploratory

Run these regardless of whether Tier 1+2 "already met target" — they may deliver further lift.

| # | Branch | Hypothesis | Success | ~Wall |
|---|---|---|---|---|
| T3.1 | `exp/wider-deeper` | `--couple-hidden-dim 384 --couple-num-residual-blocks 6` (~1.3 M params) may close an underfit gap surfaced by the better loss. | +0.5 pp; train-val gap ≤ 2 × Tier 0 | ~45 min |
| T3.2 | `exp/in-batch-neg` | DPR-style: negatives pooled across events in a batch. Per-couple scoring preserved; requires vectorizing the per-event loss loop. | +0.5 pp | ~40 min |
| T3.3 | `exp/swa` | SWA over last 20 % epochs. Complementary to T2.3 only if EMA was weak. | +0.2 pp | 26 min |

## DAG

```
T0 baseline-k60 (measures D@60_tracks)
   └─> T1.1 listmle ──> T1.2 temp ──> T1.3 smooth ──> T1.4 cosine-power
                                                          │
                                                          ├─> T2.1 hardneg ──> T2.2 pair-feats
                                                          └─> T2.3 ema
                                                                  │
                                                                  ├─> T3.1 wider-deeper
                                                                  ├─> T3.2 in-batch-neg
                                                                  └─> T3.3 swa
```

## Explicitly off the list

- Cross-couple attention / graph-over-couples / set-transformer — user constraint.
- K-sweep as baseline work — K2=60 is fixed.
- Combinatorial triplet post-processing — ruled out (<10 % GT survival).
- Boundary-negative sampling, ASL, OHEM — collapsed recall in prior experiments.
- Pairwise features inside Stage 1 message passing — latent collapse.
- RankMixup — weak evidence.
- Targeting C@Kc — analysis stat, not a structural ceiling; absolute C@K is the north star.

## Files modified (canonical paths in the `claude/server-setup` worktree)

| File | Used by |
|---|---|
| `part/train_couple_reranker.py` | T0 (seeds), T1.1 (`--couple-loss`), T1.3 (`--label-smoothing`), T2.1 (`--hardneg-*`), T2.3 (EMA + BN-recalibration ordering). |
| `weaver/weaver/nn/model/CoupleReranker.py` | T1.1 (softmax-CE branch), T1.2 (temperature), T2.1 (hard-neg sampling). |
| `part/utils/couple_features.py` | T2.2 (four new pair-kinematic features, `COUPLE_FEATURE_DIM` 51→55). |
| `part/networks/lowpt_tau_CoupleReranker.py` | T2.2 (`input_dim=55` default). |
| `part/sweep_topk2.sh` | Temperature (T1.2) and cosine-power (T1.4) sweeps at fixed K2=60. |
| `part/tests/test_couple_reranker.py` | TDD per new loss branch / per new feature. |

## Verification checklist (per experiment)

1. **TDD** — tests under `part/tests/` first; `python -m pytest tests/ -q` must show 457+ passed unchanged.
2. **Smoke test** (≤ 30 s GPU) — `python part/train_couple_reranker.py --epochs 1 --steps-per-epoch 3 --batch-size 16 --num-workers 0 --top-k2 60 <flags>`. Must finish cleanly, save a BN-calibrated checkpoint with finite running stats, print final C@100.
3. **Full run** — `--top-k2 60 --epochs 60 --batch-size 96 <flags>`.
4. **Record** — append to `part/reports/experiment_log.md`: branch, hypothesis, C@100, D@60_tracks, ΔC@100, gap_closure, non-finite count, verdict.

## Global kill

If Tier 1 + Tier 2 cumulative gap_closure < 30 %, stop and reassess — the ceiling is set by Stage 1 / Stage 2 recall and future work belongs at the prefilter / ParT layer.
