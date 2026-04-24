# Batch 6 — Position-Aware Couple-Reranker Loss Sweep

**Date:** 2026-04-22
**Server:** vast.ai port 26246
**Sweep roots:**
- `experiments/couple_batch6_20260421_223538/` — B0, B6_1, B6_2, B6_3 (first attempt)
- `experiments/couple_batch6_20260422_080428/` — B6_3 rerun after fp16 overflow fix

---

## 1. Motivation

Apr-14 `topk2_sweep` evidence: as couple-reranker input K grows from 50 → 200, deployed softmax-CE **loss halves** (0.082 → 0.041) while **mean first-GT-couple rank climbs 7×** (14 → 96) and C@100 barely moves (+0.005 pp). The loss is anti-correlated with ranking quality at large K.

Root cause: softmax-CE picks 50 random negatives per positive. At K=200 the candidate pool is ~19,900 couples; random draws almost always hit trivially-easy negatives. GT dominates them → loss → 0 even when hard competitors still rank above GT.

Batch 6 tests three orthogonal fixes at fixed K=60:

| Run | Loss + knobs | Mechanism targeted |
|---|---|---|
| `B0_v3_anchor` | v3 softmax-CE (defaults) | re-anchor |
| `B6_1_hardneg_full` | v3 softmax-CE + `hardneg_fraction=1.0` | random-negative failure mode |
| `B6_2_lambda_noK` | `lambda_ndcg2pp` + `ndcg_k=0` + σ=3 + ideal-DCG norm | fix existing rank-aware loss |
| `B6_3_approx_ndcg` | new `approx_ndcg` (Qin et al. 2010) + α=5 | principled smooth NDCG |

Shared config: K2=60, 60 epochs, 200 steps/ep, bs=256, AMP, SOAP→AdamW (lr 5e-4, cosine², wd 1e-2, grad_clip 1.0), seed 42, `projected_infersent p=32`, hidden_dim 256, 4 residual blocks, dropout 0.1, label_smoothing 0.1, cosine_power 2.0, BN calibration 200 steps, cascade = `cascade_oldprefilter_stage2_20260421` (old prefilter_best.pt + retrained stage2_best.pt).

---

## 2. Results

### 2.1 Final metrics

| Run | best C@100c | epoch | mean_rank_train (avg ep 56–60) | ρ(loss,rank) (avg ep 56–60) | gate 1 C@100 ≥ v3+0.002 | gate 3 ρ > 0.3 |
|---|---:|---:|---:|---:|:---:|:---:|
| B0_v3_anchor | **0.8286** | 58 | 17.6 | +0.569 | — (anchor) | pass |
| B6_1_hardneg_full | 0.8188 | 58 | 18.9 | +0.203 | **fail** (−0.0098) | **fail** |
| B6_2_lambda_noK | 0.8218 | 25 | 19.9 | **+0.862** | **fail** (−0.0068) | pass |
| B6_3_approx_ndcg | 0.8123 | 58 | 21.9 | +0.353 | **fail** (−0.0163) | pass |

Numbers are the training-harness `C@100c` (val-subset with GT filter), not full-val C@100. Full-val gate evaluation pending.

### 2.2 Key observations

1. **No candidate beat the v3 anchor on C@100c.** Gate 1 fails for all three.
2. **v3 softmax-CE is already position-aware** at K=60 (ρ=+0.57), contradicting the initial hypothesis that ρ≈0 for softmax-CE. The random-negative failure mode is K-dependent: at K=60 the loss still tracks rank.
3. **B6_1 (hard-neg mining) regressed ρ to +0.20** — feeding only hard competitors starved the loss of the easy-negative gradient that anchors the decision boundary. The loss became noisier, C@100 dropped.
4. **B6_2 (λ-NDCG, no K truncation, σ=3, ideal-DCG norm)** produced the strongest position-awareness signal (ρ=+0.86) but C@100 regressed by 0.007 pp. Val loss plateau hit at epoch 25 (no new best afterwards) — the loss saturates before the model fully exploits rank geometry.
5. **B6_3 (ApproxNDCG α=5)** learned a position-aware signal (ρ=+0.35) but with higher mean_rank (21.9) and worst C@100c (0.8123). The α=5 smoothing is too soft — sigmoid saturates for the wide score distribution the cascade produces; effective learning signal concentrates on the top ~10 ranks and weak discrimination beyond.

### 2.3 Interpretation

- The Apr-14 anti-correlation effect is **real but only manifests at K≥100**. At K=60 (production operating point) softmax-CE is already position-aware. Fix mechanism does not apply when the problem does not exist.
- Direct rank-aware losses (λ-NDCG, ApproxNDCG) produce much stronger ρ but at the cost of C@K — the optimization surface changes and the BN-calibrated classification head lands in a different minimum.
- **K-robustness gate 4 not yet evaluated.** A position-aware loss may still win if `mean_rank(K=200)/mean_rank(K=60) < 3.0` while v3 sits at 5.6. That is the untested acceptance path.

---

## 3. Files downloaded

`part/experiments/batch6_archive/`:
- `B0_v3_anchor/{best_model_calibrated.pt, training.log, loss_curves.png, loss_history.json}`
- `B6_1_hardneg_full/{…}`
- `B6_2_lambda_noK/{…}`
- `B6_3_approx_ndcg/{…}`
- `sweep_first.log`, `sweep_rerun_B6_3.log`

---

## 4. Follow-ups

1. **Run `diagnostics/k_robustness_eval.py --ks 60 100 150 200`** on all 4 checkpoints. If B6_2 ratio < 3.0, the rank-aware loss wins gate 4 even without gate 1 — promotion path via K-robustness.
2. **Full-val eval via composite-key-fixed `diagnostics/eval_couple_reranker.py`** on B0 + B6_2 (+ B6_3 if ratio interesting) to confirm C@100 hierarchy on the real test distribution (training-harness C@100c diverged from full-val C@100 by ~0.035 pp historically).
3. **If K-robustness gate passes for B6_2:** deploy as v4; otherwise v3 stands.
4. **σ sweep for λ-NDCG** (σ ∈ {1, 5, 10}) — current σ=3 may be off-optimum; the rank-weighting sharpness directly shapes gate-1 C@K.
5. **Adaptive-α ApproxNDCG** — schedule α from 2→10 during training to avoid sigmoid saturation early.
6. **Revisit at higher K.** If production ever moves K=60 → K=100+, rerun the whole sweep — the loss anti-correlation may return.

---

## 5. Reproduction

```bash
bash sweep_couple_batch6.sh  # overnight, screen
EXPERIMENTS="B6_2_lambda_noK" bash sweep_couple_batch6.sh  # single run
SMOKE_MODE=1 bash sweep_couple_batch6.sh                   # 1-epoch smoke
```

CLI per run in the `Arguments:` line of each `training.log`.
