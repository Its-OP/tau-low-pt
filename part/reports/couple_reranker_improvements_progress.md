# CoupleReranker Improvement Sweep — Live Progress

Plan: `part/reports/couple-reranker-improvements.md`
Branch: `claude/server-setup`
Server: `ssh -p 26893 root@69.63.236.192`
Started: 2026-04-16

Records each experiment as it completes. Headline metric: **C@100_couples** (unconditional — fraction of events with ≥1 GT couple in top-100 reranked couples; denominator = total events).

Ceiling: **D@60_tracks** — fraction of events with ≥2 GT pions in the cascade's top-60 tracks. Measured during T0 and quoted against every subsequent result as `gap_closure = (C@100_new − C@100_baseline) / (D@60 − C@100_baseline)`.

## Experiment table

| ID | Branch / flags | Status | C@100 | D@60 | ΔC@100 vs T0 | gap_closure | Non-finite | Notes |
|---|---|---|---|---|---|---|---|---|
| T0 | `exp/baseline-k60` (seed=42, pairwise) | ✅ done (ep 58/60) | **0.8257** | **0.8596** | — (baseline) | — | 0 | Best @ ep 58. Val on 9.6K events (train-time subset). Stall at ep12-19, breakout ep31-58. |

## Baseline (T0) full metrics — epoch 58 (best C@100)

| Metric | Value |
|---|---|
| C@100_couples | 0.8257 |
| C@200_couples | 0.8497 |
| RC@100_couples | 0.7902 |
| RC@200_couples | 0.8096 |
| D@60_tracks (ceiling) | **0.8596** |
| D@100_tracks | 0.8997 |
| D@200_tracks | 0.9492 |
| mean first GT rank | 16.65 |
| eligible events | 8 250 / 9 600 |
| events with full triplet | 7 850 / 9 600 |

**Gap to close:** D@60 − C@100 = 0.8596 − 0.8257 = **0.0339 (3.39 pp)**.

**Sanity note:** the eval script (`diagnostics/eval_couple_reranker.py`) on a previous run produced C@100 ≈ 0.79 against the same checkpoint, below this 0.8257. The train-time validate samples only the first 9.6K events (`max_steps = steps_per_epoch // 2 × batch_size`) and doesn't shuffle, so the biased subset likely inflates the number vs the full 52.5K-event eval. Every later entry in this log reports the same train-time C@100 metric for apples-to-apples comparison; a separate "full-val eval" pass may follow later.

## Narrative

### T0 — Baseline (K2=60, seed=42, pairwise softplus) ✅
Command: `--top-k2 60 --epochs 60 --batch-size 96 --steps-per-epoch 200 --num-workers 10 --seed 42 --k-values-tracks 30 50 60 75 100 200 --k-values-couples 50 75 100 200`.

Trajectory of best-so-far C@100:
- ep 1 → 0.7943
- ep 7 → 0.8078 (first local peak, then 6-epoch stall)
- ep 19 → 0.8207 (breakout after LR hit peak)
- ep 32 → 0.8230
- ep 38 → 0.8235
- ep 46 → 0.8241
- ep 48 → 0.8249
- **ep 58 → 0.8257 (final best)**

The 6–8-epoch stall around ep 12–19 matches the known pairwise-softplus saturation pattern. Checkpoint at `experiments/tier0/T0_baseline_k60_20260416_191527/checkpoints/best_model_calibrated.pt`.

### T1.1 — Softmax-CE loss 🏃
*Starting now.* Same config as T0 with `--couple-loss softmax-ce`.
