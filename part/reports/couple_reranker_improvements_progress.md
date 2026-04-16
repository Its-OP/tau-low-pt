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
| T0 | `exp/baseline-k60` (seed=42, pairwise) | 🏃 in progress | — | — | — | — | — | Training started epoch 1 @ 19:15 UTC |

## Narrative

### T0 — Baseline (K2=60, seed=42, pairwise softplus)
*Started* 2026-04-16 19:15 UTC. Command: `--top-k2 60 --epochs 60 --batch-size 96 --steps-per-epoch 200 --num-workers 10 --seed 42 --k-values-tracks 30 50 60 75 100 200 --k-values-couples 50 75 100 200`.

Early epoch trajectory (from `/workspace/tier0.log`):
- Epoch 1: C@100=0.7943
- Epoch 2: C@100=0.8002
- Epoch 3: C@100=0.8019
- Epoch 4: C@100=0.8046

Training still climbing. Results section will be updated when the run completes.
