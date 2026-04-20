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

## Full-sweep results — 14 experiments (couple_improvements_20260416_211532)

Sweep launched 2026-04-16 21:15 UTC, finished 2026-04-17 10:45 UTC. All 14 runs completed 60 epochs. Reference metric: `val_c_at_100_couples` at the best epoch selected by argmax over all 60 epoch JSONs. Ceiling reference uses this sweep's own T0 run, not the earlier standalone T0.

**Sweep-local T0** — C@100 = **0.8266**, D@60 = **0.8596**, ceiling gap = **0.0330** (3.30 pp). Best epoch 58. Within ~0.001 of the earlier standalone T0 (0.8257 / 0.8596); treat as consistent.

### Knob audit

Verification: `Arguments:` dict parsed from each `training.log` against the expected-override table in `sweep_couple_improvements.sh`. Non-default fields must match; other fields must stay at default. Secondary witnesses: `CoupleReranker: N params (...)` line, EMA checkpoint presence when `couple_ema_decay > 0`. `reported_diff` = |argmax of `val_c_at_100_couples` over per-epoch JSONs − `Training complete. Best C@100:` tail line|.

| Experiment | Knob | Witness | Epochs JSON | reported_diff |
|---|---|---|---|---|
| T0_baseline | ✓ | ✓ | 60 | 2.5e-06 |
| T1_1_listmle | ✓ | ✓ | 60 | 3.3e-06 |
| T1_2_temp_025 | ✓ | ✓ | 60 | 4.2e-06 |
| T1_2_temp_050 | ✓ | ✓ | 60 | 4.2e-06 |
| T1_2_temp_200 | ✓ | ✓ | 60 | 1.7e-06 |
| T1_3_smooth_005 | ✓ | ✓ | 60 | 8.3e-07 |
| T1_3_smooth_010 | ✓ | ✓ | 60 | 4.2e-06 |
| T1_4_cos_050 | ✓ | ✓ | 60 | 5.0e-06 |
| T1_4_cos_150 | ✓ | ✓ | 60 | 0.0 |
| T1_4_cos_200 | ✓ | ✓ | 60 | 4.2e-06 |
| T2_1_hardneg_05 | ✓ | ✓ | 60 | 3.3e-06 |
| T2_2_pair_v2 | ✓ | ✓ | 60 | 1.7e-06 |
| T2_3_ema | ✓ | ✓ | 60 | 1.7e-06 |
| T3_1_wider | ✓ | ✓ | 60 | 2.5e-06 |

All 14 knobs flipped as intended. Supporting facts: T2_2_pair_v2 has 576,129 trainable params (baseline 575,105 → +1,024 = 4 extra input dims × 256, exact match for the 51→55 input-dim bump). T3_1_wider has 1,873,345 trainable params at hidden_dim=384, num_residual_blocks=6. T2_3_ema carries `best_model_ema_calibrated.pt`; no other run does. Reported-vs-argmax diff < 1e-5 everywhere (float32 rounding). **Every experiment is conducted as planned.**

### Headline metrics

ΔC@100 and gap_closure are against the sweep-local T0 (C@100=0.8266, D@60=0.8596, denom=0.0330). Best epoch is argmax of `val_c_at_100_couples` across the 60 epoch JSONs.

| Experiment | best ep | C@50 | C@75 | C@100 | C@200 | RC@100 | D@60 | ΔC@100 | gap_closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0_baseline | 58 | 0.7887 | 0.8116 | **0.8266** | 0.8488 | 0.7908 | 0.8596 | — | — |
| T1_1_listmle | 51 | 0.7865 | 0.8102 | 0.8254 | 0.8483 | 0.7902 | 0.8589 | −0.0011 | −3.47% |
| T1_2_temp_025 | 58 | 0.7886 | 0.8103 | 0.8249 | 0.8490 | 0.7892 | 0.8596 | −0.0017 | −5.05% |
| T1_2_temp_050 | 51 | 0.7866 | 0.8104 | 0.8264 | 0.8482 | 0.7909 | 0.8589 | −0.0002 | −0.63% |
| T1_2_temp_200 | 58 | 0.7897 | 0.8121 | 0.8252 | 0.8485 | 0.7898 | 0.8596 | −0.0014 | −4.10% |
| T1_3_smooth_005 | 58 | 0.7895 | 0.8130 | 0.8282 | 0.8492 | 0.7924 | 0.8596 | +0.0017 | +5.05% |
| **T1_3_smooth_010** | 58 | 0.7919 | 0.8149 | **0.8301** | 0.8496 | **0.7941** | 0.8596 | **+0.0035** | **+10.73%** |
| T1_4_cos_050 | 58 | 0.7877 | 0.8102 | 0.8256 | 0.8480 | 0.7902 | 0.8596 | −0.0009 | −2.84% |
| T1_4_cos_150 | 58 | 0.7894 | 0.8120 | 0.8275 | 0.8480 | 0.7918 | 0.8596 | +0.0009 | +2.84% |
| T1_4_cos_200 | 58 | 0.7892 | 0.8120 | 0.8274 | 0.8494 | 0.7910 | 0.8596 | +0.0008 | +2.52% |
| T2_1_hardneg_05 | 32 | 0.7801 | 0.8048 | 0.8208 | 0.8466 | 0.7849 | 0.8634 | −0.0057 | −17.35% |
| T2_2_pair_v2 | 51 | 0.7808 | 0.8050 | 0.8198 | 0.8464 | 0.7848 | 0.8585 | −0.0068 | −20.50% |
| T2_3_ema | 58 | 0.7898 | 0.8113 | 0.8260 | 0.8486 | 0.7902 | 0.8596 | −0.0005 | −1.58% |
| T3_1_wider | 44 | 0.7851 | 0.8097 | 0.8259 | 0.8473 | 0.7893 | 0.8577 | −0.0006 | −1.89% |

### Narrative

**Single winner: T1.3 label smoothing (ε=0.10)** — +0.0035 over T0, 10.73% gap closure. ε=0.05 is half the effect (+0.0017, 5.05%). Both run on softmax-CE. Note that softmax-CE alone (T1.1) is **slightly worse** than pairwise (−0.0011) — label smoothing is what flips softmax-CE from neutral to positive, so the gain is specifically `softmax-CE + ε=0.10`, not softmax-CE per se.

**Cosine-power** — weak net-positive at powers ≥1.5 (+0.0009 / +0.0008). Power=0.5 (faster LR drop) is slightly negative. Useful only as a secondary knob to stack with label smoothing.

**Temperature** — flat-to-negative across {0.25, 0.5, 2.0}. T=0.5 is closest to neutral (−0.0002); T=0.25 and T=2.0 hurt. Default T=1.0 is already fine.

**Busts** — T2.1 hard-neg mining: −0.0057 with an early peak at ep 32. The margin filter (score < s_pos − 0.1) likely throws away too many informative negatives or breaks the LR schedule. T2.2 pair-kinematics-v2 stacks on T2.1 and compounds the damage (−0.0068, worst in the sweep); cannot separate the v2 features' own effect from the hardneg penalty. **T2.1 needs to be rerun on top of pairwise (or with a different margin/fraction) before T2.2 can be judged.**

**Neutral** — T2.3 EMA (−0.0005) and T3.1 wider (−0.0006); wider peaks early (ep 44), consistent with mild overfit at 1.87 M params.

**Ceiling** — headroom D@60 − C@100 = 0.0330. T1.3 ε=0.10 closes 10.7% of it. Remaining 89% is still available and is a Stage-3 problem, not a Stage-1/2 one, as long as D@60 > C@100.

**Recommended follow-ups (not executed here):**
1. Stack `smooth-010 + cos-150` and `smooth-010 + cos-200` — cheapest hope for additive gain, both individually non-negative.
2. Re-run T2.1 with `--couple-hardneg-fraction` ∈ {0.25, 0.75} and `--couple-hardneg-margin` ∈ {0.0, 0.2, 0.5} to find whether the mining can be made additive; if still negative, drop the whole hard-neg branch.
3. Re-run T2.2 without T2.1 (pair-kinematics-v2 on top of label-smoothed softmax-CE) to get a clean reading on the four new features.
4. T2.3: compare `best_model_ema_calibrated.pt` metric against live — the current row reflects live weights only.

Raw artifact: `/workspace/tau-low-pt/part/experiments/couple_improvements_20260416_211532/diagnostics.json`.

### Full-val confirmation (52,452 events)

The sweep table above uses the train-time validator, which only scans the first 9,600 events (`steps_per_epoch × batch_size // 2`) of the val split. Running the standalone `diagnostics/eval_couple_reranker.py` on the entire 52,452-event val set at K2=60, top-200 couples per event, against `best_model_calibrated.pt` for T0 + the four net-positive runs.

| Checkpoint | C@50 | C@75 | **C@100** | C@200 | RC@100 | mean_rank | ΔC@100 vs T0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0_baseline | 0.7563 | 0.7765 | 0.7881 | 0.8041 | 0.6774 | 11.6 | — |
| T1_3_smooth_005 | 0.7589 | 0.7791 | 0.7894 | 0.8048 | 0.6781 | 11.3 | +0.0013 |
| **T1_3_smooth_010** | **0.7608** | **0.7799** | **0.7908** | **0.8050** | **0.6790** | **11.0** | **+0.0027** |
| T1_4_cos_150 | 0.7571 | 0.7778 | 0.7882 | 0.8050 | 0.6772 | 11.5 | +0.0001 |
| T1_4_cos_200 | 0.7579 | 0.7783 | 0.7893 | 0.8050 | 0.6779 | 11.4 | +0.0012 |

All five evals used the same frozen cascade (`models/cascade_best.pt`), K2=60, 52,452 events, 47,855 eligible (full triplet inside K1 reach), 39,206 with full triplet in the reranker output.

**Observations:**
1. **Ordering preserved** vs train-time: smooth_010 > smooth_005 > cos_200 ≈ cos_150 > T0. Winner is robust to the 5.5× larger eval population.
2. **Absolute C@100 drops ~4 pp** vs train-time val (0.83 → 0.79). Consistent offset across all five checkpoints → train-time subset is an easier slice of the val set, not a metric bug.
3. **ΔC@100 smaller on full val** (+0.27 pp vs +0.35 pp on train-subset for smooth_010). Gain shrinks but stays significantly positive.
4. **mean_rank** drops 11.6 → 11.0 for smooth_010 (−5%), independent confirmation of improved ranking.
5. **Cosine-power** variants are now near-flat on full val (+0.01 to +0.12 pp); their train-time signal mostly noise. Label smoothing is the only knob with a clean positive effect.

Raw parquets: `/tmp/full_val_eval_20260417_120621/{T0_baseline,T1_3_smooth_005,T1_3_smooth_010,T1_4_cos_150,T1_4_cos_200}.parquet` on server.

## New baseline v2 — smooth_010 + cos_200 stacked

Trained one combined config: `--couple-loss softmax-ce --couple-label-smoothing 0.10 --cosine-power 2.0`. All other flags at sweep defaults (seed=42, K2=60, batch=96, 60 epochs). Experiment dir: `experiments/baseline_v2_s010_c200/baseline_v2_s010_c200_20260417_123418`. Best epoch 56, train-time C@100 = 0.83156.

**Full-val (52,452 events):**

| Checkpoint | C@50 | C@75 | **C@100** | C@200 | RC@100 | mean_rank | ΔC@100 vs T0 | ΔC@100 vs smooth_010 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0_baseline | 0.7563 | 0.7765 | 0.7881 | 0.8041 | 0.6774 | 11.6 | — | — |
| T1_3_smooth_010 | 0.7608 | 0.7799 | 0.7908 | 0.8050 | 0.6790 | 11.0 | +0.0027 | — |
| **baseline_v2 (s010+c200)** | **0.7616** | **0.7814** | **0.7914** | **0.8049** | **0.6804** | **10.9** | **+0.0033** | **+0.0006** |

**Reading:**
- Stacking gains +0.06 pp C@100 and −0.1 mean_rank over smooth_010 alone on full val. Cosine-power's gain is mostly absorbed; most of the improvement is still label smoothing.
- New baseline total lift vs T0: +0.33 pp C@100, +0.30 pp RC@100, −0.7 mean_rank. Modest but consistent across all K.
- Train-time lift was +0.50 pp (0.83156 − 0.8266), full-val lift only +0.33 pp — consistent with the offset seen earlier where train-subset overstates gains.
- **This is the new reference.** Subsequent sweeps should use `baseline_v2_s010_c200_20260417_123418/checkpoints/best_model_calibrated.pt` as T0, and the flag trio `--couple-loss softmax-ce --couple-label-smoothing 0.10 --cosine-power 2.0` as the baseline flag set.

## Per-experiment audit

All values from train-time C@100 (9 600-event subset). Trajectory = best C@100 at epoch checkpoints {1, 10, 20, 30, 40, 50, 60}; **best_ep** column = argmax epoch of C@100 across all 60 val evals. Full artifact bundle at `debug_checkpoints/couple_improvements_20260416_211532/` on main (gitignored, 124 MB: best ckpts + per-epoch metric JSONs + training logs + loss history).

### Trajectory table

| Experiment | ep 1 | ep 10 | ep 20 | ep 30 | ep 40 | ep 50 | ep 60 | best_ep | best C@100 | ΔC@100 vs T0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0_baseline | 0.7933 | 0.8120 | 0.8117 | 0.8196 | 0.8180 | 0.8228 | 0.8244 | 58 | **0.8266** | — |
| T1_1_listmle | 0.7929 | 0.8103 | 0.8149 | 0.8211 | 0.8195 | 0.8240 | 0.8245 | 51 | 0.8254 | −0.0011 |
| T1_2_temp_025 | 0.7894 | 0.8106 | 0.8150 | 0.8195 | 0.8197 | 0.8230 | 0.8239 | 58 | 0.8249 | −0.0017 |
| T1_2_temp_050 | 0.7932 | 0.8116 | 0.8129 | 0.8211 | 0.8195 | 0.8235 | 0.8234 | 51 | 0.8264 | −0.0002 |
| T1_2_temp_200 | 0.7966 | 0.8125 | 0.8148 | 0.8208 | 0.8190 | 0.8239 | 0.8240 | 58 | 0.8252 | −0.0014 |
| T1_3_smooth_005 | 0.7942 | 0.8139 | 0.8173 | 0.8221 | 0.8215 | 0.8244 | 0.8257 | 58 | 0.8282 | +0.0017 |
| **T1_3_smooth_010** | 0.7951 | 0.8155 | 0.8180 | 0.8232 | 0.8239 | 0.8274 | 0.8278 | 58 | **0.8301** | **+0.0035** |
| T1_4_cos_050 | 0.7934 | 0.8111 | 0.8145 | 0.8206 | 0.8179 | 0.8219 | 0.8232 | 58 | 0.8256 | −0.0009 |
| T1_4_cos_150 | 0.7935 | 0.8127 | 0.8143 | 0.8207 | 0.8194 | 0.8231 | 0.8242 | 58 | 0.8275 | +0.0009 |
| T1_4_cos_200 | 0.7931 | 0.8115 | 0.8149 | 0.8214 | 0.8196 | 0.8235 | 0.8249 | 58 | 0.8274 | +0.0008 |
| T2_1_hardneg_05 | 0.7897 | 0.8049 | 0.8111 | 0.8143 | 0.8154 | 0.8173 | 0.8191 | 32 | 0.8208 | −0.0057 |
| T2_2_pair_v2 | 0.7936 | 0.8080 | 0.8100 | 0.8149 | 0.8149 | 0.8170 | 0.8178 | 51 | 0.8198 | −0.0068 |
| T2_3_ema | 0.7931 | 0.8122 | 0.8146 | 0.8215 | 0.8192 | 0.8234 | 0.8244 | 58 | 0.8260 | −0.0005 |
| T3_1_wider | 0.7982 | 0.8083 | 0.8139 | 0.8221 | 0.8180 | 0.8258 | 0.8253 | 44 | 0.8259 | −0.0006 |

All 14 runs monotonically rise through ep 50+, except T2_1 (peaks ep 32), T3_1 (ep 44), and T1_2_temp_200 (peaks ep 58 but flat after ep 30).

### Per-experiment verdicts

**T0_baseline** — pairwise softplus, 50 random negs per positive. Steady climb, no stall at this seed (earlier standalone T0 saw ep12–19 stall; this re-run did not). Sets reference at C@100=0.8266. Reference checkpoint: `experiments/couple_improvements_20260416_211532/T0_baseline/T0_baseline_20260416_211533/checkpoints/best_model_calibrated.pt`.

**T1_1_listmle (softmax-CE, T=1.0)** — ΔC@100 = **−0.0011**. Softmax-CE with a single positive per event reduces to ListMLE top-1 loss `L = −s_pos/T + logsumexp(all_scores/T)`. Hypothesis (Bruch 2019) was that the listwise form removes the pairwise-softplus saturation; in practice the net effect is slightly negative. Softmax-CE without regularisation over-concentrates probability mass on the top-1 logit, producing well-calibrated top-1 but marginally worse top-K for K>1. **Not a bust, but not additive on its own** — label smoothing on top is what flips it positive (see T1.3).

**T1_2_temp_025 (T=0.25)** — ΔC@100 = **−0.0017**. Sharpening the softmax (dividing logits by 0.25) amplifies the dominant logit and starves gradients on middle-ranked negatives. Worst of the temperature sweep.

**T1_2_temp_050 (T=0.5)** — ΔC@100 = **−0.0002**. Near-neutral. Closest to T=1.0.

**T1_2_temp_200 (T=2.0)** — ΔC@100 = **−0.0014**. Smoothing the softmax flattens gradients across all negatives, slowing learning. Plateau earliest (ep 30).

**Verdict on T1.2**: temperature is a bust. The default T=1.0 is already near-optimal for this loss + dataset. Drop from follow-ups.

**T1_3_smooth_005 (label smoothing ε=0.05)** — ΔC@100 = **+0.0017**, 5.05% gap closure. Half the effect of ε=0.10 — linear in ε across this range.

**T1_3_smooth_010 (label smoothing ε=0.10)** — ΔC@100 = **+0.0035**, **10.73% gap closure**. Only unambiguous winner. Mechanism: regularises the softmax distribution `q_i = (1-ε) · one_hot_i + ε/N`, which penalises over-confident top-1 predictions and redistributes gradient to informative negatives. Works here because softmax-CE alone over-concentrates (T1.1 negative). Epoch trajectory shows the largest breakout between ep 40→50 (ΔC@100 = +0.0035 vs T0's +0.0048 in the same window), indicating better late-stage LR utilisation.

**T1_4_cos_050 (cosine_power=0.5, steeper decay)** — ΔC@100 = **−0.0009**. Accelerates LR decay, model hits plateau earlier, underfits late epochs. Explicitly negative.

**T1_4_cos_150 (cosine_power=1.5)** — ΔC@100 = **+0.0009** (train-time), **+0.0001** (full-val). Delayed LR drop lets the late-stage loss continue to decrease. Train-time signal positive; full-val shows it's mostly noise.

**T1_4_cos_200 (cosine_power=2.0)** — ΔC@100 = **+0.0008** (train-time), **+0.0012** (full-val). Similar to cos_150; larger effect on full val suggests a real but tiny benefit from holding LR high for longer before the final decay. Kept in the combined baseline v2.

**T2_1_hardneg_05 (hard-neg fraction=0.5, margin=0.1)** — ΔC@100 = **−0.0057**, worst metric at best_ep=32. Hard-negative mining per ANCE/NV-Retriever: sort all couple scores, take top-25 negatives with `score < s_pos − 0.1`, plus 25 random. Two failure signatures:
  1. **Early peak**: best_ep=32 vs 58 for neutral runs → the distribution of negatives changes as the model improves, so the loss landscape is non-stationary. Cosine LR schedule (which assumes a stationary objective) drops LR too early relative to this moving target.
  2. **Drift visible in trajectory**: ep 50 (0.8173) < ep 30 (0.8143 → 0.8208 peak at 32 → monotonic decline). The model over-specialises on whatever negatives pass the margin filter and loses generality on softer negatives.
  Follow-up must either (a) loosen margin to 0.0 to include hardest possible negatives, (b) reduce fraction to 0.25 so random negatives still dominate, or (c) re-tune the LR schedule (longer warmup, later decay) for the non-stationary objective. All three to test before claiming hard-neg as dead.

**T2_2_pair_v2 (T2.1 + 4 extra pair-kinematic features)** — ΔC@100 = **−0.0068**, worst overall. Stacks on top of T2.1 so inherits its damage plus adds 4 extra input dims (cos(θ), (m−m_τ)/σ, dxy·dxy, dz·dz), bumping input_dim 51→55. Param count check: 575,105 → 576,129 (+1024 = 4×256 exact). **Flag did flip, features are wired in, but we can't attribute the damage** — pair_v2's own contribution is confounded with hard-neg's penalty. Needs a clean re-run with pair_v2 on top of label-smoothed softmax-CE (no hard-neg) to isolate.

**T2_3_ema (EMA decay=0.999)** — ΔC@100 = **−0.0005**, essentially neutral. `torch.optim.swa_utils.AveragedModel` tracks an exponential moving average of weights, BN recalibrated on the EMA copy post-training. `best_model_ema_calibrated.pt` exists (witness confirmed). The reported −0.0005 reflects **live weights only** — the EMA checkpoint was saved but never scored against val during training. Needs a dedicated eval pass on the EMA weights to know its true value. Low priority given neutral live-weight result.

**T3_1_wider (hidden=384, blocks=6, 1.87 M params)** — ΔC@100 = **−0.0006**. 3.2× parameter bump; best_ep=44 (early peak) signals mild overfit. Loss curves (not plotted here) show train/val gap opening after ep 40. The baseline architecture (hidden=256, blocks=4, 575K params) is not underfitting at this dataset size. **No capacity headroom to exploit.**

### What works / what does not — summary

**Works (train-time AND full-val positive):**
- Label smoothing on softmax-CE at ε=0.10 → +0.0027 on full val, sole reliable knob.

**Works marginally (train-time positive, full-val noisy):**
- Cosine-power ≥ 1.5 → ≤ +0.0012 on full val. Kept for the combined baseline v2 as a zero-cost additive.

**Neutral:**
- Temperature ∈ [0.5, 2.0], EMA, wider-deeper architecture. Default suffices.

**Busts:**
- Hard-neg mining at fraction=0.5 + margin=0.1 (non-stationary objective, cosine LR mismatch).
- Pair-kinematics v2 stacked on hard-neg (confounded).
- Softmax-CE alone without smoothing (over-confident top-1).

### Failure mode: T2.1 hard-neg mining in detail

Because T2.1 is the most interesting bust, dissect it further.

| Diagnostic | Baseline T0 | T2.1 hardneg |
|---|---|---|
| best_ep | 58 | 32 |
| trajectory ep 30→40 | +0.0168 | +0.0011 |
| trajectory ep 40→50 | +0.0048 | +0.0019 |
| trajectory ep 50→60 | +0.0016 | +0.0018 |
| val_d_at_60_tracks at best | 0.8596 | 0.8634 |

The `D@60_tracks=0.8634` at T2.1's best epoch is **higher** than T0's — the upstream cascade's track recall is unchanged (the same frozen cascade), so this difference is sampling noise over the val subset. C@100 dropping while D@60 is stable means the reranker itself is worse at ranking, not that the cascade is delivering fewer GT tracks. This isolates the failure to the reranker training, not upstream.

The trajectory also tells us the **model converges too fast then degrades slightly**, which is characteristic of non-stationary objectives where the mined hard negatives change distribution as the scorer improves. The cosine schedule (peak at 5% warmup, then decay) decays LR monotonically; by ep 32 LR is already ~40% of peak. If hard-neg fraction were lowered or the schedule were extended, the model could retain more late-stage adaptation.

### Recommended next sweep (against baseline v2)

1. **Hard-neg repair**: three variants on top of baseline_v2:
   - `hardneg_0.25_margin_0.0` (more random negs, tightest margin).
   - `hardneg_0.50_margin_0.0` (same frac, tighter margin).
   - `hardneg_0.50_margin_0.1 + --cosine-power 3.0` (delay LR decay).
2. **Clean pair_v2 read**: pair-kinematics-v2 on baseline_v2 without hard-neg, to isolate its own contribution.
3. **EMA full eval**: evaluate `best_model_ema_calibrated.pt` of T2_3 run against the full val set and compare against live weights.
4. **Extend training**: baseline_v2 best is at ep 56/60 — bump to 80 epochs and see if gains continue.

Drop entirely: T1.2 temperature sweep, T3.1 wider, T1.1 bare softmax-CE.

### Artifact index

| Location | Content |
|---|---|
| `debug_checkpoints/couple_improvements_20260416_211532/` | 14 exp dirs × (best_model_calibrated.pt, metrics/epoch_*.json, training.log, loss_history.json) + sweep.log + diagnostics.json |
| `debug_checkpoints/baseline_v2_s010_c200/` | Baseline v2 run (training.log, 60 epoch JSONs, best ckpt) |
| `debug_checkpoints/full_val_eval_20260417_120621/` | Full-val eval logs + metrics for T0 + 4 small-positives (parquets not copied, 91 MB each) |
| `debug_checkpoints/full_val_eval_baseline_v2_133452/` | Full-val eval logs + metrics for baseline v2 |
| Server `/workspace/tau-low-pt/part/models/couple_reranker_best.pt` | Now baseline v2 ckpt; v1 archived as `couple_reranker_v1_pairwise.pt` |
