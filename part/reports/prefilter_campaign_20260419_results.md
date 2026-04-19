# Prefilter Improvement Campaign — Results Report

**Campaign:** 2026-04-16 → 2026-04-19
**Plan:** `part/reports/pre-filter-updates.md`
**Target:** C@200 > 0.95 (baseline 0.919, 40-epoch overfit run)
**Branch:** `prefilter-campaign` (commits `701637d` … `1a20fba`)
**Artifact archive:** `~/Projects/masters/experiments/prefilter_campaign_20260419/` (~260 MB, gitignored)

## Headline

- Best R@200 in campaign: **0.9227** (E2a, 40 ep, edge features + k=16 + 3 rounds) — unchanged from the original 40-epoch run that opened the campaign.
- Best R@200 within the 20-epoch budget: **0.9223** (E2c, edge + k=32 + r=3).
- **Target 0.95 was not reached.** Closest approach is +0.0007 above the original baseline, ~0.028 short of target.
- Gather rewrite lifted the BS=256 OOM ceiling for k=48; k=64 now blocked on a different line (`masked_fill`).
- Object Condensation, MPM pretrain, and self-distillation all failed to converge in the 20-ep budget (ranked `FAIL-collapse` / `FAIL-kill`).

## Setup

- Orchestrator: `part/sweep_prefilter.sh full <IDs>` — compile on, AMP on, 500 steps/epoch, 10 workers.
- Two vast.ai RTX PRO 6000 Blackwell 97 GB instances in parallel (s1 port 12971, s2 port 10783).
- 20 epochs × BS=256 is the apples-to-apples budget (matches baseline's 40 × BS=128 data/ep); server 2 retries used BS=128 because of the gather OOM.
- Artifact archive path: `~/Projects/masters/experiments/prefilter_campaign_20260419/{s1,s2}/{logs,experiments}/`.

## Results — ranked by best R@200

| # | Exp | Flags | Server / BS | Epochs | Best R@200 | Best ep | Final R@200 | Wall | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | E2a | edge, k=16, r=3 | s1 · 256 | 40 | **0.9227** | 33 | 0.9183 | 6h50m | OK (ref) |
| 2 | E2c | edge, k=32, r=3 | s1 · 256 | 20 | 0.9223 | 18 | 0.9186 | 5h20m | OK |
| 3 | E2b | edge, k=32, r=2 | s1 · 256 | 20 | 0.9212 | 18 | 0.9158 | 3h58m | OK |
| 4 | E2d | edge, k=48, r=2 | s1 · 256 (new gather) | 20 | 0.9194 | 18 | 0.9135 | 4h24m | OK (gather-fix verified) |
| 5 | E2d | edge, k=48, r=2 | s2 · 128 | 20 | 0.9186 | 15 | 0.9137 | 2h23m | OK |
| 6 | E2e | edge, k=64, r=2 | s2 · 128 | 20 | 0.9184 | 15 | 0.9119 | 2h55m | OK |
| 7 | E1 | edge | s1 · 256 | 40 | 0.9181 | 37 | 0.9105 | 5h36m | OK |
| 8 | BASELINE | (none) | s1 · 256 | 40 | 0.9166 | 33 | 0.9120 | 5h41m | OK |
| 9 | E3 | edge, listwise_ce | s1 · 256 | 20 | 0.9167 | 14 | 0.9145 | 2h42m | OK |
| 10 | E4 | edge, logit_adjust τ=1 | s1 · 256 | 20 | 0.9150 | 17 | 0.9137 | 2h54m | OK |
| 11 | E6 | edge, PNA | s2 · 128 | 20 | 0.9150 | 15 | 0.9087 | 2h21m | OK |
| 12 | E9 | edge, infonce T=0.5 | s1 · 256 | 20 | 0.9149 | 16 | 0.9082 | 2h56m | OK |
| 13 | E7 | edge, xgb-stub feature | s1 · 256 | 20 | 0.9148 | 16 | 0.9103 | 2h46m | OK |
| 14 | E11 | edge, augmentation | s1 · 256 | 20 | 0.9135 | 12 | 0.9017 | 2h46m | OK |
| 15 | E8 | r=0 (no aggregation) | s1 · 256 | 20 | 0.8641 | 16 | 0.8632 | 1h24m | OK (ablation) |
| — | E5 | edge, object_condensation | s1 · 256 | 11/20 | 0.8287 | 2 | — | 1h14m | FAIL-collapse |
| — | E10 | edge, mpm_pretrain | s1 · 256 | 7/20 | 0.4446 | 1 | — | 1h01m | FAIL-kill (diverged) |
| — | E12 | edge, self-distillation | s1 · 256 | 3/20 | 0.3652 | 2 | — | 27m | FAIL-kill (diverged) |
| — | E2d | edge, k=48, r=2 | s1 · 256 (old gather) | 0/20 | — | — | — | — | FAIL-OOM |
| — | E2e | edge, k=64, r=2 | s1 · 256 (old gather) | 0/20 | — | — | — | — | FAIL-OOM |
| — | E2e | edge, k=64, r=2 | s1 · 256 (new gather) | 0/20 | — | — | — | — | FAIL-OOM@masked_fill |
| — | E6 | edge, PNA | s1 · 256 (old/new) | 1/20 | 0.8929 | 1 | — | — | FAIL-OOM@_pna_aggregate |

`Final R@200` is always the epoch-20 (or epoch-40) number; `Best` may be earlier.

## Single-lever deltas vs. BASELINE (0.9166)

| Lever | Δ R@200 |
|---|---:|
| Edge features (E1, 40 ep) | +0.0015 |
| Edge + k=16 r=3 (E2a, 40 ep) | **+0.0061** |
| Edge + k=32 r=2 (E2b, 20 ep) | +0.0046 |
| Edge + k=32 r=3 (E2c, 20 ep) | +0.0057 |
| Edge + k=48 r=2 (E2d BS=256, 20 ep) | +0.0028 |
| Listwise CE (E3, 20 ep) | +0.0001 |
| Logit-adjust (E4, 20 ep) | −0.0016 |
| InfoNCE (E9, 20 ep) | −0.0017 |
| PNA aggregation (E6 BS=128, 20 ep) | −0.0016 |
| XGB-stub feature (E7, 20 ep) | −0.0018 |
| JetCLR augmentation (E11, 20 ep) | −0.0031 |
| r=0 (E8 ablation) | −0.0525 |

**What moves the needle:** edge features + extra message-passing round (r=3). Everything else — new losses, new features, new aggregators, data augmentation — is neutral-to-slightly-negative at this budget.

**Aggregation IS load-bearing:** removing message passing (E8) costs −0.05 R@200. Node-only MLP collapses.

## Failure autopsies

**E5 (object condensation)** — β head collapsed to uniform by epoch 5; `ranking` loss went to 0, `object_condensation` term dominated and wiped gradients. Peaked at ep 2 and never recovered. OC is known-sensitive to `q_min`, `beta_weight`; a proper retry would sweep both plus try longer warmup.

**E10 (masked particle modeling pretrain)** — val R@200 = 0.4446 at ep 1, diverged to 0.31 by ep 5. 20-ep budget folds pretrain + fine-tune into one run; that's wrong — MPM expects pretrain on reconstruction only (no label loss), then a fine-tune from that checkpoint. Killed at ep 7.

**E12 (EMA self-distillation)** — val R@200 = 0.36 at ep 3. KL between student and stale teacher dominated the total loss; teacher starts from random-init student so KL signal is noise. Killed at ep 3. A proper retry needs EMA warmup (teacher = student for the first N epochs, then decay) and `--kl-weight` an order of magnitude smaller.

**E2d / E2e / E6 at BS=256 (original gather)** — `cross_set_gather` allocated three `(B·C·M·K)` scratch tensors per call; at k=48 that peaks ~24 GiB, k=64 ~32 GiB. Fixed by the gather rewrite for E2d; E2e's next bottleneck is `neighbor_features.masked_fill` at `TrackPreFilter.py:557` (same-shape mask materialization), E6 hits `_pna_aggregate` at `TrackPreFilter.py:468` (5× channel blowup intrinsic to PNA). BS=128 s2 runs completed all three; use those for the ranking.

## Performance work delivered

- **Profiling CLI** — `--profile-steps N` in `train_prefilter.py` with lean defaults (summary-only; opt-in `--profile-record-shapes` / `--profile-memory` / `--profile-chrome-trace`). Commits `ed1e8df`, `78d75bf`.
- **Profile of E2a-config (20 active steps, BS=256)** — kernel-launch-bound, not compute-bound. `aten::copy_` 37 %, `aten::masked_fill` 14 %, `Memcpy DtoD` 13 %, `aten::max` (reduce) 9.8 %, `aten::conv1d` only 1.6 %. Confirms the sweep wasn't dominated by the MLPs — it was dominated by the surrounding tensor plumbing.
- **Compile enabled in the sweep** — `--no-compile` dropped (`e19ac5b`). Launch overhead fused away for most elementwise chains.
- **`cross_set_gather` rewrite** — body reduced to `reference_features.gather(dim=2, index=expanded_indices)` (commits `4f53572` tests, `1a20fba` impl). Peak memory 3 big scratch tensors → 1. Verified end-to-end by E2d at BS=256 on server 1 (0.9194). 513 tests green.

## What wins, concretely

- **Winning config** to go from here: `--use-edge-features --num-neighbors 16 --num-message-rounds 3` for 40 epochs. That's E2a, which we already have a checkpoint for.
- **Second-best at half the epochs:** `--use-edge-features --num-neighbors 32 --num-message-rounds 3` (E2c) — 0.9223 in 20 ep. If wall-time matters more than the last +0.0004, pick this.
- **Aggregation choice** — stick with standard max-pool; PNA neither helps metric nor fits in memory at BS=256.
- **Loss choice** — pairwise softplus remains the default; none of listwise CE / InfoNCE / logit-adjust beat it on this dataset at this budget.

## Next steps (ranked)

1. **E2c at 40 epochs (compile)** — one config, one run. E2c matched E2a (0.9223 vs 0.9227) in half the epochs; the extra 20 may push past 0.95 or re-confirm the ceiling. Two outcomes, both informative.
2. **Stack edge + k=32 r=3 + listwise_ce** — no experiment combined the k/r winner with a CE-family loss. Each was standalone-positive; stacking should be tried once before declaring the architecture maxed out.
3. **Land the `masked_fill` + `_pna_aggregate` unblockers** — both are one-function rewrites following the same pattern as `cross_set_gather`. Unblocks E2e and E6 at BS=256 for a fair comparison.
4. **If (1) and (2) together still fall short of 0.95**, the working hypothesis becomes: *the achievable ceiling on raw 16-feature input with this architecture family is ≈ 0.93–0.94.* Per-track XGBoost on the same 16 features is already "slightly worse" per the campaign plan, so the signal budget isn't huge. At that point, the campaign's conclusion is that closing the last 3 pp needs either (a) richer per-track features (reco-stage — ruled out by project constraints), or (b) a different architecture class (ruled out: no attention, no GNN, no dense compression).

## Commits touched during the campaign

`701637d` edge features + CLI knobs · `752252f` loss module · `0292fc0` deferred-experiment flags · `25784f0` sweep defaults · `4cb6218` server workflow · `ed1e8df` torch.profiler integration · `78d75bf` profiler verbosity flags · `e19ac5b` drop --no-compile · `4f53572` gather equivalence tests · `1a20fba` gather rewrite.
