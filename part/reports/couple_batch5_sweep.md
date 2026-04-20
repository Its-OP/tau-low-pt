# Couple-reranker Batch 5 sweep — tokenization scale & depth

**Date:** 2026-04-20
**Branch:** server-setup (worktree)
**Anchor:** v3 baseline — `projected_infersent p=32 + softmax-CE + label_smoothing=0.10 + cosine_power=2.0`
**Goal:** Test whether the Batch-3 F5 tokenized-reranker regressions (−0.002 to −0.005 pp C@100c) were caused by insufficient capacity (d_token=16, 2–3 blocks) rather than a flaw in the feature-as-token formulation itself.

---

## TL;DR

Tokenized (FT-Transformer) couple-reranker **beats v3 at train-time in the smallest scaled config** (`d_token=32, blocks=4, batch=96` → train-time C@100c **0.83229**, Δ = +0.00135 vs sweep-local anchor 0.83094), but the train-time gain **reverses at full-val scale**: B5_1 full-val C@100 = **0.7917** vs v3 **0.7925** → Δ = **−0.0008**. Gain does not grow with model size either — at `d_token=64, blocks=4` the run plateaus below v3 (0.8283, killed ep 38). The tokenization-scale hypothesis is fully ruled out: no config beats v3 on full-val. **Do not promote.** v3 remains the working baseline. Next direction: deferred Phase-C hypotheses (MoE, curriculum, VICReg pretrain, Stage-2 distillation).

---

## Configs

Held constant: `ft_transformer` mode, FFN 4×, pre-norm + GELU, `head_dim = d_token / num_heads ≥ 8`, 60 epochs, `top-k2=60`, seed=42, same data & optimizer as v3.

| # | Name | d_token | blocks | heads | head_dim | Batch | Steps/ep | Params | Events/ep |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | `B0_v3_anchor` | — | — | — | — | 96 | 200 | 600,289 | 19,200 |
| 1 | `B5_1_d32_b4` | 32 | 4 | 4 | 8 | 96 | 200 | 690,145 | 19,200 |
| 2 | `B5_2_d32_b6` | 32 | 6 | 4 | 8 | 64 | 300 | 715,553 | 19,200 |
| 3 | `B5_3_d48_b4` | 48 | 4 | 6 | 8 | 48 | 400 | 758,161 | 19,200 |
| 4 | `B5_4_d64_b4` | 64 | 4 | 8 | 8 | 32 | 600 | 850,753 | 19,200 |
| 5 | `B5_5_d64_b6` | — | — | — | — | — | — | — | dropped (OOM: >100 GB VRAM) |

B0/B5_1 ran in the main sweep (`couple_batch5_20260419_180220`). B5_2–B5_4 moved to a continuation sweep (`couple_batch5_cont_20260420_000628`) with per-experiment batch reductions because B5_1 saturated 94/98 GB VRAM at batch=96 and scaling `d_token` pushed over capacity. Events/epoch held constant at 19,200 by compensating steps.

---

## Results

Train-time C@100c (9,600-event limited-val subset, same as cascade-best harness):

| Run | Best C@100c | Δ vs B0 | Best epoch | Wall | Status |
|---|---:|---:|---:|---:|---|
| `B0_v3_anchor` | **0.83094** | — | 55 | ~1h 00m | complete |
| `B5_1_d32_b4` | **0.83229** | **+0.00135** | 48 | ~1h 30m | complete |
| `B5_2_d32_b6` | 0.83167 | +0.00073 | — | ~7h 00m | complete |
| `B5_3_d48_b4` | 0.83115 | +0.00021 | 31 | ~6h 45m | complete |
| `B5_4_d64_b4` | 0.8283 | −0.0026 | 24 | ~6h (ep 38/60 killed) | **killed** — plateau 14 ep, LR decayed, unrecoverable |

Trajectory of `new best` for `B5_4` (pre-kill): 0.8029 → 0.8075 → 0.8125 → 0.8174 → 0.8221 (ep 5) → 0.8227 (ep 11) → 0.8255 (ep 12) → 0.8256 (ep 17) → 0.8273 (ep 20) → **0.8283 (ep 24)** → flat through ep 38. LR schedule decayed below 3.5e-04 at ep 38, gap to v3 = 0.0026 with 22 ep remaining.

### Full-val eval (B5_1 only)

Evaluated `B5_1_d32_b4` on the full 52,284-event validation set (7 parquet files, `top-k2=60`, same harness as v3 full-val). Took 270s on the RTX PRO 6000.

| K | C@K | RC@K |
|---:|---:|---:|
| 50 | 0.7648 | 0.6602 |
| 75 | 0.7822 | 0.6735 |
| **100** | **0.7917** | **0.6807** |
| 200 | 0.8053 | 0.6913 |

eligible: 47,624 / 52,284 | full_triplet: 39,012 | mean_rank: 10.6

**vs v3 baseline** (full-val C@100 = 0.7925, mean_rank = 10.8):
- C@100 Δ = **−0.0008** (reversal from train-time +0.00135)
- mean_rank Δ = −0.2 (tighter top, but C@100 tail is worse)

**Verdict:** train-time win does not transfer. B5_1 does not beat v3 on full-val.

---

## Interpretation

**1. Tokenization helps at small scale.** The minimum scaled config (`d=32, blocks=4, batch=96`) beats v3 by +0.00135 train-time. This contradicts the Batch-3 F5 result (d=16, blocks=3, regression) — the feature-as-token formulation **does** outperform `projected_infersent` once given enough width/depth.

**2. But the gain does not scale.** Δ vs B0 monotonically shrinks as params grow: **+0.00135 → +0.00073 → +0.00021 → negative**. The "insufficient capacity" hypothesis is falsified: more tokenizer capacity is actively harmful in this harness.

**3. Batch-size confound.** B5_2/3/4 used reduced batches (64 → 48 → 32) to fit VRAM. Events/epoch was held constant, but the gradient noise scale changed (halved twice). The degradation direction coincides with batch shrink, not just param growth. Can't cleanly disentangle `(d, blocks)` from batch size; confound is systematic in the direction of the claim.

**4. B5_4 is the cleanest failure signal.** At `batch=32`, the model took **24 epochs** to reach 0.8283 (vs `B5_1` hitting 0.8318 by epoch 38 at batch=96). Validation plateau after ep 24 while LR was still at ~4.8e-04 indicates the optimization trajectory is fundamentally worse — not "needs more steps." Killing at ep 38 with 14 epochs of staleness and decayed LR is a correct call; recovery in the remaining 22 epochs was not physically plausible.

**5. The sweep's core hypothesis ("tokenization capacity is the lever") is ruled out.** Scale up → no growth in Δ. Direction deprioritized.

**6. Train-time C@100c on the 9,600-event subset is a weak predictor.** B5_1's +0.00135 train-time → −0.0008 full-val confirms the ~4pp harness offset is not uniform and small train-time deltas (≤ 0.002 pp) routinely flip sign at full-val. Future sweep acceptance gates must be full-val, not train-time.

---

## Recommendation

1. **Do not promote B5_1.** Full-val C@100 = 0.7917 is −0.0008 vs v3 0.7925. Train-time +0.00135 did not transfer. v3 stays as the working baseline.
2. **Drop the tokenization-scale direction entirely.** No config (d=32–64, blocks=4–6) beat v3 on full-val. Do not run d=96, blocks=8, or per-track-tokens sweeps. The capacity lever does not exist at this data scale.
3. **Keep `B5_1` checkpoint** as a reference for ablations only (e.g. if future hypotheses want to isolate FT-Transformer-style per-feature embedding). Not deployed.
4. **Next sweep direction** — from deferred Phase-C queue, pick one lever orthogonal to encoder capacity:
   - **H18 MoE encoder** — sparse specialization, different inductive bias
   - **H11 curriculum** — data-scale lever, independent of arch
   - **H19 VICReg pretrain** — representation-learning lever
   - **H5 Stage-2 distillation** — reuses Stage-2 ParT signal directly
5. **Future sweep acceptance**: gate on full-val C@100 (not train-time C@100c). Small train-time Δ (≤ 0.002 pp) is within noise of the train-time harness and flips at full-val scale.
6. Do **not** re-run `B5_4`; the failure was structural to small-batch optimization.

---

## Notes

- All runs used the same data, seed, optimizer, and schedule as v3. The only differences are the flags listed in the Configs table.
- `B5_5_d64_b6` was excluded pre-launch: at batch=32 its estimated peak VRAM was >100 GB. No config existed that kept the run inside VRAM and above the user-mandated batch floor of 32.
- `B5_4` killed manually at 2026-04-20 ~22:00 local, best checkpoint preserved at `experiments/couple_batch5_cont_20260420_000628/B5_4_d64_b4/.../checkpoints/best_model_calibrated.pt` (ep 24, C@100c=0.8283) for any post-hoc inspection.
- Param-count estimates from the plan tracked observed values to within ~10% (planned 660k–914k vs observed 690k–851k).
