# Cascade Full-Val Evaluation — 2026-04-21

Latest end-to-end cascade evaluation with **per-stage standalone weights**. Supersedes all prior full-val C@K numbers this session (those were biased low by a positional-alignment bug in `eval_couple_reranker.py` — see §5).

---

## 1. Model configuration

| Stage | Weights | Key metric |
|---|---|---|
| Stage 1 — TrackPreFilter | `part/models/prefilter_best.pt` (Apr-5 baseline) | training R@200 = 0.919 |
| Stage 2 — ParT CascadeReranker | `part/models/stage2_best.pt` (Apr-21 retrain, bundled cascade at `/workspace/.../cascade_oldprefilter_stage2_20260421_134102/checkpoints/best_model.pt`) | training R@50 = 0.8257 at ep 91 |
| Stage 3 — CoupleReranker v3 | `part/models/couple_v3_best.pt` (from couple_batch4 B0_v3_anchor) | training-subset C@100c = 0.8309 |

**Validation set:** 52,284 events (52,500 raw, minus 216 dropped by `selection: event_n_tracks > 0`). **All events have exactly 3 GT pions** by construction of the dataset.

---

## 2. End-to-end cascade metrics

Denominator = 52,284 events. Both computation methods (index-matching + pT-matching) shown for cross-check; **index is authoritative**, pT agrees at the stage-3 level because double-collisions are rare. pT-level disagreement at earlier stages is fp16 quantization noise (see §4).

| Metric | Description | By Indices | By pT |
|---|---|---:|---:|
| R@256 (Stage 1) | fraction of GT pions recovered in top-256 prefilter | **0.9432** | 0.9487 |
| P@256 | fraction of events with entire triplet in top-256 prefilter | **0.8787** | 0.8921 |
| R@100 (Stage 2) | fraction of GT pions recovered in top-100 ParT | **0.8778** | 0.8883 |
| R@60 (Stage 2) | fraction of GT pions recovered in top-60 ParT | **0.8359** | 0.8448 |
| P@60 | fraction of events with entire triplet in top-60 ParT | **0.7311** | 0.7359 |
| PC@60 | fraction of events with ≥2 GT pions in top-60 ParT (couple-achievable ceiling) | **0.8563** | 0.8605 |
| C@50 (Stage 3) | fraction of events with ≥1 GT couple in top-50 couples | **0.7898** | 0.7901 |
| C@75 | fraction of events with ≥1 GT couple in top-75 couples | **0.8106** | 0.8110 |
| C@100 | fraction of events with ≥1 GT couple in top-100 couples | **0.8236** | 0.8240 |
| C@200 | fraction of events with ≥1 GT couple in top-200 couples | **0.8433** | 0.8439 |

**Reference sanity check:** training-time S1_R@K1=256 ≈ 0.945; our full-val R@256 frac-recall = 0.9432 ≈ matches. Training R@200 (prefilter ep 36) = 0.919; our R@200-eq ≈ the same.

**Ceilings and utilization:**
- Prefilter triplet ceiling: P@256 = 0.8787. Bounds everything downstream.
- ParT triplet ceiling: P@60 = 0.7311. Bounds any full-triplet-completion scheme on top-60.
- Couple-achievable ceiling: PC@60 = 0.8563. The highest C@K can ever reach.
- **C@100 = 0.8236 = 96.2% of PC@60** — couple reranker is near-saturated at its input. Further Stage-3 improvements are capped unless Stage-1 or Stage-2 deliver more.

---

## 3. Prefilter perfect-recall percentiles

Unconditional percentiles of **max-GT-rank** per event (worst-ranked GT pion's 1-indexed rank in the prefilter's full sorted output). 52,544 events processed by `diagnostics/prefilter_p95.py`.

| Percentile | K needed to capture full triplet |
|---|---:|
| p50 | 26 |
| p75 | 136 |
| p90 | 292 |
| **p95** | **399** |
| p99 | 599 |
| p99.5 | 664 |
| p99.9 | 859 |

Mean max-GT-rank = 95.5.

**Implications:** the current production `K1=256` retains the full triplet for only ~88% of events (matches P@256 = 0.8787). To hit p95 coverage, K1 must grow to ~400; to hit p99, K1 ≈ 600. Stage-2 runtime scales linearly in K1, so raising K1 is a direct speed-accuracy trade.

---

## 4. fp16 quantization of `track_pt`

CMS nanoAOD source (`example_root.root`) stores `Track_pt` as **float32** but every value round-trips exactly through float16 — effective precision is float16 (10 mantissa bits + implicit 1 = 11 significant bits). Step size per binade = `2^(E−10)` where E is the unbiased exponent.

Collision rate by pT bucket (fp16 binade boundaries, 500 events sampled):

| pT range | fp16 step | tracks sampled | unique pTs | dup rate |
|---|---:|---:|---:|---:|
| 0.5–1 GeV | 4.9e-4 | 87,878 | 1,024 | **98.8%** |
| 1–2 GeV | 9.8e-4 | 31,587 | 1,024 | 96.8% |
| 2–4 GeV | 2.0e-3 | 7,365 | 985 | 86.6% |
| 4–8 GeV | 3.9e-3 → 7.8e-3 | 915 | 519 | 43.3% |
| 8–500 GeV | coarser | 92 | 81 | 12.0% |

The 1,024-unique ceiling in low-pT buckets is the **pigeonhole limit**: each fp16 binade has exactly 2¹⁰ = 1,024 representable values. Low-pT tau daughters cluster at 0.5–0.6 GeV — exactly where collisions dominate. **Only integer track indices are safe identifiers**; pT can coincide across non-GT and GT tracks.

Root-to-parquet conversion (`scripts/condor/convert_to_parquet/convert_root_to_parquet.py`) preserves within-event track ordering — no sort or shuffle ever applied. Within-event track index `i` in parquet = index `i` in original ROOT tree (restricted to pions + pT cutoff survivors).

---

## 5. Positional-alignment bug (fixed)

**Bug:** `eval_couple_reranker.py` preloaded `track_pt` by sequentially reading parquet rows (52,500) and looked up pT via `all_track_pt[global_event_offset + event_index]`. The DataLoader applies `selection: event_n_tracks > 0`, dropping 216 events — from the first drop onward, every subsequent event's pT list pointed to a **different event's tracks**.

**Downstream consequence:** `gt_pion_pt` held wrong/partial pTs for ~18.6% of events. `compute_couple_metrics.py` matches couples to GT via pT-set membership, so those events silently failed — **every full-val C@K in session memory** (`v3=0.7925`, `B5_1=0.7917`, `B4_4=0.7921`, pre-fix cascade = 0.7880) is biased low by ~0.035 pp.

**Fix:** switched `track_pt` preload to a composite-key-indexed dict (`(event_run, event_id, event_luminosity_block, source_batch_id, source_microbatch_id)`), looked up per-event via the metadata already emitted in `Z`. All 52,284 events now have the full triplet of GT pT values.

Eval-script patch landed in `part/diagnostics/eval_couple_reranker.py` (this session, uncommitted). Tests updated in `part/tests/test_eval_couple_reranker.py` (9 passed). 589 full-suite tests pass.

**Post-fix delta:** C@100 **0.7880 → 0.8236** (+0.036) for the new cascade + v3 couple. All session full-val numbers need re-evaluation before they can be cited.

---

## 6. Artifacts

| Path | Content |
|---|---|
| `experiments/results.parquet` | 52,284 events × 15 columns — end-to-end eval output (top-256 prefilter, top-100 ParT, top-200 couples + couple/prefilter/ParT pT + GT indices+pT, all composite-key-aligned) |
| `experiments/cascade_metrics.txt` | raw metric table (this data) |
| `part/models/prefilter_best.pt` | Stage 1 weights |
| `part/models/stage2_best.pt` | Stage 2 weights |
| `part/models/couple_v3_best.pt` | Stage 3 weights |
| server `/workspace/tau-low-pt/part/experiments/cascade_oldprefilter_stage2_20260421_134102/` | full Stage-2 retrain artifacts (loss curves, epoch checkpoints, training.log) |
| server `/workspace/prefilter_p95.json` | percentile raw data |

**Reproducibility:** Stage-2 retrained with `--seed 42` config at `part/train_cascade.py --stage1-num-neighbors 16 --optimizer soap --no-compile --batch-size 64 --steps-per-epoch 500 --epochs 100 --lr 1e-3 ...` (full config in the training.log). Peak val R@50 = 0.8257 at ep 91 (vs old Apr-6 cascade 0.8269 at ep 55 — within 0.0012 seed variance).
