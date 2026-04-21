# Prefilter Improvement Campaign — Experiment TODO

## Context

Stage-1 prefilter (`TrackPreFilter`) plateau **C@200 = 0.919, C@256 ≈ 0.943** on pT ≥ 0.5 GeV dataset, hit epoch 36 of `prefilter_dim256_cutoff_bs192_PreFilter_20260405_120532` run. Past epoch 36 train R@200 climb 0.932 → 0.957, val stagnate/drift — classic overfit. Per-track XGBoost on same 16 raw features "slightly worse", so current 400K-param deep model buy little over feature-engineered boosting.

Campaign target **C@200 > 0.95** (≥ +3.1 pp) within **≤ 20 epochs per experiment**, hard arch constraints (no attention, no dense-tensor compression, no GNN, raw features only, no re-run locally-failed). Downstream cascade (Stage 2 / Stage 3) frozen — experiments must preserve prefilter per-track-scores interface so cascade consume `selected_indices` unchanged (check `part/train_cascade.py:316-330`).

### User constraints

1. **No attention.** Rule out Transformer, Set Transformer ISAB, Perceiver IO, Nyström/Performer/Longformer, NPT/SAINT (intersample attention), TabR, Hopfield.
2. **No dense-tensor compression.** Rule out bottleneck pool event to fixed-size vector — 0.27 % positive signal lost.
3. **≤ 20 epochs** per experiment.
4. **No GNN.** Graph variants (GravNet / EdgeConv backbone / multi-space composite) failed locally.
5. **Raw detector-level features only.** No reconstruction-stage cheat.
6. **Biggest-expected-gain ordering.**
7. **Target C@200 > 0.95.**

### Prior failures (do NOT re-propose)

From `project_failed_experiments.md` and `project_noise_neighborhoods.md`: ASL, OHEM, pairwise-LV-in-GNN-message-passing, GravNet, EdgeConv backbone, triplet scoring, multi-space composite graphs, per-track ρ-candidate count.

### Current architecture (verified)

- `TrackPreFilter` — `weaver/weaver/nn/model/TrackPreFilter.py:23`
- Mode `mlp`; per-track Conv1d (16 → 256) + 2 rounds k-NN(k = 16, η-φ) max-pool aggregation + scorer head; 400 K params
- Neighbor dim `= 2 · hidden_dim` at `TrackPreFilter.py:162` (max-pool only). Richer `aggregation_mode = 'pna'` at line 159 exist but untested (`5 · hidden_dim`, cat mean/max/min/std).
- Wrapper `part/networks/lowpt_tau_TrackPreFilter.py:38-82` hard-code `num_neighbors = 16, num_message_rounds = 2`. No CLI flag yet.
- `ranking_temperature_start = ranking_temperature_end = 1.0` defaults (frozen). Epoch-36 run used 2.0 → 0.5 (wrapper override that branch); current wrapper already frozen.
- `build_cross_set_edge_features` exist at `HierarchicalGraphBackbone.py:232` — concat `[center, neighbor − center, pairwise_lv_fts(center_lv, neighbor_lv)]`. Prefilter `mlp` mode NOT use it today; only max-pool node features.

---

## Cross-experiment infrastructure (build ONCE before experiments 1, 2, 3, 4, 5, 8, 10, 11)

1. **CLI + wrapper plumbing** — `part/train_prefilter.py` argparse block (~line 337) gain:
   - `--num-neighbors` (int, default 16)
   - `--num-message-rounds` (int, default 2)
   - `--aggregation-mode` (choices: `mlp`, `pna`; default `mlp`)
   - `--use-edge-features` (flag)
   - `--loss-type` (choices: `pairwise`, `listwise_ce`, `infonce`, `logit_adjust`, `object_condensation`; default `pairwise`)
   - `--logit-adjust-tau` (float, default 1.0)
   - `--temperature` (float; set both start and end when given)
   - `--use-augmentation` (flag)
   - `--ssl-pretrain-ckpt` (path)
   - `--extra-feature xgb_score` (optional flag enable XGB-score input channel)

   Thread all into `part/networks/lowpt_tau_TrackPreFilter.py:22-108` via existing kwargs dict — do NOT duplicate wrapper.

2. **Edge-feature path in `mlp` aggregation** — `TrackPreFilter.py:162` + `_forward_mlp` (lines 410-434) must:
   - propagate `(B, 4, P)` `lorentz_vectors` through forward (signature already have it line 301).
   - call `build_cross_set_edge_features` from `HierarchicalGraphBackbone.py:232` on each round `cross_set_knn` output.
   - set `neighbor_input_dim = 2 · hidden_dim + pairwise_lv_fts_dim` when `use_edge_features=True`. Use `pairwise_lv_fts_dim = 4` (ln ΔR, ln m², ln kT, ln z) — inspect `ParticleTransformer.pairwise_lv_fts` confirm exact dim.
   - add charge-product channel separately (1 extra dim), reuse existing charge input feature.

3. **Loss module** — new file `weaver/weaver/nn/model/prefilter_losses.py`:
   - `listwise_ce_loss(scores, labels, mask)` — event-wise softmax over valid tracks, CE against 3-hot target (or averaged-over-positives loss).
   - `infonce_in_event(scores, labels, mask, temperature)` — 1-vs-rest InfoNCE per positive.
   - `logit_adjust_offset(num_pos, num_neg, tau)` — constant offset added to logits during training (Menon 2007.07314).
   - `object_condensation_loss(embeddings, beta, labels, mask, q_min=0.1)` — Kieseler 2002.03605 attractive/repulsive potential + β regularizer.
   - Wire into `TrackPreFilter._ranking_loss` via `loss_type` branch; default path stay unchanged for reproducibility.

4. **Set augmentation module** — `part/utils/training_utils.py` (or new `part/utils/set_augmentation.py`) with three transforms, each take `(features, points, lorentz_vectors, mask, labels)`:
   - `TrackDropout(p=0.10)` — randomly zero mask non-positive tracks.
   - `FeatureJitter(sigma=0.05)` — Gaussian smear on log features.
   - `EtaPhiRotation()` — global rotation of φ (and eta-flip p=0.5).
   Call in `train_prefilter.py` right before `compute_loss`.

5. **XGBoost score cache (one-off, only if exp 7 run)** — `part/scripts/cache_xgb_scores.py` train per-track XGBoost on same 16 features, write parquet column `track_xgb_score` aligned to existing data layout. Load by add `- track_xgb_score` to `pf_features` block in `part/data/low-pt/lowpt_tau_trackfinder.yaml`. Regenerate `.auto.yaml` after.

6. **SSL pretrain entry point (only if exp 10 run)** — `part/pretrain_prefilter_mpm.py` reuse dataloader from `train_prefilter.py:404-485`, drop label head, mask 15 % of tracks, reconstruct 16 features from k-NN aggregated context, MSE loss. Save only `track_mlp.*` and `neighbor_mlps.*` weights in checkpoint loadable by existing `TrackPreFilter.__init__`.

7. **Metrics plumbing (unchanged)** — existing `part/diagnostics/evaluate_prefilter.py` and `reports/prefilter_analysis_*.md` already produce per-event parquet + C@K tables; each experiment reuse.

### Canonical server-side command (one-off per experiment)

```bash
ssh -p 26893 root@69.63.236.192 "cd /workspace/tau-low-pt/part && \
  nohup env RUN_NAME=<exp-name> bash train_prefilter.sh <flags> > \
  /workspace/logs/<exp-name>.log 2>&1 & echo PID=\$!"
```

Monitor with pattern in `part/reports/server_workflow.md`. Each experiment must land C@K metrics in `part/reports/experiment_log.md` and checkpoint in `part/models/debug_checkpoints/<exp-name>_<timestamp>/`.

---

## TOP 3 — biggest shots at closing 0.919 → 0.95 gap

### 1. `pairwise-edge-features`
- **Hypothesis.** ParT-style pairwise edge features (ln ΔR, ln m², ln kT, ln z, q_i · q_j) injected at every k-NN aggregation round give network physics-structured inter-track signal max-pool of raw node features cannot synthesize. `build_cross_set_edge_features` exist at `HierarchicalGraphBackbone.py:232`, proven stable in Stage-2 cascade — we wire existing primitive into prefilter, not reinvent.
- **Estimated ΔC@200.** +0.020 … +0.040 (largest single-experiment estimate in campaign)
- **Change scope.**
  - `weaver/weaver/nn/model/TrackPreFilter.py:162` — switch `neighbor_input_dim` to `2 · hidden_dim + pairwise_lv_fts_dim + 1` when `use_edge_features=True`.
  - `_forward_mlp` at `TrackPreFilter.py:410-434` — thread `lorentz_vectors`, call `build_cross_set_edge_features`, `cross_set_gather` the edge features, concat with `[current, max_pooled]` before neighbor MLP.
  - `part/networks/lowpt_tau_TrackPreFilter.py:38-82` — pass `use_edge_features=True`.
  - CLI flag `--use-edge-features` (infra §1).
- **Training budget.** 20 epochs from scratch. Per-epoch overhead ~1.3 × baseline.
- **Kill signal.** Val R@200 < 0.92 by epoch 10, OR NaN loss (detach LV features inside `pairwise_lv_fts` — see `HierarchicalGraphBackbone.py:272-289` for established pattern).
- **Cascade compatibility.** Output interface unchanged → Stage 2 / 3 do NOT need retrain.

### 2. `receptive-field-sweep` (combined k and rounds, gated by exp 1)
- **Hypothesis.** Current k = 16 × 2 rounds ≤ 256 tracks; events hold 600 – 1130 tracks, so receptive field see only 24 – 45 % of event. Bigger k and/or 3 rounds widen per-track view without O(P²) blow-up of attention. Expect stack (imperfectly) with exp 1 because larger neighborhoods also have more edges.
- **Estimated ΔC@200.** +0.010 … +0.030 on top of exp 1.
- **Change scope.** Pure config — `part/networks/lowpt_tau_TrackPreFilter.py:42-43`. Sweep `(num_neighbors, num_message_rounds) ∈ {(16, 3), (32, 2), (32, 3), (48, 2), (64, 2)}`. Infra §1 expose these as CLI flags.
- **Training budget.** 5 runs × 20 epochs. Winner become baseline for exp 3 onward.
- **Kill signal.** Per-epoch wall-clock > 2.5 × baseline at k ≥ 48 → skip that grid point. No correctness risk.
- **Cascade compatibility.** Full.

### 3. `listwise-ce-loss`
- **Hypothesis.** Current pairwise softplus rank 3 positives against 50 sampled negatives per event (`TrackPreFilter.ranking_num_samples = 50` default) — discard 99 % of background structure. Event-wise softmax-CE use all 600 – 1100 in-event negatives per step, directly optimize rank position of positives, exact target of C@K metric. Theory (2411.07770) CCE upper-bound NDCG / MRR more tightly than BCE/BPR.
- **Estimated ΔC@200.** +0.010 … +0.020.
- **Change scope.**
  - `weaver/weaver/nn/model/prefilter_losses.py:listwise_ce_loss` (infra §3).
  - `TrackPreFilter._ranking_loss` — new `loss_type='listwise_ce'` branch; keep pairwise path intact.
  - `part/train_prefilter.py:148-150` — thread `args.loss_type`.
- **Training budget.** 20 epochs from scratch (apply on top of exp-2 winner config).
- **Kill signal.** Val R@200 < 0.92 by epoch 10 despite loss decrease → listwise formulation not driving separation → revert.
- **Cascade compatibility.** Full.

**Expectation:** if exp 1 + 2 + 3 each land at mid-estimate and stack with ~70 % efficiency, C@200 reach 0.919 + 0.70 · (0.030 + 0.020 + 0.015) ≈ **0.965**. If stacking weaker, land at ~0.95 anyway. If below 0.95 after these three, proceed to next tier.

---

## Next tier — run if top-3 stop short of 0.95

### 4. `logit-adjust-loss`
- **Hypothesis.** Menon 2007.07314 logit adjustment add `−τ · log(π_neg / π_pos)` at train time (≈ `−τ · log(375)` for 3/1130). Provably optimize balanced error; orthogonal to listwise CE (exp 3) so stack. Not same as ASL (failed): no hard clip, no focal down-weight.
- **Estimated ΔC@200.** +0.005 … +0.015.
- **Change scope.** `prefilter_losses.py:logit_adjust_offset`; call inside `TrackPreFilter._ranking_loss` before score differences taken. Sweep τ ∈ {0.5, 1.0, 2.0}.
- **Training budget.** 3 × 15 epochs (fine-tune from top-3 winner).
- **Kill signal.** Val R@200 regress vs top-3 winner by > 0.005 after epoch 10.
- **Cascade compatibility.** Full.

### 5. `object-condensation-loss`
- **Hypothesis.** Kieseler 2002.03605 condense "few objects in noisy set" via β-weighted potential loss — exactly our setting. Architecture-agnostic (run on any per-track embedding), no GNN needed. `clustering_dim` parameter already reserved at `lowpt_tau_TrackPreFilter.py:33` but unused.
- **Estimated ΔC@200.** +0.005 … +0.020 (high variance; OC hyperparameter-sensitive).
- **Change scope.**
  - `TrackPreFilter.scorer` at `TrackPreFilter.py:187-195` — add second Conv1d(hidden_dim, 1) β head and Conv1d(hidden_dim, clustering_dim) embedding head; keep score = `β · max(embedding · prototype)` output for downstream API.
  - `prefilter_losses.py:object_condensation_loss` — attractive/repulsive + β regularizer.
- **Training budget.** 2 × 20 epochs (one q_min=0.1, one q_min=0.5).
- **Kill signal.** β collapse to uniform (per-event β std < 0.01) by epoch 5. If both runs collapse, drop.
- **Cascade compatibility.** Output still monotonic per-track score; top-K selection unaffected.

### 6. `pna-aggregation`
- **Hypothesis.** Cheap config flip — `aggregation_mode = 'pna'` at `TrackPreFilter.py:159` concat `[current, mean, max, min, std]` instead of `[current, max]`. Double aggregation expressiveness without change complexity class; untested in repo. Should compat with edge features (exp 1) — concat channels independently.
- **Estimated ΔC@200.** +0.003 … +0.015.
- **Change scope.** Pure config — `part/networks/lowpt_tau_TrackPreFilter.py` set `aggregation_mode='pna'`. Code path at `TrackPreFilter.py:157-163` already exist.
- **Training budget.** 20 epochs (ideally combined with exp 1 edge features for final baseline).
- **Kill signal.** Val R@200 < exp-1 baseline by epoch 10.
- **Cascade compatibility.** Full.

### 7. `xgb-score-as-feature`
- **Hypothesis.** DeepGBM-style hybrid. Since per-track XGBoost ≈ current deep model, feed XGBoost score as 17-th input feature let deep model spend capacity on set-level context instead of re-learn per-track physics. Simple, no arch change.
- **Estimated ΔC@200.** +0.005 … +0.015.
- **Change scope.**
  - `part/scripts/cache_xgb_scores.py` (infra §5) — one-time XGB training + parquet write.
  - `part/data/low-pt/lowpt_tau_trackfinder.yaml:53-83` — add `- track_xgb_score` to `pf_features`. Regenerate `.auto.yaml` next to config.
  - `input_dim = 17` propagate automatically via `len(data_config.input_dicts['pf_features'])` at `lowpt_tau_TrackPreFilter.py:36`.
- **Training budget.** 15 – 30 min XGBoost + 20 prefilter epochs.
- **Kill signal.** Val R@200 < XGBoost-alone R@200 + 0.005 → model collapse to XGBoost → drop.
- **Cascade compatibility.** Full, but XGBoost score cache must also be generated on server for inference.

### 8. `aggregation-ablation`
- **Hypothesis.** Diagnostic. Train `num_message_rounds = 0` (pure per-track MLP). If C@200 barely drop, aggregation dead weight, concentrate future work on OC / listwise / XGB-feature routes. If C@200 collapse, aggregation IS doing work, focus stay on make it richer (exps 1, 2, 6).
- **Estimated ΔC@200.** −0.04 … +0.005 (expect: negative if aggregation help).
- **Change scope.** Pure config — `num_message_rounds=0` at `lowpt_tau_TrackPreFilter.py:43`. Model already tolerate via `range(self.num_message_rounds)` at `TrackPreFilter.py:409`.
- **Training budget.** 15 epochs.
- **Kill signal.** N/A — diagnostic, always run to completion. Use results re-prioritize tail of list.
- **Cascade compatibility.** Full.

### 9. `infonce-in-event`
- **Hypothesis.** Alternative to listwise-CE (exp 3). InfoNCE with all in-event background as negatives, 3 positives as anchors. Stronger representation signal than softplus pairwise. Mutually exclusive with exp 3.
- **Estimated ΔC@200.** +0.005 … +0.015 (pick exp 3 or exp 9 based on which has better baseline curve at epoch 10).
- **Change scope.** `prefilter_losses.py:infonce_in_event`; `TrackPreFilter._ranking_loss` `loss_type='infonce'` branch.
- **Training budget.** 15 epochs from scratch.
- **Kill signal.** Val R@200 < exp-3 baseline by epoch 10 → exp 3 won → stop.
- **Cascade compatibility.** Full.

### 10. `masked-particle-modeling-ssl`
- **Hypothesis.** Heinrich 2401.13537 MPM objective — reconstruct masked track features from k-NN-aggregated context — architecture-agnostic. Pretrain unlock unlabeled-regime learning; fine-tune keep labeled objective. Expose more per-track structure to encoder without change per-track output API.
- **Estimated ΔC@200.** +0.005 … +0.020 (variance high; gain depend on how much unlabeled structure encoder absorb).
- **Change scope.**
  - `part/pretrain_prefilter_mpm.py` (infra §6).
  - `train_prefilter.py:593-621` — resume block gain `--ssl-pretrain-ckpt` branch load backbone weights only (not scorer head).
- **Training budget.** 10 epochs pretrain (one-time artifact) + 15 epochs fine-tune. Fine-tune fit within 20-epoch cap.
- **Kill signal.** Pretraining reconstruction MSE hit near-variance ceiling by epoch 5 → no learnable structure → abort pretrain, drop experiment.
- **Cascade compatibility.** Full.

### 11. `jetclr-augmentations`
- **Hypothesis.** Track dropout (p=0.1) + Gaussian feature smear (σ=0.05) + η-φ rotation from JetCLR (Dillon 2108.04253). Regularize late-epoch overfit at epoch 36.
- **Estimated ΔC@200.** +0.003 … +0.010.
- **Change scope.** Infra §4 (`set_augmentation` module); call in `train_prefilter.py` right before `compute_loss`.
- **Training budget.** 20 epochs from scratch (on top of winning config).
- **Kill signal.** Val R@200 plateau earlier than baseline → augmentation break symmetry. Disable η-φ rotation first (impact parameters not rotation-invariant), retry with dropout + smear only.
- **Cascade compatibility.** Full (train-only transform).

### 12. `self-distillation-ema`
- **Hypothesis.** EMA teacher (2601.19897) — maintain exponentially-moving-average copy of student, use teacher soft scores as auxiliary regression target. Known rare-class boost; tiny compute overhead (one extra forward pass on teacher).
- **Estimated ΔC@200.** +0.003 … +0.010.
- **Change scope.**
  - `train_prefilter.py` — add EMA module (torch-standard pattern), KL-divergence loss between student and teacher logits.
  - No architectural change.
- **Training budget.** 20 epochs from scratch.
- **Kill signal.** KL loss dominate total loss (> 0.5× scalar ratio) → teacher follow student too closely; reduce EMA decay (0.999 → 0.995) or weight.
- **Cascade compatibility.** Full.

---

## Ordering and stop conditions

- **Phase A (priority):** Build infra §1 + §2 + §3, then run exps **1 → 2 → 3** in sequence. Each winning checkpoint become baseline for next. If C@200 ≥ 0.95 after exp 3, declare victory, stop campaign.
- **Phase B (parallel/cheap):** Run exp **8** (aggregation ablation) and exp **6** (PNA) early — both config-only, no code change. They inform whether aggregation worth strengthening.
- **Phase C (fallback tier):** If Phase A < 0.95, run exps **4, 5, 9** stack on Phase-A winner; then exps **7, 10, 11, 12** as tail. Exps 3 and 9 mutually exclusive — run one with better baseline curve.
- **Per-experiment success threshold:** experiment "pass" if val C@200 improve by > 0.003 over baseline (above run-to-run noise band, ≈ 0.002). Non-passers do not feed forward.

## Files to modify (critical map)

| File | Role |
|---|---|
| `weaver/weaver/nn/model/TrackPreFilter.py` | Core model — edge features (exp 1), aggregation mode (exp 6), OC scorer head (exp 5), loss-type dispatch |
| `weaver/weaver/nn/model/HierarchicalGraphBackbone.py` | `build_cross_set_edge_features` already in place — reuse for exp 1 |
| `weaver/weaver/nn/model/prefilter_losses.py` | NEW — listwise CE / InfoNCE / logit adjust / OC losses |
| `part/networks/lowpt_tau_TrackPreFilter.py` | Thread CLI flags into constructor kwargs |
| `part/train_prefilter.py` | New CLI flags, loss-type branch, augmentation call, EMA teacher |
| `part/train_prefilter.sh` | Example invocations per experiment |
| `part/pretrain_prefilter_mpm.py` | NEW — SSL pretraining entry point (exp 10) |
| `part/scripts/cache_xgb_scores.py` | NEW — XGBoost per-track score cache (exp 7) |
| `part/utils/training_utils.py` or new `part/utils/set_augmentation.py` | NEW — set-friendly augmentations (exp 11) |
| `part/data/low-pt/lowpt_tau_trackfinder.yaml` | Extra `track_xgb_score` input feature (exp 7) — regenerate `.auto.yaml` after |
| `part/reports/experiment_log.md` | Append one row per experiment with C@K metrics and commit hash |

## Verification (end-to-end test plan)

Every experiment:
1. **Smoke test on local MPS** — `bash train_prefilter.sh --run-name smoke_<exp> --epochs 1 --steps-per-epoch $((1500 / 16)) --batch-size 16 --num-workers 0 --device mps` confirm new flag / loss / aggregation compile and produce finite losses end-to-end (per `feedback_always_smoke_test.md` and `feedback_smoke_test_size.md` conventions).
2. **Server run** — launch via canonical command above, monitor with ping pattern in `part/reports/server_workflow.md`, capture training log + final checkpoint.
3. **Inference metrics** — run `part/diagnostics/evaluate_prefilter.py` on full val set; confirm C@200 > baseline + 0.003. Append to `part/reports/experiment_log.md`.
4. **Cascade passthrough check (exp 1-6 only, once per winning experiment)** — run `part/eval_couple_reranker.sh` with new prefilter checkpoint swapped into cascade, confirm C@100 does not regress by more than 0.005 from pre-experiment cascade baseline.
5. **Commit on `debug-couple-reranker-eval` (or new branch)** — one commit per experiment, match `caveman:caveman-commit` convention if active.

---

## Status of planning phases

- ✅ Phase 1 — local scrape (agent a907fd47)
- ✅ Phase 1 — audit (agent a7afdcae)
- ✅ Phase 1 — SOTA survey (agent a76c46e5)
- ✅ Phase 2 — plan synthesis (Plan agent)
- ✅ Phase 3 — spot-check verified line numbers (grep on `TrackPreFilter.py`, `HierarchicalGraphBackbone.py`, `lowpt_tau_TrackPreFilter.py`)
- ⏭️ Phase 4 — this file
- ⏭️ Phase 5 — ExitPlanMode