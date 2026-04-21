# Prefilter Improvement Campaign — Experiment TODO

## Context

The Stage-1 prefilter (`TrackPreFilter`) has plateaued at **C@200 = 0.919, C@256 ≈ 0.943** on the pT ≥ 0.5 GeV dataset, reached at epoch 36 of the `prefilter_dim256_cutoff_bs192_PreFilter_20260405_120532` run. Beyond epoch 36 train R@200 keeps climbing 0.932 → 0.957 while val stagnates/drifts — classic overfit. A per-track XGBoost on the same 16 raw features is only "slightly worse", so the current 400K-param deep model buys little over feature-engineered boosting.

The campaign targets **C@200 > 0.95** (≥ +3.1 pp) within **≤ 20 epochs per experiment**, subject to hard architectural constraints (no attention, no dense-tensor compression, no GNN, raw features only, no re-running locally-failed approaches). Downstream cascade (Stage 2 / Stage 3) stays frozen — all experiments must preserve the prefilter's per-track-scores interface so the cascade consumes `selected_indices` unchanged (checked via `part/train_cascade.py:316-330`).

### User constraints

1. **No attention.** Rules out Transformer, Set Transformer ISAB, Perceiver IO, Nyström/Performer/Longformer, NPT/SAINT (intersample attention), TabR, Hopfield.
2. **No dense-tensor compression.** Rules out any bottleneck that pools the event into a fixed-size vector — the 0.27 % positive signal would be lost.
3. **≤ 20 epochs** per experiment.
4. **No GNN.** All graph variants (GravNet / EdgeConv backbone / multi-space composite) already failed locally.
5. **Raw detector-level features only.** No reconstruction-stage cheating.
6. **Biggest-expected-gain ordering.**
7. **Target C@200 > 0.95.**

### Prior failures (do NOT re-propose)

From `project_failed_experiments.md` and `project_noise_neighborhoods.md`: ASL, OHEM, pairwise-LV-in-GNN-message-passing, GravNet, EdgeConv backbone, triplet scoring, multi-space composite graphs, per-track ρ-candidate count.

### Current architecture (verified)

- `TrackPreFilter` — `weaver/weaver/nn/model/TrackPreFilter.py:23`
- Mode `mlp`; per-track Conv1d (16 → 256) + 2 rounds of k-NN(k = 16, η-φ) max-pool aggregation + scorer head; 400 K params
- Neighbor dim `= 2 · hidden_dim` at `TrackPreFilter.py:162` (max-pool only). A richer `aggregation_mode = 'pna'` at line 159 exists but is untested (`5 · hidden_dim`, cat of mean/max/min/std).
- Wrapper `part/networks/lowpt_tau_TrackPreFilter.py:38-82` hard-codes `num_neighbors = 16, num_message_rounds = 2`. No CLI flag yet.
- `ranking_temperature_start = ranking_temperature_end = 1.0` defaults (frozen). The epoch-36 run used 2.0 → 0.5 (wrapper override in that branch); the current wrapper is already frozen.
- `build_cross_set_edge_features` already exists at `HierarchicalGraphBackbone.py:232` — concatenates `[center, neighbor − center, pairwise_lv_fts(center_lv, neighbor_lv)]`. The prefilter's `mlp` mode does NOT use it today; it only max-pools node features.

---

## Cross-experiment infrastructure (build ONCE before experiments 1, 2, 3, 4, 5, 8, 10, 11)

1. **CLI + wrapper plumbing** — `part/train_prefilter.py` argparse block (~line 337) gains:
   - `--num-neighbors` (int, default 16)
   - `--num-message-rounds` (int, default 2)
   - `--aggregation-mode` (choices: `mlp`, `pna`; default `mlp`)
   - `--use-edge-features` (flag)
   - `--loss-type` (choices: `pairwise`, `listwise_ce`, `infonce`, `logit_adjust`, `object_condensation`; default `pairwise`)
   - `--logit-adjust-tau` (float, default 1.0)
   - `--temperature` (float; sets both start and end when provided)
   - `--use-augmentation` (flag)
   - `--ssl-pretrain-ckpt` (path)
   - `--extra-feature xgb_score` (optional flag to enable the XGB-score input channel)

   Thread all into `part/networks/lowpt_tau_TrackPreFilter.py:22-108` via the existing kwargs dict — do NOT duplicate the wrapper.

2. **Edge-feature path in `mlp` aggregation** — `TrackPreFilter.py:162` + `_forward_mlp` (lines 410-434) must:
   - propagate the `(B, 4, P)` `lorentz_vectors` through the forward (signature already has it at line 301).
   - call `build_cross_set_edge_features` from `HierarchicalGraphBackbone.py:232` on each round's `cross_set_knn` output.
   - set `neighbor_input_dim = 2 · hidden_dim + pairwise_lv_fts_dim` when `use_edge_features=True`. Use `pairwise_lv_fts_dim = 4` (ln ΔR, ln m², ln kT, ln z) — inspect `ParticleTransformer.pairwise_lv_fts` to confirm exact dim.
   - add charge-product channel separately (1 extra dim), reusing the existing charge input feature.

3. **Loss module** — new file `weaver/weaver/nn/model/prefilter_losses.py`:
   - `listwise_ce_loss(scores, labels, mask)` — event-wise softmax over valid tracks, CE against the 3-hot target (or averaged-over-positives loss).
   - `infonce_in_event(scores, labels, mask, temperature)` — 1-vs-rest InfoNCE per positive.
   - `logit_adjust_offset(num_pos, num_neg, tau)` — constant offset added to logits during training (Menon 2007.07314).
   - `object_condensation_loss(embeddings, beta, labels, mask, q_min=0.1)` — Kieseler 2002.03605 attractive/repulsive potential + β regularizer.
   - Wired into `TrackPreFilter._ranking_loss` via a `loss_type` branch; default path stays unchanged for reproducibility.

4. **Set augmentation module** — `part/utils/training_utils.py` (or new `part/utils/set_augmentation.py`) with three transforms, each taking `(features, points, lorentz_vectors, mask, labels)`:
   - `TrackDropout(p=0.10)` — randomly zero mask non-positive tracks.
   - `FeatureJitter(sigma=0.05)` — Gaussian smear on log features.
   - `EtaPhiRotation()` — global rotation of φ (and eta-flip with p=0.5).
   Called in `train_prefilter.py` right before `compute_loss`.

5. **XGBoost score cache (one-off, only if exp 7 runs)** — `part/scripts/cache_xgb_scores.py` trains a per-track XGBoost on the same 16 features and writes a parquet column `track_xgb_score` aligned to the existing data layout. Loaded by adding `- track_xgb_score` to the `pf_features` block in `part/data/low-pt/lowpt_tau_trackfinder.yaml`. Regenerate the `.auto.yaml` afterwards.

6. **SSL pretrain entry point (only if exp 10 runs)** — `part/pretrain_prefilter_mpm.py` reuses the dataloader from `train_prefilter.py:404-485`, drops the label head, masks 15 % of tracks, reconstructs their 16 features from the k-NN aggregated context, MSE loss. Saves only `track_mlp.*` and `neighbor_mlps.*` weights in a checkpoint loadable by the existing `TrackPreFilter.__init__`.

7. **Metrics plumbing (unchanged)** — the existing `part/diagnostics/evaluate_prefilter.py` and `reports/prefilter_analysis_*.md` already produce per-event parquet + C@K tables; each experiment reuses them.

### Canonical server-side command (one-off per experiment)

```bash
ssh -p 26893 root@69.63.236.192 "cd /workspace/tau-low-pt/part && \
  nohup env RUN_NAME=<exp-name> bash train_prefilter.sh <flags> > \
  /workspace/logs/<exp-name>.log 2>&1 & echo PID=\$!"
```

Monitor with the pattern in `part/reports/server_workflow.md`. Each experiment must land C@K metrics in `part/reports/experiment_log.md` and a checkpoint in `part/models/debug_checkpoints/<exp-name>_<timestamp>/`.

---

## TOP 3 — biggest shots at closing the 0.919 → 0.95 gap

### 1. `pairwise-edge-features`
- **Hypothesis.** ParT-style pairwise edge features (ln ΔR, ln m², ln kT, ln z, q_i · q_j) injected at every k-NN aggregation round give the network the physics-structured inter-track signal that max-pool of raw node features cannot synthesize. `build_cross_set_edge_features` already exists at `HierarchicalGraphBackbone.py:232` and is proven stable in the Stage-2 cascade — we are wiring an existing primitive into the prefilter, not reinventing it.
- **Estimated ΔC@200.** +0.020 … +0.040 (largest single-experiment estimate in the campaign)
- **Change scope.**
  - `weaver/weaver/nn/model/TrackPreFilter.py:162` — switch `neighbor_input_dim` to `2 · hidden_dim + pairwise_lv_fts_dim + 1` when `use_edge_features=True`.
  - `_forward_mlp` at `TrackPreFilter.py:410-434` — thread `lorentz_vectors`, call `build_cross_set_edge_features`, `cross_set_gather` the edge features, concat with `[current, max_pooled]` before the neighbor MLP.
  - `part/networks/lowpt_tau_TrackPreFilter.py:38-82` — pass `use_edge_features=True`.
  - CLI flag `--use-edge-features` (infra §1).
- **Training budget.** 20 epochs from scratch. Expected per-epoch overhead ~1.3 × baseline.
- **Kill signal.** Val R@200 < 0.92 by epoch 10, OR NaN loss (detach LV features inside `pairwise_lv_fts` — see `HierarchicalGraphBackbone.py:272-289` for the established pattern).
- **Cascade compatibility.** Output interface unchanged → Stage 2 / 3 do NOT need retraining.

### 2. `receptive-field-sweep` (combined k and rounds, gated by exp 1)
- **Hypothesis.** Current k = 16 × 2 rounds ≤ 256 tracks; events hold 600 – 1130 tracks, so the receptive field sees only 24 – 45 % of the event. Bigger k and/or 3 rounds widens the per-track view without the O(P²) blow-up of attention. Expected to stack (imperfectly) with exp 1 because larger neighborhoods also have more edges.
- **Estimated ΔC@200.** +0.010 … +0.030 on top of exp 1.
- **Change scope.** Pure config — `part/networks/lowpt_tau_TrackPreFilter.py:42-43`. Sweep `(num_neighbors, num_message_rounds) ∈ {(16, 3), (32, 2), (32, 3), (48, 2), (64, 2)}`. Infra §1 exposes these as CLI flags.
- **Training budget.** 5 runs × 20 epochs. Winner becomes the baseline for exp 3 onward.
- **Kill signal.** Per-epoch wall-clock > 2.5 × baseline at k ≥ 48 → skip that grid point. No correctness risk.
- **Cascade compatibility.** Full.

### 3. `listwise-ce-loss`
- **Hypothesis.** Current pairwise softplus ranks 3 positives against 50 sampled negatives per event (`TrackPreFilter.ranking_num_samples = 50` default) — discards 99 % of the background structure. An event-wise softmax-CE uses all 600 – 1100 in-event negatives per step and directly optimizes the rank position of positives, which is the exact target of the C@K metric. Theoretically (2411.07770) CCE upper-bounds NDCG / MRR more tightly than BCE/BPR.
- **Estimated ΔC@200.** +0.010 … +0.020.
- **Change scope.**
  - `weaver/weaver/nn/model/prefilter_losses.py:listwise_ce_loss` (infra §3).
  - `TrackPreFilter._ranking_loss` — new `loss_type='listwise_ce'` branch; keep the pairwise path intact.
  - `part/train_prefilter.py:148-150` — thread `args.loss_type`.
- **Training budget.** 20 epochs from scratch (apply on top of the exp-2 winner's config).
- **Kill signal.** Val R@200 < 0.92 by epoch 10 despite loss decreasing → listwise formulation not driving separation → revert.
- **Cascade compatibility.** Full.

**Expectation:** if exp 1 + 2 + 3 each land at their mid-estimate and stack with ~70 % efficiency, C@200 reaches 0.919 + 0.70 · (0.030 + 0.020 + 0.015) ≈ **0.965**. If stacking is weaker, we land at ~0.95 anyway. If below 0.95 after these three, proceed to the next tier.

---

## Next tier — run if top-3 stops short of 0.95

### 4. `logit-adjust-loss`
- **Hypothesis.** Menon 2007.07314's logit adjustment adds `−τ · log(π_neg / π_pos)` at train time (≈ `−τ · log(375)` for 3/1130). Provably optimizes balanced error; orthogonal to listwise CE (exp 3) so it stacks. Not the same as ASL (failed): no hard clipping, no focal down-weighting.
- **Estimated ΔC@200.** +0.005 … +0.015.
- **Change scope.** `prefilter_losses.py:logit_adjust_offset`; called inside `TrackPreFilter._ranking_loss` before the score differences are taken. Sweep τ ∈ {0.5, 1.0, 2.0}.
- **Training budget.** 3 × 15 epochs (fine-tune from the top-3 winner).
- **Kill signal.** Val R@200 regresses vs top-3 winner by > 0.005 after epoch 10.
- **Cascade compatibility.** Full.

### 5. `object-condensation-loss`
- **Hypothesis.** Kieseler 2002.03605 condenses "a few objects in a noisy set" via a β-weighted potential loss — exactly our setting. Architecture-agnostic (runs on any per-track embedding), no GNN needed. The `clustering_dim` parameter is already reserved at `lowpt_tau_TrackPreFilter.py:33` but unused.
- **Estimated ΔC@200.** +0.005 … +0.020 (high variance; OC is hyperparameter-sensitive).
- **Change scope.**
  - `TrackPreFilter.scorer` at `TrackPreFilter.py:187-195` — add a second Conv1d(hidden_dim, 1) β head and a Conv1d(hidden_dim, clustering_dim) embedding head; keep the score = `β · max(embedding · prototype)` output for the downstream API.
  - `prefilter_losses.py:object_condensation_loss` — attractive/repulsive + β regularizer.
- **Training budget.** 2 × 20 epochs (one with q_min=0.1, one with q_min=0.5).
- **Kill signal.** β collapses to uniform (per-event β std < 0.01) by epoch 5. If both runs collapse, drop.
- **Cascade compatibility.** Output is still a monotonic per-track score; top-K selection unaffected.

### 6. `pna-aggregation`
- **Hypothesis.** Cheap config flip — `aggregation_mode = 'pna'` at `TrackPreFilter.py:159` concatenates `[current, mean, max, min, std]` instead of `[current, max]`. Doubles the aggregation expressiveness without changing complexity class; untested in the repo. Should be compatible with edge features (exp 1) — concat channels independently.
- **Estimated ΔC@200.** +0.003 … +0.015.
- **Change scope.** Pure config — `part/networks/lowpt_tau_TrackPreFilter.py` sets `aggregation_mode='pna'`. Code path at `TrackPreFilter.py:157-163` already exists.
- **Training budget.** 20 epochs (ideally combined with exp 1's edge features for the final baseline).
- **Kill signal.** Val R@200 < exp-1 baseline by epoch 10.
- **Cascade compatibility.** Full.

### 7. `xgb-score-as-feature`
- **Hypothesis.** DeepGBM-style hybrid. Since per-track XGBoost ≈ current deep model, feeding the XGBoost score as a 17-th input feature lets the deep model spend its capacity on set-level context instead of re-learning per-track physics. Simple, no arch change.
- **Estimated ΔC@200.** +0.005 … +0.015.
- **Change scope.**
  - `part/scripts/cache_xgb_scores.py` (infra §5) — one-time XGB training + parquet write.
  - `part/data/low-pt/lowpt_tau_trackfinder.yaml:53-83` — add `- track_xgb_score` to `pf_features`. Regenerate `.auto.yaml` next to the config.
  - `input_dim = 17` propagates automatically via `len(data_config.input_dicts['pf_features'])` at `lowpt_tau_TrackPreFilter.py:36`.
- **Training budget.** 15 – 30 min XGBoost + 20 prefilter epochs.
- **Kill signal.** Val R@200 < XGBoost-alone R@200 + 0.005 → model collapsing to XGBoost → drop.
- **Cascade compatibility.** Full, but the XGBoost score cache must also be generated on the server for inference.

### 8. `aggregation-ablation`
- **Hypothesis.** Diagnostic. Trains `num_message_rounds = 0` (pure per-track MLP). If C@200 barely drops, aggregation is dead weight and we should concentrate future work on OC / listwise / XGB-feature routes. If C@200 collapses, the aggregation IS doing work and the focus stays on making it richer (exps 1, 2, 6).
- **Estimated ΔC@200.** −0.04 … +0.005 (expected: negative if aggregation helps).
- **Change scope.** Pure config — `num_message_rounds=0` at `lowpt_tau_TrackPreFilter.py:43`. Model already tolerates this via `range(self.num_message_rounds)` at `TrackPreFilter.py:409`.
- **Training budget.** 15 epochs.
- **Kill signal.** N/A — diagnostic, always run to completion. Use results to re-prioritize the tail of this list.
- **Cascade compatibility.** Full.

### 9. `infonce-in-event`
- **Hypothesis.** Alternative to listwise-CE (exp 3). InfoNCE with all in-event background as negatives, 3 positives as anchors. Stronger representation signal than softplus pairwise. Mutually exclusive with exp 3.
- **Estimated ΔC@200.** +0.005 … +0.015 (pick exp 3 or exp 9 based on which has better baseline curve at epoch 10).
- **Change scope.** `prefilter_losses.py:infonce_in_event`; `TrackPreFilter._ranking_loss` `loss_type='infonce'` branch.
- **Training budget.** 15 epochs from scratch.
- **Kill signal.** Val R@200 < exp-3 baseline by epoch 10 → exp 3 won → stop.
- **Cascade compatibility.** Full.

### 10. `masked-particle-modeling-ssl`
- **Hypothesis.** Heinrich 2401.13537's MPM objective — reconstruct masked track features from the k-NN-aggregated context — is architecture-agnostic. Pretraining unlocks unlabeled-regime learning; fine-tuning keeps the labeled objective. Exposes more per-track structure to the encoder without changing the per-track output API.
- **Estimated ΔC@200.** +0.005 … +0.020 (variance high; gain depends on how much unlabeled structure the encoder can absorb).
- **Change scope.**
  - `part/pretrain_prefilter_mpm.py` (infra §6).
  - `train_prefilter.py:593-621` — resume block gains a `--ssl-pretrain-ckpt` branch that loads backbone weights only (not scorer head).
- **Training budget.** 10 epochs pretrain (one-time artifact) + 15 epochs fine-tune. Fine-tune fits within the 20-epoch cap.
- **Kill signal.** Pretraining reconstruction MSE hits near-variance ceiling by epoch 5 → no learnable structure → abort pretraining, drop experiment.
- **Cascade compatibility.** Full.

### 11. `jetclr-augmentations`
- **Hypothesis.** Track dropout (p=0.1) + Gaussian feature smearing (σ=0.05) + η-φ rotation from JetCLR (Dillon 2108.04253). Regularizes the late-epoch overfit observed at epoch 36.
- **Estimated ΔC@200.** +0.003 … +0.010.
- **Change scope.** Infra §4 (`set_augmentation` module); call in `train_prefilter.py` right before `compute_loss`.
- **Training budget.** 20 epochs from scratch (on top of the winning config).
- **Kill signal.** Val R@200 plateaus earlier than baseline → augmentation breaking a symmetry. Disable η-φ rotation first (impact parameters are not rotation-invariant), retry with dropout + smear only.
- **Cascade compatibility.** Full (train-only transform).

### 12. `self-distillation-ema`
- **Hypothesis.** EMA teacher (2601.19897) — maintain an exponentially-moving-average copy of the student, use teacher's soft scores as an auxiliary regression target. Known rare-class boost; tiny compute overhead (one extra forward pass on the teacher).
- **Estimated ΔC@200.** +0.003 … +0.010.
- **Change scope.**
  - `train_prefilter.py` — add EMA module (torch-standard pattern), KL-divergence loss between student and teacher logits.
  - No architectural change.
- **Training budget.** 20 epochs from scratch.
- **Kill signal.** KL loss dominates total loss (> 0.5× scalar ratio) → teacher following student too closely; reduce EMA decay (e.g., 0.999 → 0.995) or weight.
- **Cascade compatibility.** Full.

---

## Ordering and stop conditions

- **Phase A (priority):** Build infra §1 + §2 + §3, then run exps **1 → 2 → 3** in sequence. Each one's winning checkpoint becomes the baseline for the next. If C@200 ≥ 0.95 after exp 3, declare victory and stop the campaign.
- **Phase B (parallel/cheap):** Run exp **8** (aggregation ablation) and exp **6** (PNA) early — both are config-only, no code change. They inform whether aggregation is worth strengthening.
- **Phase C (fallback tier):** If Phase A < 0.95, run exps **4, 5, 9** stacking on the Phase-A winner; then exps **7, 10, 11, 12** as the tail. Exps 3 and 9 are mutually exclusive — run the one with the better baseline curve.
- **Per-experiment success threshold:** an experiment "passes" if val C@200 improves by > 0.003 over its baseline (above the run-to-run noise band, ≈ 0.002). Non-passers do not feed forward.

## Files to modify (critical map)

| File | Role |
|---|---|
| `weaver/weaver/nn/model/TrackPreFilter.py` | Core model — edge features (exp 1), aggregation mode (exp 6), OC scorer head (exp 5), loss-type dispatch |
| `weaver/weaver/nn/model/HierarchicalGraphBackbone.py` | `build_cross_set_edge_features` already in place — reuse for exp 1 |
| `weaver/weaver/nn/model/prefilter_losses.py` | NEW — listwise CE / InfoNCE / logit adjust / OC losses |
| `part/networks/lowpt_tau_TrackPreFilter.py` | Thread CLI flags into constructor kwargs |
| `part/train_prefilter.py` | New CLI flags, loss-type branch, augmentation call, EMA teacher |
| `part/train_prefilter.sh` | Example invocations for each experiment |
| `part/pretrain_prefilter_mpm.py` | NEW — SSL pretraining entry point (exp 10) |
| `part/scripts/cache_xgb_scores.py` | NEW — XGBoost per-track score cache (exp 7) |
| `part/utils/training_utils.py` or new `part/utils/set_augmentation.py` | NEW — set-friendly augmentations (exp 11) |
| `part/data/low-pt/lowpt_tau_trackfinder.yaml` | Extra `track_xgb_score` input feature (exp 7) — regenerate `.auto.yaml` afterwards |
| `part/reports/experiment_log.md` | Append one row per experiment with C@K metrics and commit hash |

## Verification (end-to-end test plan)

For every experiment:
1. **Smoke test on local MPS** — `bash train_prefilter.sh --run-name smoke_<exp> --epochs 1 --steps-per-epoch $((1500 / 16)) --batch-size 16 --num-workers 0 --device mps` to confirm the new flag / loss / aggregation compiles and produces finite losses end-to-end (per the `feedback_always_smoke_test.md` and `feedback_smoke_test_size.md` conventions).
2. **Server run** — launch via the canonical command above, monitor with the ping pattern in `part/reports/server_workflow.md`, capture the training log + final checkpoint.
3. **Inference metrics** — run `part/diagnostics/evaluate_prefilter.py` on the full val set; confirm C@200 > baseline + 0.003. Append to `part/reports/experiment_log.md`.
4. **Cascade passthrough check (exp 1-6 only, once per experiment that wins)** — run `part/eval_couple_reranker.sh` with the new prefilter checkpoint swapped into the cascade and confirm C@100 does not regress by more than 0.005 from the pre-experiment cascade baseline.
5. **Commit on `debug-couple-reranker-eval` (or a new branch)** — one commit per experiment, matching the `caveman:caveman-commit` convention if active.

---

## Status of the planning phases

- ✅ Phase 1 — local scrape (agent a907fd47)
- ✅ Phase 1 — audit (agent a7afdcae)
- ✅ Phase 1 — SOTA survey (agent a76c46e5)
- ✅ Phase 2 — plan synthesis (Plan agent)
- ✅ Phase 3 — spot-check verified line numbers (grep on `TrackPreFilter.py`, `HierarchicalGraphBackbone.py`, `lowpt_tau_TrackPreFilter.py`)
- ⏭️ Phase 4 — this file
- ⏭️ Phase 5 — ExitPlanMode
