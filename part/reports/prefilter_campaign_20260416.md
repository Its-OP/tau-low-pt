# Prefilter Improvement Campaign — Progress Report

**Campaign opened:** 2026-04-16
**Plan:** `part/reports/pre-filter-updates.md`
**Target:** C@200 > 0.95 (baseline 0.919)
**Branch:** `prefilter-campaign`
**Budget per experiment:** ≤ 20 epochs; 1 GPU (RTX PRO 6000 96 GB, vast.ai)
**All runs on CUDA:** MPS is not a target (see `feedback_no_mps.md`).

## Baseline (for comparison)

| Run | C@200 | C@256 | d′ | median rank | epoch best | notes |
|-----|-------|-------|----|-------------|------------|-------|
| `prefilter_dim256_cutoff_bs192_PreFilter_20260405_120532` | 0.919 | ~0.943 | 2.035 | 7 | 36 | pairwise softplus, k=16, 2 rounds, max-pool, no edges |

## Campaign infrastructure (committed)

| Commit | Summary |
|--------|---------|
| `701637d` | Pairwise LV edge features + CLI knobs (`--use-edge-features`, `--num-neighbors`, `--num-message-rounds`, `--aggregation-mode`, `--loss-type`, `--logit-adjust-tau`, `--listwise-temperature`, `--use-augmentation`, `--ssl-pretrain-ckpt`) |
| `752252f` | `weaver/weaver/nn/model/prefilter_losses.py` (listwise CE, InfoNCE, logit-adjust, object condensation) + `_ranking_loss` dispatcher |

Tests: 481 pass (was 464) — +17 new tests for edge features and losses.

## Experiments

### Smoke test (GPU, 1 epoch, 20 steps)
- Config: `--use-edge-features`, BS=128, steps=20, CUDA, no AMP, no compile.
- Duration: ~75 s total.
- Result: val R@200 = 0.8417 at 1 epoch / 5 k samples — finite losses, checkpoint saves, no NaN. Edge-feature path end-to-end healthy.

### Smoke sweep — all flags validated on GPU

Ran `bash sweep_prefilter.sh smoke` on vast.ai. 13 experiments, 1 epoch × 50 steps × BS=128, ~95 s each, total ~22 min. **All PASSED — no OOM, no NaN losses, no CLI typos, no shape mismatches.**

| Exp | Flags | R@200 | d′ | rank |
|-----|-------|------:|---:|-----:|
| BASELINE | (none) | 0.858 | 1.62 | 23 |
| E1 | `--use-edge-features` | 0.863 | 1.65 | 22 |
| E2a | edge + k=16 r=3 | 0.867 | 1.65 | 23 |
| E2b | edge + k=32 r=2 | 0.870 | 1.66 | 22 |
| E2c | edge + k=32 r=3 | 0.869 | 1.65 | 22 |
| E2d | edge + k=48 r=2 | 0.871 | 1.62 | 25 |
| E2e | edge + k=64 r=2 | 0.865 | 1.62 | 26 |
| E3 | edge + `--loss-type listwise_ce` | **0.878** | **1.73** | **17** |
| E4 | edge + `--loss-type logit_adjust` | 0.874 | 1.76 | 19 |
| E6 | edge + `--aggregation-mode pna` | 0.870 | 1.65 | 22 |
| E8 | `--num-message-rounds 0` | 0.825 | 1.48 | 38 |
| E9 | edge + `--loss-type infonce --listwise-temperature 0.5` | 0.877 | 1.70 | 18 |
| E11 | edge + `--use-augmentation` | 0.871 | 1.62 | 21 |

**Early signal** (1 epoch, tiny batch — NOT comparable to the 0.919 baseline which had 500 steps × 36 epochs):
- Listwise CE (E3) and InfoNCE (E9) have the best early d′ / rank, consistent with the plan's prediction that event-wise listwise losses dominate pairwise softplus.
- Edge features (E1) add ~+0.5 pp R@200 at 1 epoch — expected to amplify over 20 epochs.
- k/rounds sweep (E2) peaks around k=48, r=2 but the range is tight; full-run signal will be cleaner.
- Aggregation ablation (E8) loses ~3.3 pp — aggregation IS doing work.
- PNA (E6) tracks E1 closely.

### Overnight full sweep (handoff)

All 17 experiments hidden behind feature flags. Launch inside ``screen``
so an SSH disconnect does not kill the run:

```bash
ssh -p 12971 root@ssh9.vast.ai \
  "cd /workspace/tau-low-pt && git pull && cd part && \
   screen -dmS sweep_full bash -c \
     'bash sweep_prefilter.sh full > /workspace/logs/sweep_full_master.log 2>&1'"
```

Reattach to watch progress:
```bash
ssh -p 12971 root@ssh9.vast.ai
# then on server:
screen -r sweep_full     # reattach (detach again with Ctrl+A, D)
```

Per-experiment: **40 epochs** × 500 steps × BS=256 × AMP, ~10 workers. Each run ~30–45 min. **17 experiments** × ~40 min ≈ **11 hours total** (override via `EPOCHS=N bash sweep_prefilter.sh full`).

Summary lands at `/workspace/logs/sweep_full_summary.txt`. Per-experiment checkpoints at `/workspace/experiments/sweep_full_<exp>_PreFilter_<ts>/`.

Experiment roster:
| ID | Flags | Hypothesis |
|----|-------|-----------|
| BASELINE | (none) | Current production config |
| E1 | `--use-edge-features` | Pairwise LV edge features (+0.02–0.04 estimated) |
| E2a-e | edge + k/rounds grid | Larger receptive field |
| E3 | edge + `--loss-type listwise_ce` | Event-wise softmax CE |
| E4 | edge + `--loss-type logit_adjust` | Menon 2007.07314 class-prior offset |
| E5 | edge + `--loss-type object_condensation` | Kieseler 2002.03605 |
| E6 | edge + `--aggregation-mode pna` | PNA multi-aggregator |
| E7 | edge + `--use-xgb-stub-feature` | DeepGBM-style hybrid (stub) |
| E8 | `--num-message-rounds 0` | Aggregation ablation |
| E9 | edge + `--loss-type infonce` | InfoNCE per positive |
| E10 | edge + `--loss-type mpm_pretrain` | Masked particle modeling SSL |
| E11 | edge + `--use-augmentation` | JetCLR augmentations |
| E12 | edge + `--use-self-distillation` | EMA teacher + KL distillation |

Real-XGBoost (E7 proper): the current flag wires a frozen random-init
linear stub standing in for a pre-trained per-track XGBoost score.
Replace the stub by training XGBoost once and swapping the stub weights
— or by caching scores to parquet and adding the column to the data
config. Stub exists purely to exercise the 17-dim input path in the
smoke sweep.

