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

### Exp 1 — `pairwise-edge-features` (in flight)
- Config: `--use-edge-features`, BS=256, 500 steps/epoch, 20 epochs, AMP, 10 workers, CUDA.
- Launched: 2026-04-16 (server time 19:28).
- Status: **running** — log `/workspace/logs/exp1_edge.log`, experiment dir `/workspace/experiments/exp1_edge_PreFilter_<ts>/`.
- Success criterion: C@200 > 0.922 (+0.003 over baseline 0.919).

_Results to be appended below as epochs complete._

