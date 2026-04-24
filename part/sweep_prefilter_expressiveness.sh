#!/bin/bash
# =============================================================================
# Prefilter expressiveness sweep — target P@256 > 0.90.
#
# Usage:
#   bash sweep_prefilter_expressiveness.sh smoke            # 1-ep validation, all experiments
#   bash sweep_prefilter_expressiveness.sh full             # 40-ep campaign, all experiments
#   bash sweep_prefilter_expressiveness.sh full P1 P2       # only P1, P2
#
# Modes:
#   smoke — 1 epoch, BS=128, 50 steps/ep, 4 workers, no AMP. Shape/wiring check.
#   full  — 40 epochs, BS=256, 500 steps/ep, 10 workers, AMP on. Real campaign run.
#
# Every experiment selects checkpoints by perfect_at_256 (not R@200 —
# the diagnostic chain established P@256 as the primary target).
#
# Experiment list:
#   E2a_anchor  — baseline re-run (E2a config), same harness, seed floor.
#   P1          — per-feature embedding (grouped 1×1 conv, embed_dim=32).
#   P2          — SE-style feature gate on track_mlp output.
#   P3          — FiLM event-context modulation of track_mlp output.
#   P4          — soft-attention aggregation replaces max-pool per round.
#   P6          — two-tier prefilter (all tracks → top-600 → top-256).
# =============================================================================
set -euo pipefail

MODE="${1:-smoke}"
shift || true
REQUESTED_EXPERIMENTS=("$@")

if [ "${MODE}" != "smoke" ] && [ "${MODE}" != "full" ]; then
    echo "ERROR: first arg must be 'smoke' or 'full', got '${MODE}'"
    exit 1
fi

# ---- Resolve conda ----
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
elif [ -d "/opt/miniforge3" ]; then
    CONDA_BASE="/opt/miniforge3"
elif [ -d "/root/miniconda3" ]; then
    CONDA_BASE="/root/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
else
    echo "ERROR: conda not found."
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate part

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${LOG_DIR:-/workspace/logs}"
EXP_DIR="${EXP_DIR:-/workspace/experiments}"
mkdir -p "${LOG_DIR}" "${EXP_DIR}"

# ---- Mode-specific defaults (each overridable via env var) ----
if [ "${MODE}" = "smoke" ]; then
    EPOCHS="${EPOCHS:-1}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    AMP_FLAG="${AMP_FLAG:-}"
    KEEP_BEST_K="${KEEP_BEST_K:-1}"
else
    EPOCHS="${EPOCHS:-40}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
    NUM_WORKERS="${NUM_WORKERS:-10}"
    AMP_FLAG="${AMP_FLAG:---amp}"
    KEEP_BEST_K="${KEEP_BEST_K:-3}"
fi

# ---- Shared invariant args ----
# Every experiment selects checkpoints by P@256 (the sweep target).
COMMON_ARGS_SINGLE_TIER=(
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml
    --data-dir data/low-pt/train/
    --val-data-dir data/low-pt/val/
    --network networks/lowpt_tau_TrackPreFilter.py
    --experiments-dir "${EXP_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --steps-per-epoch "${STEPS_PER_EPOCH}"
    --num-workers "${NUM_WORKERS}"
    --device cuda:0
    --save-every 2
    --keep-best-k "${KEEP_BEST_K}"
    --checkpoint-criterion perfect_at_256
)
COMMON_ARGS_TWO_TIER=(
    --data-config data/low-pt/lowpt_tau_trackfinder.yaml
    --data-dir data/low-pt/train/
    --val-data-dir data/low-pt/val/
    --network networks/lowpt_tau_TwoTierPreFilter.py
    --experiments-dir "${EXP_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --steps-per-epoch "${STEPS_PER_EPOCH}"
    --num-workers "${NUM_WORKERS}"
    --device cuda:0
    --save-every 2
    --keep-best-k "${KEEP_BEST_K}"
    --checkpoint-criterion perfect_at_256
)
if [ -n "${AMP_FLAG}" ]; then
    COMMON_ARGS_SINGLE_TIER+=("${AMP_FLAG}")
    COMMON_ARGS_TWO_TIER+=("${AMP_FLAG}")
fi

# ---- Experiment definitions ----
# Value = two-element tuple (network_family, extra_args_string).
# ``network_family`` is either "single" or "two_tier" — selects which
# COMMON_ARGS block to prepend.
declare -A NETWORK_FAMILY=(
    [E2a_anchor]="single"
    [P1]="single"
    [P2]="single"
    [P3]="single"
    [P4]="single"
    [P6]="two_tier"
)
declare -A EXPERIMENT_ARGS=(
    # Baseline = train_prefilter.py + wrapper defaults: edges ON, k=16,
    # r=3, dropout=0.1, P1 per-feature embedding ON (per_feature, dim=32)
    # as of the 2026-04-23 sweep. Each follow-up row lists ONLY what it
    # changes on top of that baseline.
    #
    # E2a_anchor reproduces the pre-P1 baseline by explicitly turning
    # the per-feature embedding OFF; P1 is now a no-op row kept for
    # lineage documentation only.
    [E2a_anchor]="--feature-embed-mode none"
    [P1]=""
    [P2]="--feature-gate --feature-gate-bottleneck 16"
    [P3]="--film-head --film-context-dim 32"
    [P4]="--soft-attention-aggregation --soft-attention-bottleneck 64"
    [P6]="--two-tier-top-n 600 --two-tier-coarse-hidden-dim 128 --two-tier-refine-hidden-dim 384 --two-tier-coarse-neighbors 16 --two-tier-refine-neighbors 32 --two-tier-coarse-rounds 2 --two-tier-refine-rounds 3"
)

# Scheduling order — cheapest / lowest-risk first.
EXPERIMENT_ORDER=(
    E2a_anchor
    P2
    P3
    P1
    P4
    P6
)

# If user specified experiments, filter the order.
if [ "${#REQUESTED_EXPERIMENTS[@]}" -gt 0 ]; then
    FILTERED_ORDER=()
    for requested in "${REQUESTED_EXPERIMENTS[@]}"; do
        if [ -n "${EXPERIMENT_ARGS[${requested}]+set}" ]; then
            FILTERED_ORDER+=("${requested}")
        else
            echo "WARNING: unknown experiment '${requested}' — skipping"
        fi
    done
    EXPERIMENT_ORDER=("${FILTERED_ORDER[@]}")
fi

# ---- Run banner ----
echo "============================================"
echo "  Prefilter expressiveness sweep — mode=${MODE}"
echo "  Target:    P@256 > 0.90 (from 0.879 E2a)"
echo "============================================"
echo "Experiments:  ${EXPERIMENT_ORDER[*]}"
echo "Epochs:       ${EPOCHS}"
echo "Steps/epoch:  ${STEPS_PER_EPOCH}"
echo "Batch:        ${BATCH_SIZE}"
echo "Workers:      ${NUM_WORKERS}"
echo "AMP:          ${AMP_FLAG:-off}"
echo "Log dir:      ${LOG_DIR}"
echo "Exp dir:      ${EXP_DIR}"
echo ""

SUMMARY_FILE="${LOG_DIR}/sweep_expressiveness_${MODE}_summary.txt"
: > "${SUMMARY_FILE}"
echo "Summary:      ${SUMMARY_FILE}"
echo ""

# ---- Execute ----
OVERALL_FAILED=0
for experiment in "${EXPERIMENT_ORDER[@]}"; do
    FAMILY="${NETWORK_FAMILY[${experiment}]}"
    EXTRA_ARGS_STRING="${EXPERIMENT_ARGS[${experiment}]}"
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS_STRING})
    MODEL_NAME="sweep_expressiveness_${MODE}_${experiment}"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

    if [ "${FAMILY}" = "two_tier" ]; then
        COMMON_ARGS=("${COMMON_ARGS_TWO_TIER[@]}")
    else
        COMMON_ARGS=("${COMMON_ARGS_SINGLE_TIER[@]}")
    fi

    echo "---- [$(date +%H:%M:%S)] ${experiment} (${FAMILY}): ${EXTRA_ARGS_STRING:-(no extra flags)}"
    echo "     log: ${LOG_FILE}"

    set +e
    python train_prefilter.py \
        "${COMMON_ARGS[@]}" \
        --model-name "${MODEL_NAME}" \
        "${EXTRA_ARGS_ARRAY[@]}" \
        > "${LOG_FILE}" 2>&1
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        FINAL_METRIC=$(grep -E 'Epoch [0-9]+ val' "${LOG_FILE}" | tail -1)
        echo "[OK]  ${experiment}: ${FINAL_METRIC:-completed}" | tee -a "${SUMMARY_FILE}"
    else
        echo "[FAIL] ${experiment}: exit ${exit_code} (see ${LOG_FILE})" \
            | tee -a "${SUMMARY_FILE}"
        OVERALL_FAILED=1
        # Smoke mode: stop on first failure to aid debugging.
        # Full mode: continue so the overnight sweep isn't killed by one bad run.
        if [ "${MODE}" = "smoke" ]; then
            echo ""
            echo "Smoke failure — stopping sweep. Log tail:"
            tail -30 "${LOG_FILE}" | tr '\r' '\n' | tail -20
            exit 2
        fi
    fi
    echo ""
done

echo "============================================"
echo "  Sweep complete"
echo "============================================"
cat "${SUMMARY_FILE}"
exit ${OVERALL_FAILED}
