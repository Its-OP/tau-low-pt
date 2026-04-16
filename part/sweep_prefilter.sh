#!/bin/bash
# =============================================================================
# Prefilter-improvement campaign — experiment sweep orchestrator.
#
# Usage:
#   bash sweep_prefilter.sh smoke           # 1-epoch validation run over ALL experiments
#   bash sweep_prefilter.sh full            # 20-epoch campaign run over ALL experiments
#   bash sweep_prefilter.sh smoke E1 E3     # smoke-run only specified experiments
#   bash sweep_prefilter.sh full E1 E2a E3  # full-run specified experiments
#
# Modes:
#   smoke — 1 train epoch + 1 val epoch, BS=128, 50 steps/epoch, 4 workers, no AMP.
#           Detects immediate failures (CLI flag typos, shape mismatches, OOMs).
#   full  — 20 train epochs, BS=256, 500 steps/epoch, 10 workers, AMP.
#           Real campaign run — produces publishable C@K numbers.
#
# Each experiment:
#   - Writes log to   /workspace/logs/sweep_<mode>_<exp>.log
#   - Writes exp dir  /workspace/experiments/sweep_<mode>_<exp>_PreFilter_<ts>/
#   - Runs sequentially (1 GPU). Stops on first failure unless --continue set.
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
COMMON_ARGS=(
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
    --no-compile
    --save-every 2
    --keep-best-k "${KEEP_BEST_K}"
)
if [ -n "${AMP_FLAG}" ]; then
    COMMON_ARGS+=("${AMP_FLAG}")
fi

# ---- Experiment definitions ----
# Each experiment is a bash array of extra args appended after COMMON_ARGS.
# The key is the experiment ID used on the CLI and in filenames.
declare -A EXPERIMENTS=(
    [BASELINE]=""
    [E1]="--use-edge-features"
    [E2a]="--use-edge-features --num-neighbors 16 --num-message-rounds 3"
    [E2b]="--use-edge-features --num-neighbors 32 --num-message-rounds 2"
    [E2c]="--use-edge-features --num-neighbors 32 --num-message-rounds 3"
    [E2d]="--use-edge-features --num-neighbors 48 --num-message-rounds 2"
    [E2e]="--use-edge-features --num-neighbors 64 --num-message-rounds 2"
    [E3]="--use-edge-features --loss-type listwise_ce"
    [E4]="--use-edge-features --loss-type logit_adjust --logit-adjust-tau 1.0"
    [E5]="--use-edge-features --loss-type object_condensation --clustering-dim 8"
    [E6]="--use-edge-features --aggregation-mode pna"
    [E7]="--use-edge-features --use-xgb-stub-feature"
    [E8]="--num-message-rounds 0"
    [E9]="--use-edge-features --loss-type infonce --listwise-temperature 0.5"
    [E10]="--use-edge-features --loss-type mpm_pretrain --mpm-mask-ratio 0.15"
    [E11]="--use-edge-features --use-augmentation"
    [E12]="--use-edge-features --use-self-distillation --ema-decay 0.999 --kl-weight 0.1"
)

# Order matters — this is the scheduling order for the sweep.
EXPERIMENT_ORDER=(
    BASELINE
    E1
    E2a E2b E2c E2d E2e
    E3
    E4
    E5
    E6
    E7
    E8
    E9
    E10
    E11
    E12
)

# If user specified experiments, filter the order.
if [ "${#REQUESTED_EXPERIMENTS[@]}" -gt 0 ]; then
    FILTERED_ORDER=()
    for requested in "${REQUESTED_EXPERIMENTS[@]}"; do
        if [ -n "${EXPERIMENTS[${requested}]+set}" ]; then
            FILTERED_ORDER+=("${requested}")
        else
            echo "WARNING: unknown experiment '${requested}' — skipping"
        fi
    done
    EXPERIMENT_ORDER=("${FILTERED_ORDER[@]}")
fi

# ---- Run banner ----
echo "============================================"
echo "  Prefilter sweep — mode=${MODE}"
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

SUMMARY_FILE="${LOG_DIR}/sweep_${MODE}_summary.txt"
: > "${SUMMARY_FILE}"
echo "Summary:      ${SUMMARY_FILE}"
echo ""

# ---- Execute ----
OVERALL_FAILED=0
for experiment in "${EXPERIMENT_ORDER[@]}"; do
    EXTRA_ARGS_STRING="${EXPERIMENTS[${experiment}]}"
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS_STRING})
    MODEL_NAME="sweep_${MODE}_${experiment}"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"

    echo "---- [$(date +%H:%M:%S)] ${experiment}: ${EXTRA_ARGS_STRING:-(no extra flags)}"
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
        # Extract final val R@200 if present
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
