#!/bin/bash
# =============================================================================
# Couple-reranker improvement sweep — Batch 2.
#
# Builds on Batch 1 (completed 2026-04-17). All experiments run on top of
# the new baseline v2 (softmax-ce + label_smoothing=0.10 + cosine_power=2.0,
# which are now the CLI defaults in `train_couple_reranker.py`).
#
# Focus: pluggable couple-embedding block (A1-A5), clean reads of Batch 1
# confounded results (B1), hardneg repair (C1), and an extended-epochs
# sanity check (D1).
#
# Usage:
#   bash sweep_couple_batch2.sh                   # full run (detached screen)
#   SMOKE_MODE=1 bash sweep_couple_batch2.sh      # 1-epoch smoke
#   NO_SCREEN=1 bash sweep_couple_batch2.sh       # run inline
#   EXPERIMENTS="A1_infersent C1_hardneg_025_m0" bash ...  # subset
#
# Reattach detached sweep:
#   screen -r couple_batch2
# Kill:
#   screen -S couple_batch2 -X quit
#
# Outputs:
#   experiments/couple_batch2_{smoke_,}<timestamp>/
#     <experiment_name>/
#       training.log
#       metrics/epoch_<N>.json
#       loss_history.json
#       checkpoints/best_model_calibrated.pt
#     sweep.log
# =============================================================================
set -euo pipefail

SMOKE_MODE="${SMOKE_MODE:-0}"
SESSION_NAME="${SESSION_NAME:-couple_batch2}"

# ---- Self-relaunch in detached screen (unless we're already inside
# screen or the caller set NO_SCREEN=1). STY is set by screen for its
# children, so we use that to detect the inner run.
if [ -z "${STY:-}" ] && [ "${NO_SCREEN:-0}" != "1" ]; then
    if ! command -v screen &>/dev/null; then
        echo "ERROR: screen not installed. Install it or pass NO_SCREEN=1." >&2
        exit 1
    fi
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}"; then
        echo "Screen session '${SESSION_NAME}' already exists." >&2
        echo "Reattach: screen -r ${SESSION_NAME}" >&2
        echo "Kill:     screen -S ${SESSION_NAME} -X quit" >&2
        exit 1
    fi

    CHILD_ENV=""
    for var in SMOKE_MODE EPOCHS STEPS_PER_EPOCH BATCH_SIZE NUM_WORKERS \
               KEEP_BEST_K BN_CALIBRATION_STEPS TOP_K2 SEED DEVICE \
               DATA_CONFIG DATA_DIR VAL_DATA_DIR NETWORK CASCADE_CHECKPOINT \
               EXPERIMENTS; do
        if [ -n "${!var:-}" ]; then
            CHILD_ENV+="${var}='${!var}' "
        fi
    done

    screen -dmS "${SESSION_NAME}" bash -c "
        ${CHILD_ENV} NO_SCREEN=1 bash '${BASH_SOURCE[0]}'
        echo ''
        echo '--- Sweep finished. Press Enter to close this screen session. ---'
        read -r
    "
    echo "Sweep launched in detached screen session '${SESSION_NAME}'."
    echo "Reattach: screen -r ${SESSION_NAME}"
    echo "Kill:     screen -S ${SESSION_NAME} -X quit"
    exit 0
fi

# ---- Resolve python interpreter ----
# SSH-launched screens do not auto-activate conda, so rely on an absolute
# path to the `part` env's python. Overridable via PYTHON env var.
PYTHON="${PYTHON:-/venv/part/bin/python}"
if [ ! -x "${PYTHON}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "ERROR: python not found (tried ${PYTHON} and PATH)" >&2
        exit 1
    fi
fi
echo "Using python: ${PYTHON}"

# ---- Common training config (overridable via env vars) ----
DATA_CONFIG="${DATA_CONFIG:-data/low-pt/lowpt_tau_trackfinder.yaml}"
DATA_DIR="${DATA_DIR:-data/low-pt/train/}"
VAL_DATA_DIR="${VAL_DATA_DIR:-data/low-pt/val/}"
NETWORK="${NETWORK:-networks/lowpt_tau_CoupleReranker.py}"
CASCADE_CHECKPOINT="${CASCADE_CHECKPOINT:-models/cascade_best.pt}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
TOP_K2="${TOP_K2:-60}"
K_VALUES_TRACKS="30 50 60 75 100 200"
K_VALUES_COUPLES="50 75 100 200"

if [ "${SMOKE_MODE}" = "1" ]; then
    EPOCHS="${EPOCHS:-1}"
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-3}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    NO_IN_MEMORY="${NO_IN_MEMORY:-1}"
    KEEP_BEST_K="${KEEP_BEST_K:-1}"
    BN_CALIBRATION_STEPS="${BN_CALIBRATION_STEPS:-0}"
    SWEEP_ROOT_PREFIX="couple_batch2_smoke"
else
    EPOCHS="${EPOCHS:-60}"
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
    BATCH_SIZE="${BATCH_SIZE:-96}"
    NUM_WORKERS="${NUM_WORKERS:-10}"
    NO_IN_MEMORY="${NO_IN_MEMORY:-0}"
    KEEP_BEST_K="${KEEP_BEST_K:-3}"
    BN_CALIBRATION_STEPS="${BN_CALIBRATION_STEPS:-200}"
    SWEEP_ROOT_PREFIX="couple_batch2"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${SCRIPT_DIR}/experiments/${SWEEP_ROOT_PREFIX}_${TIMESTAMP}"
mkdir -p "${SWEEP_ROOT}"
SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

# ---- Experiment list: "name|extra flags" ----
#
# Each row toggles a single feature on top of the new baseline v2 defaults
# (softmax-ce + smoothing=0.10 + cos=2.0). B0_baseline_v2 is included as
# the reference anchor — running with no extra flags reproduces baseline v2
# under the identical sweep harness (same seed, same val subset, same BN
# calibration) so ΔC@100 comparisons are not confounded by harness drift.
declare -a ALL_EXPERIMENTS=(
    "B0_baseline_v2|"
    "A1_infersent|--couple-embed-mode infersent"
    "A2_symmetric|--couple-embed-mode symmetric"
    "A3_bilinear_lrb|--couple-embed-mode bilinear_lrb"
    "A4_proj_infersent_p16|--couple-embed-mode projected_infersent --couple-projector-dim 16"
    "A5_proj_infersent_p32|--couple-embed-mode projected_infersent --couple-projector-dim 32"
    "B1_pair_v2_clean|--pair-kinematics-v2"
    "C1_hardneg_025_m0|--couple-hardneg-fraction 0.25 --couple-hardneg-margin 0.0"
    "D1_extended_80ep|--epochs 80"
)

# Filter via EXPERIMENTS env var (space-separated names). Default = all.
if [ -n "${EXPERIMENTS:-}" ]; then
    declare -a EXPERIMENTS_FILTER
    IFS=' ' read -r -a EXPERIMENTS_FILTER <<< "${EXPERIMENTS}"
    declare -a FILTERED=()
    for entry in "${ALL_EXPERIMENTS[@]}"; do
        name="${entry%%|*}"
        for wanted in "${EXPERIMENTS_FILTER[@]}"; do
            if [ "${name}" = "${wanted}" ]; then
                FILTERED+=("${entry}")
                break
            fi
        done
    done
    ALL_EXPERIMENTS=("${FILTERED[@]}")
fi

NUM_TOTAL=${#ALL_EXPERIMENTS[@]}

echo "================================================================"
echo "  Couple-reranker Batch 2 sweep"
echo "================================================================"
echo "Mode:              $([ "${SMOKE_MODE}" = "1" ] && echo "SMOKE" || echo "FULL")"
echo "Sweep root:        ${SWEEP_ROOT}"
echo "Experiments:       ${NUM_TOTAL}"
echo "Epochs/run:        ${EPOCHS}"
echo "Steps/epoch:       ${STEPS_PER_EPOCH}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Num workers:       ${NUM_WORKERS}"
echo "BN calib steps:    ${BN_CALIBRATION_STEPS}"
echo "top_k2:            ${TOP_K2}"
echo "Device:            ${DEVICE}"
echo "Cascade ckpt:      ${CASCADE_CHECKPOINT}"
echo "Seed:              ${SEED}"
echo "Started:           $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

NUM_OK=0
NUM_FAILED=0
FAILED_NAMES=""

for entry in "${ALL_EXPERIMENTS[@]}"; do
    NAME="${entry%%|*}"
    EXTRA_FLAGS="${entry#*|}"
    EXP_DIR="${SWEEP_ROOT}/${NAME}"
    mkdir -p "${EXP_DIR}"
    EXP_LOG="${EXP_DIR}/training.log"

    echo "----------------------------------------------------------------"
    echo "  [$(($NUM_OK + $NUM_FAILED + 1))/${NUM_TOTAL}]  ${NAME}"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Flags:    ${EXTRA_FLAGS:-<baseline v2>}"
    echo "----------------------------------------------------------------"

    NO_IN_MEMORY_FLAG=""
    if [ "${NO_IN_MEMORY}" = "1" ]; then
        NO_IN_MEMORY_FLAG="--no-in-memory"
    fi

    set +e
    # shellcheck disable=SC2086
    "${PYTHON}" train_couple_reranker.py \
        --data-config "${DATA_CONFIG}" \
        --data-dir "${DATA_DIR}" \
        --val-data-dir "${VAL_DATA_DIR}" \
        --network "${NETWORK}" \
        --cascade-checkpoint "${CASCADE_CHECKPOINT}" \
        --top-k2 "${TOP_K2}" \
        --k-values-tracks ${K_VALUES_TRACKS} \
        --k-values-couples ${K_VALUES_COUPLES} \
        --seed "${SEED}" \
        --model-name "${NAME}" \
        --experiments-dir "${EXP_DIR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --steps-per-epoch "${STEPS_PER_EPOCH}" \
        --num-workers "${NUM_WORKERS}" \
        --keep-best-k "${KEEP_BEST_K}" \
        --bn-calibration-steps "${BN_CALIBRATION_STEPS}" \
        --device "${DEVICE}" \
        ${NO_IN_MEMORY_FLAG} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${EXP_LOG}"
    STATUS=${PIPESTATUS[0]}
    set -e

    if [ "${STATUS}" -eq 0 ]; then
        NUM_OK=$((NUM_OK + 1))
        echo "  OK: ${NAME} finished at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        NUM_FAILED=$((NUM_FAILED + 1))
        FAILED_NAMES="${FAILED_NAMES} ${NAME}"
        echo "  FAILED: ${NAME} (exit ${STATUS}) at $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo ""
done

echo "================================================================"
echo "  Batch 2 sweep complete: ${NUM_OK} OK, ${NUM_FAILED} failed, ${NUM_TOTAL} total"
echo "  Finished:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Root:           ${SWEEP_ROOT}"
echo "================================================================"
if [ -n "${FAILED_NAMES}" ]; then
    echo "Failed:${FAILED_NAMES}"
    exit 1
fi
