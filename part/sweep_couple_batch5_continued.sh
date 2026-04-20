#!/bin/bash
# =============================================================================
# Batch 5 continuation — B5_2/3/4 with reduced batch sizes to fit VRAM.
#
# B5_1 at (d=32, blocks=4, batch=96) used 94/98 GB VRAM during Batch 5 main
# sweep. Scaling d or blocks pushes this over. To keep per-run inside
# ~80 GB with the batch floor of 32, each experiment gets its own
# (batch, steps) pair tuned so `batch * steps` = 19 200 events per epoch
# (same dataset coverage as batch=96 × 200 steps).
#
# B5_5 (d=64, blocks=6) dropped — even at batch=32 it would be > 100 GB.
# Usage:
#   bash sweep_couple_batch5_continued.sh                    # screen run
#   SMOKE_MODE=1 bash sweep_couple_batch5_continued.sh       # smoke
#   NO_SCREEN=1 bash sweep_couple_batch5_continued.sh        # inline
# =============================================================================
set -euo pipefail

SMOKE_MODE="${SMOKE_MODE:-0}"
SESSION_NAME="${SESSION_NAME:-couple_batch5_cont}"

if [ -z "${STY:-}" ] && [ "${NO_SCREEN:-0}" != "1" ]; then
    if ! command -v screen &>/dev/null; then
        echo "ERROR: screen not installed." >&2
        exit 1
    fi
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}"; then
        echo "Screen '${SESSION_NAME}' exists." >&2
        exit 1
    fi
    CHILD_ENV=""
    for var in SMOKE_MODE TOP_K2 SEED DEVICE PYTHON NO_IN_MEMORY \
               DATA_CONFIG DATA_DIR VAL_DATA_DIR NETWORK CASCADE_CHECKPOINT \
               EXPERIMENTS KEEP_BEST_K BN_CALIBRATION_STEPS; do
        if [ -n "${!var:-}" ]; then
            CHILD_ENV+="${var}='${!var}' "
        fi
    done
    screen -dmS "${SESSION_NAME}" bash -c "
        ${CHILD_ENV} NO_SCREEN=1 bash '${BASH_SOURCE[0]}'
        echo ''
        echo '--- Sweep finished. Press Enter to close. ---'
        read -r
    "
    echo "Sweep launched in screen '${SESSION_NAME}'."
    exit 0
fi

PYTHON="${PYTHON:-/venv/part/bin/python}"
if [ ! -x "${PYTHON}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "ERROR: python not found." >&2
        exit 1
    fi
fi

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
NO_IN_MEMORY="${NO_IN_MEMORY:-0}"
KEEP_BEST_K="${KEEP_BEST_K:-3}"
BN_CALIBRATION_STEPS="${BN_CALIBRATION_STEPS:-200}"

if [ "${SMOKE_MODE}" = "1" ]; then
    SMOKE_EPOCHS=1
    SMOKE_STEPS=3
    SMOKE_BATCH=16
    SMOKE_WORKERS=2
    SMOKE_NO_IN_MEMORY=1
    SMOKE_KEEP=1
    SMOKE_BN=0
    SWEEP_ROOT_PREFIX="couple_batch5_cont_smoke"
else
    EPOCHS="${EPOCHS:-60}"
    NUM_WORKERS="${NUM_WORKERS:-10}"
    SWEEP_ROOT_PREFIX="couple_batch5_cont"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${SCRIPT_DIR}/experiments/${SWEEP_ROOT_PREFIX}_${TIMESTAMP}"
mkdir -p "${SWEEP_ROOT}"
SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

# Each row: name | batch | steps | extra flags
# batch*steps = 19 200 events/epoch (same as main sweep).
declare -a ALL_EXPERIMENTS=(
    "B5_2_d32_b6|64|300|--couple-embed-mode ft_transformer --tokenize-d 32 --tokenize-blocks 6 --tokenize-heads 4"
    "B5_3_d48_b4|48|400|--couple-embed-mode ft_transformer --tokenize-d 48 --tokenize-blocks 4 --tokenize-heads 6"
    "B5_4_d64_b4|32|600|--couple-embed-mode ft_transformer --tokenize-d 64 --tokenize-blocks 4 --tokenize-heads 8"
)

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
echo "  Batch 5 continuation (VRAM-aware per-exp batch sizes)"
echo "================================================================"
echo "Mode:              $([ "${SMOKE_MODE}" = "1" ] && echo "SMOKE" || echo "FULL")"
echo "Sweep root:        ${SWEEP_ROOT}"
echo "Experiments:       ${NUM_TOTAL}"
echo "Started:           $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

NUM_OK=0
NUM_FAILED=0
FAILED_NAMES=""

for entry in "${ALL_EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME BATCH_SIZE STEPS_PER_EPOCH EXTRA_FLAGS <<< "${entry}"
    EXP_DIR="${SWEEP_ROOT}/${NAME}"
    mkdir -p "${EXP_DIR}"
    EXP_LOG="${EXP_DIR}/training.log"

    if [ "${SMOKE_MODE}" = "1" ]; then
        RUN_EPOCHS=${SMOKE_EPOCHS}
        RUN_STEPS=${SMOKE_STEPS}
        RUN_BATCH=${SMOKE_BATCH}
        RUN_WORKERS=${SMOKE_WORKERS}
        RUN_NO_IN_MEMORY=${SMOKE_NO_IN_MEMORY}
        RUN_KEEP=${SMOKE_KEEP}
        RUN_BN=${SMOKE_BN}
    else
        RUN_EPOCHS=${EPOCHS}
        RUN_STEPS=${STEPS_PER_EPOCH}
        RUN_BATCH=${BATCH_SIZE}
        RUN_WORKERS=${NUM_WORKERS}
        RUN_NO_IN_MEMORY=${NO_IN_MEMORY}
        RUN_KEEP=${KEEP_BEST_K}
        RUN_BN=${BN_CALIBRATION_STEPS}
    fi

    echo "----------------------------------------------------------------"
    echo "  [$(($NUM_OK + $NUM_FAILED + 1))/${NUM_TOTAL}]  ${NAME}"
    echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Batch:    ${RUN_BATCH}  Steps/epoch: ${RUN_STEPS}"
    echo "  Flags:    ${EXTRA_FLAGS}"
    echo "----------------------------------------------------------------"

    NO_IN_MEMORY_FLAG=""
    if [ "${RUN_NO_IN_MEMORY}" = "1" ]; then
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
        --epochs "${RUN_EPOCHS}" \
        --batch-size "${RUN_BATCH}" \
        --steps-per-epoch "${RUN_STEPS}" \
        --num-workers "${RUN_WORKERS}" \
        --keep-best-k "${RUN_KEEP}" \
        --bn-calibration-steps "${RUN_BN}" \
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
echo "  Continuation complete: ${NUM_OK} OK, ${NUM_FAILED} failed"
echo "  Finished:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Root:           ${SWEEP_ROOT}"
echo "================================================================"
if [ -n "${FAILED_NAMES}" ]; then
    echo "Failed:${FAILED_NAMES}"
    exit 1
fi
