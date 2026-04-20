#!/bin/bash
# =============================================================================
# Couple-reranker Batch 5 sweep — tokenized reranker scale + depth.
#
# Tests the hypothesis that Batch-3 F5 tokenization failures (regressions of
# −0.002 to −0.005 pp C@100 at d_token=16, 2–3 blocks) were caused by
# insufficient capacity, not a flaw in the feature-as-token idea itself.
# Holds mode = ft_transformer, FFN ratio = 4×, pre-norm + GELU. Sweeps
# (d_token, num_blocks) along a geometric ladder. head_dim stays ≥ 8.
#
# Usage:
#   bash sweep_couple_batch5.sh                    # overnight (detached screen)
#   SMOKE_MODE=1 bash sweep_couple_batch5.sh       # 1-epoch smoke
#   NO_SCREEN=1 bash sweep_couple_batch5.sh        # run inline (debug)
#   EXPERIMENTS="B5_4_d64_b4" bash sweep_couple_batch5.sh   # subset
# =============================================================================
set -euo pipefail

SMOKE_MODE="${SMOKE_MODE:-0}"
SESSION_NAME="${SESSION_NAME:-couple_batch5}"

if [ -z "${STY:-}" ] && [ "${NO_SCREEN:-0}" != "1" ]; then
    if ! command -v screen &>/dev/null; then
        echo "ERROR: screen not installed. Install it or pass NO_SCREEN=1." >&2
        exit 1
    fi
    if screen -list 2>/dev/null | grep -q "\.${SESSION_NAME}"; then
        echo "Screen '${SESSION_NAME}' exists. Reattach or kill first." >&2
        exit 1
    fi
    CHILD_ENV=""
    for var in SMOKE_MODE EPOCHS STEPS_PER_EPOCH BATCH_SIZE NUM_WORKERS \
               KEEP_BEST_K BN_CALIBRATION_STEPS TOP_K2 SEED DEVICE \
               DATA_CONFIG DATA_DIR VAL_DATA_DIR NETWORK CASCADE_CHECKPOINT \
               EXPERIMENTS PYTHON; do
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
    SWEEP_ROOT_PREFIX="couple_batch5_smoke"
else
    EPOCHS="${EPOCHS:-60}"
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
    BATCH_SIZE="${BATCH_SIZE:-96}"
    NUM_WORKERS="${NUM_WORKERS:-10}"
    NO_IN_MEMORY="${NO_IN_MEMORY:-0}"
    KEEP_BEST_K="${KEEP_BEST_K:-3}"
    BN_CALIBRATION_STEPS="${BN_CALIBRATION_STEPS:-200}"
    SWEEP_ROOT_PREFIX="couple_batch5"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${SCRIPT_DIR}/experiments/${SWEEP_ROOT_PREFIX}_${TIMESTAMP}"
mkdir -p "${SWEEP_ROOT}"
SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

# B0 is the v3 concat+projector baseline, no tokenization; every B5_* row
# is ft_transformer with varying (d_token, num_blocks) and matching num_heads
# so head_dim = d_token / num_heads stays ≥ 8.
declare -a ALL_EXPERIMENTS=(
    "B0_v3_anchor|"
    "B5_1_d32_b4|--couple-embed-mode ft_transformer --tokenize-d 32 --tokenize-blocks 4 --tokenize-heads 4"
    "B5_2_d32_b6|--couple-embed-mode ft_transformer --tokenize-d 32 --tokenize-blocks 6 --tokenize-heads 4"
    "B5_3_d48_b4|--couple-embed-mode ft_transformer --tokenize-d 48 --tokenize-blocks 4 --tokenize-heads 6"
    "B5_4_d64_b4|--couple-embed-mode ft_transformer --tokenize-d 64 --tokenize-blocks 4 --tokenize-heads 8"
    "B5_5_d64_b6|--couple-embed-mode ft_transformer --tokenize-d 64 --tokenize-blocks 6 --tokenize-heads 8"
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
echo "  Couple-reranker Batch 5 sweep (tokenization scale & depth)"
echo "================================================================"
echo "Mode:              $([ "${SMOKE_MODE}" = "1" ] && echo "SMOKE" || echo "FULL")"
echo "Sweep root:        ${SWEEP_ROOT}"
echo "Experiments:       ${NUM_TOTAL}"
echo "Epochs/run:        ${EPOCHS}"
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
    echo "  Flags:    ${EXTRA_FLAGS:-<v3 baseline>}"
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
echo "  Batch 5 sweep complete: ${NUM_OK} OK, ${NUM_FAILED} failed, ${NUM_TOTAL} total"
echo "  Finished:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Root:           ${SWEEP_ROOT}"
echo "================================================================"
if [ -n "${FAILED_NAMES}" ]; then
    echo "Failed:${FAILED_NAMES}"
    exit 1
fi
