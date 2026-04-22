#!/bin/bash
# =============================================================================
# Couple-reranker Batch 6 sweep — position-aware loss fix.
#
# The Apr-14 topk2 sweep showed that the deployed softmax-CE loss halves
# (0.082 → 0.041) as input K grows from 50 → 200, while the mean rank of
# the best GT couple climbs 7× (14 → 96). The loss is anti-correlated
# with ranking quality. Batch 6 tests three fixes:
#
#   B6_1  hard-negative mining: softmax-CE still, but all 50 negatives
#         per positive are drawn from the hardest currently-misranked
#         couples (hardneg_fraction=1.0). Attacks the
#         "random negatives are trivially easy" failure mode.
#   B6_2  lambda_ndcg2pp with K-truncation disabled (ndcg_k=0), σ=3, and
#         proper ideal-DCG normalisation. Re-runs Batch-3's B3_1
#         candidate with the truncation + normalisation bugs fixed.
#   B6_3  approx_ndcg (Qin et al. 2010): smooth differentiable NDCG,
#         rank approximated by a pairwise sigmoid; no truncation, no
#         sampling. α=5.
#
# B0 re-anchors v3 on the same harness (stage1 old prefilter_best.pt +
# retrained stage2 + v3 couple defaults, top_k2=60, 60 epochs).
#
# Usage:
#   bash sweep_couple_batch6.sh                 # overnight (detached screen)
#   SMOKE_MODE=1 bash sweep_couple_batch6.sh    # 1-epoch smoke
#   NO_SCREEN=1 bash sweep_couple_batch6.sh     # run inline (debug)
#   EXPERIMENTS="B6_2_lambda_noK" bash sweep_couple_batch6.sh   # subset
# =============================================================================
set -euo pipefail

SMOKE_MODE="${SMOKE_MODE:-0}"
SESSION_NAME="${SESSION_NAME:-couple_batch6}"

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
    SWEEP_ROOT_PREFIX="couple_batch6_smoke"
else
    EPOCHS="${EPOCHS:-60}"
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
    BATCH_SIZE="${BATCH_SIZE:-96}"
    NUM_WORKERS="${NUM_WORKERS:-10}"
    NO_IN_MEMORY="${NO_IN_MEMORY:-0}"
    KEEP_BEST_K="${KEEP_BEST_K:-3}"
    BN_CALIBRATION_STEPS="${BN_CALIBRATION_STEPS:-200}"
    SWEEP_ROOT_PREFIX="couple_batch6"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${SCRIPT_DIR}/experiments/${SWEEP_ROOT_PREFIX}_${TIMESTAMP}"
mkdir -p "${SWEEP_ROOT}"
SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

# All runs share v3 defaults (projected_infersent p=32, softmax-CE
# label_smoothing=0.10, cosine_power=2.0) unless overridden by the row's
# extra flags.
declare -a ALL_EXPERIMENTS=(
    "B0_v3_anchor|"
    "B6_1_hardneg_full|--couple-hardneg-fraction 1.0"
    "B6_2_lambda_noK|--couple-loss lambda_ndcg2pp --ndcg-k 0 --lambda-sigma 3.0"
    "B6_3_approx_ndcg|--couple-loss approx_ndcg --ndcg-alpha 5.0"
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
echo "  Couple-reranker Batch 6 sweep (position-aware loss fix)"
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
        --amp \
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
echo "  Batch 6 sweep complete: ${NUM_OK} OK, ${NUM_FAILED} failed, ${NUM_TOTAL} total"
echo "  Finished:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Root:           ${SWEEP_ROOT}"
echo "================================================================"
if [ -n "${FAILED_NAMES}" ]; then
    echo "Failed:${FAILED_NAMES}"
    exit 1
fi
