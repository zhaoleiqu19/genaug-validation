#!/usr/bin/env bash
# Fine-tune + evaluate one (domain, shot, seed) cell of the FT-FSOD CDFSOD baseline.
# Usage: run_one.sh <domain> <shot> <seed> <gpu_id> <port>
set -euo pipefail

DOMAIN="$1"      # ArTaxOr | clipart1k | DIOR | FISH | NEU-DET | UODD
SHOT="$2"        # 1 | 5 | 10
SEED="$3"
GPU="$4"
PORT="$5"

FTFSOD_REPO="${FTFSOD_REPO:-$HOME/external/FT-FSOD}"
WORK_ROOT="${WORK_ROOT:-/data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

CONFIG="${FTFSOD_REPO}/configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_${DOMAIN}_${SHOT}shot.py"
RUN_NAME="swinB_${DOMAIN}_${SHOT}shot_seed${SEED}"
TRAIN_WORK_DIR="${WORK_ROOT}/${RUN_NAME}"
TEST_WORK_DIR="${WORK_ROOT}/${RUN_NAME}_test"

if [ ! -f "${CONFIG}" ]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${TRAIN_WORK_DIR}" "${TEST_WORK_DIR}" "${RESULTS_DIR}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
cd "${FTFSOD_REPO}"

echo "[run_one] train: domain=${DOMAIN} shot=${SHOT} seed=${SEED} gpu=${GPU}"
START_TS=$(date +%s)
conda run -n ftfsod ./tools/dist_train.sh "${CONFIG}" 1 "${PORT}" "${GPU}" \
    --work-dir "${TRAIN_WORK_DIR}" \
    --cfg-options randomness.seed="${SEED}"

CKPT_PATH=$(find "${TRAIN_WORK_DIR}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)
if [ -z "${CKPT_PATH}" ]; then
  echo "[run_one] no best checkpoint found in ${TRAIN_WORK_DIR}" >&2
  exit 1
fi

echo "[run_one] test: ckpt=${CKPT_PATH}"
# NOTE: tools/dist_test.sh does NOT forward extra args to test.py (it has
# no "${@:6}", unlike dist_train.sh's "${@:5}" — confirmed by inspecting
# both scripts byte-for-byte during Task 5). Using it here would silently
# drop --work-dir/--out and every test run would collide in test.py's
# default relative work_dir. Replicate what dist_test.sh does internally,
# with proper forwarding, instead of calling the wrapper.
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTHONPATH="${FTFSOD_REPO}:${PYTHONPATH:-}" \
conda run -n ftfsod python -m torch.distributed.launch \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --nproc_per_node=1 --master_port="${PORT}" \
    "${FTFSOD_REPO}/tools/test.py" \
    "${CONFIG}" "${CKPT_PATH}" --launcher pytorch \
    --work-dir "${TEST_WORK_DIR}" \
    --out "${TEST_WORK_DIR}/${DOMAIN}_${SHOT}shot.pkl"
END_TS=$(date +%s)

# Find the one JSON under TEST_WORK_DIR that actually holds the metric
# (path depth confirmed empirically in Task 5 — search broadly rather than
# hardcode depth, since it's driven by mmengine's own timestamped layout).
METRIC_JSON=$(conda run -n ftfsod python3 -c "
import glob, json, sys
for f in glob.glob('${TEST_WORK_DIR}/**/*.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if isinstance(d, dict) and 'coco/bbox_mAP' in d:
        print(f)
        sys.exit(0)
sys.exit(1)
")
if [ -z "${METRIC_JSON}" ]; then
  echo "[run_one] no metrics json with coco/bbox_mAP found under ${TEST_WORK_DIR}" >&2
  exit 1
fi

cp "${METRIC_JSON}" "${RESULTS_DIR}/${RUN_NAME}.json"
echo "${RUN_NAME},${DOMAIN},${SHOT},${SEED},${GPU},${START_TS},${END_TS},${METRIC_JSON}" \
    >> "${RESULTS_DIR}/run_manifest.csv"

echo "[run_one] done: ${RUN_NAME} ($(( (END_TS-START_TS)/60 )) min)"
