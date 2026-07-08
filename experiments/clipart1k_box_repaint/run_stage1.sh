#!/usr/bin/env bash
# Stage 1 (1-shot only) launch script for the clipart1k box-repaint
# rung-1 experiment. See docs/specs/2026-07-08-clipart1k-box-repaint-rung1-design.md.
#
# Usage: experiments/clipart1k_box_repaint/run_stage1.sh <gpu_id> <port>
set -euo pipefail

GPU="$1"
PORT="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIPART_SRC="/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k"
GENAUG_ROOT="/data1/qushiduo/datasets/genaug/clipart1k"
RAW_DIR="${GENAUG_ROOT}/raw_generations/1shot"
MANIFEST="${GENAUG_ROOT}/raw_generations/manifest_1shot.csv"
MERGED_DIR="${GENAUG_ROOT}/1shot"
RESULTS_DIR="${REPO_ROOT}/experiments/clipart1k_box_repaint/results"

# `python3 -m generation.box_repaint.generate` etc. resolve their package
# imports off the current working directory — must run from repo root.
cd "${REPO_ROOT}"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}"

echo "[stage1] generating box-repaint variants (N=4, strength=0.4)"
conda run -n flux2 python3 -m generation.box_repaint.generate \
    --shot-json "${CLIPART_SRC}/annotations/1_shot.json" \
    --images-dir "${CLIPART_SRC}/train" \
    --model-dir /data1/qushiduo/models/flux2/FLUX.1-dev \
    --out-dir "${RAW_DIR}" \
    --manifest "${MANIFEST}" \
    --n-variants 4 --strength 0.4 --gpu "${GPU}"

echo "[stage1] building merged annotations"
python3 -m generation.box_repaint.build_annotations \
    --shot-json "${CLIPART_SRC}/annotations/1_shot.json" \
    --real-images-dir "${CLIPART_SRC}/train" \
    --manifest "${MANIFEST}" \
    --out-dir "${MERGED_DIR}"

echo "[stage1] training 3 seeds (42, 43, 44)"
for SEED in 42 43 44; do
    TRAIN_DATA_ROOT="${MERGED_DIR}" \
    TRAIN_ANN_FILE="annotations.json" \
    TRAIN_IMG_PREFIX="images/" \
    RUN_TAG="_boxrepaint" \
    RESULTS_DIR="${RESULTS_DIR}" \
    "${REPO_ROOT}/baselines/ftfsod_cdfsod/run_one.sh" clipart1k 1 "${SEED}" "${GPU}" "${PORT}"
done

echo "[stage1] comparing against frozen baselines"
python3 -m experiments.clipart1k_box_repaint.compare_results \
    --baseline-dir "${REPO_ROOT}/baselines/ftfsod_cdfsod/results" \
    --augmented-dir "${RESULTS_DIR}" \
    --out "${RESULTS_DIR}/stage1_report.md"
