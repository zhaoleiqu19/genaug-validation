# FT-FSOD CDFSOD Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up FT-FSOD (MM Grounding DINO Swin-B) as the reproducible, un-augmented K-shot detector baseline on CDFSOD, per `docs/superpowers/specs/2026-07-03-cdfsod-ftfsod-baseline-design.md`.

**Architecture:** Clone upstream FT-FSOD to `~/external/FT-FSOD` (untouched mmdet fork with custom Hybrid-Ensemble-Decoder modules), point it directly at the read-only CDFSOD dataset via `src_path.py` (no data copying — verified the raw annotation files it expects already match what's on disk), and wrap its train/test CLI in our own repo-tracked scripts under `baselines/ftfsod_cdfsod/` that add seed control, GPU selection, and result aggregation the upstream one-off script doesn't have.

**Tech Stack:** Python 3.10, PyTorch 2.6.0+cu124, mmengine/mmcv 2.1.0/mmdet (OpenMMLab, FT-FSOD's bundled fork), transformers 4.57.1, pycocotools (via mmdet), bash wrapper scripts, `python3 -m pytest` for the aggregator unit test.

## Global Constraints

- PyTorch ≤ 2.6.0 (CentOS 7 / glibc 2.17 ceiling) — use `torch==2.6.0+cu124`.
- `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/` is read-only — never write, edit, move, or delete anything there.
- Our own `/data*/qushiduo/` folders (confirmed: `/data/qushiduo`, `/data1/qushiduo`, `/data1035/qushiduo`) are ours to write freely — weights and heavy run artifacts (checkpoints, tfevents) go there, never in the repo.
- Code / configs / wrapper scripts / lightweight result summaries live in this repo, under `baselines/ftfsod_cdfsod/`.
- Third-party code (FT-FSOD itself) stays out of this repo, cloned to `~/external/FT-FSOD` (same convention as `~/external/OV-DEIM`).
- Every reported number states dataset, shots, split file, seed, model variant, training budget, GPU. Headline numbers need ≥3 seeds.
- HF downloads: unset proxy vars + `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`; pip via `-i https://pypi.tuna.tsinghua.edu.cn/simple`; `git clone` from GitHub still needs the proxy (`HTTP_PROXY=http://127.0.0.1:8119` etc., already set in this shell's env).
- This repo is private — never push to a public remote.
- The repo-level `python3` (what `python3 -m pytest` actually invokes, confirmed via `which python3` / `python3 --version` in this shell) resolves to the **system Python 3.6.8** (`/usr/bin/python3`), not any conda env — the `ftfsod` conda env is only for FT-FSOD's own train/test process. Any code under `baselines/ftfsod_cdfsod/` that `python3 -m pytest` imports (i.e. `aggregate_results.py` and its test) must be Python-3.6-compatible: no PEP 585 builtin generics (`dict[str, int]`, `list[float]`), no `from __future__ import annotations` (that's 3.7+). Use `typing.Dict`/`typing.List`/`typing.Tuple` instead.

## Confirmed facts from repo inspection (do not re-derive)

- FT-FSOD cloned at `~/external/FT-FSOD` (already done during design research). `src_path.py` at its root holds `CDFSOD_PATH`, `MMGDINOB_PATH` placeholders to fill in.
- `configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_{DOMAIN}_{SHOT}shot.py` exists for `DOMAIN ∈ {ArTaxOr, clipart1k, DIOR, FISH, NEU-DET, UODD}` × `SHOT ∈ {1, 5, 10}` — 18 configs total, exactly matching directory names already present under `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/`.
- Each config uses `ann_file='annotations/{shot}_shot.json'` and `data_prefix=dict(img='train/')` for training, `annotations/test.json` + `test/` for val/test — i.e. it reads the **raw** (non-`_converted`) annotation files already on the read-only disk. No dataset copy/conversion needed.
- `randomness=dict(seed=42, deterministic=True, ...)` is hardcoded per-config; `tools/train.py` also hardcodes `seed(42)` at import time, but `mmengine.Runner` re-seeds from `cfg.randomness` before training starts, and `tools/train.py` accepts `--cfg-options key=value` (mmengine `DictAction`, supports dotted nested keys) — so `--cfg-options randomness.seed=<N>` reliably controls the actual training seed per run.
- `tools/dist_train.sh CONFIG GPUS PORT CUDA_VISIBLE_DEVICES [extra train.py args...]` and `tools/dist_test.sh CONFIG CHECKPOINT GPUS PORT CUDA_VISIBLE_DEVICES` both prepend the repo root to `PYTHONPATH`, which is what makes `import mmdet` resolve to FT-FSOD's bundled fork (custom `GroundingDINO_ParallelDecoder_15_DNQuery_rand` detector, `BBoxHeadFirstHook6` hook) instead of any pip-installed mmdet — **always launch via these wrapper scripts** (or manually set `PYTHONPATH` to the repo root), never bare `python tools/train.py`.
- The upstream `run_mmgdinob_traineval_cdfsod.sh` has a bug: its glob `grounding_dino_swin-b_finetune_A*.py` only matches `ArTaxOr` configs (the only domain starting with "A"), and only the 1-shot block is uncommented. We do not reuse this script — we write our own loop covering all 6 domains × 3 shots × N seeds.
- `analyze_results_cdfsod.py` (upstream) expects directories named `swinB_all_{dataset}_{shot}shot` each containing a JSON with a top-level `coco/bbox_mAP` key. We don't reuse this script either (our own seeds need mean/std aggregation it doesn't do) but we mirror its `coco/bbox_mAP` key convention.
- Existing conda envs `embed`/`flux2`/`srgs` already run `torch==2.6.0+cu124` successfully — confirms this wheel is already cached locally (installs in the new env should hit local pip cache, not the slow proxy path, for the torch/torchvision packages at least).
- Weight/model convention already used elsewhere on this machine: `/data1/qushiduo/models/<project>/...` (e.g. `/data1/qushiduo/models`). We follow it: `/data1/qushiduo/models/ftfsod/`.
- `lang_model_name = 'bert-base-uncased'` is set once in `configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py:6` and consumed as both `model.language_model.name` and a tokenizer name later in the same file — editing this one line (to a local BERT path) propagates through every CDFSOD finetune config, since they all `_base_`-chain through this file.

## File Structure

New files, all in this repo unless noted:

- `baselines/ftfsod_cdfsod/run_one.sh` — trains + evaluates exactly one (domain, shot, seed) cell.
- `baselines/ftfsod_cdfsod/run_matrix.sh` — loops `run_one.sh` over the full domain × shot × seed grid.
- `baselines/ftfsod_cdfsod/aggregate_results.py` — parses per-run result JSONs into a mean±std markdown/CSV table; pure-Python, unit-testable without a GPU.
- `baselines/ftfsod_cdfsod/tests/test_aggregate_results.py` — pytest unit test for the aggregator.
- `baselines/ftfsod_cdfsod/results/` — tracked in git: per-run metric JSON copies, `run_manifest.csv`, final aggregated table. (Heavy artifacts — checkpoints, `.pkl` predictions, tfevents — go to `/data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/`, untracked.)
- `baselines/ftfsod_cdfsod/README.md` — setup + run instructions + results summary (the "conclusion note" required by repo convention).

Modified files outside this repo (not git-tracked by us, but document the exact edits — same status as any other local environment setup step):

- `~/external/FT-FSOD/src_path.py` — fill in `CDFSOD_PATH` and `MMGDINOB_PATH`.
- `~/external/FT-FSOD/configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py:6` — point `lang_model_name` at the local BERT checkpoint.
- The `ftfsod` conda env's installed `mmengine` package — two files patched per FT-FSOD's documented "deterministic hack".

---

### Task 1: Create the `ftfsod` conda environment

**Files:** none (environment only).

**Interfaces:**
- Produces: a conda env named `ftfsod` with `torch`, `mmengine`, `mmcv`, `mmdet` (pip package, will be shadowed by FT-FSOD's bundled fork at run time), and FT-FSOD's `requirements.txt` deps importable.

- [ ] **Step 1: Create the env and install PyTorch**

```bash
conda create -n ftfsod python=3.10 -y
conda run -n ftfsod pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

- [ ] **Step 2: Verify CUDA-enabled torch**

```bash
conda run -n ftfsod python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.6.0+cu124 True`

- [ ] **Step 3: Install the OpenMMLab stack**

```bash
conda run -n ftfsod pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
conda run -n ftfsod mim install "mmengine"
conda run -n ftfsod mim install "mmcv==2.1.0"
conda run -n ftfsod mim install mmdet
```

If `mmcv==2.1.0` fails to find a prebuilt wheel for this CUDA/torch/glibc combo (known CentOS 7 risk flagged in the spec), fall back to source build:

```bash
conda run -n ftfsod pip install mmcv==2.1.0 --no-binary mmcv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- [ ] **Step 4: Install FT-FSOD's remaining requirements**

```bash
conda run -n ftfsod pip install -r ~/external/FT-FSOD/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- [ ] **Step 5: Verify the full import chain, including FT-FSOD's bundled mmdet fork**

```bash
cd ~/external/FT-FSOD
PYTHONPATH="$(pwd):$PYTHONPATH" conda run -n ftfsod python3 -c "
import mmengine, mmcv, torch
import mmdet
print('mmdet module path:', mmdet.__file__)
from mmdet.models.detectors.grounding_dino_HED import GroundingDINO_ParallelDecoder_15_DNQuery_rand
from mmdet.engine.hooks.stage_lr_hook import BBoxHeadFirstHook6
print('OK: FT-FSOD custom modules import cleanly')
"
```

Expected: `mmdet module path` points inside `~/external/FT-FSOD/mmdet/...` (not a site-packages path), and the final line prints without ImportError.

- [ ] **Step 6: Record the environment for reproducibility**

```bash
conda run -n ftfsod pip freeze > /tmp/claude-2041/-home-qushiduo-projects-genaug-validation/64fed25a-55a5-45c7-81e7-f48bf791d1a6/scratchpad/ftfsod_env_freeze.txt
```

(No commit yet — this is a checkpoint for Task 6's README, not a deliverable on its own.)

---

### Task 2: Apply the MMEngine deterministic-mode patch

**Files:** two files inside the `ftfsod` conda env's `mmengine` install (not in this repo).

**Interfaces:**
- Consumes: the `ftfsod` env from Task 1.
- Produces: an `mmengine` where `Runner.set_randomness` and `mmengine.runner.utils.set_random_seed` accept `warn_only`, matching what every FT-FSOD CDFSOD config passes (`randomness=dict(..., warn_only=True)`).

- [ ] **Step 1: Locate the two target files**

```bash
conda run -n ftfsod python3 -c "import mmengine, os; print(os.path.dirname(mmengine.__file__))"
```

Note the printed path as `$MMENGINE_DIR`. The two files to patch are `$MMENGINE_DIR/runner/runner.py` and `$MMENGINE_DIR/runner/utils.py`.

- [ ] **Step 2: Confirm current (unpatched) signatures**

```bash
grep -n "def set_random_seed" "$MMENGINE_DIR/runner/utils.py"
grep -n "set_randomness\|set_random_seed(" "$MMENGINE_DIR/runner/runner.py"
```

Read the matched function bodies (`Read` tool) before editing — line numbers will differ from the README's references (which point at a specific upstream commit).

- [ ] **Step 3: Patch `runner/utils.py`**

In `set_random_seed`, add a `warn_only: bool = True` parameter and change the `torch.use_deterministic_algorithms(True)` call to `torch.use_deterministic_algorithms(True, warn_only=warn_only)`. Use the `Edit` tool with the exact surrounding lines found in Step 2 (do not guess line numbers — match on the real function signature and call site text).

- [ ] **Step 4: Patch `runner/runner.py`**

In the method that calls `set_random_seed` (search result from Step 2, `set_randomness` per the README), add `warn_only: bool = True` to its own signature and pass `warn_only=warn_only` through to `set_random_seed(...)`.

- [ ] **Step 5: Verify the patch**

```bash
conda run -n ftfsod python3 -c "
from mmengine.runner.utils import set_random_seed
import inspect
sig = inspect.signature(set_random_seed)
assert 'warn_only' in sig.parameters, sig
print('OK:', sig)
"
```

Expected: prints the signature including `warn_only`, no AssertionError.

---

### Task 3: Download weights

**Files:** none in this repo — writes to `/data1/qushiduo/models/ftfsod/`.

**Interfaces:**
- Produces: `/data1/qushiduo/models/ftfsod/bert-base-uncased/` (config + weights + tokenizer files), `/data1/qushiduo/models/ftfsod/grounding_dino_swin-b_pretrain_all-f9818a7c.pth`.

- [ ] **Step 1: Download BERT-base-uncased via hf-mirror**

```bash
mkdir -p /data1/qushiduo/models/ftfsod
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  conda run -n ftfsod python3 -c "
from transformers import BertConfig, BertModel, AutoTokenizer
config = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
out = '/data1/qushiduo/models/ftfsod/bert-base-uncased'
config.save_pretrained(out)
model.save_pretrained(out)
tokenizer.save_pretrained(out)
print('saved to', out)
"
```

- [ ] **Step 2: Verify BERT files landed completely**

```bash
ls -la /data1/qushiduo/models/ftfsod/bert-base-uncased/
```

Expected: `config.json`, `pytorch_model.bin` (or `model.safetensors`), `tokenizer.json`/`vocab.txt`, `tokenizer_config.json` all present and non-zero size. If any file is 0 bytes (the known proxy-truncation failure mode — see project memory on Claude Code's own launcher for the same symptom), re-run Step 1; `from_pretrained`/`save_pretrained` will re-fetch missing pieces from the HF cache.

- [ ] **Step 3: Download NLTK data**

```bash
conda run -n ftfsod python3 -c "
import nltk
nltk.download('punkt', download_dir='/home/qushiduo/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='/home/qushiduo/nltk_data')
"
```

- [ ] **Step 4: Download the MMGDINO-B checkpoint (via proxy, GitHub/openmmlab CDN)**

```bash
curl -L -o /data1/qushiduo/models/ftfsod/grounding_dino_swin-b_pretrain_all-f9818a7c.pth \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
```

(Proxy env vars are already set in this shell per Global Constraints — leave them on for this non-HF host, unlike Step 1.)

- [ ] **Step 5: Verify checkpoint integrity**

```bash
conda run -n ftfsod python3 -c "
import torch
ckpt = torch.load('/data1/qushiduo/models/ftfsod/grounding_dino_swin-b_pretrain_all-f9818a7c.pth', map_location='cpu', weights_only=False)
print(type(ckpt), list(ckpt.keys())[:5] if isinstance(ckpt, dict) else 'n/a')
"
```

Expected: loads without error (a truncated/corrupted download raises `RuntimeError`/`EOFError` here — catch it now, not mid-training).

---

### Task 4: Wire real paths into the FT-FSOD config

**Files:**
- Modify: `~/external/FT-FSOD/src_path.py`
- Modify: `~/external/FT-FSOD/configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py:6`

**Interfaces:**
- Consumes: `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets` (read-only, unchanged), weight paths from Task 3.
- Produces: a loadable FT-FSOD CDFSOD config with all paths resolved to real files.

- [ ] **Step 1: Edit `src_path.py`**

Replace the placeholder lines with:

```python
RF100_VL_FSOD_PATH = 'your_dataset_path/rf100-vl/rf100-vl-fsod'
ODINW_PATH = 'your_dataset_path/OdinW/odinw_13'
CDFSOD_PATH = '/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets'
CDMixed_PATH = 'your_dataset_path/CDMixed'

MMGDINOB_PATH = '/data1/qushiduo/models/ftfsod/grounding_dino_swin-b_pretrain_all-f9818a7c.pth'
MMGDINOL_PATH = 'your_checkpoint_path/grounding_dino_swin-l_pretrain_all-56d69e78.pth'
```

(Leave the RF100/ODinW/CDMixed/Swin-L placeholders — out of scope per the spec.)

- [ ] **Step 2: Edit `configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py`**

Change line 6 from:
```python
lang_model_name = 'bert-base-uncased'
```
to:
```python
lang_model_name = '/data1/qushiduo/models/ftfsod/bert-base-uncased'
```

- [ ] **Step 3: Verify the FISH 1-shot config resolves correctly**

```bash
cd ~/external/FT-FSOD
PYTHONPATH="$(pwd):$PYTHONPATH" conda run -n ftfsod python3 -c "
from mmengine.config import Config
cfg = Config.fromfile('configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_FISH_1shot.py')
print('data_root:', cfg.train_dataloader.dataset.data_root)
print('train ann_file:', cfg.train_dataloader.dataset.ann_file)
print('load_from:', cfg.load_from)
print('lang model name:', cfg.model.language_model.name)
"
```

Expected: `data_root` ends in `.../datasets/FISH`, `train ann_file` is `annotations/1_shot.json`, `load_from` and `lang model name` point at the Task 3 downloads — all real, existing paths (verify each with a quick `test -e`).

---

### Task 5: Smoke run — FISH, 1-shot, seed 42

**Files:** none in this repo (writes to `/data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/`, read-only source data untouched).

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: a confirmed, empirically-verified location and shape of the final test-metrics JSON — this is the input contract Task 6's `run_one.sh`/`aggregate_results.py` are built against.

- [ ] **Step 1: Pick a free GPU**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Pick the index with the lowest `memory.used` (GPU 4 or 5 had ~587MB used as of design time — confirm current state, it changes).

- [ ] **Step 2: Train**

```bash
mkdir -p /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs
cd ~/external/FT-FSOD
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
conda run -n ftfsod ./tools/dist_train.sh \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_FISH_1shot.py \
  1 9999 <GPU_ID_FROM_STEP_1> \
  --work-dir /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/smoke_FISH_1shot_seed42
```

Expected: runs to completion (16 epochs per the config), logs a `coco/bbox_mAP` line each epoch (`val_interval=1`), writes at least one `best_coco_bbox_mAP_iter_*.pth` under the work-dir.

- [ ] **Step 3: Confirm the checkpoint landed**

```bash
find /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/smoke_FISH_1shot_seed42 -name "best_coco_bbox_mAP_iter_*.pth"
```

Expected: exactly one match. Record its path as `$CKPT`.

- [ ] **Step 4: Test**

```bash
mkdir -p /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/smoke_FISH_1shot_seed42_test
conda run -n ftfsod ./tools/dist_test.sh \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_FISH_1shot.py \
  "$CKPT" 1 9999 <GPU_ID_FROM_STEP_1> \
  --work-dir /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/smoke_FISH_1shot_seed42_test \
  --out /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/smoke_FISH_1shot_seed42_test/FISH_1shot.pkl
```

Expected: prints a final COCO-style AP table to stdout.

- [ ] **Step 5: Locate the metrics JSON empirically**

```bash
find /data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs/smoke_FISH_1shot_seed42_test -name "*.json" | xargs -I{} sh -c 'echo ==={}===; python3 -c "import json; d=json.load(open(\"{}\")); print(\"coco/bbox_mAP\" in d, list(d.keys())[:8])"'
```

Record which file (there should be exactly one containing the `coco/bbox_mAP` key) and its path pattern relative to the test work-dir (e.g. `{timestamp}/{timestamp}.json` or `{timestamp}/vis_data/scalars.json`) — **this exact pattern is required input for Task 6, Step 1**. If no file contains the key, check stdout from Step 4 for the printed AP table and cross-reference `mmdet/evaluation/metrics/` in the FT-FSOD repo for how `CocoMetric` publishes results, then re-search.

- [ ] **Step 6: Sanity-check the number**

Print the mAP value found in Step 5. It should be a plausible detection mAP (roughly in `[0, 60]` as a percentage, or `[0, 0.6]` as a fraction depending on how `analyze_results_cdfsod.py`'s `*100` convention maps here) — not zero (would indicate a broken class/label mapping) and not suspiciously perfect (would indicate test-set leakage into the 1-shot training split). Do not proceed to Task 6 until this looks sane; if it doesn't, this is a debugging checkpoint — apply `superpowers:systematic-debugging` rather than guessing.

---

### Task 6: Repo-tracked matrix runner + result aggregator

**Files:**
- Create: `baselines/ftfsod_cdfsod/run_one.sh`
- Create: `baselines/ftfsod_cdfsod/run_matrix.sh`
- Create: `baselines/ftfsod_cdfsod/aggregate_results.py`
- Create: `baselines/ftfsod_cdfsod/tests/test_aggregate_results.py`
- Create: `baselines/ftfsod_cdfsod/results/.gitkeep`

**Interfaces:**
- Consumes: the metrics-JSON path pattern confirmed in Task 5 Step 5 (referred to below as `<CONFIRMED_JSON_GLOB>` — fill in with the real pattern before writing `run_one.sh`).
- Produces: `aggregate_results.py:load_results(results_dir: str) -> Dict[Tuple[str, str], List[float]]` (keyed by `(domain, shot)`, values are the list of mAP floats across seeds) and `aggregate_results.py:summarize(results: dict) -> List[dict]` (rows with `domain`, `shot`, `mean`, `std`, `n`) — used by later reporting work, so keep these names stable. Use `typing.Dict/List/Tuple`, not builtin `dict[...]`/`list[...]` subscripting — this file is imported by `python3 -m pytest`, which per Global Constraints runs under system Python 3.6.8.

- [ ] **Step 1: Write `run_one.sh`**

```bash
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
conda run -n ftfsod ./tools/dist_test.sh "${CONFIG}" "${CKPT_PATH}" 1 "${PORT}" "${GPU}" \
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
```

```bash
chmod +x baselines/ftfsod_cdfsod/run_one.sh
```

- [ ] **Step 2: Write `run_matrix.sh`**

```bash
#!/usr/bin/env bash
# Full CDFSOD baseline sweep: 6 domains x {1,5,10} shots x N seeds.
# Usage: run_matrix.sh <gpu_id> [seeds="42 43 44"]
set -euo pipefail

GPU="${1:?usage: run_matrix.sh <gpu_id> [seeds]}"
SEEDS="${2:-42 43 44}"
PORT_BASE=9999

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAINS="ArTaxOr clipart1k DIOR FISH NEU-DET UODD"
SHOTS="1 5 10"

i=0
for domain in $DOMAINS; do
  for shot in $SHOTS; do
    for seed in $SEEDS; do
      port=$((PORT_BASE + i))
      i=$((i + 1))
      "${SCRIPT_DIR}/run_one.sh" "${domain}" "${shot}" "${seed}" "${GPU}" "${port}"
    done
  done
done
```

```bash
chmod +x baselines/ftfsod_cdfsod/run_matrix.sh
```

- [ ] **Step 3: Write the failing test for the aggregator**

```python
# baselines/ftfsod_cdfsod/tests/test_aggregate_results.py
import json
import statistics

from baselines.ftfsod_cdfsod.aggregate_results import load_results, summarize


def test_load_and_summarize_groups_by_domain_and_shot(tmp_path):
    results_dir = tmp_path
    (results_dir / "swinB_FISH_1shot_seed42.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.30}))
    (results_dir / "swinB_FISH_1shot_seed43.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.34}))
    (results_dir / "swinB_ArTaxOr_1shot_seed42.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.50}))

    results = load_results(str(results_dir))

    assert results[("FISH", "1")] == [30.0, 34.0]
    assert results[("ArTaxOr", "1")] == [50.0]

    rows = summarize(results)
    fish_row = next(r for r in rows if r["domain"] == "FISH" and r["shot"] == "1")
    assert fish_row["n"] == 2
    assert fish_row["mean"] == statistics.mean([30.0, 34.0])
    assert fish_row["std"] == statistics.stdev([30.0, 34.0])

    artaxor_row = next(r for r in rows if r["domain"] == "ArTaxOr" and r["shot"] == "1")
    assert artaxor_row["n"] == 1
    assert artaxor_row["std"] == 0.0
```

- [ ] **Step 4: Run it to verify it fails**

```bash
cd /home/qushiduo/projects/genaug-validation
python3 -m pytest baselines/ftfsod_cdfsod/tests/test_aggregate_results.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'baselines.ftfsod_cdfsod.aggregate_results'`.

- [ ] **Step 5: Write `aggregate_results.py`**

```python
#!/usr/bin/env python3
"""Aggregate per-run FT-FSOD CDFSOD result JSONs into a mean +/- std table.

Usage: python3 aggregate_results.py <results_dir> [--out table.md]
"""
import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Tuple

RUN_NAME_RE = re.compile(r"^swinB_(?P<domain>.+)_(?P<shot>\d+)shot_seed(?P<seed>\d+)$")


def load_results(results_dir: str) -> Dict[Tuple[str, str], List[float]]:
    """Group per-run mAP values by (domain, shot). Values are percentages."""
    results = {}  # type: Dict[Tuple[str, str], List[float]]
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        run_name = os.path.splitext(os.path.basename(path))[0]
        match = RUN_NAME_RE.match(run_name)
        if not match:
            continue
        with open(path) as f:
            data = json.load(f)
        map_value = data.get("coco/bbox_mAP")
        if map_value is None:
            continue
        key = (match.group("domain"), match.group("shot"))
        results.setdefault(key, []).append(round(map_value * 100, 4))
    return results


def summarize(results: Dict[Tuple[str, str], List[float]]) -> List[dict]:
    """Turn grouped results into sorted (domain, shot, mean, std, n) rows."""
    rows = []
    for (domain, shot), values in results.items():
        rows.append({
            "domain": domain,
            "shot": shot,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        })
    rows.sort(key=lambda r: (r["domain"], int(r["shot"])))
    return rows


def render_markdown(rows: List[dict]) -> str:
    lines = ["| Domain | Shot | mAP (mean ± std) | N seeds |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['domain']} | {r['shot']} | {r['mean']:.2f} ± {r['std']:.2f} | {r['n']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    rows = summarize(results)
    table = render_markdown(rows)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add `__init__.py` files so the test import resolves as a package**

```bash
touch baselines/__init__.py baselines/ftfsod_cdfsod/__init__.py baselines/ftfsod_cdfsod/tests/__init__.py
```

- [ ] **Step 7: Run the test again to verify it passes**

```bash
cd /home/qushiduo/projects/genaug-validation
python3 -m pytest baselines/ftfsod_cdfsod/tests/test_aggregate_results.py -v
```

Expected: PASS (2 assertions groups, no failures).

- [ ] **Step 8: Create the results dir placeholder and commit**

```bash
mkdir -p baselines/ftfsod_cdfsod/results
touch baselines/ftfsod_cdfsod/results/.gitkeep
git add baselines/__init__.py baselines/ftfsod_cdfsod/
git commit -m "feat: FT-FSOD CDFSOD matrix runner and result aggregator

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Run the full baseline sweep and publish results

**Files:**
- Create: `baselines/ftfsod_cdfsod/README.md`
- Modify: `report/` — add the CDFSOD baseline table (create `report/` content if this is the first entry; check current state with `ls report/` before deciding whether to create a new file or append).

**Interfaces:**
- Consumes: `run_matrix.sh`, `aggregate_results.py` from Task 6.
- Produces: the reference table every later generation-augmentation experiment compares against.

- [ ] **Step 1: Re-pick a free GPU (state may have changed since Task 5)**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

- [ ] **Step 2: Launch the full sweep in the background**

54 runs (6 domains × 3 shots × 3 seeds), each involving a 16-epoch fine-tune + eval — expect a long wall-clock (hours). Launch with `run_in_background` so this doesn't block:

```bash
cd /home/qushiduo/projects/genaug-validation
nohup baselines/ftfsod_cdfsod/run_matrix.sh <GPU_ID> "42 43 44" \
  > /data1/qushiduo/experiments/ftfsod_cdfsod/run_matrix.log 2>&1 &
echo "launched, pid $!"
```

- [ ] **Step 3: Monitor to completion**

```bash
tail -f /data1/qushiduo/experiments/ftfsod_cdfsod/run_matrix.log
```

Expected: 18 `[run_one] done: ...` lines × 3 seeds = 54 total, no `[run_one]` error exits. If any cell fails, note it — do not silently drop it from the report (per Global Constraints, incomplete cells must be flagged, not hidden).

- [ ] **Step 4: Confirm result count**

```bash
wc -l baselines/ftfsod_cdfsod/results/run_manifest.csv
ls baselines/ftfsod_cdfsod/results/*.json | wc -l
```

Expected: 54 lines in the manifest, 54 JSON files (assuming no failures from Step 3; otherwise document the shortfall).

- [ ] **Step 5: Generate the aggregated table**

```bash
python3 baselines/ftfsod_cdfsod/aggregate_results.py \
  baselines/ftfsod_cdfsod/results \
  --out baselines/ftfsod_cdfsod/results/summary_table.md
cat baselines/ftfsod_cdfsod/results/summary_table.md
```

- [ ] **Step 6: Write `baselines/ftfsod_cdfsod/README.md`**

```markdown
# FT-FSOD CDFSOD Baseline

Un-augmented K-shot detector baseline for the genaug-validation mainline.
Detector: FT-FSOD (MM Grounding DINO Swin-B), upstream at
https://github.com/Intellindust-AI-Lab/FT-FSOD (cloned locally to
`~/external/FT-FSOD`, not vendored into this repo).

Design doc: `docs/superpowers/specs/2026-07-03-cdfsod-ftfsod-baseline-design.md`

## Setup

1. Conda env `ftfsod` (Python 3.10, torch 2.6.0+cu124, mmengine/mmcv 2.1.0/mmdet,
   FT-FSOD's `requirements.txt`) — see plan
   `docs/superpowers/plans/2026-07-03-ftfsod-cdfsod-baseline.md` Task 1 for exact
   commands, including the MMEngine deterministic-mode patch (Task 2).
2. Weights at `/data1/qushiduo/models/ftfsod/` (BERT-base-uncased,
   MMGDINO-B checkpoint) — Task 3.
3. `~/external/FT-FSOD/src_path.py` and
   `configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py:6` point at the
   real dataset/weight paths — Task 4.

## Data

Read directly from `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/`
(read-only, never modified) using the official fixed 1/5/10-shot support
splits (`annotations/{shot}_shot.json`) and full `test.json` for evaluation.

## Running

```bash
baselines/ftfsod_cdfsod/run_one.sh FISH 1 42 <gpu_id> 9999   # single cell
baselines/ftfsod_cdfsod/run_matrix.sh <gpu_id> "42 43 44"    # full sweep
python3 baselines/ftfsod_cdfsod/aggregate_results.py baselines/ftfsod_cdfsod/results
```

## Results

<paste `summary_table.md` contents here>

## Conclusion

<one paragraph: does this reproduction land in a reasonable band around
FT-FSOD's published CDFSOD numbers? Any domain/shot cells that diverged and
why, if known. This is the reference every generation-augmentation
experiment compares against going forward.>
```

Fill in the `<...>` sections with the actual Step 5 table and a real comparison against FT-FSOD's published numbers (check their paper/repo's reported CD-FSOD table) before committing — no placeholders in the committed version.

- [ ] **Step 7: Add the table to `report/`**

```bash
ls report/
```

If empty, create `report/cdfsod-baseline.md` with the same table plus a one-line pointer back to `baselines/ftfsod_cdfsod/README.md` for setup details. If `report/` already has a running document (check before assuming), append a dated section there instead — follow whatever structure already exists.

- [ ] **Step 8: Commit**

```bash
git add baselines/ftfsod_cdfsod/README.md baselines/ftfsod_cdfsod/results/ report/
git commit -m "feat: FT-FSOD CDFSOD baseline results (6 domains x 1/5/10-shot x 3 seeds)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** environment (Task 1–2), weights (Task 3), data wiring (Task 4, confirmed no conversion needed — spec's conditional copy step doesn't trigger), smoke run (Task 5), full matrix + seeds (Task 6–7), report table (Task 7), pytest unit test (Task 6) — all spec sections have a task.
- **Deferred by spec, correctly absent here:** EdgeCrafter, generation-strategy ladder, Swin-L, support-set resampling — none appear in this plan.
- **Known open risk carried forward, not hidden:** mmcv 2.1.0 build on CentOS 7 (Task 1 Step 3 has a fallback but the fallback itself may need further debugging once hit for real — flagged, not silently assumed away).
- **One genuine unknown resolved via an embedded discovery step, not a placeholder:** the exact on-disk path of the test-metrics JSON (Task 5 Step 5) — Task 6's `run_one.sh` searches for the `coco/bbox_mAP` key content-first specifically so it doesn't hardcode a guessed path depth.
