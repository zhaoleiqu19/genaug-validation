# genaug-validation

Business-oriented validation: does generative-image augmentation measurably improve few-shot object detection? Testbeds = the CDFSOD benchmark (NTIRE challenge data) + 2 business scenarios (TBD with mentor — expect only tens of real training images each).

## Tech Stack

Python 3 via conda envs (system default `python` is 2.7 — always use `python3`). PyTorch ≤ 2.6.0 (CentOS 7 / glibc 2.17 ceiling). Detector baselines: CD-ViTO / DE-ViT (detectron2, ships with the CDFSOD repo) and/or a deployment-style detector (TBD).

## Commands

- Test: `python3 -m pytest`
- Lint: (none yet)
- Run: per-experiment entry scripts under `experiments/` (each experiment dir documents its own run command)

## Architecture

- `experiments/` — one dir per experiment: config, launch script, results, and a short conclusion note
- `baselines/` — K-shot real-data-only detector baselines (the reference numbers everything else is compared against)
- `generation/` — synthetic-data pipelines (inpainting / LoRA-adapted generators / compositional routes)
- `report/` — the running evidence report for the mentor (what works, what doesn't, at what cost)
- `notes/` — working notes
- Literature lives in the separate public KG repo (`~/projects/flux2_playground/docs/kg/`); this repo links to it, never copies it.

## Data & shared-disk rules (hard rules)

- **Shared datasets on `/data*` disks are read-only. Never write, edit, move, or delete anything there** — above all in other people's folders. CDFSOD lives at `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/` (a labmate's folder): read-only, no exceptions. If an experiment needs modified annotations (e.g. converted shot files), copy them into this repo or our own storage first.
- Our own `/data*/qushiduo/` folders are for **large artifacts only** (model weights, big generated outputs). No code, no configs, no notes there.
- All project code / configs / docs live here, under `/home/qushiduo/projects/`.
- This repo is **private and must stay private**: small business-scenario samples (tens of images/labels) may be committed here, but never pushed to any public remote.

## Environment quirks

- HF weight downloads: local proxy blocks large LFS bodies → unset proxy env vars + `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`; pip via `-i https://pypi.tuna.tsinghua.edu.cn/simple`; `git clone` from GitHub still needs the proxy.
- arXiv fetches: use `/abs/` (or `/html/`) pages, never `/pdf/`.

## Conventions

- Every reported number states its exact setup (dataset, shots, seeds, detector, training budget) — few-shot variance is large; single-seed numbers are noise. Minimum 3 seeds for headline claims.
- Datasets are referenced by absolute path in configs, never copied into the repo (tiny business samples are the one exception).
- `handoff.md` / `agent-debrief.md` are volatile session state — gitignored, never committed.

## Current Work

See `handoff.md` for current in-flight state (maintained by the `/handoff` skill).
