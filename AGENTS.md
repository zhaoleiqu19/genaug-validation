# genaug-validation

Business-oriented validation: does generative-image augmentation measurably improve few-shot object detection? Testbeds = the CDFSOD benchmark (NTIRE challenge data) + 2 business scenarios (TBD with mentor — expect only tens of real training images each).

## Tech Stack

Python 3 via conda envs (system default `python` is 2.7, and the repo-level `python3`/pytest is 3.6.8 — code imported by `python3 -m pytest` must stay 3.6-compatible). PyTorch ≤ 2.6.0 (CentOS 7 / glibc 2.17 ceiling). Detector baselines: FT-FSOD (MM Grounding DINO Swin-B, mmdet fork at `~/external/FT-FSOD`, conda env `ftfsod`) on CDFSOD; EdgeCrafter/ECDet reserved for the business scenarios.

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
- `docs/specs/` — approved design specs; `docs/plans/` — implementation plans. Agent skills that default to `docs/superpowers/...` write here instead (user preference overrides the skill default).
- Literature lives in the separate public KG repo (`~/projects/flux2_playground/docs/kg/`); this repo links to it, never copies it.

## Data & shared-disk rules (hard rules)

- **Shared datasets on `/data*` disks are read-only. Never write, edit, move, or delete anything there** — above all in other people's folders. CDFSOD lives at `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/` (a labmate's folder): read-only, no exceptions. If an experiment needs modified annotations (e.g. converted shot files), copy them into this repo or our own storage first.
- Our own `/data*/qushiduo/` folders are **ours to write freely** — download new models/weights there, write large generated outputs there, reorganize as needed. Just keep code / configs / notes out of them (those live in this repo).
- All project code / configs / docs live here, under `/home/qushiduo/projects/`.
- This repo is **private and must stay private**: small business-scenario samples (tens of images/labels) may be committed here, but never pushed to any public remote.

## Environment quirks

- HF weight downloads: local proxy blocks large LFS bodies → unset proxy env vars + `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`; pip via `-i https://pypi.tuna.tsinghua.edu.cn/simple`; `git clone` from GitHub still needs the proxy.
- arXiv fetches: use `/abs/` (or `/html/`) pages, never `/pdf/`.

## Conventions

- Every reported number states its exact setup (dataset, shots, seeds, detector, training budget) — few-shot variance is large; single-seed numbers are noise. Minimum 3 seeds for headline claims.
- Datasets are referenced by absolute path in configs, never copied into the repo (tiny business samples are the one exception).
- `handoff.md` / `agent-debrief.md` are volatile session state — gitignored, never committed.
- `README.md` carries a dated changelog + roadmap for mentor alignment — update it whenever a milestone lands (new baseline numbers, direction changes, big findings), not just at the end.

## Current Work

See `handoff.md` for current in-flight state (maintained by the `/handoff` skill).
