# CDFSOD Baseline: FT-FSOD (Design)

Date: 2026-07-03
Status: awaiting user review

## Context & Goal

Mainline question: does generative-image augmentation measurably improve few-shot
object detection? Per mentor (xuanlong) guidance the experiment structure is:

- **Detector baseline on CDFSOD = FT-FSOD** (his prior work; MM Grounding DINO
  Swin-B based, SOTA on CD-FSOD, official one-shot script for the exact NTIRE
  data we have). All generation-strategy experiments on CDFSOD compare against —
  and run on top of — this detector.
- **Generation side is intentionally open** (later, separate spec). Current
  working direction — to be firmed up as we go, not binding: generative-model
  routes (different generators / generation strategies) evaluated incrementally
  against the no-aug baseline, roughly in a cumulative
  `baseline → +trick1 → +trick1+trick2 → …` style so each addition's marginal
  contribution stays attributable. Specific tricks, ordering, and even the exact
  comparison structure get decided during the survey/experiment loop.
- **EdgeCrafter (ECDet) is deferred to the business scenarios** — it will be the
  only detector used on business data; it does not need CDFSOD numbers now.

**This spec covers only the first deliverable: FT-FSOD un-augmented K-shot
baselines on CDFSOD.**

## Scope

- **Phase 1 (this run): 2 domains — clipart1k and FISH.** Revised down from
  the original 6-domain plan after Task 5's smoke run revealed the official
  configs use wildly uneven `max_epochs` per domain (FISH=16, clipart1k=50,
  everyone else=100) combined with `val_interval=1` (full test-set eval every
  epoch) — DIOR (5000 test images × 100 epochs) alone projects to ~136 GPU-hours
  for 3-shot×3-seed, ArTaxOr to ~38h; the full 6-domain×3-seed matrix would run
  ~4-5 days even on 2 GPUs. clipart1k (~47min/run) and FISH (~34min/run) are
  the two cheapest domains at official settings (no `val_interval` deviation
  needed) and, being cross-domain multi-class (clipart1k, 20 classes) and
  already-validated single-class (FISH, from Task 5), give a working
  baseline→generation-comparison pipeline on real, defensible numbers without
  the multi-day cost. **ArTaxOr/DIOR/NEU-DET/UODD are deferred**, added later
  only if/when a specific business or research need calls for that domain —
  not run wholesale up front. Each generation-augmentation experiment will
  need its target domain's baseline established before comparison; this is a
  per-domain, as-needed cost, not a one-time full-matrix cost.
- 1/5/10-shot, **official fixed support splits** (comparable to FT-FSOD paper /
  NTIRE leaderboard; no support resampling in this phase)
- Model: MMGDINO-**B** (Swin-B) — FT-FSOD's primary config; Swin-L deferred
  (24GB 4090D headroom unverified, and B is the paper's main comparison point)
- Variance: **3 training seeds** per cell for headline numbers (few-shot
  fine-tuning is unstable — FT-FSOD's own README says so even with fixed seeds).
  Smoke phase (Task 5) ran 1 seed on FISH 1-shot only.
- **We do not compare our generation-augmented results directly against
  FT-FSOD's published paper numbers.** Different infra/seeds/exact commit
  would confound the augmentation's effect. Our own no-augmentation baseline
  (this phase) — run on identical infra/seeds as the future augmented runs —
  is the actual control group. The paper's numbers remain useful only as an
  external sanity check that our reproduction is in a plausible band (see
  Acceptance Criteria), not as a substitute baseline.

## Non-goals

- No generative augmentation yet (next spec).
- No EdgeCrafter/ECDet work (waits for business data).
- No support-set resampling script (optional later add-on).
- No Swin-L, no ODinW-13 / RF100-VL benchmarks.

## Plan of record

### 1. Environment (new conda env `ftfsod`)

Per FT-FSOD README: Python 3.10, torch 2.6.0 + torchvision 0.21.0 (cu124 — the
exact combo already proven in this machine's `embed`/`flux2`/`srgs` envs, and
within the CentOS 7 / glibc 2.17 + torch ≤ 2.6.0 ceiling), then the OpenMMLab
stack: mmengine, mmcv 2.1.0, mmdet via `mim`, plus BERT-base-uncased & NLTK
data. FT-FSOD also requires a documented deterministic-mode patch to MMEngine.

Known risk: **mmcv 2.1.0 on CentOS 7** may need source compilation (glibc /
prebuilt-wheel mismatch). Mitigation: try prebuilt cu124 wheel first; fall back
to source build with the env's gcc toolchain. pip via tsinghua mirror; HF
weights (BERT) via hf-mirror with proxy unset.

### 2. Code & weights placement

- `~/external/FT-FSOD` — upstream clone (third-party code stays out of this
  repo; same convention as `~/external/OV-DEIM`). Cloned via proxy.
- `/data*/qushiduo/weights/ftfsod/` — MMGDINO-B pretrained checkpoint (from
  OpenMMLab model zoo), BERT weights.
- All experiment configs / launch wrappers / logs-of-record live in this repo
  under `baselines/ftfsod_cdfsod/`.

### 3. Data wiring

- Datasets referenced by absolute path from the read-only labmate folder
  `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/` (never written to).
- If FT-FSOD's expected annotation layout differs (id types, file naming — the
  dataset folder carries both raw string-id and `_converted` int-id shot files),
  converted/renamed copies go to `/data*/qushiduo/datasets/cdfsod_ftfsod/`, with
  the conversion script committed here.

### 4. Execution order

1. **Smoke run**: 1 domain × 1-shot × 1 seed (FISH — single-class, smallest
   annotation surface) through FT-FSOD's `run_mmgdinob_traineval_cdfsod.sh`
   (or its per-run command extracted from the script), verifying train +
   eval + COCO-mAP output end to end.
2. **Phase-1 matrix**: clipart1k + FISH × {1,5,10}-shot × 3 seeds (see Scope),
   launched via a wrapper script in `baselines/ftfsod_cdfsod/` that records
   per-run config, seed, GPU, and wall-clock. Other domains added later,
   as-needed, using the same wrapper.
3. **Report**: results table (mean ± std over seeds) in `report/`, plus a short
   conclusion note in `baselines/ftfsod_cdfsod/` comparing against FT-FSOD's
   published numbers to validate our reproduction before anything builds on it.

### 5. Acceptance criteria

- Reproduced FT-FSOD CDFSOD numbers within a reasonable band of the published
  ones (exact tolerance judged per domain; large gaps get investigated, not
  reported).
- Every reported cell carries: dataset, shots, split file used, seed count,
  model variant, training budget, GPU. (Repo convention: numbers without full
  setup are not reported.)
- `report/` table + per-run artifacts reproducible from committed configs.

## Later phases (for orientation, not in scope)

1. Generation-strategy experiments on CDFSOD × FT-FSOD (shape and specifics
   deliberately open; separate brainstorm + spec once we survey generators).
2. Business scenarios × ECDet (EdgeCrafter), once mentor provides data.
