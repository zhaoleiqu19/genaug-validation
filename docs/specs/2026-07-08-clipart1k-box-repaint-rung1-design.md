# Generation Rung 1: Box Repaint (clipart1k) — Design

Date: 2026-07-08
Status: approved, ready for implementation plan.

Supersedes the method section of
`docs/specs/2026-07-04-genaug-rung0-img2img-design.md` (already marked
superseded in that file). This spec operationalizes the confirmed
replacement method from the precheck into a real generation pipeline +
training protocol.

## Context & Goal

Phase-1 no-aug baselines are frozen (`report/cdfsod-baseline.md`,
`README.md`). `report/genaug-rung0-precheck.md` tested three generation
routes visually (no detector training) on real K-shot support images and
found: naive global img2img falsified on FISH; box-local background
inpainting failed 0/6 on clipart1k (needs FLUX.1-Fill-dev, parked); **box
repaint — freeze background, repaint inside the GT box at strength<1 —
worked 6/6 on clipart1k at strength 0.4**, and partially on FISH (2/5, the
rest parked as a domain-inherent problem).

This spec is rung 1 of the experiment ladder (`no-aug → gen-baseline →
+trick1 → …`, see `AGENTS.md`) — the first working generation method,
replacing the falsified naive-img2img rung-0 design. Its own known
limitation (single fixed strength, no diversity beyond seed, no quality
filtering) is a deliberate simplification: whatever it doesn't do becomes an
attributable target for the next trick.

## Method

- **Generator:** FLUX.1-dev, `FluxInpaintPipeline`, `flux2` conda env
  (unchanged from the precheck).
- **Mechanism:** freeze the background pixel-exact (paste the frozen region
  back after generation), repaint only inside the GT box at
  `strength = 0.4` (SDEdit-style variation anchored on the real object's
  latents, not generated from scratch). Prompt names the class:
  `"a {cat}, flat-color cartoon clipart illustration, bold black outlines"`,
  `{cat}` = the category name from the shot json's `categories` list
  (already human-readable: sheep, chair, boat, ...).
- **Per-(image, annotation) generation, not per-image:** iterate over
  annotations, not images. Each annotation's box is the sole repaint
  target for that generation call; every other region — including a second
  GT box on the same image (present in 8/92 images in the clipart1k 5-shot
  split, confirmed by inspecting `5_shot.json`) — stays frozen background.
  The output variant carries the full original annotation list for that
  source image unchanged: box coordinates never move, the untouched box is
  pixel-identical to source, the targeted box is a same-class repaint.
- **N = 8 variants per annotation**, seed = variant index (0–7), for
  reproducibility (variant index → seed, same convention as the superseded
  rung-0 spec).
- **Strength fixed at 0.4** for every variant — the confirmed-safe floor
  from the precheck (6/6 clean). No per-image strength tuning, no mixing in
  0.6/0.8 variants: that granularity (needs per-image confirmation per the
  precheck's bird case, which drifted species at 0.8) is deferred to a
  later trick, not built into rung 1.
- **No quality filtering.** All `N × |annotations|` generated variants are
  kept and merged into training data unfiltered. Deliberate rung-1
  simplification, consistent with the ladder philosophy: a pre-registered,
  attributable weakness, not an oversight.

## Scope

clipart1k only, 1-shot (20 annotations) and 5-shot (100 annotations)
support splits — matching the frozen baseline's 1/5-shot cells. 10-shot
deferred (smallest expected gain from more real images, so smallest
expected augmentation payoff too — same reasoning as the superseded spec).
FISH stays parked: the precheck found 3/5 support images dissolve to an
empty box at every strength tested, including the 0.4 floor — a
domain-inherent VAE-downsample/contrast floor, not fixable by mask
direction or strength, so not something this pipeline addresses.

## Pipeline architecture (two-stage)

1. **Generate** (`generation/box_repaint/generate.py`) — GPU stage. Reads
   the official `{shot}_shot.json`, and for each annotation × 8 variants
   runs `FluxInpaintPipeline` with the box mask, writing raw PNGs plus a
   manifest (source image, box, category, variant index, strength, seed,
   output path). Resumable/idempotent (skip outputs that already exist),
   following the precheck script's pattern.
2. **Build annotations** (`generation/box_repaint/build_annotations.py`) —
   CPU-only stage. Reads the manifest + the original shot json, copies the
   real support images alongside the generated ones, and emits a merged
   COCO-format training json to `/data1/qushiduo/datasets/genaug/clipart1k/`.
   Decoupled from generation on purpose: re-merging (e.g. dropping specific
   variants later, changing which variants get included) never requires
   re-running FLUX.
- The prompt template lives as a small data structure in the generation
  module (category name → prompt string), not a hardcoded literal
  duplicated across scripts.

## Data & code placement

- `generation/box_repaint/` — `generate.py`, `build_annotations.py`,
  prompt template data, unit tests for the annotation-merge logic.
- `/data1/qushiduo/datasets/genaug/clipart1k/` — synthetic images + merged
  COCO json + copied real support images (self-contained; the read-only
  CDFSOD source at `/data6022/xuanlong/...` is never written).
- `experiments/clipart1k_box_repaint/` — config/launch script, results,
  conclusion note, per repo convention.
- Training reuses `baselines/ftfsod_cdfsod/run_one.sh` (or a thin wrapper)
  extended with a dataset-path `--cfg-options` override pointing the train
  annotation/image paths at the merged set; everything else about the
  official config is untouched (same epoch budget, same `test.json`, same
  eval). Exact wrapper mechanism is left to the implementation plan.

## Experiment protocol

- **Cells:** clipart1k × {1, 5}-shot × 3 seeds (42/43/44) = 6 runs, trained
  on the real+synthetic merged set, compared against the same 6 frozen
  Phase-1 cells (`report/cdfsod-baseline.md`).
- **Decision criteria (fixed in advance, unchanged from the superseded
  spec):** per-cell Δ vs the frozen baseline; `|Δ| > 2×` the baseline
  cell's seed-std counts as signal. A negative or null result is a valid,
  reportable outcome.
- **Reported alongside mAP:** generation cost (wall-clock, #images
  generated) and an equivalent-real-images reading off the baseline's
  1→5→10-shot curve.
- **Declared limitation, carried over:** augmented runs see 9× more images
  per epoch (1 real + 8 synthetic per annotation) at the same epoch count
  as the baseline, confounding "more steps" with "more data" — the honest
  practitioner comparison ("same recipe, more data"), not a controlled
  ablation of steps alone. Cheap control if the rung-1 Δ is positive:
  no-aug clipart1k re-trained with ~9× epochs, 1-shot × 3 seeds only (skip
  entirely if Δ ≈ 0 or negative — nothing to explain away).

## Non-goals

- No FISH generation (parked — see Scope).
- No 10-shot.
- No quality filtering, no per-image strength tuning, no strength > 0.4
  variants — all candidate future tricks, not rung 1.
- No FLUX.1-Fill-dev / background-repaint route (parked in the precheck;
  needs a different checkpoint not yet downloaded).
- No business data, no other CDFSOD domains.

## Acceptance criteria

- 6 formal cells reported as mean±std beside the frozen Phase-1 baseline,
  same format as `report/cdfsod-baseline.md`, plus Δ and an
  equivalent-real-images reading.
- Every number carries full setup (generator, strength, N, prompt
  template, seeds, budget).
- Synthetic data regenerable from committed code + the documented manifest
  (source image, box, category, variant, seed).
- A conclusion note stating plainly whether box-repaint augmentation
  helps, hurts, or does nothing on clipart1k at 1-shot and 5-shot — and
  what that implies for the next trick to build (e.g. higher-strength
  variants, quality filtering, LoRA domain customization).
