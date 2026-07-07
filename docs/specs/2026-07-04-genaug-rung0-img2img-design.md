# Generation Ladder Rung 0: img2img Baseline (Design)

Date: 2026-07-04
Status: **superseded by pre-check results (2026-07-07)** — see
`report/genaug-rung0-precheck.md`. The naive-global-img2img method below was
falsified on FISH (no global strength is label-consistent across the support
set) before any training run. The working method going forward is box-local
foreground partial repaint (freeze background, repaint inside the GT box at
strength<1) — confirmed 6/6 on clipart1k, narrow/domain-limited on FISH.
This document is kept as the historical record of the original design and
its rejection; do not implement the method below as specified.

## Context & Goal

Phase-1 no-aug baselines are frozen (`report/cdfsod-baseline.md`: FT-FSOD,
clipart1k + FISH, 1/5/10-shot × 3 seeds). The mentor-agreed experiment
structure is a cumulative ablation ladder over generation strategies:
`no-aug → gen-baseline → +trick1 → +trick1+trick2 → …`, generative-model
routes only.

**This spec covers only rung 0 — the generation baseline**: the simplest
defensible generative augmentation, whose measured Δ (positive or negative)
becomes the reference point every later trick's marginal contribution is
attributed against. Rung 0's known weaknesses are chosen deliberately: they
give whatever tricks come later a clearly attributable target. **The tricks
themselves are NOT decided** — candidate directions from the survey
(foreground-preserving inpainting, LoRA domain customization,
post-generation filtering, generator swap) are a working list only, to be
finalized with the mentor as rung-0 results come in; per the earlier
direction decision, tricks are chosen iteratively, not upfront.

Method provenance: the mechanism is SDEdit-style img2img; using it as
training-data augmentation follows DA-Fusion (ICLR 2024, classification) —
rung 0 is that recipe minus textual inversion, transplanted to detection
with inherited boxes. The box-drift risk of naive editing is a documented
failure mode in the detection-generation literature (the reason
layout-conditioned and foreground-preserving routes exist), which is
precisely what makes it the right baseline weakness to measure.

A calibration fact from the baseline curves: +4 real images (1→5-shot) buys
+2.26 mAP on FISH and +3.53 on clipart1k. Synthetic-data gains will be
measured on this same scale, including an "equivalent real images" reading —
the business-facing metric.

## Method (rung 0, pre-registered)

- **Generator:** FLUX.1-dev (local, `flux2` conda env), confirmed via visual
  pre-check — see `README.md` ("生成侧方案收敛") for the full candidate list
  and rationale. FLUX.2-klein-4B was the original candidate but failed the
  same pre-check: its step-distilled sampling gives a near-binary strength
  response (no-op below ~0.8, catastrophic box drift above ~0.85, no usable
  middle ground), unlike FLUX.1-dev (guidance-distilled only) and a
  non-distilled control (SDXL-Inpaint), both of which showed a graceful
  faithfulness-diversity continuum on the same images. Fixed for the whole
  ladder so later rungs stay comparable; running multiple generators as
  parallel formal arms remains a separate lateral axis, not rung 0. The exact
  strength value is a separate open item — see below.
- **Operation:** img2img on each official K-shot support image, at a fixed
  strength, with a fixed per-domain prompt template (e.g. FISH: underwater
  photo of fish; clipart1k: cartoon/clipart illustration of {classes}).
  N = 8 variants per real image, fixed generation seeds (variant index →
  seed) for reproducibility.
- **Labels:** bboxes inherited unchanged from the source image. This is the
  known approximation of rung 0: higher strength → object drift → box noise;
  lower strength → little diversity. That faithfulness–diversity tension is
  exactly what t1 (background-only inpainting) later addresses.
- **Strength selection (bounded, before any detector training):** try 2-3
  candidate strengths spanning the confirmed generator's actual
  faithfulness-diversity range — not assumed in advance (klein's cliff showed
  the usable range can't be guessed from architecture alone, it must be
  measured) — across all support images in the target K-shot split, not a
  handful (cliff position was found to vary by image content). Pick ONE by
  visual plausibility of object preservation, document the choice. No tuning
  against test mAP, ever.

## Experiment protocol (pre-registered)

- **Pipeline bring-up:** FISH 1-shot, 1 seed end-to-end first.
- **Formal cells:** {clipart1k, FISH} × {1, 5}-shot × 3 training seeds
  (42/43/44) = 12 runs. 10-shot deferred (smallest expected gain).
- **Training:** FT-FSOD official configs untouched; only the train
  annotation/data paths are overridden (via `--cfg-options`) to point at our
  merged real+synthetic set on our own disk. Same 3 seeds, same test.json,
  same eval as the baseline.
- **Declared limitation (epoch-based budget):** keeping the official recipe
  means the augmented runs take ~9× more gradient steps (9× data, same
  epochs). This is the honest practitioner comparison ("same recipe, more
  data") but confounds "more steps" with "more data". Cheap control to
  separate them: no-aug FISH 1-shot × 3 seeds re-trained with 9× epochs
  (3 short runs). Include if the rung-0 Δ is positive; skip if Δ ≈ 0 or
  negative (nothing to explain away).
- **Decision criteria (fixed in advance):** per-cell Δ vs the frozen
  baseline; |Δ| > 2× the baseline cell's seed-std counts as signal. A
  negative or null result is a valid, reportable outcome — rung 0 exists to
  measure, not to succeed.
- **Reported alongside mAP:** generation cost (wall-clock, #images) and the
  equivalent-real-images reading off the 1→5→10 curves.

## Data & code placement

- Synthetic images + merged annotations: `/data1/qushiduo/datasets/genaug/`
  (own disk; the read-only CDFSOD source is never written — real support
  images are *copied* into the merged set so each experiment dir is
  self-contained).
- Generation pipeline code: `generation/img2img_baseline/` in this repo
  (generate script targeting the `flux2` env, annotation builder, prompt
  templates as data).
- Experiment record: `experiments/e01_rung0_img2img/` (config overrides,
  launch script, results, conclusion note), per repo convention.
- Detector training reuses `baselines/ftfsod_cdfsod/run_one.sh` extended (or
  thinly wrapped) to accept dataset-override cfg-options — mechanism decided
  at plan time.

## Non-goals

- No tricks yet (candidates like inpainting/LoRA/filtering are undecided;
  whichever gets picked becomes its own small spec/plan after rung-0 results).
- Generator *selection* (confirming one working generator before rung-0
  runs) is a prerequisite gate, not covered by this non-goal. Still out of
  scope: running multiple generators as parallel formal experimental arms —
  that comparison stays a separate lateral axis, later.
- No 10-shot, no ArTaxOr/DIOR/NEU-DET/UODD, no business data, no prompt
  engineering beyond the fixed template.

## Acceptance criteria

- 12 formal cells reported as mean±std beside the frozen baseline, same
  format as `report/cdfsod-baseline.md`, plus Δ and equivalent-real-images.
- Every number carries full setup (generator, strength, N, prompt template,
  seeds, budget).
- Synthetic data regenerable from committed code + documented seeds.
- A conclusion note saying plainly whether naive img2img augmentation helps,
  hurts, or does nothing on each domain/shot — and what that implies for
  which trick to build next.
