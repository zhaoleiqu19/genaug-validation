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
- **N = 4 variants per annotation**, seed = variant index (0–3), for
  reproducibility (variant index → seed, same convention as the superseded
  rung-0 spec). **N is not evidence-tuned** — no dose-response ablation has
  been run (see Non-goals: an N ablation is a candidate future trick, not
  rung 1). N=4 was chosen for two practical reasons, not for training
  quality: it roughly halves the generation cost of the
  originally-inherited N=8 (itself carried over from the falsified rung-0
  img2img design without being re-justified for this method — measured at
  ~45s/image at strength 0.4 on one GPU from the precheck logs, N=8 would
  cost ≈12 GPU-hours across both shot tiers vs ≈6 for N=4); and for 1-shot
  specifically it makes `1 real + 4 synthetic = 5` images per class — an
  exact count match to the real 5-shot support set, which Stage 1 of the
  experiment protocol below uses for a zero-extra-cost secondary
  comparison.
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

**Execution is staged, not committed upfront.** 5-shot generation and
training only happen if the 1-shot stage clears a pre-registered decision
gate (see Experiment protocol). This avoids spending 5-shot compute before
there is any evidence the method direction works at all, consistent with
the ladder's iterative-investment philosophy (`AGENTS.md`: tricks are
decided as results come in, not upfront).

## Pipeline architecture (two-stage)

1. **Generate** (`generation/box_repaint/generate.py`) — GPU stage. Reads
   the official `{shot}_shot.json`, and for each annotation × N variants
   (N decided per stage — see Experiment protocol; N=4 for Stage 1) runs
   `FluxInpaintPipeline` with the box mask, writing raw PNGs plus a
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

### Stage 1 — 1-shot only (run first)

- **Generate:** 20 annotations × N=4 = 80 synthetic images (≈1 GPU-hour at
  the measured ~45s/image).
- **Train:** clipart1k 1-shot × 3 seeds (42/43/44) = 3 cells, on the
  1-real + 4-synthetic merged set (5 images/class total).
- **Two readings, both read off the same 3 trained cells — neither needs
  extra generation or training:**
  1. **Primary — "does augmentation help at a fixed real-data budget?"**
     (1real+4synth) vs the frozen 1-shot real baseline
     (`report/cdfsod-baseline.md`). This is the open question rung 1
     exists to answer.
  2. **Secondary — "can synthetic substitute for real data collection?"**
     (1real+4synth, exactly 5 images/class) vs the frozen 5-shot real
     baseline — an exact count match, so the comparison isolates
     real-vs-synthetic quality holding total count fixed. **Expected to
     underperform**, not treated as an open hypothesis: strength-0.4
     box repaint is anchored to the source object's pose/shape by
     construction (SDEdit-style, confirmed in the precheck) and cannot
     produce the viewpoint/pose/background diversity that 4 genuinely
     different real photos would. Reported to quantify the gap and
     document a known rung-1 limitation for the next trick, not to test
     whether it might go the other way.
- **Decision gate (reuses the existing decision criteria below, not a new
  standard):** if the **primary** reading's `|Δ| > 2×` the 1-shot
  baseline's seed-std and Δ is positive → proceed to Stage 2. If Δ is
  ≈0 or negative → stop here; report the null/negative result plainly and
  do not invest in Stage 2 — revisit what the next trick should change
  instead of mechanically running 5-shot regardless of outcome.

### Stage 2 — 5-shot (conditional on the Stage 1 gate)

- Only executed if Stage 1's primary reading clears the gate.
- N for the 5-shot generation is decided at Stage 2 kickoff, informed by
  Stage 1 — candidates are N=4 (cost parity with Stage 1) or N=5 (exact
  count match to the existing frozen 10-shot baseline, enabling the same
  kind of free secondary reading Stage 1 got at zero extra training cost).
  Not fixed now.
- Same structure as Stage 1 otherwise: clipart1k 5-shot × 3 seeds, primary
  reading vs the frozen 5-shot baseline, optional secondary reading vs the
  frozen 10-shot baseline if N is chosen to count-match.

### Decision criteria (fixed in advance, unchanged from the superseded spec)

Per-cell Δ vs the frozen baseline; `|Δ| > 2×` the baseline cell's
seed-std counts as signal. A negative or null result is a valid,
reportable outcome.

### Reporting

Alongside mAP: generation cost (wall-clock, #images generated) and,
where applicable, the count-matched secondary reading described above (in
place of the earlier curve-interpolation "equivalent real images" idea —
the exact-count comparison is more directly interpretable).

### Declared limitation, carried over

Augmented Stage-1 runs see 5× more images per epoch (1 real + 4 synthetic
per annotation) at the same epoch count as the baseline, confounding
"more steps" with "more data" — the honest practitioner comparison ("same
recipe, more data"), not a controlled ablation of steps alone. Cheap
control if the Stage 1 Δ is positive: no-aug clipart1k re-trained with
~5× epochs, 1-shot × 3 seeds only (skip entirely if Δ ≈ 0 or negative —
nothing to explain away).

## Non-goals

- No FISH generation (parked — see Scope).
- No 10-shot.
- No quality filtering, no per-image strength tuning, no strength > 0.4
  variants — all candidate future tricks, not rung 1.
- No N (variants-per-annotation) ablation / dose-response sweep. N=4 is a
  cost/convenience choice, not a tuned one (see Method); a systematic
  sweep is a candidate future trick, conditional on Stage 1 showing a
  positive signal worth optimizing further, not something to build into
  rung 1 upfront. Also avoids the "tuning against test mAP" pitfall the
  superseded spec explicitly banned for strength selection — the same
  principle applies to N.
- No Stage 2 (5-shot) work unless Stage 1's decision gate passes — see
  Experiment protocol.
- No FLUX.1-Fill-dev / background-repaint route (parked in the precheck;
  needs a different checkpoint not yet downloaded).
- No business data, no other CDFSOD domains.

## Acceptance criteria

- **Stage 1 (always delivered):** 3 formal 1-shot cells reported as
  mean±std beside the frozen 1-shot baseline, same format as
  `report/cdfsod-baseline.md`, plus the primary Δ (vs 1-shot baseline)
  and the secondary count-matched reading (vs 5-shot baseline). A
  conclusion note stating plainly whether the Stage 1 decision gate
  passed or not, and why.
- **Stage 2 (delivered only if the Stage 1 gate passes):** 3 more formal
  5-shot cells, same reporting format, primary Δ vs the 5-shot baseline
  plus the optional secondary reading vs the 10-shot baseline if N was
  chosen to count-match.
- Every number carries full setup (generator, strength, N, prompt
  template, seeds, budget).
- Synthetic data regenerable from committed code + the documented manifest
  (source image, box, category, variant, seed).
- A final conclusion note (once Stage 1, and Stage 2 if triggered, are
  both done) stating plainly whether box-repaint augmentation helps,
  hurts, or does nothing on clipart1k — and what that implies for the
  next trick to build (e.g. higher-strength variants, quality filtering,
  LoRA domain customization, an N ablation if the signal is worth
  optimizing further).
