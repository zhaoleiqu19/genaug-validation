# clipart1k Rung-1: Box Repaint — Stage 1 (1-shot)

Design doc: `docs/specs/2026-07-08-clipart1k-box-repaint-rung1-design.md`
Implementation plan: `docs/plans/2026-07-08-clipart1k-box-repaint-stage1.md`
Precheck that selected this method: `report/genaug-rung0-precheck.md` (Route 3)

## Method

Freeze everything outside the GT box, repaint inside it at strength 0.4
(SDEdit-style, FLUX.1-dev, `flux2` env), N=4 synthetic variants per
annotation, per-category prompt `"a {cat}, flat-color cartoon clipart
illustration, bold black outlines"`. 1-shot split: 20 real images/20
annotations -> 80 synthetic images generated, merged into a 100-image
training set (1 real + 4 synthetic per class). Full mechanism detail
(resolution round-trip, mask construction, background-freeze guarantee):
`docs/notes/2026-07-09-box-repaint-pipeline-qa.md` (gitignored, local only).

Training reuses the frozen FT-FSOD 1-shot config completely unmodified
(same `batch_size=4`, `AdamW lr=1e-4`, `max_epochs=50`) except for the
train-data path, via `run_one.sh`'s `TRAIN_DATA_ROOT`/`TRAIN_ANN_FILE`/
`TRAIN_IMG_PREFIX` overrides. 3 seeds (42/43/44).

## Why "anchored to source pose" — mechanism, not just an empirical observation

`FluxInpaintPipeline`'s `strength` controls two things together: how many of
the `num_inference_steps` actually run (`init_timestep = min(steps *
strength, steps)`), and how heavily-noised the starting latents are (derived
from that same truncated timestep range). At `strength=0.4, steps=28`, only
the last 11 (least-noised) steps run, starting from a lightly-noised version
of the source box content — a small "creative budget," not a fresh
generation from pure noise. Separately, on every one of those steps the
pipeline re-noises the *original* image's latents to that step's noise level
and force-overwrites everything outside the mask with them, so only the
box-interior latents genuinely evolve — the background one sees during the
box's evolution is always the true background, not an approximation. This
latent-level mask blending is what `freeze_outside_box`'s final pixel-level
hard-paste backs up (VAE encode/decode isn't perfectly lossless, so the
hard-paste removes the residual reconstruction error the mask blending
alone wouldn't catch). Net effect: strength directly trades off box-content
novelty against pose/shape fidelity to the source — 0.4 was chosen (in the
rung-0 precheck) specifically to stay label-consistent, at the cost of the
low variant-to-variant diversity this Stage 1 result reflects.

## Results

### Primary reading — does augmentation help at a fixed real-data budget?

(1real+4synth) vs the frozen 1-shot real baseline:

| | seed42 | seed43 | seed44 | mean ± std |
|---|---|---|---|---|
| 1-shot baseline (real only) | 56.90 | 56.40 | 56.40 | 56.57 ± 0.29 |
| 1real+4synth (box repaint) | 56.00 | 56.90 | 55.60 | 56.17 ± 0.67 |

**Delta = -0.40, signal threshold (2x baseline std) = 0.58.**
**Decision gate: NO SIGNAL.**

### Secondary reading — can synthetic substitute for real data collection?

(1real+4synth, 5 images/class) vs the frozen 5-shot real baseline (exact
count match; predicted in the design spec to underperform, not treated as
an open hypothesis):

| | seed42 | seed43 | seed44 | mean ± std |
|---|---|---|---|---|
| 5-shot baseline (real only) | 59.80 | 60.10 | 60.40 | 60.10 ± 0.30 |
| 1real+4synth (box repaint) | 56.00 | 56.90 | 55.60 | 56.17 ± 0.67 |

**Delta = -3.93.**

## Conclusion

**Box repaint at strength 0.4 shows no measurable effect on clipart1k
1-shot detection.** The primary Δ (-0.40) is well inside 3-seed noise
(threshold 0.58) — indistinguishable from zero, and if anything slightly
negative. Per the pre-registered decision gate, this is a stop: **Stage 2
(5-shot) is not run**, and the ~5x-epoch step-count control is also skipped
(nothing positive to explain away — see design spec's declared limitation).

The secondary reading confirms the design spec's prediction: 4 synthetic
variants anchored to a single source image's exact pose/background cannot
substitute for 4 genuinely different real photos (Δ = -3.93 vs 5-shot).
This is a known ceiling of strength-anchored, single-source repaint, not a
pipeline bug — the generation pipeline itself was verified correct
end-to-end (80/80 images generated cleanly across 4 batches, annotation
merge verified with no missing files, output quality spot-checked).

**Why no signal, plausible reading:** strength=0.4 keeps every synthetic
variant close to its single real source image (same pose, same background,
only texture/color inside the box differs) — the augmentation adds volume
without adding the kind of instance diversity (viewpoint, pose, background)
that would plausibly move few-shot detection mAP. This doesn't mean
generative augmentation can't work for this problem; it means this specific
low-diversity route, at this strength, doesn't.

## Implications for the next trick

Not pursuing: N sweep, Stage 2 (5-shot), or the epoch-count control on this
method — none would be informative given Δ ≈ 0. The next rung should target
*diversity* directly rather than tuning this route's parameters further —
e.g. a compositional route (place the object in genuinely different
backgrounds/poses) rather than local repaint anchored to one source image.
Candidate directions are open, not yet decided.

## Pipeline notes (for reruns / other domains)

- Generation is resumable and was run in 4 batches (`--limit 5/10/15/20`)
  specifically to catch failures early rather than discovering a problem
  only after a full 80-image run — worth keeping as the default pattern for
  future generation runs on this shared machine.
- `generate.py` uses `pipe.enable_group_offload(offload_type="block_level",
  num_blocks_per_group=1, use_stream=True)`, not the sequential offload from
  the original Stage 1 plan doc. Benchmarked on an idle GPU: ~24s/image at
  ~9 GiB peak, vs. sequential offload's ~85-124s/image at <2 GiB peak, and
  vs. the `enable_model_cpu_offload` mode that OOM'd twice last session at
  ~18-21 GiB peak. Verified output-image-identical (same weights/precision,
  only execution scheduling differs) before switching production code.
- Training and generation seeds are independent: generation seeds
  (0-3, one per variant) are baked into the already-generated image files;
  training seeds (42/43/44) only control data-loading order and
  augmentation randomness on top of that fixed dataset.
