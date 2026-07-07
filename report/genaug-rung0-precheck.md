# Generation Route Pre-Checks: img2img → Box-Local Repaint

Cheap visual pre-checks (no detector training) run before committing to a
generation route, on the real K-shot support images. Generator throughout:
FLUX.1-dev (`FluxImg2ImgPipeline` / `FluxInpaintPipeline`, base checkpoint,
not the dedicated Fill-dev inpainting model), `flux2` conda env. Rationale
for pre-checking first: in detection (unlike classification) any spurious
same-class object or a dissolved/relocated target corrupts the *label*, not
just the pixel quality — cheap to catch before spending a training run on it.

## Route 1 — naive global img2img (originally-specced rung-0): FALSIFIED on FISH

`FluxImg2ImgPipeline`, strength grid `[0.25, 0.3, 0.35, 0.4]`, all 5 FISH
5-shot support images. No single global strength was label-consistent across
all 5. Two failure modes, confirmed by follow-up controls to be method
ceilings rather than tuning artifacts:

- **Extra unlabeled fish.** The prompt + FLUX's prior fill open water with
  fish not in the ground truth → missing-annotation noise. A constrained
  prompt + negative-prompt (`true_cfg_scale=4.0`) can suppress this, but then
  *relocates* the one surviving fish out of the GT box — prompt/CFG control
  what/how-many, never *where*.
- **Empty box for tiny/low-contrast targets.** Two small support fish
  dissolve out of their GT box even at strength 0.15–0.2 (near no-op) — the
  VAE's 8× downsample/upsample erases a sub-resolution, low-contrast target
  regardless of denoising strength.

Root cause: naive global img2img has no mechanism to enforce box locality.
This motivated the pivot to mask-based (spatial) routes below.

## Route 2 — box-local background inpainting: FAILED on clipart1k, not pursued further

Freeze the GT box (padded), repaint everything else; hard-paste the box
region back after generation so the label region is pixel-exact by
construction. 6 clipart1k support images (spread across categories, incl. an
extreme ~7× upscale case), strengths 0.9–1.0.

**0/6 clean.** Base FLUX.1-dev is not an inpainting-trained checkpoint: it
free-generates a new scene rather than conditioning on the frozen region, so
the frozen box reads as a rectangle pasted onto an unrelated new
character/scene, or gets "reframed" as a picture-within-the-scene (e.g. a
framed painting, a photo held by a character) — coherent as an image, wrong
as a label (the object is no longer really "in" the depicted scene). One case
also spawned a second same-class object (a second dog) in the repainted
background.

Diagnosis: a model-capability ceiling, not a mask-logic problem.
**FLUX.1-Fill-dev** (dedicated inpainting checkpoint, not yet downloaded) is
the natural fix if this route is revisited — parked in favor of Route 3
below, which needed no new model.

## Route 3 — box-local foreground partial repaint: WORKS on clipart1k, narrow/domain-limited on FISH

Inverse of Route 2: freeze the background (background pixel-frozen by direct
paste — always seam-perfect by construction), repaint only inside the box at
`strength < 1` (SDEdit-style variation anchored on the real object's latents,
not generated from scratch). Class name included in the prompt to anchor
identity (deliberately opposite of Route 2, which omitted it to avoid
spawning extra instances). Strengths 0.4 / 0.6 / 0.8.

**clipart1k** (same 6 support images as Route 2): **6/6** kept class identity
and a filled box at strength 0.4 (genuine make/color/pose variety, e.g. the
car case cleanly varies model and color across all 3 strengths); 5/6 still
clean at 0.8. One case (bird) gained extra in-box instances at 0.6 and
drifted to a different species by 0.8. → **0.4 is a safe global floor;
0.6–0.8 usable but needs per-image confirmation.**

**FISH** (all 5 real 5-shot support images): **2/5** (the F7 fish and one
large, sharp acanthopagrus) had a narrow usable window — clean at 0.4,
degraded by 0.6, unusable (color hallucination / species drift) by 0.8.
**3/5** (both small Gerres targets + one blurry/foreshortened acanthopagrus)
dissolved to an empty box at **every** strength tested, including the 0.4
floor. This confirms Route 1's root cause (VAE downsample/contrast floor on
small or low-contrast targets) is **independent of mask direction or
strength** — box-locality fixes the "extra unlabeled object" failure mode but
not the "target dissolves" one.

## Takeaways

1. **Route 3 (foreground-anchored partial repaint) is the confirmed working
   method for clipart1k rung-1** — no new model download needed.
2. **Route 2 (background repaint) needs FLUX.1-Fill-dev** to be viable at
   all; parked rather than pursued, since Route 3 already gives a working
   path forward.
3. **FISH's problem is domain-inherent** (small / low-contrast / murky
   targets), not fixable by mask direction or strength. Reinforces the
   earlier decision not to invest further compute in FISH generative
   augmentation without a resolution/contrast eligibility gate on which
   support images to even attempt.
4. **General method lesson** (candidate for promotion beyond this repo):
   label-completeness in detection augmentation requires spatial/mask
   control — prompt/CFG alone controls what/how-many, never *where*; small
   or low-contrast targets are lost at the VAE bottleneck regardless of
   strength or mask direction (a resolution/contrast floor, not a tunable).

## Next steps

- clipart1k rung-1: build the real generation pipeline on Route 3
  (strength ≈ 0.4 default; optional per-category strength check for the
  0.6–0.8 range), then run FT-FSOD training vs the frozen Phase-1 baseline.
- FISH: no further generative-augmentation investment planned without a
  data-quality eligibility gate — parked.
- FLUX.1-Fill-dev: not downloaded; only fetch if a future need specifically
  requires the background-diversity axis (Route 2).

## Artifacts (external, gitignored — scripts + annotated-PNG evidence)

`~/projects/flux2_playground/_archive/flux-experiments/genaug_rung0_check/`:
`gen_check_flux1dev.py` + `gen_check_controls.py` (Route 1 + controls),
`gen_check_clipart_inpaint.py` (Route 2), `gen_check_clipart_fg.py` +
`gen_check_fish_fg.py` (Route 3), `outputs_*/` (per-route annotated outputs).
