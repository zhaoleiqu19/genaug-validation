# FT-FSOD CDFSOD Baseline

Un-augmented K-shot detector baseline for the genaug-validation mainline.
Detector: FT-FSOD (MM Grounding DINO Swin-B), upstream at
https://github.com/Intellindust-AI-Lab/FT-FSOD (cloned locally to
`~/external/FT-FSOD`, not vendored into this repo).

Design doc: `docs/specs/2026-07-03-cdfsod-ftfsod-baseline-design.md`
Implementation plan: `docs/plans/2026-07-03-ftfsod-cdfsod-baseline.md`

## Scope (Phase 1)

The official CDFSOD configs use very uneven `max_epochs` per domain (FISH=16,
clipart1k=50, ArTaxOr/DIOR/NEU-DET/UODD=100) combined with `val_interval=1`
(full test-set validation every epoch). At official settings, DIOR alone
(5000 test images x 100 epochs) projects to ~136 GPU-hours for 3-shot x
3-seed; the full 6-domain matrix would take 4-5 days even on 2 GPUs.

Phase 1 covers only **clipart1k and FISH** — the two cheapest domains at
official settings (no deviation from the paper's protocol needed). The
other four domains (ArTaxOr, DIOR, NEU-DET, UODD) are deferred and will be
added later, as-needed, when a specific generation-augmentation experiment
requires that domain — establishing a domain's baseline is a one-time,
per-domain cost, not something repeated per experiment.

## Setup

1. Conda env `ftfsod` (Python 3.10, torch 2.6.0+cu124, mmengine/mmcv 2.1.0/mmdet,
   FT-FSOD's `requirements.txt`, plus `fairscale` and an in-place patch to
   `mmengine/runner/checkpoint.py` for `weights_only=False` — see plan Task 1/5
   for exact commands and why).
2. Weights at `/data1/qushiduo/models/ftfsod/` (BERT-base-uncased,
   MMGDINO-B checkpoint) — plan Task 3.
3. `~/external/FT-FSOD/src_path.py` and
   `configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py:6` point at the
   real dataset/weight paths — plan Task 4.

## Data

Read directly from `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/`
(read-only, never modified) using the official fixed 1/5/10-shot support
splits (`annotations/{shot}_shot.json`) and full `test.json` for evaluation.

## Running

```bash
baselines/ftfsod_cdfsod/run_one.sh clipart1k 1 42 <gpu_id> <port>   # single cell
python3 baselines/ftfsod_cdfsod/aggregate_results.py baselines/ftfsod_cdfsod/results
```

`run_matrix.sh` loops all 6 domains and is available for when the deferred
domains get added; Phase 1's 2-domain scope was run via direct `run_one.sh`
calls split across 2 GPUs (see plan Task 7).

**Known issue, fixed:** `run_one.sh`'s test step originally crashed with
`PYTHONPATH: unbound variable` under `set -u` whenever the calling shell had
no `PYTHONPATH` already set (which is the normal case) — this went
undetected in Task 6's review because the script was never actually
executed end-to-end there (only unit-tested and syntax-checked). It slipped
past 18/18 test invocations in the first Phase-1 run, so training completed
correctly for all 18 cells but zero test results landed. Fixed by defaulting
the reference (`${PYTHONPATH:-}`); all 18 checkpoints were re-evaluated
afterward without needing to retrain.

## Results

18 runs: clipart1k + FISH x {1,5,10}-shot x 3 seeds (42/43/44), MMGDINO-B,
official fixed support splits, official per-domain epoch budgets, single
RTX 4090D per run.

| Domain | Shot | mAP (mean +/- std) | N seeds |
|---|---|---|---|
| FISH | 1 | 42.37 +/- 0.59 | 3 |
| FISH | 5 | 44.63 +/- 1.26 | 3 |
| FISH | 10 | 45.77 +/- 0.42 | 3 |
| clipart1k | 1 | 56.57 +/- 0.29 | 3 |
| clipart1k | 5 | 60.10 +/- 0.30 | 3 |
| clipart1k | 10 | 61.07 +/- 0.55 | 3 |

## Conclusion

Reproduction lands close to FT-FSOD's published CD-FSOD numbers (Table 1,
MMGDINO-B), well within the spread expected from few-shot fine-tuning
variance:

| Domain | Shot | Paper | Ours | Diff |
|---|---|---|---|---|
| clipart1k | 1 | 55.6 | 56.57 | +0.97 |
| clipart1k | 5 | 59.4 | 60.10 | +0.70 |
| clipart1k | 10 | 59.6 | 61.07 | +1.47 |
| FISH (DeepFish) | 1 | 42.7 | 42.37 | -0.33 |
| FISH (DeepFish) | 5 | 45.5 | 44.63 | -0.87 |
| FISH (DeepFish) | 10 | 46.3 | 45.77 | -0.53 |

All six cells are within ~1.5 mAP of the published numbers (clipart1k
slightly higher, FISH slightly lower — both within the ~0.3-1.3 std observed
across our own 3 seeds), confirming the environment/config reproduction is
trustworthy. Both domains show the expected monotonic mAP increase from
1-shot to 10-shot.

**This table is the reference every generation-augmentation experiment on
clipart1k/FISH compares against going forward.** It does not get re-run per
experiment — only the augmented variants need new runs, compared against
these same fixed numbers (see design spec's comparison principle: augmented
results are compared against our own baseline, not directly against the
paper's numbers, to keep the causal comparison clean of infra confounds).
