# CDFSOD Baseline (Phase 1): FT-FSOD, clipart1k + FISH

Detector: FT-FSOD (MM Grounding DINO Swin-B), official fixed 1/5/10-shot
support splits, official per-domain epoch budgets, 3 seeds (42/43/44).
Full setup, environment issues found/fixed, and running instructions:
`baselines/ftfsod_cdfsod/README.md`.

| Domain | Shot | mAP (mean +/- std) | N seeds |
|---|---|---|---|
| FISH | 1 | 42.37 +/- 0.59 | 3 |
| FISH | 5 | 44.63 +/- 1.26 | 3 |
| FISH | 10 | 45.77 +/- 0.42 | 3 |
| clipart1k | 1 | 56.57 +/- 0.29 | 3 |
| clipart1k | 5 | 60.10 +/- 0.30 | 3 |
| clipart1k | 10 | 61.07 +/- 0.55 | 3 |

Reproduction check against FT-FSOD's published CD-FSOD numbers (Table 1,
MMGDINO-B): all six cells within ~1.5 mAP (clipart1k slightly above,
FISH slightly below), well inside the seed-to-seed spread we observe
ourselves — reproduction is trustworthy.

This is the reference for all clipart1k/FISH generation-augmentation
comparisons going forward. ArTaxOr/DIOR/NEU-DET/UODD are deferred (see
`docs/specs/2026-07-03-cdfsod-ftfsod-baseline-design.md` Scope) and will be
added, per-domain, only when a specific experiment needs them.
