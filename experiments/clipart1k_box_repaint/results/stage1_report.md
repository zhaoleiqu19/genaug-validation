## Stage 1 primary reading: augmentation vs 1-shot baseline

| | mean +/- std | n |
|---|---|---|
| 1-shot baseline | 56.57 +/- 0.29 | 3 |
| 1real+4synth | 56.17 +/- 0.67 | 3 |

Delta = -0.40, signal threshold (2x baseline std) = 0.58
Decision gate: NO SIGNAL -> stop, do not run Stage 2

## Stage 1 secondary reading: augmentation vs 5-shot baseline (expected to underperform)

| | mean +/- std | n |
|---|---|---|
| 5-shot baseline | 60.10 +/- 0.30 | 3 |
| 1real+4synth | 56.17 +/- 0.67 | 3 |

Delta = -3.93
