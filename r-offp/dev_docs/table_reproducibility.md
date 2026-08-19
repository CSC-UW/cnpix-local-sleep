---
title: The manuscript tables are not bitwise reproducible
updated: 2026-08-18
---

# Run-to-run variation in the manuscript tables

Rebuilding the supplement twice from identical code, data and renv library does not
produce identical tables. Measured 2026-08-18, two consecutive
`run_all_analyses.R all` + `run_all_locality_analyses.R` passes in one clone:

| Table | Cells differing between two identical runs |
| --- | --- |
| `manuscript_s1a_homeostasis.csv` | 13 / 370 |
| `manuscript_s3a_locality.csv` | 2 / 95 |
| everything else (S0, S1b, S2a, S2b, S3b, S4a, S4b, S5) | 0 |

Magnitude. Third-to-fourth significant figure. p-values move like
`0.0064` <-> `0.0065`; CI bounds move in their last digit, most often bounds sitting
near zero (`-0.0073` -> `-0.0074` -> `-0.0075` across three runs of the same code).
No significance star, point estimate or effect size has been observed to change.

Values oscillate rather than drift. On one S1a cell the three observed runs gave
`0.0065`, `0.0064`, `0.0065`. Do not read a difference as a direction.

A variance component at the boundary reads as `0` or `1.3e-05` interchangeably.
S0's `SD struct` for Total OFFness / Medium+Large / NREM is a singular fit; both
strings mean "estimated at zero". A build logs ~174 `boundary (singular) fit` warnings.

## What this means for comparisons

Byte-comparing `_output_manuscript/*.csv` across two builds will report differences
that are not changes. When checking that a refactor left the numbers alone, compare
with tolerance and check the things that carry meaning: stars, point estimates, effect
sizes, and CI *signs*. A refactor is clean if it stays inside the envelope above.

Worked example: verifying the checkpoint-3 data split (2026-08-18), a clean clone was
compared against a supplement built on 2026-08-14. Seven of ten tables were
byte-identical; the three that differed (18 cells) differed no more than two identical
runs of the same code did (15 cells), so the gap was attributed to this effect rather
than to the refactor.

## Suspected mechanism (conjecture, not verified)

R here links threaded OpenBLAS (`libopenblasp-r0.3.30.so`) on a 224-core machine
with `OMP_NUM_THREADS` unset, so the thread count (and therefore floating-point
summation order) varies with machine load. Near-singular mixed-model fits sit on a
flat likelihood surface where that is enough to move the optimizer's stopping point.
This would explain why only the lme4 post-hoc tables vary while the seeded permutation
(S5) and resampling diagnostics (S4a/S4b) are stable, but it has not been tested.
The obvious test is to pin `OMP_NUM_THREADS=1` and re-run twice; if the tables then
match bitwise, the mechanism is confirmed and pinning it becomes a cheap way to make
manuscript builds comparable.
