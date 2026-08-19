---
title: Excess globality (observed vs windowed null) companion pipeline
updated: 2026-07-26
---

# Excess globality (observed vs windowed null)

Another fully separable companion pipeline tests whether cross-structure OFF
overlap is greater than chance: is an OFF's observed overlap degree larger than
the duration-matched windowed local-shift null? It mirrors the locality
companion, swapping `overlap_status` for a two-level `quantity` factor (`null`
reference, `observed`). It is the r-offp port of cnpix_local_sleep's
`cross_structure_offs.test_excess_above_chance` (the subject-level paired test +
intercept-only mixed model behind the *Plot 5 statistics* cell of
`group_cross_structure_offs.ipynb`). Additive: deleting the excess-globality
files (`R/excess_globality_data_loading.R`, `R/excess_globality_analysis.R`,
`config/excess_globality.yaml`, `scripts/run_excess_globality_analysis.R`,
`tests/testthat/test-excess_globality_analysis.R`, `_output_excess_globality/`)
and re-running `devtools::document()` removes it without touching anything else.

- Data (LLAS only, whole-recording NREM; exported by cnpix_local_sleep's
  `off-analysis export-excess-globality-offs`):
  `summarized_excess_globality_offs.parquet`: one row per
  `(subject, structure, quantity)` with `value` (the cell's mean overlap degree),
  `count`, `clade`, `condition == "NREM"`, and provenance (`null_scope`,
  `window`, `n_shuffles`). Loaded via `load_excess_globality_summary()`;
  `prepare_excess_globality_data()` factors `quantity` (`null` reference).
- One question: across multi-cortical subjects, is observed overlap degree
  greater than the windowed-shift null? The LRT main effect of `quantity` (full
  `value ~ quantity + (1|subject) + (1|subject:structure)` vs intercept-only null)
  is the "observed > chance" test; the `"observed - null"` posthoc gives
  direction and Cohen's d. Unit of analysis is the per-`(subject, structure)`
  mean (no per-OFF pseudoreplication), pairing carried by `(1|subject:structure)`.
  Singular fits expected on weak/zero effects; the FE LRT is valid.
- Run: `Rscript scripts/run_excess_globality_analysis.R [dataset]`,
  writing to the separate
  `_output_excess_globality/<dataset>/state-NREM/value/excess_paired/`
  root. Run `off-analysis export-excess-globality-offs` then `renv::install('.')`
  (refreshes `extdata`) first.
