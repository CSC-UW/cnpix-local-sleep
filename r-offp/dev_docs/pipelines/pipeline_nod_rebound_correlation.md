---
title: NOD-rebound correlation companion pipeline
updated: 2026-08-19
---

# NOD-rebound correlation

Another fully separable companion pipeline tests whether a whole-period NOD
OFF metric predicts that same metric's `NREM.Rebound`
(`Early.REC.NREM - Early.REC.NREM.Match`), across `(subject, probe, structure)`
combos. It exists because the exploratory version of this comparison
quantifies it with an ordinary Pearson correlation that pools every combo as
independent (~29 combos from 15 subjects): pseudoreplication. Here the slope of
the NOD predictor is estimated in a linear mixed model with `subject` (and
optionally `structure`) as random intercepts. Additive: deleting
`R/correlation_analysis.R`, `config/nod_rebound_correlation.yaml`,
`scripts/*correlation*.R`, `tests/testthat/test-correlation_analysis.R`, the
`nod_rebound_correlation_*_offs.parquet` extdata, and `_output_correlations/`
(then `devtools::document()`) removes it without touching anything else.

- Data: `nod_rebound_correlation_{off_type}_offs.parquet`, exported by
  cnpix_local_sleep's `off-analysis export-nod-rebound-correlation` (conditions: the two
  whole-period predictors `NOD` (Wake+NREM) and `NOD.Wake` (wake-only), plus
  `Early.REC.NREM`, `Early.REC.NREM.Match`). Loaded via
  `load_nod_rebound_correlation_data()`; reshaped to one row per
  `(subject, probe, structure)` (`x` = metric@predictor, `y` = the response
  metric's NREM.Rebound) by `build_correlation_frame()`.
- Cross-metric response (`response_metric:` on a metric entry, optional): by
  default `y` is the predictor metric's own NREM.Rebound. Setting `response_metric`
  makes `y` the NREM.Rebound of a *different* quantity, currently
  `mean_zlog_delta` (cortical delta power), sourced from the band-power condition
  means (`summarized_full48h_bandpower_offs.parquet`, via
  `load_bandpower_condition_means()`) and joined to `x` on
  `(subject, probe, structure)`. `build_correlation_frame(..., response_metric,
  response_data)` and `run_nod_rebound_correlation(..., response_metric,
  response_data)` carry this; the orchestrators load the band-power frame once
  when any entry needs it. Cross-metric outputs use a combined metric path segment
  `<metric>__vs__<response_metric>/` (preserving the fixed 4-level output depth the
  diagnostic scripts glob). The delta rebound is a full-cortical property
  independent of LAS filtering, so `y` is identical across llas/clas/llas_exclusive;
  only `x` differs by dataset.
- Predictors (`predictor_conditions:` in the config, a list): the analysis is
  run once per predictor and outputs are separated at the
  `.../<metric>/<predictor>/` level, so `NOD` and `NOD.Wake` never collide. Both
  are present in every parquet, so switching predictors needs no re-export. A
  scalar `predictor_condition:` is still honored for backward compatibility.
- Models (`config/nod_rebound_correlation.yaml`): two named random-effects
  structures: `subject` (`y ~ x + (1|subject)`, the headline) and
  `subject_structure` (`y ~ x + (1|subject) + (1|structure)`, the crossed
  robustness check). Slope tested by LRT of full vs `y ~ 1 + <re_terms>`; Cohen's
  f² via the reused `subtract_ranef_get_fsquared()` (handles crossed intercepts).
- Metrics: predictors `total_area_norm` and `rate`, each fit both as
  self-rebound and cross-metric (`response_metric: mean_zlog_delta`), ×
  `llas`/`clas`/`blas`/`llas_exclusive` (`llas & ~clas`; from
  `export-nod-rebound-correlation --category llas_exclusive`) × 2 predictors
  (`NOD`, `NOD.Wake`).
- Run: `Rscript scripts/run_all_correlation_analyses.R [dataset]`
  (or `run_correlation_analysis.R <metric> [model] [dataset]`), sweeps every
  predictor, writing to the separate
  `_output_correlations/<dataset>/<metric>/<predictor>/<model>/`
  root (`results.json`/`.rds`, `summary.txt`, `variance_components.csv`, an SVG
  subject-coloured scatter with the fixed-effect line + CI ribbon).
- Robustness diagnostics (additive; do NOT replace base outputs; each sweeps
  all predictors, with predictor-scoped outputs):
  `run_correlation_xtransform_sensitivity.R` (rank(x)/log(x) leverage-robust
  refits), `run_correlation_influence_diagnostics.R` (Cook's D / LOSO / LOCO /
  leave-2 & 3-subject-out -> `_output_correlations/influence_diagnostics/<predictor>/`),
  and the `plot_correlation_{xtransform,influence_diagnostics}.R` SVG plotters;
  each transform-comparison SVG is colocated at the fit's
  `xtransform_sensitivity/comparison.svg` path.
  `plot_correlation_xtransform_publication.R` exports the subject-only
  self-rebound publication-grid data; its Python companion renders 7-inch-wide
  LLAS/CLAS SVGs with editable Figma text, one per predictor.
  Findings: `gfys_workspace/docs/reports/2026-07-08_correlation_influence_and_transform_sensitivity.md`.

## Commands

Every command sweeps BOTH predictors in the config's `predictor_conditions`
(`NOD` = Wake+NREM, `NOD.Wake` = wake-only), writing to separate
`.../<metric>/<predictor>/` subtrees.

```bash
Rscript scripts/run_all_correlation_analyses.R all           # base LMM fits (all predictors)
Rscript scripts/run_correlation_xtransform_sensitivity.R     # rank(x)/log(x) leverage-robust refits
Rscript scripts/run_correlation_influence_diagnostics.R      # Cook's D / LOSO / LOCO / leave-2 & 3-out
Rscript scripts/plot_correlation_xtransform.R                # SVG: raw|rank|log scatter per fit
Rscript scripts/plot_correlation_xtransform_publication.R    # export publication-grid model data
python scripts/plot_correlation_xtransform_publication.py    # Figma-ready SVG: 7in LLAS/CLAS grids
Rscript scripts/plot_correlation_influence_diagnostics.R     # SVG: leverage scatter / LOSO / leave-2 (run diagnostics first)
```
