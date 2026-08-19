---
title: Band-power homeostasis (per-condition SWA) companion pipeline
updated: 2026-07-26
---

# Band-power homeostasis (per-condition SWA)

A fully separable companion pipeline that runs the standard
condition-homeostasis analysis on a measure that is not an OFF property: the
per-condition mean of the z-scored log10 instantaneous bipolar band power (the
*same* trace that annotates OFF periods). It reuses the identical
`run_cx_homeostasis_analysis` model core (LRT main effect of `condition`, Cohen's
f², rebound/surge/decline post-hocs, crossed subject/structure random effects) by
passing the band-power table in as pre-loaded `data`, so the six-condition
violins and significance bars render in exactly the same style as the OFF
measures. Additive: deleting the bandpower files
(`config/bandpower_homeostasis.yaml`, `scripts/run_bandpower_analysis.R`,
`scripts/run_all_bandpower_analyses.R`, `config/plots/bandpower_homeostasis.yaml`,
`config/summary_tables/bandpower_summary.yaml`, `load_bandpower_condition_means()`
in `R/data_loading.R`, `_output_bandpower/`) removes it. The only core touch is
that `run_cx_homeostasis_analysis` validates `off_type` against the OFF enum
only when it loads the data itself (`is.null(data)`); with pre-loaded `data`
it is a free provenance label.

- Data (single `bandpower` dataset, no llas/clas/blas split; exported by
  cnpix_local_sleep's `off-analysis export-bandpower-condition-means`):
  `summarized_full48h_bandpower_offs.parquet`: one row per
  `(subject, probe, structure, condition)` with `mean_zlog_delta`,
  `mean_zlog_eta` (plus `mean_log_*`/raw `mean_*`) and struct metadata
  `clade`/`AP.Coord`/`Cx.AP.group`. Loaded via `load_bandpower_condition_means()`.
- Model: `mean_zlog_delta ~ condition + (1|subject) + (1|structure) +
  (1|subject:structure)` (`crossed_interaction`, as for BLAS `total_area_norm`),
  under the `six`/`nrem`/`wake` condition sets.
- Run: `Rscript scripts/run_all_bandpower_analyses.R`, writing to
  `_output_bandpower/bandpower/<condition_set>/<response_var>/<model>/`.
  Then `python plot_results.py config/plots/bandpower_homeostasis.yaml` (violins +
  sig bars) and `python summarize_results.py
  config/summary_tables/bandpower_summary.yaml`. Run
  `off-analysis export-bandpower-condition-means` then `renv::install('.')`
  (refreshes `extdata`) first.
- Grid-cell-sized panels: `python plot_results_cell.py
  config/plots/bandpower_homeostasis.yaml` emits each violin as a bare, figma-
  styled SVG sized to exactly one `cx_homeostasis_grid.svg` cell (nrem -> 360×180
  pt, wake -> 180×180 pt) as `<panel>_adjusted_cell_violin.svg`, so a standalone
  band-power violin lines up beside the OFF-measure grid in Figma. The cell
  geometry comes from `plot_results_grid.grid_cell_figsize()`; the script is
  layout-independent and works for any plot YAML.

## Commands

```bash
Rscript scripts/run_all_bandpower_analyses.R
Rscript scripts/run_bandpower_analysis.R mean_zlog_delta all nrem
python plot_results.py config/plots/bandpower_homeostasis.yaml           # violins + sig bars
python summarize_results.py config/summary_tables/bandpower_summary.yaml
```
