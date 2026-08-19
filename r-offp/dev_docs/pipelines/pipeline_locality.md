---
title: Locality (Local vs Overlapping) companion pipeline
updated: 2026-08-19
---

# Locality analyses (Local vs Overlapping)

Another fully separable companion pipeline tests cross-structure OFF
"locality": whether an OFF in one cortical structure temporally overlaps OFFs in
other structures (`overlap_status` `Local` = overlaps none; `Overlapping` =
overlaps >=1). It splits OFFs by `overlap_status` where the core pipeline splits
them by condition alone, and answers the inferential questions behind the
exploratory cross-structure OFF plots.
Additive: deleting the locality files (`R/locality_data_loading.R`,
`R/locality_analysis.R`, `config/locality.yaml`, `scripts/*locality*.R`,
`config/summary_tables/locality_summary.yaml`, `config/plots/locality.yaml`,
`plot_locality_results.py`,
`tests/testthat/test-locality_analysis.R`, `_output_locality/`) and re-running
`devtools::document()` removes it without touching anything else.

- Data (LLAS only; exported by cnpix_local_sleep's `off-analysis export-locality-offs`):
  - `summarized_locality_overlap_offs.parquet`: `(subject, structure, condition)`
    with `mean_overlap_degree`. Loaded via `load_locality_overlap_summary()`.
  - `summarized_locality_per_condition_llas_offs.parquet`:
    `(subject, probe, structure, condition, overlap_status)` medians of
    `median_duration`/`median_span`/`median_area`. Loaded via
    `load_locality_per_condition_summary()`.
  - `summarized_locality_full48h_llas_offs.parquet`:
    `(subject, probe, structure, state, overlap_status)`, `state in {NREM, Wake}`,
    overlap computed over the whole 48 h recording. Loaded via
    `load_locality_full48h_summary()`.
  - `prepare_locality_data()` factors `overlap_status` (`Local` reference).
- Three questions (`config/locality.yaml`):
  1. condition contrasts on `mean_overlap_degree`, routed through
     `run_cx_homeostasis_analysis` (ordinary `condition` fixed effect; includes
     the `Late.NOD.Wake - Early.NOD.Wake` posthoc).
  2. within each condition, Local vs Overlapping per measure:
     `run_locality_main_effect_analysis` (one fit per condition), plus the
     condition × overlap_status interaction: `run_locality_interaction_analysis`
     (six/nrem/wake).
  3. within whole-recording NREM / Wake, Local vs Overlapping per measure
     (`run_locality_main_effect_analysis`), plus the state × overlap_status
     interaction.
- Models (paired within `subject:structure`): single condition/state
  `~ overlap_status + (1|subject) + (1|structure) + (1|subject:structure)`;
  interaction `~ condition * overlap_status + (1|subject) + (1|structure) +
  (1|subject:structure)`. Singular fits expected; the FE LRT is valid.
- Two ways to get per-condition Local-vs-Overlapping: the separate
  per-condition fits above, OR (from the SAME pooled interaction model)
  `build_locality_simple_effect_contrasts()`: the interaction result's
  `interaction$simple` holds the per-condition simple effects (one FWER-adjusted
  `glht` family) and `interaction$overlap_main` the marginal overlap effect
  (averaged across conditions).
- Weight the pooled interaction (`locality_interaction_weighted`): the OLS
  pooled fit assumes one residual variance across cells, which its RVF/QQ plots
  show is violated when a window mixes high-variance (NREM) and low-variance
  (Wake) cells; the OLS `six` simple effects then understate Wake (e.g.
  `median_duration` Early.NOD.Wake p~0.13). Inverse cell-variance WLS
  (`weighted: true`) resolves the heteroscedasticity (homogeneous Pearson
  residuals; near-normal QQ) and recovers the Wake effects (Early.NOD.Wake
  p~8e-8) with essentially unchanged point estimates; this is the principled fix,
  not window-splitting. CAVEAT: `cohens_d_analogue`/`f^2` sum VarCorr variances, and
  the weighted residual variance is on the weighted (not response) scale, so the
  weighted d/f^2 are NOT comparable to OLS; compare on estimates + p-values.
- Transform variants of the `six` interaction (`locality_interaction_{log,sqrt,
  weighted_log,weighted_sqrt}`): weighting alone leaves a heavy upper tail / positive
  skew in the QQ. A concave transform helps; the `six` window is fit across all 6
  OLS/WLS x identity/log/sqrt combos and each interaction fit now writes
  `figures/diagnostics.txt` (Shapiro-Wilk on residuals) for comparison. Empirically
  (all three measures) WLS-log is best (ordering log > sqrt > identity; weighting
  helps on top of any transform): Shapiro W ~0.91->0.97 (duration), ~0.84->0.95 (area).
  Even WLS-log doesn't pass the strict Shapiro test (n=234 -> hypersensitive) and area
  keeps a mild upper tail, but the departure is now a few extreme points, not skew.
  Inference is robust across all variants (all 6 simple effects sig, Wake included).
  Log estimates are on the multiplicative (log-ratio) scale.
- Run: `Rscript scripts/run_all_locality_analyses.R [dataset]`
  (or `run_locality_analysis.R <rv> [kind] [dataset]`), writing to the separate
  `_output_locality/<dataset>/<analysis_kind>/<response_var>/<model>/`
  root (`analysis_kind` in `overlap_degree`/`cond-<condition>`/`state-<NREM|Wake>`/
  `interaction-<set>`). Run `off-analysis export-locality-offs` then
  `renv::install('.')` (refreshes `extdata`) first.
- Summary: `python summarize_results.py config/summary_tables/locality_summary.yaml`
  for the main-effect (Overlapping-vs-Local) table; request 1 and the interaction
  results have distinct JSON shapes; read their `summary.txt`/`results.json`.
- Plots: `python plot_locality_results.py config/plots/locality.yaml` (reuses
  `plot_results.py`'s machinery unchanged; SVGs -> `_output_locality/figures/`).
  Two plot types from one config: (1) request 1: top-level `entries` draw the
  `mean_overlap_degree` NREM + wake violin panels (between-condition bars); (2)
  request 2: the `local_vs_overlapping` section draws the notebook `4b` figure:
  per condition, dodged Local/Overlapping violins with a within-condition bracket
  whose p-value is the pooled interaction model's `interaction$simple` (point at
  the chosen model, e.g. `locality_interaction_weighted_log`). Conditions for (2)
  come from `condition_windows.six`; Local/Overlapping colors match
  `locality_palette()`.
