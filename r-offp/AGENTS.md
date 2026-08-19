# AGENTS.md: r-offp (`offp`)

Do not assume Visual Studio Code is the IDE, even if you are a VSIX extension "for
VS Code"; this may be running alongside Positron.

## What this is

An R package (`offp`) for statistical analysis and visualization of OFF-period
detection data in sleep/wake neuroscience research. Companion to the Python
[`cnpix_local_sleep`](https://github.com/CSC-UW/cnpix-local-sleep) package,
which produces everything in `inst/extdata/`.
Both are developed by the same developer, so cross-package changes
are acceptable. You can request a symlink to the `cnpix_local_sleep` source if that is more
token-efficient than exploring via GitHub.

## Environment Setup

R and system-level dependencies are managed via `pixi`; R package dependencies are
declared in `DESCRIPTION` and managed by `renv` (initialized: `renv.lock` and
`.Rprofile` are committed).

```bash
pixi global install --environment my-r-env r-base radian r-renv
```

## Build & Test Commands

```bash
# Install the local package (see Known Issues; use renv, not pak/devtools::install)
R --quiet --no-restore --no-save -e "renv::install('.')"

# Regenerate documentation (Roxygen2 -> man/ and NAMESPACE)
R --quiet --no-restore --no-save -e "devtools::document()"

# Test / check
R --quiet --no-restore --no-save -e "devtools::test()"
R --quiet --no-restore --no-save -e "devtools::check()"

# Single analysis: run_analysis.R <rv> [model] [dataset] [condition_set]
Rscript scripts/run_analysis.R rate               # all models, LLAS
Rscript scripts/run_analysis.R rate crossed       # one named model
Rscript scripts/run_analysis.R rate all clas      # specific dataset
Rscript scripts/run_analysis.R rate all llas wake # specific condition set

# All analyses
Rscript scripts/run_all_analyses.R                # every dataset
Rscript scripts/run_all_analyses.R llas           # one dataset

# Summary tables
python summarize_results.py config/summary_tables/summary.yaml
```

### Figure assembly

```bash
# Single-page 6x4 grid SVG (rows = measures, cols = LLAS/CLAS x Wake/NREM) from the
# same config/plots/cx_homeostasis.yaml entries. Each cell's model is resolved
# unambiguously from the entries (errors on any ambiguity); bare figure (no
# ticks/labels/titles) for Figma annotation. Plots adjusted data (random effects
# removed) by default. --boxplot honored.
python plot_results_grid.py config/plots/cx_homeostasis.yaml         # adjusted (RE removed)
python plot_results_grid.py --raw config/plots/cx_homeostasis.yaml   # raw values

# Individual bare panels sized to ONE grid cell (nrem -> 360x180 pt, wake -> 180x180
# pt), so a standalone violin lines up beside cx_homeostasis_grid.svg in Figma. Same
# figma style / chrome-stripping as the grid; layout-independent (iterates any plot
# YAML's entries).
python plot_results_cell.py config/plots/bandpower_homeostasis.yaml
python plot_results_cell.py --response-var mean_zlog_delta config/plots/bandpower_homeostasis.yaml
python plot_results_cell.py --raw config/plots/cx_homeostasis.yaml

# Assembled draft of the replacement Supplementary Figure S2 (size-adjusted ON/OFF
# transition synchrony): mechanism panels from the cnpix_local_sleep validation notebook's cached
# tables, result panels drawn through plot_results' own violin functions so the draft
# and the Figma panels cannot drift apart. Writes into
# docs/ms/figures_draft/. Needs the adj_mean_*_mad analyses and one execution
# of the validation notebook. Also emits each mechanism panel (b-e) as a standalone
# SVG under S2_panels/, drawn by the same panel functions as the assembly; panel f's
# violins come from plot_results_cell.py instead.
python plot_s2_replacement.py                  # assembled draft + panels + comparison
python plot_s2_replacement.py --no-comparison --no-panels
```

### Manuscript tables

SLEEP-ready editable `.docx` + `.xlsx` for the "Neuropixels view of local sleep"
paper. Config-driven (`config/summary_tables/manuscript_*.yaml`); reports ONLY All
OFFs (`llas`) / Medium+Large (`clas`) / Small (`llas_exclusive`); never BLAS.
Cells give <=2 sig-fig, adjusted two-sided p with stars, the contrast estimate with its
95% CI, and the Cohen's-d analogue. Table types: `homeostasis` / `correlation_slope` /
`correlation_robustness` / `depth_profile` / `per_event_meta` /
`epoched_partial` / `locality` / `size_globality` / `model_reference`. Outputs ->
`_output_manuscript/`.

```bash
python manuscript_tables.py config/summary_tables/manuscript_s1a_homeostasis.yaml   # one table
python manuscript_tables.py --supplement config/summary_tables/manuscript_supplement.yaml  # combined S0-S5
./make_manuscript_tables.sh                       # all tables (+ the S1b companion) from current outputs
./make_manuscript_tables.sh --rerun-diagnostics   # + refit S4a transforms & S4b leave-out battery (slow)
./make_manuscript_tables.sh --all                 # + re-run R analyses + refresh cnpix_local_sleep correlation CSVs

# Table S1b, the companion DATA workbook for Table S1a: the actual per-(subject,
# structure, condition) values behind the S1a contrasts (OFF count, OFF rate, Total
# OFFness, duration, span, area, residual MUA, ON/OFF-transition synchrony, delta
# power), one sheet per property, blocked by All/Medium+Large/Small. Reads the
# summarized_full48h_*_offs parquets + the S1a config (so it tracks S1a). SEPARATE
# file -> _output_manuscript/manuscript_s1b_values.{xlsx,csv}.
python manuscript_s1b_values.py
```

## Analysis Axes

Two orthogonal axes select what is fit: `off_type` (LAS filter) and
`condition_set` (which conditions enter the fit).

### OFF types (`OFF_TYPES`)

`llas`, `clas`, `blas`, plus `llas_exclusive` (`llas & ~clas`) on some pipelines.
`blas` is not reported in the manuscript but stays wired up, since review may ask
to see it.

### OFF detection

There is one OFF source: `summarized_full48h_{off_type}_offs.parquet`: full-48h
morphological OFFs, detected whole-recording with state-aware thresholds and subset
to the six statistical conditions. The historical per-condition detection (one
pass per scored condition window) is retired; nothing manuscript-facing read it,
and the `off_source` parameter it selected with is gone from the R, Python, YAML
and `_output` path layers.

The `summarized_full48h_*` tables are committed, so the analyses run from a bare
clone. Only the three event-level `full48h_{llas,clas,blas}_offs.parquet` are not (347 MB;
GitHub Release assets, and no R code reads them -- see `../docs/DATA.md`). To regenerate
either, from cnpix_local_sleep:

```bash
cd gfys_workspace && uv run off-analysis export-full48h-offs   # -> ../cnpix-local-sleep/r-offp/inst/extdata
cd gfys_workspace && uv run off-analysis publish-release-data   # re-host the event-level tables
```

### Condition sets (`condition_set`)

Conditions and post-hoc contrasts are bundled into named sets in a top-level
`condition_sets:` block in `config/cx_homeostasis.yaml`.

| `condition_set` | conditions | post-hoc contrasts |
| ------ | ------ | ------ |
| `six` | 4 sleep + 2 wake | 3 NREM + the wake (Late-Early NOD) contrast |
| `nrem` | the 4 sleep conditions | 3 NREM contrasts |
| `wake` | the 2 wake conditions | the wake (Late-Early NOD) contrast |

Each response variable lists `analyses` (a list of `{condition_set, model}` pairs)
instead of a bare `models` list. A measure can be fit under several sets (each an
independent fit with its own variance components). Default wiring preserves the
historical selection: LLAS variables use `six`, CLAS/BLAS use `nrem`. Results nest as
`_output/<dataset>/<condition_set>/<response_var>/<model>/`, and the
Python `plot_results.py` / `summarize_results.py` entries are
`[response_variable, dataset, condition_set, model]` 4-tuples. The plotter draws the
NREM panel for sets covering NREM conditions and the wake panel for sets covering wake
conditions (`six` draws both).

## Companion Pipelines

Each is fully separable and additive: deleting its files and re-running
`devtools::document()` removes it without touching the core analyses. Details in
`dev_docs/pipelines/`:

| Pipeline | Question | Output root | Doc |
| --- | --- | --- | --- |
| NOD-rebound correlation | does a whole-period NOD metric predict its own `NREM.Rebound`? | `_output_correlations/` | [pipeline_nod_rebound_correlation.md](dev_docs/pipelines/pipeline_nod_rebound_correlation.md) |
| Locality | Local vs Overlapping (cross-structure OFF overlap) | `_output_locality/` | [pipeline_locality.md](dev_docs/pipelines/pipeline_locality.md) |
| Excess globality | is observed overlap above the windowed-shift null? | `_output_excess_globality/` | [pipeline_excess_globality.md](dev_docs/pipelines/pipeline_excess_globality.md) |
| Band-power homeostasis | per-condition SWA through the identical homeostasis model | `_output_bandpower/` | [pipeline_bandpower_homeostasis.md](dev_docs/pipelines/pipeline_bandpower_homeostasis.md) |
| Size-adjusted edge synchrony | ON/OFF transition synchrony with the estimator's mechanical dependence on OFF size removed | `_output/` (`adj_mean_*_mad`) | [pipeline_edge_adjusted.md](dev_docs/pipelines/pipeline_edge_adjusted.md) |

Each requires its extdata exported from cnpix_local_sleep first (see the individual doc), then
`renv::install('.')` to refresh `inst/extdata`.

## Package Architecture

### Data Flow

```
Python cnpix_local_sleep -> inst/extdata/summarized_full48h_{llas,clas,blas}_offs.parquet
       v
  load_offs_summary(off_type)  [R/data_loading.R]
       v
  Filter: clade="Cx", layer="None", detection_mode="spatial"
  Filter: conditions (from the selected condition_set: six/nrem/wake)
       v
  fit_models(fe_terms, re_terms)  [R/model_fitting.R]
       v
  test_main_effect()  [R/statistical_tests.R]
       v
  subtract_ranef_get_fsquared()  [R/effect_sizes.R]
       v
  Plots + JSON/RDS/text outputs  [R/analysis.R, R/plotting.R]
```

### Function call graph

```
run_cx_homeostasis_analysis()           [R/analysis.R]
  |-- load_offs_summary()               [R/data_loading.R]
  |-- build_condition_comparisons()     [R/contrasts.R]
  |-- fit_models()                      [R/model_fitting.R]
  |-- get_condition_palette()           [R/colors.R]
  |-- plot_rvf() / plot_qqline() / plot_distributions_by_condition()  [R/plotting.R]
  |-- generate_figure_diagnostics()     [R/analysis.R]
  |-- test_main_effect()                [R/statistical_tests.R]
  |   |-- subtract_ranef_get_fsquared() [R/effect_sizes.R]
  |   |   |-- subtract_random_effects() / strip_random_effects() / cohens_local_fsquared()
  |   `-- cohens_d_analogue()           [R/effect_sizes.R]
  |-- build_json_summary() / build_text_summary()  [R/analysis.R]
  `-- format_posthoc_summary()          [R/summaries.R]
```

### R Source Files

| File | Purpose |
| --- | --- |
| `R/analysis.R` | High-level pipeline: `run_cx_homeostasis_analysis()` and result serialization |
| `R/data_loading.R` | `load_offs_summary()`: loads parquet from `inst/extdata/` |
| `R/model_fitting.R` | `fit_models()`: full/null models for given FE/RE terms; `validate_model_def()` |
| `R/effect_sizes.R` | Cohen's f² via random effects subtraction (Selya et al. 2012) |
| `R/statistical_tests.R` | LRT-based main effect test with optional post-hoc contrasts |
| `R/plotting.R` | ggplot2 diagnostic and distribution plots |
| `R/colors.R` | `get_condition_palette()`: RColorBrewer-based condition colors |
| `R/contrasts.R` | `build_condition_comparisons()`: builds `multcomp::mcp` from config posthoc strings |
| `R/summaries.R` | `format_posthoc_summary()`: text formatting for post-hoc results |
| `R/offp-package.R` | Package-level documentation |
| `summarize_results.py` | Reads a summary YAML config, produces markdown + CSV tables from `results.json` files |

Computation is separated from presentation: `scripts/run_analysis.R` and
`scripts/run_all_analyses.R` produce results in `_output/`; `summarize_results.py`
and `manuscript_tables.py` read those pre-computed results.

### Output Structure

```
_output/llas/six/rate/
  |-- crossed_weighted/
  |   |-- results.rds       # Full R result list (models, data, tests)
  |   |-- results.json      # Machine-readable: p-values, effect sizes, significance
  |   |-- summary.txt       # Human-readable text summary
  |   |-- variance_components.csv
  |   `-- figures/          # *_residuals/_qq/_violins/_violins_adjusted .png + .rds
  |       `-- diagnostics.txt      # Shapiro-Wilk, group medians/IQRs
  `-- crossed_interaction_weighted/
      `-- (same structure)
```

The `results.json` posthoc section contains `contrasts`, `pvalues` (adjusted),
`estimates`, `cohens_d`, `ci_lower`, `ci_upper`: parallel arrays in the same order as
the posthoc strings in `cx_homeostasis.yaml`.

## Data Schema

All summarized parquets share one schema:

| Column | Type | Description |
| --- | --- | --- |
| `subject` | character | Animal identifier (random factor) |
| `condition` | character | Experimental condition |
| `structure` | character | Brain structure (random factor) |
| `clade` | character | Cell type classification ("Cx", etc.) |
| `layer` | character | Cortical layer ("None" for no layer distinction) |
| `detection_mode` | character | OFF detection algorithm ("spatial", etc.) |
| `median_duration`, `rate`, ... | numeric | Response variables (16 total) |

Six conditions, sleep: `Early.BSL.NREM`, `Early.REC.NREM.Match`,
`Early.REC.NREM`, `Late.REC.NREM`; wake: `Early.NOD.Wake`, `Late.NOD.Wake`. Which
enter a fit is set by the condition set, not by `off_type`.

The cx_homeostasis analyses filter to `clade == "Cx"`, `layer == "None"`,
`detection_mode == "spatial"`, then to the condition set's conditions, and validate one
structure entry per subject-condition combination.

## Statistical Methodology

Models are fit with configurable random effects structures (defined per response
variable in YAML), then full vs null compared by likelihood ratio test.

1. Model definitions (all fit with ML, not REML): `fe_terms` (always
   `"condition"`), `re_terms` (e.g. `c("(1 | subject)", "(1 | structure)")`),
   `weighted` (boolean), and an optional `transform` (one of
   `names(RESPONSE_TRANSFORMS)`: `identity` (default), `log`, `log10`, `log1p`,
   `sqrt`). The transform is applied to the response column in place before
   weighting and fitting, so weights, random-effect subtraction, effect sizes, and all
   diagnostic plots operate on one consistent scale (a `log(y)` *formula* term would
   break the by-name random-effect subtraction in `effect_sizes.R`). Because output
   dirs key on the model `name`, a transform must be carried by a distinct model name
   (e.g. `crossed_interaction_sqrt`), exactly like `weighted`. The null model is
   derived automatically by removing `fe_terms`. Common configurations use YAML
   anchors; custom ones can be specified inline per response variable.
2. Testing: LRT via `anova()` comparing full vs null.
3. Effect sizes (Selya et al. 2012), the core novel methodology: subtract
   estimated random intercepts from raw data, refit as fixed-effects-only `lm()` on
   adjusted data, compute Cohen's f² = (R²_full - R²_null) / (1 - R²_full). This
   enforces identical random structure across models.
4. Post-hoc contrasts: `multcomp::glht()` with a contrast matrix built by
   `build_condition_comparisons()`. Effect sizes as a Cohen's d analogue
   (mean / sqrt(sum of variance components)).

## Code Style

- Use package namespaces for non-base functions (`dplyr::filter()`, `multcomp::glht()`)
- `snake_case` for variables and functions
- Roxygen2 with markdown enabled

## Known Issues

- `renv` symlinks library packages to a shared cache. Tools like `pak` and
  `devtools::install()` may fail trying to overwrite these symlinks. Use
  `renv::install(".")` to install the local package.
