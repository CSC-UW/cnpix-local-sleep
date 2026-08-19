---
title: Size-adjusted ON/OFF transition synchrony companion pipeline
updated: 2026-08-19
---

# Size-adjusted ON/OFF transition synchrony

An additive companion that adds two response variables,
`adj_mean_onset_mad` and `adj_mean_offset_mad`, carrying the same construct as the
published `mean_{onset,offset}_mad` but with the mechanical dependence on OFF-period
size removed. They are fitted by the *existing* `run_cx_homeostasis_analysis` core with
no change to the model, the LRT, the `glht` post-hocs, the effect sizes, the plotting
or the summary tables: they are ordinary columns on the summarized frame.

Deleting the companion parquets and the `join_adjusted_edge_statistics` helper removes
the pipeline; every other response variable is untouched.

## Why the columns exist

`onset_mad`/`offset_mad` are the median absolute deviation of an OFF period's
per-channel edge times, computed across the channels the event spans, so the number of
channels spanned is also the *sample size* of the statistic. The detector's four-channel
spatial structuring element guarantees at least four channels enter an event
simultaneously at its leading edge, which forces `MAD == 0` for any event spanning
120 µm or less whatever its true edge dispersion, and the statistic keeps climbing with
span well above that. A forward simulation with a fixed, size-independent latent edge
dispersion reproduces the entire observed MAD-vs-span curve, so the dependence belongs
to the estimator rather than to the events. OFF size itself varies with condition, so a
condition contrast in the raw cell mean is partly a contrast in OFF size.

## Data

Exported by cnpix_local_sleep's `off-analysis export-adjusted-edge-statistics` (no NFS, ~45 s)
from `full48h_llas_offs.parquet`, which is a GitHub Release asset rather than a
committed file and is fetched to a cache if it is not already in `inst/extdata`
(`cnpix-local-sleep/docs/DATA.md`):

- `inst/extdata/summarized_full48h_{llas,clas,llas_exclusive}_edge_adjusted.parquet`:
  one row per `(subject, probe, structure, condition)` with `adj_mean_onset_mad`,
  `adj_mean_offset_mad` (milliseconds, unlike the published `mean_*_mad` columns,
  which are seconds), the unadjusted `mean_*_mad` for comparison, and per-row
  bookkeeping: `{edge}_size_coding`, `{edge}_size_lambda` (the fitted amplitude of the
  shared size curve, ms), `{edge}_positivity` (share of the prediction population that
  is interpolation rather than extrapolation), `{edge}_interaction`, `n_events`.

`load_offs_summary()` left-joins these through `join_adjusted_edge_statistics()`
(`R/data_loading.R`), which is a no-op when the file is absent.
`plot_results.py` mirrors it on the Python side.

Every cell of the summarized frame gets an adjusted value; there is no minimum event
count, so the adjusted panels cover exactly the cells the published panels do.

## Model and wiring

`adj_mean_{onset,offset}_mad ~ condition + (1|subject) + (1|structure) +
(1|subject:structure)` (`crossed_interaction`), under the `nrem` and `wake` condition
sets, for the `clas` (Medium+Large) dataset only. The `llas` and `llas_exclusive`
entries in `config/cx_homeostasis.yaml`, `config/plots/cx_homeostasis.yaml` and
`config/summary_tables/manuscript_s1a_homeostasis.yaml` are commented out with the reason
rather than deleted: in the Small class a third of events are hard-floored at `MAD == 0`,
and All OFFs is a mixture of the two whose composition moves with condition, the very
thing being adjusted for.

## Run

```bash
cd gfys_workspace && uv run off-analysis export-adjusted-edge-statistics
cd ../r-offp && R -q -e "renv::install('.')"       # extdata is copied into the library
for cs in nrem wake; do
  for rv in adj_mean_onset_mad adj_mean_offset_mad; do
    Rscript scripts/run_analysis.R $rv crossed_interaction clas $cs
  done
done
python plot_results.py config/plots/cx_homeostasis.yaml                       # violins
python plot_results_cell.py --response-var adj_mean_onset_mad config/plots/cx_homeostasis.yaml
python manuscript_tables.py config/summary_tables/manuscript_s1a_homeostasis.yaml
```

Results land in `_output/clas/<condition_set>/adj_mean_*_mad/crossed_interaction/`,
alongside every other response variable.

## The assembled figure draft

```bash
python plot_s2_replacement.py     # -> docs/ms/figures_draft/S2_replacement_draft.{png,pdf}
                                  #    + S2_panels/S2_panel_{b,c,d,e}.svg
                                  #    + S2_published_vs_adjusted.png (review only)
```

Six panels: the MAD floor, the mechanical apportionment, the shared size-adjustment curve
with each pair's own profile behind it, the duration control (the covariate deliberately
*not* adjusted for), and the four result violins. The mechanism panels read the validation
notebook's cached simulation tables (execute it once first); the result panels are drawn
through `plot_results.plot_{nrem,wake}_violin`, the same functions that emit the
standalone Figma SVGs, so the draft cannot drift away from the panels that ship.

### Per-panel SVGs for Figma

`build_panel()` re-draws one mechanism panel on its own canvas through the *same* panel
function the assembly calls, so a panel imported alone cannot drift from the same panel
in the draft. Panel a is a placeholder for the reused protocol schematic and is not
emitted; panel f's four violins come from `plot_results_cell.py`, sized to grid cells:

```
_output/figures/clas/{nrem,wake}/adj_mean_{onset,offset}_mad/crossed_interaction/{nrem,wake}_adjusted_violin.svg
```

Panels are saved with `bbox_inches="tight"`, so panel d's canvas is wider than panel c's:
its title is longer than a half-width axes. Delete the titles in Figma if the columns
need to line up.

## Gotchas

- Reinstall after every re-export. `renv::install('.')` copies `inst/extdata` into
  the R library; without it the fits silently read the previous export.
- Units. `adj_*` columns are milliseconds; the published `mean_*_mad` columns are
  seconds. Contrast estimates and CIs scale with them; Cohen's d and f² do not.
- The wake fits are singular: all three random-effect variances collapse to zero,
  so the model degenerates to OLS over the wake cells. This is a property of the wake
  panel, not of the adjustment: the published `mean_*_mad` wake fits are singular the
  same way. Only 16 of 26 wake combos contribute both conditions.
