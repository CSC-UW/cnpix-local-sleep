# Notebooks

## `figures/`

Notebook paths are relative to `notebooks/figures/`; other paths are relative to the
repository root.

| Manuscript item | Code |
| --- | --- |
| Fig 1b, 1c, LFP/MUA rows of 1e, 3-D view in 1f | `fig1_signal.ipynb` |
| Fig 2b (OFF counts, E->L) | `incline_magnitudes.ipynb` |
| Fig 2c,d (sleep intrusions) | `intrusion_analysis.ipynb` |
| Fig 2e (intrusion re-admission sweep) | `intrusion_sweep.ipynb` |
| Fig 2f (delta power) | `r-offp/plot_results_cell.py` with `config/plots/bandpower_homeostasis.yaml` |
| Fig 2g,h (OFF properties, OFF rate by size class) | `r-offp/plot_results_cell.py` with `config/plots/cx_homeostasis.yaml` |
| Fig 3 (all panels) | `scripts/figure3.py` (below), from `has_value.ipynb` (a-d) and `added_value/incremental_added_value.ipynb`, `added_value/incremental_added_value_wake.ipynb`, `added_value/sequential_added_value.ipynb` (e-h) |
| Fig 4a (overlap fractions) | `cross_structure_off_relationships.ipynb` |
| Fig 4b (overlap degree) | `r-offp/plot_locality_results.py` with `config/plots/locality.yaml` (shared-y adjusted panel) |
| Fig 4c-g (overlap and OFF properties) | `group_cross_structure_offs.ipynb` |
| Fig 5 (SD-rebound correlation) | `r-offp/scripts/plot_correlation_xtransform_publication.R`, then `.py` |
| Table 1 (OFF counts) | `r-offp/manuscript_table1_values.py` |
| Supp Fig 1, 2, 3, 6, 7 | `scripts/supp_figure1.py`, `supp_figure2.py`, `supp_figure3.py`, `supp_figure6.py`, `supp_figure7.py` (below) |
| Supp Fig 4 + Table S5 (depth profile) | `laminar_trimodality_null.ipynb`; panel b reads the output of `r-offp/scripts/depth_profile_summary.R` |
| Supp Fig 5b (OFF-property grid) | `r-offp/plot_results_grid.py` with `config/plots/cx_homeostasis.yaml` |
| Supp Fig 5c (edge asynchrony) | `r-offp/plot_results_cell.py --response-var adj_mean_onset_mad` (and `adj_mean_offset_mad`) with `config/plots/cx_homeostasis.yaml` |
| Supplementary Methods, size-adjusted edge statistics | `edge_synchrony_validation/edge_synchrony_validation.ipynb` |
| Supp Table S6a (detector vs manual, early recovery NREM) | `batch_manual_vs_banded_and_morphological.ipynb` |
| Supp Tables S6b, S6c (baseline NREM, sleep deprivation, two annotations of the last hour) and the spike check of small OFFs | `manual_annotation_sets.ipynb` (reads the Table S6a results) |
| Supplementary Tables | `r-offp/make_manuscript_tables.sh` (see the top-level README) |

Two further notebooks in `added_value/` support Fig. 3:

| Notebook | Notes |
| --- | --- |
| `added_value/static_added_value.ipynb` | `sequential_added_value.ipynb` reuses its cached OFF table (`outputs/static_added_value/cache/offs_direct_48h.parquet`), so the Fig 3e-h chain does not run without it. |
| `added_value/added_value_figures.ipynb` | Shared figure assembly for the same tier of results. |

The assembled Fig. 3 (all panels, SVG + PNG at 7.09 x 8.86 in) is drawn by
`scripts/figure3.py` into `figures/outputs/figure3/`. It reads the per-event cache that
`has_value.ipynb` writes and the `added_value/outputs/added_value_data/` parquets, so both
must exist; no NFS access is needed.

Supplementary Fig. 1 (morphological OFF detection, step by step) is drawn by
`scripts/supp_figure1.py` into `figures/outputs/supp_figure1/`. It reads one MUA segment,
its channel thresholds and the hypnograms from NFS; there is no cache.

Supplementary Fig. 2 (detection and anatomical borders on two example probes) is drawn
by `scripts/supp_figure2.py` into `figures/outputs/supp_figure2/`. It reads 10 min of
each probe's MUA traces, its structure and detection-border tables and the hypnograms
from NFS (under a minute, 11 GB peak); there is no cache.

The threshold drift numbers in the Supplementary Methods (block-wise threshold
estimation) are printed by `scripts/methods/threshold_drift.py`, which writes into
`figures/outputs/threshold_drift/`. It reads every pair's MUA trace from NFS (about 2 h
with 4 workers, up to 65 GB per worker) and caches one parquet per pair in `per_block/`;
`--summary-only` re-summarises that cache without NFS.

The numbers on manually annotated wake OFF periods in the Supplementary Methods
(agreement between detectors and manual annotation) are printed by two scripts that
share `figures/outputs/annotated_wake_offs/`. `scripts/methods/annotated_wake_offs.py`
prints the counts and the correlations with delta power. It reads the labels, MUA
traces, hypnograms and delta power of 11 probes from NFS (about 20 min with 4 workers)
and writes `offs.parquet` and `rates.parquet`; `--summary-only` reprints the
correlations from `offs.parquet` without NFS. `scripts/methods/annotated_wake_offs.R`
fits the mixed models from those two files and is run from `r-offp/`.

Supplementary Fig. 3 (wake OFF rate on the sleep deprivation and baseline days) is drawn
by `scripts/supp_figure3.py` into `figures/outputs/supp_figure3/`. Its per-pair table
reads OFF tables and hypnograms from NFS and is cached there as
`baseline_wake_control.parquet`; `--plot-only` redraws from that file.

Supplementary Fig. 6 (overlapping vs. local OFF size at matched duration) is drawn by
`scripts/supp_figure6.py` into `figures/outputs/supp_figure6/`, with the plotted tables
as parquet beside it. It reads the event-level LLAS OFF table through
`cnpix_local_sleep.release_data` and needs no NFS.

Supplementary Fig. 7 (fixed-threshold occupancy) is drawn by `scripts/supp_figure7.py`
into `figures/outputs/supp_figure7/`. Its per-pair table reads MUA traces from NFS and is
cached there as `fixed_threshold_occupancy.parquet`; `--plot-only` redraws from that file.

## Outputs and caches

Each notebook writes beside itself, into `outputs/` relative to the notebook's own
directory.

Outputs read by downstream code:

- `figures/added_value/outputs/sequential_added_value/*/identity_checks.csv`

Committed as result records, read by nothing:

- `figures/outputs/full48h/comparison_summary.csv`
- `figures/outputs/full48h/intrusion_sweep_table.csv`

Both are written by `morphological.mua.pipeline.full48h`, whose `DEFAULT_OUTPUT_DIR` and
`INTRUSION_OUTPUT_DIR` point at `figures/outputs/{full48h,intrusion_sweep}/`.

Two caches are expensive to rebuild (both require mounted NFS and a long pass over the
full-48h detection) and are kept out of git:

| Cache | Size | Rebuilt by |
| --- | ---: | --- |
| `figures/outputs/bandpower_vs_off/full48h_morphological_offs_with_bandpower.parquet` | 1.7 GB | `has_value.ipynb` |
| `figures/added_value/outputs/static_added_value/cache/offs_direct_48h.parquet` | 1.3 GB | `static_added_value.ipynb` |

`laminar_trimodality_null.ipynb` caches instead to
`~/.cache/offproj_laminar_trimodality_null` (the directory name predates the carve and
is kept so existing caches stay valid).

## Running them

Most require mounted production storage. From the WISC internal workspace:

```bash
cd gfys_workspace
uv run --all-extras --group dev jupyter lab
```

`edge_synchrony_validation.ipynb` is the exception; it reads committed and
Release-hosted tables only, and needs no NFS.
