# Notebooks

## `figures/`

| Manuscript item | Notebook |
| --- | --- |
| Fig 1b-d (LFP/MUA panels + OFF outlines) | `fig1_signal.ipynb` |
| Fig 2b (OFF counts, E->L) | `incline_magnitudes.ipynb` |
| Fig 2c,d (sleep intrusions) | `intrusion_analysis.ipynb` |
| Fig 2e (intrusion re-admission sweep) | `intrusion_sweep.ipynb` |
| Fig 3a-d + Table S2a (per-event ρ) | `has_value.ipynb` |
| Fig 3e-h + Table S2b (epoched partial) | `added_value/incremental_added_value.ipynb`, `added_value/incremental_added_value_wake.ipynb`, `added_value/sequential_added_value.ipynb` |
| Fig 4a,b (overlap fractions) | `cross_structure_off_relationships.ipynb`, `group_cross_structure_offs.ipynb` |
| Supp Fig 1 + S5 (depth profile) | `laminar_trimodality_null.ipynb` |
| Supp Fig 2 + S1a (edge synchrony) | `edge_synchrony_validation/edge_synchrony_validation.ipynb` |
| Table 1 (detector vs manual) | `batch_manual_vs_banded_and_morphological.ipynb` |

Two further notebooks in `added_value/` support Fig. 3:

| Notebook | Notes |
| --- | --- |
| `added_value/static_added_value.ipynb` | `sequential_added_value.ipynb` reuses its cached OFF table (`outputs/static_added_value/cache/offs_direct_48h.parquet`), so the Fig 3e-h chain does not run without it. |
| `added_value/added_value_figures.ipynb` | Shared figure assembly for the same tier of results. |

## Outputs and caches

Each notebook writes beside itself, into `outputs/` relative to the notebook's own
directory.

Outputs read by downstream code:

- `figures/edge_synchrony_validation/outputs/*.csv`: read by
  `r-offp/plot_s2_replacement.py`, which requires `mechanical_surface_fit.csv`,
  `mechanical_surface_grid.csv` and `duration_invariance_simulated.csv` to exist before
  it will run.
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
