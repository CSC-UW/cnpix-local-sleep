# cnpix_local_sleep.morphological - morphological OFF period detection

Implementation of the quantile-threshold OFF period detection method developed by Tom
Bugnon at WISC, the manuscript's *morphological* detector. The current default
workflow detects OFF periods by applying manually set quantile-based thresholds on MUA
traces, followed by image-morphology cleaning and connected-component labeling.

## Pipeline overview

`detect-offs-full`: load the whole-recording MUA traces, take the per-structure NREM and
Wake quantile thresholds from `mua/data/quantile_thresholds.csv`, threshold to a binary
mask with state-dependent thresholds, clean it with image-morphology operations, label
connected components, extract per-OFF properties. Writes `offs.parquet` and
`off_label_indices.parquet` under `method=morphological/probe=*/structure=*/`.

Thresholds are stored per (time bin, channel), each bin matching the trace zarr's
native dask chunks (~5.5 min).

Input is 500 Hz MUA traces built upstream by `cnpix.mua` (`cnpix-mua
write-mua-traces`), with Gaussian pre-smoothing applied at load time to match the
original Bugnon spectral profile. The legacy Tom-Bugnon variant (300 Hz preprocessed
AP traces) and its preprocessing commands were deleted on 2026-08-11.

## Modules

| Module | Description |
| --- | --- |
| `detect` | Shared kernels: per-bin quantile thresholds, laminar (supra/infra) areas |
| `detect_full` | Whole-recording (48 h) detection |
| `morphology` | The detection kernel: `clean_binary_mask`, `get_off_properties`, `detect_offs` |
| `common` | `MorphologicalSourceConfig`: binds a variant's files module, trace reader and thresholds |
| `agg` | Contrast/summary aggregation over detected OFFs |
| `types` | `DetectionOpts`, `NdImageFilterKwargs`, `validate_detection_opts` |
| `detection_opts/` | YAML detection-parameter templates |
| `manual_validation` | Per-structure scoring against manual labels (true per-pixel masks) |
| `full48h_eval` | Experiment-wide scoring of the full-48h OFFs against manual labels |
| `correlation_stats` | OFF-property vs bandpower statistics |
| `edge_synchrony_validation`, `laminar_null` | Supplementary-figure analyses |
| `cli`, `analysis_cli` | `morphological-offs` and `off-analysis` entry points |
| `mua/` | The morphological variant: `files`, `readers`, `interactive` (tuner backend), `SOURCE_CONFIG` |
| `pipeline/` | Postprocessing kernels, full-48h aggregation, bandpower and the `export-*` steps |

## CLI usage

Available as `morphological-offs` after `uv sync`:

```bash
morphological-offs detect-offs-full CNPIX15-Claude imec0 V1 --options-path opts.yml
```

Post-detection steps (the `export-*` subcommands, `cross-structure-offs`, bandpowers)
live under `off-analysis`. The interactive threshold tuner's cache is built by
`offproj-dev precompute-interactive-thresholds`.

## Key parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| Gaussian smoothing | 20 Hz cutoff (~8 ms sigma) | Single scale, applied at load |
| Threshold quantiles | Per-structure, per-state | `nrem_quantile_threshold` / `wake_quantile_threshold` in `mua/data/quantile_thresholds.csv`; hand-set with the tuner, not computed |
| Image-morphology closing | `n_channels_connect: 3` | `mua/data/spatial_detection_opts.yml` |

## File paths

Detection artifacts are stored under a `method=` segment on disk:

- `method=morphological/probe=*/structure=*/` for the whole-recording outputs; path helpers
  `get_full_offs_path`, `get_full_off_label_indices_path` and
  `get_full_channel_thresholds_path` in `cnpix_local_sleep.morphological.mua.files`

`method=tom-bugnon/` still exists on disk, but only as the annotation grid: the
`processed_ap.zarr` the napari stacks and every manual OFF label were drawn on. Its
path helper is `cnpix_local_sleep.files.get_preprocessed_ap_path` and its reader is
`cnpix_local_sleep.trace_io.open_preprocessed_traces_as_xarray`. Nothing detects into
that tree; the tom-bugnon *detection* variant was deleted on 2026-08-11.

## Cross-method scoring

Scoring these OFFs against *another detector's* output rather than against the manual
labels lives in `cnpix_local_sleep.evaluation`
(`head_to_head`, `banded_vs_morphological`).
