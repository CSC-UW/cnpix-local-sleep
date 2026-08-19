# AGENTS.md: cnpix-local-sleep

## What this is

The manuscript repository for *Local sleep during wake: A Neuropixels view*: OFF-period
detection, the export pipelines that build the summary tables, the evaluation kernels
behind Table 1, the figure notebooks, and the `offp` R package that fits every reported
model.

Two packages live here. The Python package is `src/cnpix_local_sleep/`. The `offp` R
package is nested at `r-offp/` and keeps its own `.gitignore`, `DESCRIPTION` and renv
project; run R commands from `r-offp/` so `.Rprofile` and renv autoload. See
[`r-offp/AGENTS.md`](r-offp/AGENTS.md).

The majority of the code is authored by humans -- agents should strive to match its style.

Scope rule. Anything needed to reproduce a published number belongs here. Everything
else (exploratory analysis, the two interactive OFF tuners, the Harding GMM code,
cell-type firing, OFF-locked PETHs, links to thalamic activity) belong in another repo `offproj`, which depends on this package.
The dependency runs dev -> manuscript and must never be inverted.

## Build & Test

If you are working on behalf of the author (Graham) on WISC infrastructure (tononi-1 or tononi-2), run everything through the workspace project so the local editable siblings are used:

```bash
cd gfys_workspace
uv run --all-extras --group dev pytest ../cnpix-local-sleep/tests
uv run --all-extras --group dev ruff check ../cnpix-local-sleep/src ../cnpix-local-sleep/tests
```

`RUN_NFS_TESTS=1` additionally runs the production-data tests (`@pytest.mark.requires_nfs`).
Most require `/Volumes/npx_nfs/` mounted.

Other users who clone this repository are not expected (and do not need) to have this workspace structure or venv, nor access to the NFS productiond data mounts.

## Console Scripts

| Script | Covers |
| --- | --- |
| `off-analysis` | Post-detection pipeline steps: postprocess, aggregate, plot, bandpower, the `export-*` subcommands, and `publish-release-data`. Not a detection method. |
| `morphological-offs` | `detect-offs` / `detect-offs-full` |
| `unit-based-offs` | `detect` / `detect-experiment` / `detect-banded` / `detect-banded-experiment` |

The dev-only commands (`offproj-dev`, `harding-offs`) live in `offproj`.

## Layout

| Path | What |
| --- | --- |
| `src/cnpix_local_sleep/` | Python package: see the module table below |
| `r-offp/` | The `offp` R package: statistical models, manuscript tables, plot scripts |
| `notebooks/figures/` | One notebook per numbered manuscript item; see `notebooks/README.md` |
| `tests/` | Test suite |
| `scripts/` | Standalone entry points not exposed through the CLIs |

Notebooks are committed without outputs.

### Python modules

| Module | Purpose |
| --- | --- |
| `const.py` | Experiment name, conditions, contrasts, frequency bands |
| `files.py` | `get_path()`: parametric path construction with `DEFAULT_PATH_SCHEMA`; `get_r_offp_extdata_dir()` |
| `channel_anatomy.py` | Which structure and layer each probe channel sits in (htsv tables, not the atlas) |
| `trace_io.py` | Open / scale / smooth the zarr recordings, including the annotation-grid reader |
| `off_tables.py` | The `Off` row schema, the per-condition OFF loader, and the LLAS/CLAS/BLAS filters (single point of truth) |
| `hyp.py` | Hypnogram utilities |
| `plots.py` | Plotting utilities, including the condition and LLAS/CLAS/BLAS palettes |
| `units.py` | Unit/spike train utilities |
| `atlas.py` | Brain atlas integration |
| `sps_conf.py` | Subject-probe-structure config (cross-method; bundled CSV in `data/`) |
| `release_data.py` | The GitHub Release hosting the event-level OFF tables: fetch, cache, publish |
| `morphological/` | The morphological method: `common.py` (`MorphologicalSourceConfig`), `morphology.py` (the detection kernel), `detect.py`, `detect_full.py`, `manual_validation.py`, `full48h_eval.py`, shared CLI/types, `mua/` subpackage |
| `morphological/pipeline/` | Postprocess, aggregate, and the `export-*` implementations |
| `unit_based/` | The spatially tiled unit-based method: `banded.py`, `banded_eval.py`, `banded_plots.py` |
| `evaluation/` | Manual OFF labels, grid geometry, rasterizer and metric kernels, plus the cross-method drivers behind Table 1: `head_to_head.py`, `banded_vs_morphological.py` |
| `stacks/` | OFF period stacking and the napari annotation grid |

## Detection Methods

Two methods,

| Method | `method=` on disk | Code | CLI |
| --- | --- | --- | --- |
| morphological | `method=morphological` | `morphological/` | `morphological-offs` |
| unit-based (spatially tiled) | `method=unit_based` | `unit_based/` | `unit-based-offs` |

`morphological` was renamed from `mua-bugnon`.
`tom-bugnon` is a third, retired detection variant.

Where a scorer lives follows the method it is specific to. A driver that scores
one method against the manual labels sits in that method's package
(`morphological/manual_validation.py`, `morphological/full48h_eval.py`,
`unit_based/banded_eval.py`); a driver that scores one detector against *another
detector's* output sits in `evaluation/` (`head_to_head.py`,
`banded_vs_morphological.py`), as do the method-agnostic kernels.

## Path System

All output files use `cnpix_local_sleep.files.get_path()`:

```python
from cnpix_local_sleep import files
path = files.get_path(
    "off_df.parquet",
    subject="CNPIX15-Claude", probe="imec0", structure="PPC",
    condition="Early.NOD.Wake",
)
```

Schema order: `project > experiment > subject > package > method > model > probe >
structure > layer > detection_mode > threshold_group > condition > filters`.

In addition to being an internal package name,
**`offproj` is also the name of a WNE *data project* on the WISC NFS**.  
`wet.get_sglx_project("offproj")`, `package=offproj` path segments, `offproj_s3`,
and `root.attrs["offproj_version"]` all refer to that data project.

## Conditions and Contrasts

Six experimental conditions:

- Sleep: `Early.BSL.NREM`, `Early.REC.NREM.Match`, `Early.REC.NREM`, `Late.REC.NREM`
- Wake: `Early.NOD.Wake`, `Late.NOD.Wake`

LLAS analyses use all 6; CLAS and BLAS use only the 4 sleep conditions.

| Contrast | Comparison |
| --- | --- |
| `NOD.Incline` | `Late.NOD.Wake` vs `Early.NOD.Wake` |
| `NREM.Rebound` | `Early.REC.NREM` vs `Early.REC.NREM.Match` |
| `NREM.Surge` | `Early.REC.NREM` vs `Early.BSL.NREM` |
| `NREM.REC.Decline` | `Early.REC.NREM` vs `Late.REC.NREM` |
| `NREM.BSL.Decline` | `Early.BSL.NREM` vs `Early.REC.NREM.Match` |

## Export Pipelines (-> r-offp)

Each `off-analysis export-*` subcommand is an additive, fully separable companion that
writes a tidy table into `r-offp/inst/extdata/`, never to NFS. That directory is
resolved once, by `files.get_r_offp_extdata_dir()`; do not re-derive it.

## Data

`r-offp/inst/extdata/` is split by whether the table build needs the file *offline*.
The 24 summarized tables are committed (1.31 MB); the three event-level OFF tables
(`full48h_{llas,clas,blas}_offs.parquet`, 347 MB together) are GitHub Release assets,
because one exceeds GitHub's 100 MB per-file limit and only Supplementary Figure S2 /
the edge-adjusted path reads them. `cnpix_local_sleep.release_data` holds the repo/tag
config, the cache-backed fetcher (a copy in `inst/extdata` always wins) and the
`off-analysis publish-release-data` implementation. No R code reads the event-level
tables, so `offp` needs no download client. Details, including the tag policy:
[`docs/DATA.md`](docs/DATA.md).

## See Also

- Root `AGENTS.md` (workspace): code style, data rules, record-keeping
- `notebooks/README.md`: manuscript item -> notebook index
- `r-offp/AGENTS.md`: the R statistical analysis
- `offproj/AGENTS.md`: the development repository that depends on this one
