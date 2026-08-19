# cnpix-local-sleep

Code and data behind *Local sleep during wake: A Neuropixels view*.

## Layout

| Path | What |
| --- | --- |
| `src/cnpix_local_sleep/` | Python package: preprocessing, detection, postprocessing, aggregation, data export (for R), and more |
| `r-offp/` | The `offp` R package: statistical models, manuscript tables, figure components, and other plots |
| `notebooks/figures/` | Python figure components by numbered manuscript item (see [index](notebooks/README.md)), and other plots |
| `scripts/` | Entry points not exposed through the CLIs (see below) |
| `tests/` | Test suite. `RUN_NFS_TESTS=1` additionally runs production-data tests (i.e. these tests are for internal use only and won't run outside the WISC data storage environment) |

Three CLIs / console scripts are installed: `off-analysis` (the `export-*` pipelines),
`morphological-offs` (detection), and `unit-based-offs`.

## Data

`r-offp/inst/extdata/` carries the 24 summarized tables (1.31 MB) every manuscript
table is built from.

The three event-level OFF tables (`full48h_{llas,clas,blas}_offs.parquet`, 347 MB
together) are not committed to this repo since they exceed GitHub's 100 MB per-file limit,
and only the Supplementary Figure S2 / edge-adjusted path reads them.
They are published as Release assets and fetched on demand into a local cache.
See [`docs/DATA.md`](docs/DATA.md).

## Reproducing the tables

```bash
cd r-offp
R -q -e "renv::restore()" && R -q -e "renv::install('.')"
bash make_manuscript_tables.sh --rerun-R --rerun-diagnostics
```

The bare `make_manuscript_tables.sh` only reads existing `.rds` files, so on a fresh
clone it has nothing to read: `--rerun-R` refits the analyses from `inst/extdata`, and
`--rerun-diagnostics` produces the two correlation summaries that Tables S4a and S4b
read. Without the latter the build still succeeds and says so, but the supplement comes
out missing S4.

## A note on terminology

The manuscript refers to "Small", "Medium", and "Large" OFFs.
These are derived from set operations on what the code refers to as
"LLAS", "CLAS", and "BLAS" OFFs, short for "liberal", "conservative", and "broad"
"low amplitude segments (LAS)", a nod to the concept introduced in:

> Harding et al., Detection of neuronal OFF periods as low amplitude neural activity segments. BMC Neurosci. 2023

Though we ultimately did not use that method, a Python port of it, including an extension
and generalization to the spatial domain for use with Neuropixels, can be found [here](https://github.com/CSC-UW/harding).

## License

The code is MIT licensed; see [`LICENSE`](LICENSE). The data tables under
`r-offp/inst/extdata/` are CC BY-NC-ND 4.0; see [`r-offp/LICENSE`](r-offp/LICENSE).
