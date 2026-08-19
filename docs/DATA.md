---
title: Data (what is committed, what is hosted, how to get it)
updated: 2026-08-19
---

# Data

`r-offp/inst/extdata/` holds every table the manuscript's models and figures are
built from. It is split in two, and the line is whether the table build needs the
file offline, not file size, though the two coincide here.

| | Files | Size | Where |
| --- | ---: | ---: | --- |
| Summarized tables (`summarized_*`, `nod_rebound_correlation_*`, `full48h_condition_durations`, `manuscript_*.csv`) | 24 | 1.31 MB | Committed |
| Event-level OFF tables (`full48h_{llas,clas,blas}_offs.parquet`) | 3 | 347 MB | Release assets |

Every reported model reads only the committed half, so a bare clone plus
`renv::install('.')` rebuilds the whole supplement with no network and no
credentials. The event-level tables are hosted because one of them
(`full48h_llas_offs.parquet`, 213 MB) exceeds GitHub's 100 MB per-file limit, and
because only one path reads them: the size-adjusted edge statistics behind
Supplementary Figure S2.

The Release carries both halves. The committed 24 are uploaded too, so one asset
set is a complete, self-contained copy of the data, for archiving, or for a later
move to Zenodo.

## Getting the event-level tables

Nothing to do on a machine that has run `off-analysis export-full48h-offs`: a copy in
`inst/extdata/` always wins.

Otherwise they are fetched on demand and cached. The only reader is
`edge_synchrony_validation.load_events`, so `plot_s2_replacement.py`, the S2 notebook
and `off-analysis export-adjusted-edge-statistics` all get them without asking:

```python
from cnpix_local_sleep import release_data
release_data.get_event_table_path("llas")   # -> cached path, downloading if needed
```

- Cache location: `$CNPIX_LOCAL_SLEEP_CACHE`, else the platform cache directory
  (`~/.cache/cnpix_local_sleep/release_data/<tag>/`). Never `inst/extdata`; an
  installed package directory is often read-only and is wiped by the next
  `renv::install('.')`.
- Transport: `gh release download` when `gh` is on PATH, else plain HTTPS. Neither
  needs authentication.
- A re-uploaded asset invalidates the cache: the fetcher compares the asset's
  `updatedAt` against the cached file's mtime. If that check cannot be made (offline,
  no `gh`), the cached copy is used rather than the call failing.

No R code reads these tables. All seven R extdata call sites read summarized
tables only, so `offp` needs no download client and no `piggyback` dependency.

## Publishing

```bash
off-analysis publish-release-data --dry-run   # list what would go up
off-analysis publish-release-data
```

Reads whatever the exporters already wrote and computes nothing, so re-uploading
never means recomputing 347 MB. Creates the release if absent and replaces assets in
place (`--clobber`). `--event-level-only` skips the 24 committed tables.

## Tags

One rolling tag, `latest`. A re-export replaces the assets in place and the tag moves
to the current commit, which is what should happen while numbers are still moving.

The cost is that a clone does not pin its own data: whoever fetches gets whatever
`latest` holds that day. Pin it at acceptance by cutting a fixed tag and repointing
`release_data.TAG`. The cache is keyed by tag, so repointing re-fetches rather than
reusing what `latest` held.

A later move to Zenodo is not a flag flip: Zenodo's GitHub integration archives a
release's auto-generated *source* archive, not its binary assets, so the three files
would have to be uploaded there.

## One committed file is an input, not a product

`manuscript_size_globality_correlations.csv` (Table S3b) cannot be regenerated from
this repository alone: `off-analysis export-size-globality-correlations` re-runs the
windowed-shift null against mounted production storage. It ships as an input. Every
other committed table is reproducible from the exporters given that same storage.
