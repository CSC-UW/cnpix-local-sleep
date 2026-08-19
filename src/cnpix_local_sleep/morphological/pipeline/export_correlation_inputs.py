"""Export NOD-vs-NREM.Rebound correlation-input parquets for r-offp.

An additive, fully separable companion to :mod:`aggregate_experiment_offs`.
It builds the per-(subject, probe, structure, condition) summarized metrics for
exactly the conditions the r-offp correlation analysis needs:

- ``NOD``: the whole-period NOD predictor (Wake+NREM epochs; x-axis),
- ``NOD.Wake``: the wake-only variant of that predictor (the same NOD window
  restricted to Wake epochs, dropping NREM intrusions/microsleeps), and
- ``Early.REC.NREM`` / ``Early.REC.NREM.Match``: whose difference is the
  ``NREM.Rebound`` response (y-axis), computed on the r-offp side.

Both ``NOD`` and ``NOD.Wake`` are exported so the r-offp side can fit the
correlation against either predictor without re-exporting; each is a distinct
overlapping condition window (keys of
``hyp.load_statistical_condition_hypnograms``), so an OFF in the wake portion of
NOD is emitted once per covering window.

For each category it writes a flat ``nod_rebound_correlation_{cat}_offs.parquet``
into the destination directory (default: ``r-offp/inst/extdata``).
Categories are the three LAS filters plus the ``llas_exclusive`` = ``llas & ~clas``
adjacent-partition complement (the per-condition analogue of the committed
``summarized_full48h_llas_exclusive_offs.parquet``). Never writes to NFS.

Delete this module and the ``export-nod-rebound-correlation`` CLI command (in
``analysis_cli.py``) to remove the feature entirely; no existing artifact is
touched.
"""

from __future__ import annotations

import pathlib

from cnpix_local_sleep.morphological.pipeline.aggregate_experiment_offs import (
    summarize_48h_offs_for_conditions,
)

#: Conditions exported for the correlation analysis. ``NOD`` (Wake+NREM) and its
#: wake-only variant ``NOD.Wake`` are the two selectable whole-period predictors;
#: the two REC conditions form the ``NREM.Rebound`` difference
#: (``Early.REC.NREM - Early.REC.NREM.Match``) on the r-offp side. These are keys
#: of ``hyp.load_statistical_condition_hypnograms`` (``NOD`` / ``NOD.Wake`` overlap
#: each other and the REC windows, which ``summarize_48h_offs_for_conditions``
#: handles by emitting one row per covering condition).
CORRELATION_CONDITIONS = [
    "NOD",
    "NOD.Wake",
    "Early.REC.NREM",
    "Early.REC.NREM.Match",
]

#: Correlation categories: the three LAS filters plus the ``llas_exclusive``
#: (``llas & ~clas``) adjacent-partition complement.
CORRELATION_CATEGORIES = ("llas", "clas", "blas", "llas_exclusive")


def export_nod_rebound_correlation(
    output_dir: pathlib.Path | str,
    *,
    categories: tuple[str, ...] | None = None,
    grouped_boxcox: bool = False,
) -> None:
    """Write ``nod_rebound_correlation_{cat}_offs.parquet`` files.

    For each category, summarize the full-48h morphological OFFs over
    :data:`CORRELATION_CONDITIONS` (via
    :func:`aggregate_experiment_offs.summarize_48h_offs_for_conditions`, which
    tags each OFF by every covering condition window and reuses the canonical
    filter / Box-Cox / summary helpers) and write a flat parquet into
    ``output_dir``. Requires NFS (reads the raw whole-recording detection).

    Parameters
    ----------
    output_dir : pathlib.Path | str
        Destination directory for the flat parquet files (created if needed).
    categories : tuple[str, ...] | None
        Categories to export; defaults to :data:`CORRELATION_CATEGORIES`. Pass a
        subset (e.g. ``("llas_exclusive",)``) to write only those parquets without
        touching the others.
    grouped_boxcox : bool
        If True, also fit per-(subject, probe, structure) Box-Cox lambdas.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories or CORRELATION_CATEGORIES:
        print(
            f"Summarizing {cat} OFFs for correlation conditions "
            f"{CORRELATION_CONDITIONS}..."
        )
        summ = summarize_48h_offs_for_conditions(
            CORRELATION_CONDITIONS, cat, grouped_boxcox=grouped_boxcox
        )
        out_path = output_dir / f"nod_rebound_correlation_{cat}_offs.parquet"
        print(f"Saving {out_path.name} ({len(summ)} rows)...")
        summ.to_parquet(out_path)
