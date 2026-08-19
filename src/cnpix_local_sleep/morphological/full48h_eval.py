"""Evaluate full-48h morphological OFF detections against manual ground truth.

Rasterizes the full-recording morphological parquet (``full48h_{filter}_offs.parquet``,
events as seconds × µm) onto the image-stack grid and scores it against the manual
labels with the same pixel/event metrics used for the SAM3 model, using the
per-condition image-selection convention (NREM: labeled chunks only; Wake: all
chunks).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs
from cnpix_local_sleep.evaluation import config, grid, labels, metrics, rasterize


def _load_full48h(filter_name: str, condition: str) -> pd.DataFrame:
    """Condition-subset full-48h OFFs for one filter, filtered to one condition.

    Derived in memory (no persisted ``full48h_*`` parquet) via
    :func:`aggregate_experiment_offs.load_subset_of_48h_offs`, then narrowed to
    the requested condition.
    """
    df = aggregate_experiment_offs.load_subset_of_48h_offs(filter_name)
    return df[df["condition"] == condition].reset_index(drop=True)


def evaluate_all_full48h(
    eval_name: str,
    filters: tuple[str, ...] | None = None,
    *,
    manual_version: str = "latest",
) -> pd.DataFrame:
    """Score morphological full-48h OFFs against manual labels, all pairs and filters.

    Loads each filter's parquet once (condition-filtered) and reuses it across all
    (subject, probe) pairs. Pairs that fail to load are skipped with a warning.
    """
    cfg = config.EVAL_CONFIGS[eval_name]
    condition = cfg["condition"]
    filters = filters or cfg["filters"]

    offs_by_filter = {f: _load_full48h(f, condition) for f in filters}
    pairs = labels._get_subject_probe_pairs_with_labels(condition)

    rows: list[dict] = []
    skipped: list[str] = []
    for i, (subject, probe) in enumerate(pairs, 1):
        print(
            f"  [full48h_eval:{eval_name}] ({i}/{len(pairs)}) {subject}/{probe}", flush=True
        )
        # Per-pair shared loads (manual labels, grid geometry) reused across filters.
        try:
            manual_raw = labels.load_manual_labels(
                subject, probe, condition, version=manual_version
            )
            manual, _v = labels.qc_and_fix_labels(manual_raw)
            ts_flat = grid.load_stack_times_flat(subject, probe, condition)
            y_coords = grid.channel_depths(subject, probe)
        except Exception as exc:  # noqa: BLE001 - report and continue
            msg = f"{subject}/{probe}: {exc}"
            warnings.warn(msg, stacklevel=2)
            skipped.append(msg)
            continue

        chunks = labels.select_chunks(manual, cfg["chunks"])
        row_mask = np.ones(manual.shape[1], dtype=bool)
        n_labeled = int(np.any(manual > 0, axis=(1, 2)).sum())

        for filter_name in filters:
            sp_offs = offs_by_filter[filter_name]
            sp_offs = sp_offs[
                (sp_offs["subject"] == subject) & (sp_offs["probe"] == probe)
            ]
            try:
                raster = rasterize.rasterize_offs(
                    sp_offs, ts_flat, y_coords, manual.shape, eval_chunks=chunks
                )
                px = metrics.compute_pixel_metrics(manual, raster, chunks, row_mask)
                ev = metrics.summarize_event_ious(
                    metrics.compute_event_metrics(
                        manual, raster, chunks, row_mask, with_iou=False
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                msg = f"{subject}/{probe}/{filter_name}: {exc}"
                warnings.warn(msg, stacklevel=2)
                skipped.append(msg)
                continue

            rows.append(
                {
                    "eval": eval_name,
                    "condition": condition,
                    "source": "morphological",
                    "filter_name": filter_name,
                    "model": None,
                    "subject": subject,
                    "probe": probe,
                    "chunks_mode": cfg["chunks"],
                    "n_chunks_evaluated": int(len(chunks)),
                    "n_labeled_chunks": n_labeled,
                    "n_off_events": int(len(sp_offs)),
                    **px,
                    **ev,
                }
            )

    if skipped:
        print(f"\n[full48h_eval:{eval_name}] {len(skipped)} item(s) skipped:")
        for msg in skipped:
            print(f"  {msg}")

    return pd.DataFrame(rows)
