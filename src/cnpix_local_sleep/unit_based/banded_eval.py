"""Score banded unit-based OFFs against manual labels and morphological OFFs. PROVISIONAL.

Reuses the method-agnostic :mod:`cnpix_local_sleep.evaluation` kernels (the same ones that score
morphological and SAM3): rasterize an OFF frame onto the image-stack grid, then compute
pixel/event metrics against a reference label array. Two references are supported:

- manual labels (ground truth): :func:`evaluate_banded_vs_manual`
- morphological spatial OFFs (same-modality cross-check): :func:`evaluate_banded_vs_morphological`,
  rasterizing both detectors onto the same grid (spatial-to-spatial, no 1-D collapse).

Banded detection is per-structure, but the manual labels / grid are whole-probe, so by
default scoring is restricted to the structure's channel rows via :func:`structure_row_mask`.
"""

from __future__ import annotations

import numpy as np

import pandas as pd

from cnpix_local_sleep.evaluation import config, grid, labels, metrics, rasterize


def _consecutive_groups(idx: np.ndarray):
    """Yield runs of consecutive integers from a sorted array."""
    if not len(idx):
        return
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for grp in np.split(idx, splits):
        yield grp


def labeled_chunk_bouts(
    subject: str,
    probe: str,
    eval_name: str,
    *,
    state: str = "NREM",
    manual_version: str = "latest",
) -> pd.DataFrame:
    """Bouts (start/end/duration/state) covering the evaluation's stack chunks.

    Detecting banded OFFs only over these windows bounds compute to the scoring
    domain (the labeled chunks for NREM; all chunks for Wake) instead of the full
    48 h. Consecutive chunks are merged into single bouts to limit boundary
    fragmentation. The returned frame is suitable as a ``bouts_by_pass`` value for
    :func:`cnpix_local_sleep.unit_based.banded.detect_structure_banded`.
    """
    cfg = config.EVAL_CONFIGS[eval_name]
    condition = cfg["condition"]
    manual_raw = labels.load_manual_labels(subject, probe, condition, version=manual_version)
    manual, _ = labels.qc_and_fix_labels(manual_raw)
    ts_flat = grid.load_stack_times_flat(subject, probe, condition)
    n_chunks = manual.shape[0]
    spc = len(ts_flat) // n_chunks
    chunks = np.sort(labels.select_chunks(manual, cfg["chunks"]))

    rows = []
    for grp in _consecutive_groups(chunks):
        c0, c1 = int(grp[0]), int(grp[-1])
        t0 = float(ts_flat[c0 * spc])
        t1 = float(ts_flat[c1 * spc + spc - 1])
        rows.append(
            {"start_time": t0, "end_time": t1, "duration": t1 - t0, "state": state}
        )
    return pd.DataFrame(rows)


def structure_row_mask(y_coords: np.ndarray, depth_lo: float, depth_hi: float) -> np.ndarray:
    """Boolean ``(n_rows,)`` mask over stack rows for channels in ``[depth_lo, depth_hi]``.

    Stack rows are the flipped channel axis (``stack_row = (n_full-1) - channel_index``;
    see :mod:`cnpix_local_sleep.evaluation.rasterize`), so this returns the mask in the same row
    space as the manual labels / rasters.
    """
    n_full = len(y_coords)
    chan_idx = np.where((y_coords >= depth_lo) & (y_coords <= depth_hi))[0]
    mask = np.zeros(n_full, dtype=bool)
    mask[(n_full - 1) - chan_idx] = True
    return mask


def _paint_boxes_into(out, boxes, label, ts_flat, y_coords, eval_chunks=None):
    """Paint ``[start,end]×[lo,hi]`` boxes with a single ``label`` into ``out`` in place.

    Mirrors :func:`cnpix_local_sleep.evaluation.rasterize.rasterize_offs`'s inner painting (same
    searchsorted time mapping + flipped channel rows), but assigns *one* label to a
    *set* of boxes, so a merged OFF's constituent band rectangles paint as one event.
    """
    n_chunks, n_rows, spc = out.shape
    n_full = len(y_coords)
    n_flat = n_chunks * spc
    keep_chunk = None
    if eval_chunks is not None:
        keep_chunk = np.zeros(n_chunks, dtype=bool)
        keep_chunk[eval_chunks] = True
    for s, e, lo, hi in boxes:
        lo_i = int(np.searchsorted(ts_flat, s, side="left"))
        hi_i = min(int(np.searchsorted(ts_flat, e, side="right")), n_flat)
        if hi_i <= lo_i:
            continue
        flat = np.arange(lo_i, hi_i)
        chunk_ids = flat // spc
        within = flat % spc
        if keep_chunk is not None:
            m = keep_chunk[chunk_ids]
            if not m.any():
                continue
            chunk_ids, within = chunk_ids[m], within[m]
        chan_idx = np.where((y_coords >= lo) & (y_coords <= hi))[0]
        if chan_idx.size == 0:
            continue
        rows = (n_full - 1) - chan_idx
        nt, nr = chunk_ids.size, rows.size
        out[np.repeat(chunk_ids, nr), np.tile(rows, nt), np.repeat(within, nr)] = label


def rasterize_banded_union(
    off_df, all_bands_on_off_df, ts_flat, y_coords, label_shape, *, eval_chunks=None
):
    """Rasterize each merged OFF as the UNION of its constituent band boxes.

    Faithful depth×time footprint (vs the bounding-box rasterization of the mapped Off
    frame): merged OFF ``k`` paints its member band rectangles (looked up via
    ``merged_band_offs_indices``) with label ``k+1``: correct pixel *and* event metrics.
    """
    out = np.zeros(label_shape, dtype=np.int32)
    if not len(off_df):
        return out
    cols = ["start_time", "end_time", "lo", "hi"]
    for k, idxs in enumerate(off_df["merged_band_offs_indices"], start=1):
        boxes = all_bands_on_off_df.loc[idxs, cols].to_numpy(dtype=float)
        _paint_boxes_into(out, boxes, k, ts_flat, y_coords, eval_chunks=eval_chunks)
    return out


def banded_raster(
    off_frame, off_df, all_bands_on_off_df, footprint, ts_flat, y_coords, shape, chunks
):
    """Predicted raster for a banded OFF result: ``union`` footprint or ``bbox``."""
    if footprint == "union" and off_df is not None and all_bands_on_off_df is not None:
        return rasterize_banded_union(
            off_df, all_bands_on_off_df, ts_flat, y_coords, shape, eval_chunks=chunks
        ), "union"
    if footprint == "union":
        print("  footprint='union' needs off_df + all_bands_on_off_df; using bbox.")
    return rasterize.rasterize_offs(
        off_frame, ts_flat, y_coords, shape, eval_chunks=chunks
    ), "bbox"


def eval_geometry(subject, probe, eval_name, *, manual_version="latest"):
    """Shared per-(subject, probe, eval) geometry: manual labels, grid, eval chunks."""
    cfg = config.EVAL_CONFIGS[eval_name]
    condition = cfg["condition"]
    manual_raw = labels.load_manual_labels(subject, probe, condition, version=manual_version)
    manual, _violations = labels.qc_and_fix_labels(manual_raw)
    ts_flat = grid.load_stack_times_flat(subject, probe, condition)
    y_coords = grid.channel_depths(subject, probe)
    chunks = labels.select_chunks(manual, cfg["chunks"])
    return cfg, condition, manual, ts_flat, y_coords, chunks


def resolve_row_mask(
    y_coords, manual, depth_lo, depth_hi, restrict_to_structure, row_mask=None
):
    if row_mask is not None:
        return np.asarray(row_mask, dtype=bool)
    if restrict_to_structure:
        if depth_lo is None or depth_hi is None:
            raise ValueError("depth_lo/depth_hi required when restrict_to_structure=True")
        return structure_row_mask(y_coords, depth_lo, depth_hi)
    return np.ones(manual.shape[1], dtype=bool)


def evaluate_banded_vs_manual(
    subject: str,
    probe: str,
    structure: str,
    off_frame,
    eval_name: str,
    *,
    depth_lo: float | None = None,
    depth_hi: float | None = None,
    off_df=None,
    all_bands_on_off_df=None,
    footprint: str = "union",
    restrict_to_structure: bool = True,
    row_mask: np.ndarray | None = None,
    manual_version: str = "latest",
) -> dict:
    """Score a banded OFF result against manual labels (mirrors ``full48h_eval``).

    ``off_frame`` is the :class:`cnpix_local_sleep.off_tables.Off` frame from
    :func:`cnpix_local_sleep.unit_based.banded.detect_structure_banded`. ``footprint="union"``
    (default) rasterizes the true union-of-band-boxes footprint and needs ``off_df`` +
    ``all_bands_on_off_df`` (from ``return_artifacts=True``); ``"bbox"`` rasterizes the
    bounding box of ``off_frame``. ``eval_name`` is ``"NREM"`` or ``"Wake"``.

    Pass an explicit ``row_mask`` (boolean, ``(n_rows,)``) to score over an arbitrary
    row set, e.g. the morphological detection channels for a common-scope head-to-head,
    overriding ``depth_lo/depth_hi`` and ``restrict_to_structure``.
    """
    cfg, condition, manual, ts_flat, y_coords, chunks = eval_geometry(
        subject, probe, eval_name, manual_version=manual_version
    )
    row_mask = resolve_row_mask(
        y_coords, manual, depth_lo, depth_hi, restrict_to_structure, row_mask
    )

    raster, used = banded_raster(
        off_frame, off_df, all_bands_on_off_df, footprint,
        ts_flat, y_coords, manual.shape, chunks,
    )
    px = metrics.compute_pixel_metrics(manual, raster, chunks, row_mask)
    ev = metrics.summarize_event_ious(
        metrics.compute_event_metrics(manual, raster, chunks, row_mask, with_iou=False)
    )
    return {
        "eval": eval_name,
        "condition": condition,
        "source": "banded-unit-based",
        "reference": "manual",
        "footprint": used,
        "subject": subject,
        "probe": probe,
        "structure": structure,
        "chunks_mode": cfg["chunks"],
        "restricted_to_structure": restrict_to_structure,
        "n_chunks_evaluated": int(len(chunks)),
        "n_off_events": int(len(off_frame)),
        **px,
        **ev,
    }
