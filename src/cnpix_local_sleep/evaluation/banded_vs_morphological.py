"""Cross-method scoring: banded unit-based OFFs against morphological OFFs.

Both detectors are rasterized onto the same depth×time grid over an evaluation's
chunks and scored spatially against each other, no 1-D collapse. The
morphological raster uses its true per-pixel masks
(:func:`rasterize_morphological_masks`), never bounding boxes.

This is the only place either method's output is scored against the *other's*
rather than against manual labels, which is why it sits here rather than in
:mod:`cnpix_local_sleep.morphological` or :mod:`cnpix_local_sleep.unit_based`. The
banded-specific rasterization and the shared evaluation geometry are reused from
:mod:`cnpix_local_sleep.unit_based.banded_eval`.
"""

from __future__ import annotations

import numpy as np

from cnpix_local_sleep.evaluation import grid, metrics
from cnpix_local_sleep.unit_based import banded_eval


def rasterize_morphological_masks(
    subject, probe, structure, condition, label_shape, *,
    filter_name="clas", off_source="full48h",
):
    """Rasterize morphological spatial OFFs onto the stack grid using their TRUE per-pixel
    masks (not bounding boxes).

    Loads the OFFs + ``off_label_indices`` (``time_ixs``/``chan_ixs``) for the requested
    ``off_source`` (``"full48h"`` default, or ``"per_condition"``) via the shared loader
    :func:`cnpix_local_sleep.morphological.manual_validation._load_translated_mua_offs` (which applies the
    LAS ``filter_name``, remaps the MUA detection sample indices to stack-grid samples,
    and (for full-48h) restricts to the condition), then paints each OFF's exact pixels
    via :func:`cnpix_local_sleep.morphological.manual_validation._build_morphological_label_array`. Returns an
    int32 ``(n_chunks, n_rows, spc)`` array (0 = background).
    """
    from cnpix_local_sleep.morphological import manual_validation as mv
    from cnpix_local_sleep.morphological import mua as morphological_mua
    from cnpix_local_sleep import trace_io

    ts_flat = grid.load_stack_times_flat(subject, probe, condition)
    offs = mv._load_translated_mua_offs(
        subject, probe, structure, condition, filter_name,
        off_source=off_source, source_config=morphological_mua.SOURCE_CONFIG,
        stack_times_flat=ts_flat,
    )
    if not len(offs):
        return np.zeros(label_shape, dtype=np.int32)

    da_full = trace_io.open_preprocessed_traces_as_xarray(
        subject, probe, structure=None, condition=condition,
        apply_detection_channel_mask=False,
    )
    da_det = trace_io.open_preprocessed_traces_as_xarray(
        subject, probe, structure=structure, condition=condition,
        apply_detection_channel_mask=True,
    )
    return mv._build_morphological_label_array(
        offs,
        det_channels=da_det.channel.values,
        full_channels=da_full.channel.values,
        n_full_channels=da_full.sizes["channel"],
        label_shape=label_shape,
    )



def evaluate_banded_vs_morphological(
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
    filter_name: str = "clas",
    mua_raster=None,
    off_source: str = "full48h",
    restrict_to_structure: bool = True,
    row_mask: np.ndarray | None = None,
    manual_version: str = "latest",
) -> dict:
    """Spatial-to-spatial agreement of banded OFFs with morphological OFFs.

    Rasterizes both detectors onto the same depth×time grid (over the eval chunks) and
    scores banded (predicted) against morphological (reference). The morphological raster uses
    its true per-pixel masks (:func:`rasterize_morphological_masks`), not bounding boxes;
    pass a precomputed ``mua_raster`` to avoid recomputing it (it is structure-, not
    config-, dependent). ``off_source`` selects the morphological OFF source
    (``"full48h"``/``"per_condition"``). ``footprint`` selects the banded representation
    (``"union"`` of band boxes, default, or ``"bbox"``). Pass an explicit ``row_mask`` to
    score over an arbitrary row set.
    """
    cfg, condition, manual, ts_flat, y_coords, chunks = banded_eval.eval_geometry(
        subject, probe, eval_name, manual_version=manual_version
    )
    row_mask = banded_eval.resolve_row_mask(
        y_coords, manual, depth_lo, depth_hi, restrict_to_structure, row_mask
    )

    if mua_raster is None:
        mua_raster = rasterize_morphological_masks(
            subject, probe, structure, condition, manual.shape,
            filter_name=filter_name, off_source=off_source,
        )
    banded_raster, used = banded_eval.banded_raster(
        off_frame, off_df, all_bands_on_off_df, footprint,
        ts_flat, y_coords, manual.shape, chunks,
    )
    # Reference = morphological (true masks), predicted = banded unit-based.
    px = metrics.compute_pixel_metrics(mua_raster, banded_raster, chunks, row_mask)
    ev = metrics.summarize_event_ious(
        metrics.compute_event_metrics(
            mua_raster, banded_raster, chunks, row_mask, with_iou=False
        )
    )
    n_mua = int(len(np.unique(mua_raster[chunks])) - 1)  # events in the scored chunks
    return {
        "eval": eval_name,
        "condition": condition,
        "source": "banded-unit-based",
        "reference": f"morphological:{filter_name}",
        "mua_off_source": off_source,
        "footprint": used,
        "subject": subject,
        "probe": probe,
        "structure": structure,
        "chunks_mode": cfg["chunks"],
        "restricted_to_structure": restrict_to_structure,
        "n_chunks_evaluated": int(len(chunks)),
        "n_banded_events": int(len(off_frame)),
        "n_mua_events": n_mua,
        **px,
        **ev,
    }
