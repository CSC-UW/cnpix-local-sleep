"""Two annotations of the same wake images, compared with each other and with each detector.

The last-hour annotation (evaluation ``Wake``, every image inspected) and the
sleep-deprivation annotation (evaluation ``SD``) both cover part of the last hour of sleep
deprivation. The ``SD`` labels are copied onto the ``Wake`` stack grid by timestamp, and the
scored images are the ``Wake`` images whose every sample the ``SD`` grid contains and that end
no later than the last image the ``SD`` annotator labeled, so both annotators inspected all of
them; every such image is scored, labeled or not. Each annotation is in turn the reference for
the other annotation and for each detector, over the morphological detection rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cnpix_local_sleep import const
from cnpix_local_sleep.evaluation import banded_vs_morphological, grid, head_to_head
from cnpix_local_sleep.unit_based import banded_eval

ANNOTATIONS = ("Wake", "SD")


def move_to_grid(
    labels_src: np.ndarray, ts_src: np.ndarray, ts_dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Copy labels from one ``(n_chunks, samples_per_chunk)`` stack grid to another.

    Returns the labels on ``ts_dst`` (0 where a sample is not on ``ts_src``) and, per
    ``ts_dst`` image, whether every one of its samples is on ``ts_src``.
    """
    src_f, dst_f = ts_src.reshape(-1), ts_dst.reshape(-1)
    ix = np.clip(np.searchsorted(src_f, dst_f), 0, len(src_f) - 1)
    exact = np.abs(src_f[ix] - dst_f) < 1e-6
    _, n_rows, spc = labels_src.shape
    flat = labels_src.transpose(0, 2, 1).reshape(-1, n_rows)
    out = np.zeros((dst_f.size, n_rows), np.int32)
    out[exact] = flat[ix[exact]]
    m = ts_dst.shape[0]
    return out.reshape(m, spc, n_rows).transpose(0, 2, 1), exact.reshape(m, spc).all(1)


def overlap_structures() -> list[tuple[str, str, str]]:
    """Structures scored against both annotations."""
    both = set(head_to_head.labeled_cortical_structures("SD"))
    return [k for k in head_to_head.labeled_cortical_structures("Wake") if k in both]


def overlap_structure(
    subject: str,
    probe: str,
    structure: str,
    *,
    mua_filters=("llas", "clas"),
    post_ms: int = 80,
    banded_passes: dict | None = None,
) -> pd.DataFrame:
    """Score both annotations against each other and every detector on the shared images.

    One row per (reference annotation, prediction): the other annotation
    (``method="annotation"``), morphological per filter and each banded pass, the banded
    detector fitted and run on the scored images (:func:`head_to_head.banded_rasters`).
    Empty when the annotations share no image.
    """
    _, condition, wake, ts_flat, y_coords, _ = banded_eval.eval_geometry(subject, probe, "Wake")
    _, _, sd, sd_ts_flat, _, _ = banded_eval.eval_geometry(subject, probe, "SD")
    ts = ts_flat.reshape(wake.shape[0], -1)
    sd_ts = sd_ts_flat.reshape(sd.shape[0], -1)
    last = int(np.flatnonzero(sd.reshape(len(sd), -1).any(1))[-1])
    sd_end = sd_ts[last, -1]
    sd, covered = move_to_grid(sd, sd_ts, ts)
    chunks = np.flatnonzero(covered & (ts[:, -1] <= sd_end))
    if not len(chunks):
        return pd.DataFrame()

    row_mask, _ = head_to_head._mua_detection_row_masks(
        subject, probe, condition, wake.shape[1], structure
    )
    preds = {
        ("morphological", f): banded_vs_morphological.rasterize_morphological_masks(
            subject, probe, structure, condition, wake.shape, filter_name=f
        )
        for f in mua_filters
    }
    bouts = banded_eval.fit_bouts(subject, probe, "Wake", ts, chunks)
    rasters = head_to_head.banded_rasters(
        subject, probe, structure, bouts,
        banded_passes or dict(head_to_head.DEFAULT_BANDED_PASSES), (post_ms,),
        ts_flat, y_coords, wake.shape, chunks,
    )
    for (label, _), (raster, _, _) in rasters.items():
        preds[("banded", label)] = raster

    refs = dict(zip(ANNOTATIONS, (wake, sd)))
    meta = {
        "subject": subject, "probe": probe, "structure": structure, "condition": condition,
        "n_chunks": len(chunks),
        "minutes": len(chunks) * ts.shape[1] * grid.sample_period(ts_flat) / 60.0,
    }
    rows = []
    for ref_name, ref in refs.items():
        other = next(a for a in ANNOTATIONS if a != ref_name)
        for (method, label), pred in {("annotation", other): refs[other], **preds}.items():
            rows.append({
                **meta, "reference": ref_name, "method": method, "label": label,
                **head_to_head._score_raster(ref, pred, chunks, row_mask),
            })
    return pd.DataFrame(rows)


def results_path() -> Path:
    """Where ``manual_annotation_sets.ipynb`` saves the :func:`overlap_structure` rows on NFS."""
    import wisc_ecephys_tools as wet

    return (
        wet.get_sglx_project("offproj").get_experiment_directory(const.EXPERIMENT)
        / "manual_annotation_overlap_full48h.parquet"
    )
