"""Head-to-head: banded unit-based OFFs vs morphological OFFs, both scored against manual
labels on a common row set.  PROVISIONAL.

This is the synthesis engine behind the ``banded_and_morphological_vs_manual`` notebooks. Where
:mod:`cnpix_local_sleep.unit_based.banded_eval` scores banded OFFs over the *banded* structure extent
and :mod:`cnpix_local_sleep.morphological.manual_validation` scores morphological over its own detection/structure
channels, this module puts both detectors on the same footing: it rasterizes each onto the
stack grid and scores them against the manual labels over the morphological detection-channel
row set (``scope="detection"``): the depth band where morphological even attempts detection,
so their F1/IoU/sensitivity/precision are directly comparable (mirrors the M2 head-to-head
table).

Both detectors use their faithful footprints: morphological via true per-pixel masks
(never bounding boxes), banded via the union of its constituent band boxes. Banded
detection is re-run bounded to the labeled chunks (the validated path), once per structure,
then the post-merge union-duration floor is swept cheaply.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.evaluation import banded_vs_morphological, labels as ev_labels, metrics
from cnpix_local_sleep.unit_based import banded, banded_eval
from cnpix_local_sleep.unit_based import files as ub_files
from cnpix_local_sleep import trace_io

# Operating point: post-merge union-duration floor (ms) for the banded detector. 80 ms is
# the rollout operating point; pass a wider tuple (e.g. range(50, 101, 10)) to find the
# per-structure F1 peak.
DEFAULT_POST_MS: tuple[int, ...] = (80,)
DEFAULT_MUA_FILTERS: tuple[str, ...] = ("llas", "clas", "blas")
# Banded detector passes scored alongside morphological: ``{label: config_or_None}`` (a None
# config means ROLLOUT_CONFIG). The structure is loaded once and shared across passes; a
# pass that raises (e.g. greedy_fr on a low-FR structure) is warned and skipped.
DEFAULT_BANDED_PASSES: dict = {"banded-rollout": None}


def labeled_cortical_structures(eval_name: str = "NREM") -> list[tuple[str, str, str]]:
    """Cortical unit_based-included (subject, probe, structure) tuples with manual labels."""
    cond = banded_eval.config.EVAL_CONFIGS[eval_name]["condition"]
    pairset = set(ev_labels._get_subject_probe_pairs_with_labels(cond))
    spsl = sps_conf.get_subject_probe_structure_list(
        method=ub_files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    return [(s, p, st) for (s, p, st) in spsl if (s, p) in pairset]


def _mua_detection_row_masks(subject, probe, condition, n_label_rows, structure):
    """``(det_row_mask, struct_row_mask)`` for the morphological channels of ``structure``.

    Built from the tom traces the manual labels were annotated on (same construction as
    :func:`cnpix_local_sleep.morphological.manual_validation.compare_structure`): the *detection* mask is the
    depth band the morphological detector actually detects on, the *structure* mask the full structure extent.
    """
    da_full = trace_io.open_preprocessed_traces_as_xarray(
        subject, probe, structure=None, condition=condition,
        apply_detection_channel_mask=False,
    )
    da_struct = trace_io.open_preprocessed_traces_as_xarray(
        subject, probe, structure=structure, condition=condition,
        apply_detection_channel_mask=False,
    )
    da_det = trace_io.open_preprocessed_traces_as_xarray(
        subject, probe, structure=structure, condition=condition,
        apply_detection_channel_mask=True,
    )
    return ev_labels._build_channel_maps(da_full, da_struct, da_det, n_label_rows)


def _score_raster(manual, raster, chunks, row_mask):
    """Pixel + event metrics for a predicted ``raster`` vs ``manual`` over a row set."""
    px = metrics.compute_pixel_metrics(manual, raster, chunks, row_mask)
    ev = metrics.summarize_event_ious(
        metrics.compute_event_metrics(manual, raster, chunks, row_mask, with_iou=False)
    )
    n_off = int(len(np.unique(raster[chunks])) - 1)  # events present in the scored chunks
    return {**px, **ev, "n_off": n_off}


def head_to_head_structure(
    subject: str,
    probe: str,
    structure: str,
    *,
    eval_name: str = "NREM",
    mua_source: str = "full48h",
    mua_filters=DEFAULT_MUA_FILTERS,
    post_ms=DEFAULT_POST_MS,
    banded_passes: dict | None = None,
    scope: str = "detection",
    manual_version: str = "latest",
) -> pd.DataFrame:
    """Score banded and morphological OFFs vs manual on a common row set for one structure.

    Returns a tidy DataFrame: one row per morphological filter, and one row per
    (banded pass × ``post_ms``). All rows share the scored ``scope`` (``"detection"`` =
    morphological detection channels, the common comparison set; or ``"structure"``), the
    labeled NREM chunks, and the manual reference. ``banded_passes`` is
    ``{label: config_or_None}`` (None -> :data:`cnpix_local_sleep.unit_based.banded.ROLLOUT_CONFIG`),
    default :data:`DEFAULT_BANDED_PASSES`; each pass's ``min_merged_off_duration`` is dropped
    so the post-merge floor can be swept via ``post_ms``. The structure is loaded once and
    shared across passes (``preloaded=``); morphological is rasterized once per filter. A
    banded pass that raises (e.g. greedy_fr on a low-FR structure) is warned and skipped.
    """
    cfg, condition, manual, ts_flat, y_coords, chunks = banded_eval.eval_geometry(
        subject, probe, eval_name, manual_version=manual_version
    )
    det_row_mask, struct_row_mask = _mua_detection_row_masks(
        subject, probe, condition, manual.shape[1], structure
    )
    if scope == "detection":
        row_mask = det_row_mask
    elif scope == "structure":
        row_mask = struct_row_mask
    else:
        raise ValueError(f"scope must be 'detection' or 'structure', got {scope!r}")

    meta = {
        "subject": subject, "probe": probe, "structure": structure,
        "eval": eval_name, "condition": condition, "scope": scope,
        "n_chunks": int(len(chunks)),
    }
    rows = []

    # morphological (true masks), one row per LAS filter
    for filter_name in mua_filters:
        raster = banded_vs_morphological.rasterize_morphological_masks(
            subject, probe, structure, condition, manual.shape,
            filter_name=filter_name, off_source=mua_source,
        )
        rows.append({
            **meta, "method": "morphological", "label": filter_name,
            "filter_name": filter_name, "mua_off_source": mua_source, "post_ms": np.nan,
            **_score_raster(manual, raster, chunks, row_mask),
        })

    # --- banded (union footprint): load the structure ONCE, run each banded pass on the
    # shared trains (preloaded=), sweeping the post-merge floor. ---
    if banded_passes is None:
        banded_passes = dict(DEFAULT_BANDED_PASSES)
    bundle = banded.load_structure_inputs(subject, probe, structure)
    nrem_bouts = banded_eval.labeled_chunk_bouts(subject, probe, eval_name)
    if bundle is not None and len(nrem_bouts) and float(nrem_bouts["duration"].sum()) > 0:
        for label, bcfg in banded_passes.items():
            detect_cfg = dict(banded.ROLLOUT_CONFIG if bcfg is None else bcfg)
            detect_cfg.pop("min_merged_off_duration", None)  # swept via post_ms instead
            try:
                _, infos, artifacts = banded.detect_structure_banded(
                    subject, probe, structure,
                    bouts_by_pass={eval_name: nrem_bouts}, preloaded=bundle,
                    return_artifacts=True, verbose=False, **detect_cfg,
                )
            except Exception as exc:  # noqa: BLE001 - e.g. greedy_fr on a low-FR structure
                warnings.warn(
                    f"banded pass {label!r} failed for "
                    f"{subject}/{probe}/{structure}: {exc!r}",
                    stacklevel=2,
                )
                continue
            art = artifacts.get(eval_name, {})
            off_df_full = art.get("off_df")
            all_bands = art.get("all_bands_on_off_df")
            n_bands = int(infos.get(eval_name, {}).get("n_bands", 0))
            if off_df_full is None or not len(off_df_full):
                continue
            for post in post_ms:
                off_df = off_df_full[
                    off_df_full["union_duration"] >= post / 1000.0
                ].reset_index(drop=True)
                if not len(off_df):
                    continue
                raster = banded_eval.rasterize_banded_union(
                    off_df, all_bands, ts_flat, y_coords, manual.shape, eval_chunks=chunks,
                )
                rows.append({
                    **meta, "method": "banded", "label": label, "config": label,
                    "post_ms": int(post),
                    "band_definition": detect_cfg.get("band_definition"),
                    "band_sizes": str(detect_cfg.get("band_sizes")),
                    "n_bands": n_bands,
                    **_score_raster(manual, raster, chunks, row_mask),
                })

    return pd.DataFrame(rows)


def head_to_head_experiment(
    *,
    eval_name: str = "NREM",
    structures: list[tuple[str, str, str]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run :func:`head_to_head_structure` over every labeled cortical structure.

    Continues past per-structure failures (warns + skips). ``structures`` overrides the
    enumerated target list; remaining ``kwargs`` pass through to
    :func:`head_to_head_structure`. Returns one concatenated tidy DataFrame.
    """
    targets = structures if structures is not None else labeled_cortical_structures(eval_name)
    frames = []
    for subject, probe, structure in targets:
        try:
            df = head_to_head_structure(
                subject, probe, structure, eval_name=eval_name, **kwargs
            )
            if len(df):
                frames.append(df)
        except Exception as exc:  # noqa: BLE001 - best-effort sweep; report and continue
            warnings.warn(
                f"head_to_head failed for {subject}/{probe}/{structure}: {exc!r}",
                stacklevel=2,
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
