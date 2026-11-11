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
from pathlib import Path

import numpy as np
import pandas as pd

from cnpix_local_sleep import atlas, const, sps_conf
from cnpix_local_sleep.evaluation import banded_vs_morphological, grid, labels as ev_labels, metrics
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
EVENT_AGREEMENT_FNAME = "manual_event_agreement.csv"
DETECTOR_AGREEMENT_FNAME = "manual_detector_agreement.csv"
# Structures the annotators left essentially unlabeled in every label set; not scored
# against manual labels.
UNANNOTATED_STRUCTURES = frozenset({("CNPIX8-Allan", "imec0", "IL")})
# Structures left incompletely labeled in one evaluation's label set; not scored against it.
INCOMPLETE_STRUCTURES: dict[str, frozenset] = {
    "SD": frozenset({
        ("CNPIX10-Charles", "imec0", "CLA"),
        ("CNPIX4-Doppio", "imec1", "MO"),
        ("CNPIX7-Giuseppe", "imec0", "PIR"),
    }),
}


def labeled_cortical_structures(eval_name: str = "NREM") -> list[tuple[str, str, str]]:
    """Cortical unit_based-included (subject, probe, structure) tuples with manual labels.

    Excludes :data:`UNANNOTATED_STRUCTURES` and the evaluation's
    :data:`INCOMPLETE_STRUCTURES`.
    """
    cond = banded_eval.config.EVAL_CONFIGS[eval_name]["condition"]
    excluded = UNANNOTATED_STRUCTURES | INCOMPLETE_STRUCTURES.get(eval_name, frozenset())
    pairset = set(ev_labels._get_subject_probe_pairs_with_labels(cond))
    spsl = sps_conf.get_subject_probe_structure_list(
        method=ub_files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    return [
        (s, p, st)
        for (s, p, st) in spsl
        if (s, p) in pairset and (s, p, st) not in excluded
    ]


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


def banded_rasters(
    subject, probe, structure, bouts_by_pass, banded_passes, post_ms,
    ts_flat, y_coords, shape, chunks,
) -> dict:
    """Union-footprint rasters of each banded pass at each post-merge floor.

    Returns ``{(label, post_ms): (raster, detect_cfg, n_bands)}``. ``banded_passes`` is
    ``{label: config_or_None}`` (None -> ROLLOUT_CONFIG); each config's
    ``min_merged_off_duration`` is dropped in favor of the ``post_ms`` floors. The
    structure is loaded once and shared across passes. Each fit window in
    ``bouts_by_pass`` is fitted separately and its OFFs are painted into the one raster;
    ``n_bands`` is the largest band count over the windows. A pass that raises (e.g.
    greedy_fr on a low-FR structure) is warned and left out; one that keeps no OFF at a
    floor gives an empty raster.
    """
    if not any(len(b) and float(b["duration"].sum()) > 0 for b in bouts_by_pass.values()):
        return {}
    bundle = banded.load_structure_inputs(subject, probe, structure)
    if bundle is None:
        return {}
    out = {}
    for label, bcfg in banded_passes.items():
        detect_cfg = dict(banded.ROLLOUT_CONFIG if bcfg is None else bcfg)
        detect_cfg.pop("min_merged_off_duration", None)  # swept via post_ms instead
        try:
            _, infos, artifacts = banded.detect_structure_banded(
                subject, probe, structure,
                bouts_by_pass=bouts_by_pass, preloaded=bundle,
                return_artifacts=True, verbose=False, **detect_cfg,
            )
        except Exception as exc:  # noqa: BLE001 - e.g. greedy_fr on a low-FR structure
            warnings.warn(
                f"banded pass {label!r} failed for "
                f"{subject}/{probe}/{structure}: {exc!r}",
                stacklevel=2,
            )
            continue
        n_bands = max((int(i.get("n_bands", 0)) for i in infos.values()), default=0)
        for post in post_ms:
            raster = np.zeros(shape, dtype=np.int32)
            for art in artifacts.values():
                off_df = art.get("off_df")
                if off_df is None or not len(off_df):
                    continue
                off_df = off_df[off_df["union_duration"] >= post / 1000.0].reset_index(drop=True)
                if not len(off_df):
                    continue
                r = banded_eval.rasterize_banded_union(
                    off_df, art.get("all_bands_on_off_df"), ts_flat, y_coords, shape,
                    eval_chunks=chunks,
                )
                painted = r > 0
                raster[painted] = r[painted] + raster.max()
            out[(label, int(post))] = (raster, detect_cfg, n_bands)
    return out


def head_to_head_structure(
    subject: str,
    probe: str,
    structure: str,
    *,
    eval_name: str = "NREM",
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
        "minutes": len(chunks) * manual.shape[2] * grid.sample_period(ts_flat) / 60.0,
    }
    rows = []

    # morphological (true masks), one row per LAS filter
    for filter_name in mua_filters:
        raster = banded_vs_morphological.rasterize_morphological_masks(
            subject, probe, structure, condition, manual.shape,
            filter_name=filter_name,
        )
        rows.append({
            **meta, "method": "morphological", "label": filter_name,
            "filter_name": filter_name, "mua_off_source": "full48h", "post_ms": np.nan,
            **_score_raster(manual, raster, chunks, row_mask),
        })

    # banded (union footprint), one row per (pass, post-merge floor)
    if banded_passes is None:
        banded_passes = dict(DEFAULT_BANDED_PASSES)
    bouts_by_pass = banded_eval.fit_bouts(
        subject, probe, eval_name, ts_flat.reshape(manual.shape[0], -1), chunks
    )
    rasters = banded_rasters(
        subject, probe, structure, bouts_by_pass, banded_passes, post_ms,
        ts_flat, y_coords, manual.shape, chunks,
    )
    for (label, post), (raster, detect_cfg, n_bands) in rasters.items():
        rows.append({
            **meta, "method": "banded", "label": label, "config": label,
            "post_ms": post,
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


def experiment_results_path(mua_source: str = "full48h", eval_name: str = "NREM") -> Path:
    """Where ``batch_manual_vs_banded_and_morphological.ipynb`` saves
    :func:`head_to_head_experiment` (the Table 1 scores) on NFS."""
    import wisc_ecephys_tools as wet

    return (
        wet.get_sglx_project("offproj").get_experiment_directory(const.EXPERIMENT)
        / f"manual_vs_banded_and_morphological_{mua_source}_{eval_name}.parquet"
    )


def export_event_agreement(
    output_dir: Path, *, mua_source: str = "full48h", eval_name: str = "NREM"
) -> Path:
    """Write the per-structure event recall and precision of the morphological OFFs.

    Reads the saved :func:`head_to_head_experiment` table and keeps the morphological
    rows scored over the detection channels, one row per
    ``(subject, probe, structure, off_type)``. Event precision is
    ``1 - event_false_discovery_rate``, so it is missing where nothing was detected.
    Structures carry their consolidated OFF-analysis labels, like every other r-offp
    input.
    """
    df = pd.read_parquet(experiment_results_path(mua_source, eval_name))
    df = df[(df["method"] == "morphological") & (df["scope"] == "detection")]
    out = pd.DataFrame({
        "subject": df["subject"],
        "probe": df["probe"],
        "structure": df["structure"].map(atlas.consolidate_structure_label),
        "off_type": df["label"],
        "event_recall": df["event_sensitivity"],
        "event_precision": 1.0 - df["event_false_discovery_rate"],
    }).sort_values(["off_type", "subject", "probe", "structure"])
    path = Path(output_dir) / EVENT_AGREEMENT_FNAME
    out.to_csv(path, index=False)
    return path


def export_detector_agreement(output_dir: Path, *, mua_source: str = "full48h") -> Path:
    """Write each detector's per-structure agreement with every manual annotation set.

    Reads the saved :func:`head_to_head_experiment` tables of the ``NREM``, ``Match`` and
    ``SD`` evaluations and the :mod:`~cnpix_local_sleep.evaluation.annotator_overlap`
    table, and keeps the structures of the morphological analysis set. One row per
    ``(annotation, reference, subject, probe, structure, detector)``: ``annotation`` is
    the evaluation or ``overlap``; ``reference`` is the annotation scored against (the
    evaluation itself, or ``Wake`` / ``SD`` on the overlap images); ``detector`` is
    ``llas``, ``clas``, a banded pass, or ``annotation`` for the other annotation on the
    overlap images. Structures carry their consolidated OFF-analysis labels. The source of
    Supplementary Tables S6a-c.
    """
    from cnpix_local_sleep.evaluation import annotator_overlap

    frames = [
        pd.read_parquet(experiment_results_path(mua_source, ev)).assign(annotation=ev, reference=ev)
        for ev in ("NREM", "Match", "SD")
    ]
    frames.append(pd.read_parquet(annotator_overlap.results_path()).assign(annotation="overlap"))
    df = pd.concat(frames, ignore_index=True)
    analysis_set = set(sps_conf.get_subject_probe_structure_list(method="morphological"))
    in_set = np.array([k in analysis_set for k in zip(df["subject"], df["probe"], df["structure"])])
    scored = df["scope"].isna() | (df["scope"] == "detection")
    detector = df["label"].where(df["method"] != "annotation", "annotation")
    reported = detector.isin(["llas", "clas", "annotation"]) | (df["method"] == "banded")
    df = df.assign(detector=detector)[in_set & scored & reported]
    keys = ["annotation", "reference", "subject", "probe", "structure", "detector"]
    out = pd.DataFrame({
        "annotation": df["annotation"],
        "reference": df["reference"],
        "subject": df["subject"],
        "probe": df["probe"],
        "structure": df["structure"].map(atlas.consolidate_structure_label),
        "detector": df["detector"],
        "minutes": df["minutes"],
        "n_manual_events": df["n_manual_events"],
        "n_predicted_events": df["n_predicted_events"],
        "offs_per_minute": df["n_predicted_events"] / df["minutes"],
        "F1": df["F1"],
        "IoU": df["IoU"],
        "sensitivity": df["sensitivity"],
        "precision": df["precision"],
        "event_recall": df["event_sensitivity"],
        "event_precision": df["event_precision"],
    }).sort_values(keys)
    assert not out.duplicated(keys).any(), "consolidated structure labels collide"
    path = Path(output_dir) / DETECTOR_AGREEMENT_FNAME
    out.to_csv(path, index=False)
    return path
