"""Per-condition comparison of morphological detections against manual OFF labels.

The method-agnostic pieces (manual-label loaders, instance-label QC,
channel->row mapping, and the pixel/event metric kernels) now live in
``cnpix_local_sleep.evaluation`` and are re-exported here for backward compatibility.
This module retains the *morphological-specific* driver that rasterizes label indices
(``time_ixs``/``chan_ixs``) into manual-label space and scores them. Both OFF
sources are supported via the ``off_source`` argument and both use the true
per-pixel masks (never bounding boxes):

- ``off_source="per_condition"``: the per-condition ``offs.parquet`` (frozen,
  pre-48h-optimization thresholds); ``time_ixs`` are condition-MUA sample indices.
- ``off_source="full48h"`` (default): the whole-recording ``offs.parquet`` +
  ``off_label_indices.parquet`` (re-optimized thresholds); ``time_ixs`` are
  full-recording MUA sample indices, restricted to the condition by its hypnogram.

(``cnpix_local_sleep.morphological.full48h_eval`` also reads the full-48h offs but only as
bounding boxes; use this module when you need true masks.)

This is a heavy module; import it directly::

    from cnpix_local_sleep.morphological import manual_validation
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from cnpix_local_sleep import hyp
from cnpix_local_sleep.morphological.common import MorphologicalSourceConfig
from cnpix_local_sleep.evaluation import grid, labels as eval_labels, metrics
from cnpix_local_sleep import off_tables
from cnpix_local_sleep import trace_io

# Re-exported for backward compatibility (definitions live in cnpix_local_sleep.evaluation).
load_manual_labels = eval_labels.load_manual_labels
qc_and_fix_labels = eval_labels.qc_and_fix_labels
_build_channel_maps = eval_labels._build_channel_maps
_get_subject_probe_pairs_with_labels = (
    eval_labels._get_subject_probe_pairs_with_labels
)
compute_pixel_metrics = metrics.compute_pixel_metrics
compute_per_event_pixel_metrics = metrics.compute_per_event_pixel_metrics
compute_event_metrics = metrics.compute_event_metrics


@functools.lru_cache(maxsize=4)
def _get_mua_condition_times(subject: str, probe: str, condition: str) -> np.ndarray:
    """Return condition-masked timestamps from the MUA zarr.

    Loads only the time coordinate (traces stay lazy on disk), applies the same
    ``hotfix_times`` filtering and condition masking that the detection pipeline
    uses, then returns the resulting 1-D array of timestamps. Cached per
    (subject, probe, condition) so a sweep over structures on a probe loads it once.
    """
    from cnpix.mua import files as mua_files

    path = mua_files.get_mua_traces_path(subject, probe)
    da = trace_io.open_si_zarr_recording_as_xarray(path)
    hg = hyp.load_statistical_condition_hypnograms(subject, probe)[condition]
    mask = hg.covers_time(da.time)
    return da.time.values[mask]


@functools.lru_cache(maxsize=4)
def _get_mua_full_times(subject: str, probe: str) -> np.ndarray:
    """Return the FULL-recording (unmasked) MUA timestamps from the MUA zarr.

    The full-48h detection runs on the whole recording (``condition=None``), so its
    ``time_ixs`` index into these unmasked timestamps; the condition-masked variant
    above is exactly this array masked by the condition hypnogram. Cached per
    (subject, probe).
    """
    from cnpix.mua import files as mua_files

    path = mua_files.get_mua_traces_path(subject, probe)
    da = trace_io.open_si_zarr_recording_as_xarray(path)
    return da.time.values


def _build_morphological_label_array(
    morphological_offs: pd.DataFrame,
    det_channels: np.ndarray,
    full_channels: np.ndarray,
    n_full_channels: int,
    label_shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert morphological OFF label indices to manual-label array space.

    Parameters
    ----------
    morphological_offs
        DataFrame with ``label``, ``time_ixs``, and ``chan_ixs`` columns.
    det_channels
        Detection channel names (from the detection DataArray).
    full_channels
        Full-probe channel names.
    n_full_channels
        Number of full-probe channels.
    label_shape
        Shape of the manual label array ``(n_chunks, n_rows, samples)``.
    """
    full_chan_to_idx = {ch: i for i, ch in enumerate(full_channels)}
    det_chan_to_stack_row = np.array(
        [(n_full_channels - 1) - full_chan_to_idx[ch] for ch in det_channels],
        dtype=np.intp,
    )

    n_label_chunks, n_label_rows, samples_per_chunk = label_shape
    morphological_label_arr = np.zeros(label_shape, dtype=np.int32)

    for _, row in morphological_offs.iterrows():
        label = row["label"]
        time_ixs = np.asarray(row["time_ixs"])
        chan_ixs = np.asarray(row["chan_ixs"])

        stack_rows = det_chan_to_stack_row[chan_ixs]
        chunk_ids = time_ixs // samples_per_chunk
        within_chunk = time_ixs % samples_per_chunk

        valid = (
            (chunk_ids < n_label_chunks)
            & (stack_rows < n_label_rows)
            & (within_chunk < samples_per_chunk)
        )
        morphological_label_arr[chunk_ids[valid], stack_rows[valid], within_chunk[valid]] = (
            label
        )

    return morphological_label_arr


def _normalize_label_indices(offs):
    """Coerce ``time_ixs``/``chan_ixs`` to paired 1-D int arrays, dropping non-finite
    pixels.

    ``off_label_indices`` occasionally stores a 0-d scalar or a NaN for a degenerate OFF;
    ``np.asarray`` then yields a 0-d / float array that breaks the index arithmetic
    downstream (``mua_condition_times[np.asarray(ix)]`` raises, or a NaN casts to
    int64-min and indexes out of bounds). Each ``(time_ixs, chan_ixs)`` pair is flattened,
    truncated to the common length, and filtered to entries finite in both.
    """
    def _pair(t, c):
        t = np.atleast_1d(np.asarray(t)).ravel().astype(float)
        c = np.atleast_1d(np.asarray(c)).ravel().astype(float)
        n = min(t.size, c.size)
        t, c = t[:n], c[:n]
        keep = np.isfinite(t) & np.isfinite(c)
        return t[keep].astype(np.int64), c[keep].astype(np.int64)

    offs = offs.copy()
    pairs = [_pair(t, c) for t, c in zip(offs["time_ixs"], offs["chan_ixs"])]
    offs["time_ixs"] = [p[0] for p in pairs]
    offs["chan_ixs"] = [p[1] for p in pairs]
    return offs


def _load_translated_mua_offs(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    filter_name: str,
    *,
    off_source: str,
    source_config: MorphologicalSourceConfig,
    stack_times_flat: np.ndarray | None = None,
) -> pd.DataFrame:
    """Load morphological OFFs with TRUE per-pixel masks and translate ``time_ixs`` into
    stack-grid flat sample indices (ready for :func:`_build_morphological_label_array`).

    ``off_source``:

    - ``"per_condition"``: per-condition ``offs.parquet`` via
      :func:`cnpix_local_sleep.off_tables.load_subject_offs`; for ``morphological`` the
      ``time_ixs`` are condition-MUA sample indices, translated via
      :func:`_get_mua_condition_times`.
    - ``"full48h"``: whole-recording ``offs.parquet`` + ``off_label_indices.parquet``
      (per-structure paths), filtered by LAS (:func:`off_tables.filter_offs`) and
      restricted to the condition by its hypnogram; for ``morphological`` the ``time_ixs``
      are full-recording MUA sample indices, translated via :func:`_get_mua_full_times`.

    Both paths return the true ``time_ixs``/``chan_ixs`` masks (never bounding boxes).
    The ``time_ixs`` translation only applies to the ``morphological`` variant (the
    ``tom-bugnon`` traces already share the stack timebase).
    """
    if off_source not in ("per_condition", "full48h"):
        raise ValueError(
            f"off_source must be 'per_condition' or 'full48h', got {off_source!r}"
        )
    is_mua = source_config.variant == "morphological"
    if is_mua and stack_times_flat is None:
        raise ValueError("stack_times_flat is required for the morphological variant")

    if off_source == "per_condition":
        offs = off_tables.load_subject_offs(
            subject,
            filter_name=filter_name,
            with_label_indices=True,
            files_module=source_config.files_module,
        )
        offs = offs[
            (offs["probe"] == probe)
            & (offs["structure"] == structure)
            & (offs["condition"] == condition)
        ].reset_index(drop=True)
        offs = _normalize_label_indices(offs)
        if is_mua and len(offs):
            assert stack_times_flat is not None  # guaranteed by the guard above
            mua_times = _get_mua_condition_times(subject, probe, condition)
            offs["time_ixs"] = offs["time_ixs"].map(
                lambda ix: np.searchsorted(stack_times_flat, mua_times[ix])
            )
        return offs

    # off_source == "full48h" (validated above).
    fm = source_config.files_module
    offs = pd.read_parquet(fm.get_full_offs_path(subject, probe, structure))
    offs = off_tables.filter_offs(offs, filter_name)
    # The full-recording offs carry `state` (NREM/Wake), not `condition`; restrict
    # to this condition's coverage via its hypnogram before translating times.
    hg = hyp.load_statistical_condition_hypnograms(subject, probe)[condition]
    offs = offs[hg.covers_time(offs["start_time"].to_numpy())].reset_index(drop=True)
    if not len(offs):
        return offs
    lbls = pd.read_parquet(
        fm.get_full_off_label_indices_path(subject, probe, structure),
        columns=["label", "time_ixs", "chan_ixs"],
    )
    lbls = lbls[lbls["label"].isin(set(offs["label"]))]
    offs = offs.merge(lbls, on="label", how="left")
    offs = _normalize_label_indices(offs)
    if is_mua and len(offs):
        assert stack_times_flat is not None  # guaranteed by the guard above
        full_times = _get_mua_full_times(subject, probe)
        offs["time_ixs"] = offs["time_ixs"].map(
            lambda ix: np.searchsorted(stack_times_flat, full_times[ix])
        )
    return offs


def compare_structure(
    subject: str,
    probe: str,
    structure: str,
    filter_name: str,
    source_config: MorphologicalSourceConfig,
    condition: str = "Early.REC.NREM",
    *,
    manual_labels: np.ndarray,
    da_full: xr.DataArray,
    stack_times_flat: np.ndarray | None = None,
    off_source: str = "full48h",
) -> pd.DataFrame:
    """Run a full comparison for one (subject, probe, structure, filter).

    Accepts pre-loaded ``manual_labels`` and ``da_full`` to avoid redundant I/O
    when multiple structures share a probe. ``off_source`` selects which morphological
    OFFs to score: ``"full48h"`` (default, whole-recording) or ``"per_condition"``,
    both using true per-pixel masks. Returns two rows (scope = ``"detection"`` and
    ``"structure"``).
    """
    # Load structure and detection DataArrays (always from tom traces, since the
    # manual labels were annotated on stacks built from them).
    da_struct = trace_io.open_preprocessed_traces_as_xarray(
        subject,
        probe,
        structure=structure,
        condition=condition,
        apply_detection_channel_mask=False,
    )
    da_det = trace_io.open_preprocessed_traces_as_xarray(
        subject,
        probe,
        structure=structure,
        condition=condition,
        apply_detection_channel_mask=True,
    )

    # Build row masks
    n_label_rows = manual_labels.shape[1]
    det_row_mask, struct_row_mask = _build_channel_maps(
        da_full, da_struct, da_det, n_label_rows
    )

    # Load the OFFs (per-condition or full-48h) with true per-pixel masks, already
    # translated into stack-grid sample indices. The translation needs the stack's
    # flattened timestamps for the morphological variant.
    if source_config.variant == "morphological" and stack_times_flat is None:
        stack_times_flat = grid.load_stack_times_flat(subject, probe, condition)
    morphological_offs = _load_translated_mua_offs(
        subject,
        probe,
        structure,
        condition,
        filter_name,
        off_source=off_source,
        source_config=source_config,
        stack_times_flat=stack_times_flat,
    )

    # Build the morphological label array
    morphological_label_arr = _build_morphological_label_array(
        morphological_offs,
        det_channels=da_det.channel.values,
        full_channels=da_full.channel.values,
        n_full_channels=da_full.sizes["channel"],
        label_shape=manual_labels.shape,
    )

    # Determine labeled chunks
    labeled_chunks = np.where(np.any(manual_labels > 0, axis=(1, 2)))[0]

    # Compute metrics for each scope
    rows = []
    for scope_name, row_mask in [
        ("detection", det_row_mask),
        ("structure", struct_row_mask),
    ]:
        px = compute_pixel_metrics(
            manual_labels, morphological_label_arr, labeled_chunks, row_mask
        )
        ev = metrics.summarize_event_ious(
            compute_event_metrics(
                manual_labels, morphological_label_arr, labeled_chunks, row_mask
            )
        )
        rows.append(
            {
                "subject": subject,
                "probe": probe,
                "structure": structure,
                "filter_name": filter_name,
                "variant": source_config.variant,
                "off_source": off_source,
                "scope": scope_name,
                **px,
                **ev,
            }
        )

    return pd.DataFrame(rows)


def compare_all(
    source_config: MorphologicalSourceConfig,
    filter_names: list[str] | None = None,
    condition: str = "Early.REC.NREM",
    *,
    off_source: str = "full48h",
) -> pd.DataFrame:
    """Run the comparison for all cortical tuples with manual labels.

    Iterates over (subject, probe) pairs that have manual label files, loads
    shared data once per probe, then iterates over cortical structures and filter
    names. ``off_source`` selects the morphological OFF source (``"full48h"`` default or
    ``"per_condition"``); both use true masks.
    """
    if filter_names is None:
        filter_names = ["llas", "clas", "blas"]

    # Get cortical structures from config
    spsl = source_config.get_subject_probe_structure_list(
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )

    # Get (subject, probe) pairs with manual labels
    sp_with_labels = set(_get_subject_probe_pairs_with_labels())

    # Group structures by (subject, probe)
    sp_to_structures: dict[tuple[str, str], list[str]] = {}
    for s, p, st in spsl:
        if (s, p) in sp_with_labels:
            sp_to_structures.setdefault((s, p), []).append(st)

    all_results = []
    skipped = []

    for (subject, probe), structures in sorted(sp_to_structures.items()):
        print(f"Processing {subject} / {probe} ...")

        # Load shared data once per probe
        try:
            raw_labels = load_manual_labels(subject, probe, condition)
            manual_labels, violations = qc_and_fix_labels(raw_labels)
            if violations:
                n_orig = len(np.unique(raw_labels)) - 1
                n_fixed = len(np.unique(manual_labels)) - 1
                print(
                    f"  Label QC: {len(violations)} violations, "
                    f"{n_orig} -> {n_fixed} labels"
                )
        except Exception as exc:
            msg = f"  Skipping {subject}/{probe}: failed to load manual labels: {exc}"
            warnings.warn(msg, stacklevel=2)
            skipped.append(msg)
            continue

        try:
            da_full = trace_io.open_preprocessed_traces_as_xarray(
                subject,
                probe,
                structure=None,
                condition=condition,
                apply_detection_channel_mask=False,
            )
        except Exception as exc:
            msg = (
                f"  Skipping {subject}/{probe}: "
                f"failed to load preprocessed traces: {exc}"
            )
            warnings.warn(msg, stacklevel=2)
            skipped.append(msg)
            continue

        # Pre-load the MUA timebase (warming the per-probe cache) and stack
        # timestamps once per probe (only needed for the morphological variant). The
        # relevant timebase depends on off_source: full-recording vs condition-masked.
        stack_times_flat = None
        if source_config.variant == "morphological":
            try:
                if off_source == "full48h":
                    _get_mua_full_times(subject, probe)
                else:
                    _get_mua_condition_times(subject, probe, condition)
                stack_times_flat = grid.load_stack_times_flat(
                    subject, probe, condition
                )
            except Exception as exc:
                msg = (
                    f"  Skipping {subject}/{probe}: "
                    f"failed to load MUA/stack times: {exc}"
                )
                warnings.warn(msg, stacklevel=2)
                skipped.append(msg)
                continue

        for structure in structures:
            for filter_name in filter_names:
                try:
                    df = compare_structure(
                        subject,
                        probe,
                        structure,
                        filter_name,
                        source_config,
                        condition,
                        manual_labels=manual_labels,
                        da_full=da_full,
                        stack_times_flat=stack_times_flat,
                        off_source=off_source,
                    )
                    all_results.append(df)
                except Exception as exc:
                    msg = (
                        f"  Skipping {subject}/{probe}/{structure}/{filter_name}: {exc}"
                    )
                    warnings.warn(msg, stacklevel=2)
                    skipped.append(msg)

    if skipped:
        print(f"\n{len(skipped)} items skipped:")
        for msg in skipped:
            print(msg)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
