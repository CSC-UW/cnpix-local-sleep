"""Detect OFF periods using preprocessed AP-band data (morphological method).

This module combines threshold computation and OFF detection into a single
workflow, loading data once and performing both operations before writing
results to disk.

Runtimes:
- Spatial detection for all cortical (subject, probe, structure)
  combinations: 72m.
- Spatial detection for (Segundo, imec0, PPC): 2m.
  - Threshold computation accounts for ~30s of this.
"""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from cnpix_local_sleep import atlas, const
from cnpix_local_sleep.morphological.common import MorphologicalSourceConfig
from cnpix_local_sleep.morphological.types import DetectionOpts
from cnpix_local_sleep.morphological import morphology
from cnpix_local_sleep import channel_anatomy
from cnpix_local_sleep.morphological.pipeline import utils
from cnpix_local_sleep.off_tables import Off


# -------------------- Threshold quantile computation --------------------


def _get_threshold_quantile_from_value(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    source_config: MorphologicalSourceConfig,
) -> float:
    """Get the variant's per-(subject, probe, structure, condition) quantile threshold.

    Reads from the per-condition thresholds CSV bundled at
    ``<source_config.thresholds_package>.data/quantile_thresholds_per_condition.csv``
    (falling back to ``quantile_thresholds.csv`` for variants without it). This
    is deliberately separate from the 48h-optimized thresholds that
    full-recording detection (``detect_full``) consumes.
    """
    return source_config.get_per_condition_quantile_threshold(
        subject, probe, structure, condition
    )


def _get_quantile_threshold_for_condition(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    threshold_method: Literal["from_value"],
    source_config: MorphologicalSourceConfig,
) -> float:
    """Get threshold quantile for a condition based on threshold method.

    Args:
        subject: Subject identifier
        probe: Probe identifier
        structure: Brain structure name
        condition: Experimental condition
        threshold_method: Must be ``"from_value"``; read the quantile from
            the method's per-condition thresholds CSV.

    Returns:
        Threshold quantile value for the condition

    Raises:
        ValueError: If threshold_method is not "from_value".
    """
    if threshold_method != "from_value":
        raise ValueError(
            f"Unknown threshold_method: {threshold_method!r}. Must be "
            "'from_value'. The 'from_manual_off_detection' and 'mixed' "
            "methods were removed with the retired tom-bugnon variant, which "
            "held their only "
            "implementation."
        )
    return _get_threshold_quantile_from_value(
        subject, probe, structure, condition, source_config
    )


def get_channel_threshold_values_from_quantile(
    da: xr.DataArray,
    quantile: float,
    threshold_method: Literal["from_value"],
    ndimage_filter_type: str | None,
    ndimage_filter_kwargs: dict | None,
) -> xr.DataArray:
    """Compute channel-wise threshold values from a given quantile.

    Whole-recording variant. Morphological detection uses
    :func:`compute_per_bin_thresholds` instead; this is kept for callers that
    want a single threshold per channel over the whole recording.

    Args:
        da: Input xarray DataArray with dimensions (time, channel). Must
            be computed (numpy-backed), not dask-backed.
        quantile: Quantile value (between 0 and 1) to compute thresholds.
        threshold_method: Threshold method used (stored in attrs for
            provenance).
        ndimage_filter_type: Filter type used (stored in attrs for
            provenance).
        ndimage_filter_kwargs: Filter kwargs used (stored in attrs for
            provenance).

    Returns:
        DataArray with dimension (channel,) containing threshold value
        for each channel.
    """
    # Compute quantile along time axis for each channel
    threshold_values = np.quantile(da.values, quantile, axis=0)

    return xr.DataArray(
        data=threshold_values,
        dims=("channel",),
        coords={"channel": da.channel, "y": ("channel", da.y.data)},
        name="Detection threshold",
        attrs={
            "quantile": quantile,
            "threshold_method": threshold_method,
            "ndimage_filter_type": ndimage_filter_type,
            "ndimage_filter_kwargs": ndimage_filter_kwargs,
        },
    )


def compute_per_bin_thresholds(
    da: xr.DataArray,
    quantile: float,
    bin_boundaries: np.ndarray,
    *,
    derivation_mask: np.ndarray | None = None,
    threshold_method: Literal["from_value"],
    ndimage_filter_type: str | None,
    ndimage_filter_kwargs: dict | None,
) -> xr.DataArray:
    """Compute per-bin per-channel quantile thresholds.

    Splits the trace into time bins given by ``bin_boundaries`` and
    computes ``quantile`` per (bin, channel) over the optional
    ``derivation_mask``-selected samples within each bin. Bins whose
    mask selects no samples receive NaN; callers should fill these
    (typically nearest-bin forward/backward fill) before applying.

    Args:
        da: Numpy-backed DataArray with dims (time, channel).
        quantile: Target quantile in [0, 1].
        bin_boundaries: Sample-index boundaries of shape (n_bins+1,);
            ``boundaries[0] == 0`` and ``boundaries[-1] == n_time``.
            Bin ``i`` covers samples ``[boundaries[i], boundaries[i+1])``.
        derivation_mask: Optional boolean array of shape (n_time,)
            selecting samples eligible for the quantile (e.g., a state
            mask). If None, every sample contributes.
        threshold_method: Provenance metadata.
        ndimage_filter_type: Provenance metadata.
        ndimage_filter_kwargs: Provenance metadata.

    Returns:
        DataArray with dims ``("bin", "channel")``, shape
        ``(n_bins, n_channels)``. Coords: ``channel``, ``y`` per
        channel; ``bin_start_sample`` per bin (``= boundaries[:-1]``).
        ``attrs["bin_boundaries"]`` carries the full
        ``(n_bins+1,)`` array; ``attrs["quantile"]`` and the filter
        provenance match the whole-recording function.
    """
    n_time, n_channels = da.values.shape
    boundaries = np.asarray(bin_boundaries, dtype=np.int64)
    if boundaries.ndim != 1 or boundaries[0] != 0 or boundaries[-1] != n_time:
        raise ValueError(
            f"bin_boundaries must start at 0 and end at n_time={n_time}; "
            f"got first={boundaries[0]}, last={boundaries[-1]}"
        )
    n_bins = len(boundaries) - 1

    if derivation_mask is None:
        derivation_mask = np.ones(n_time, dtype=bool)
    elif derivation_mask.shape != (n_time,):
        raise ValueError(
            f"derivation_mask shape {derivation_mask.shape} does not "
            f"match n_time={n_time}"
        )

    arr = da.values
    out = np.full((n_bins, n_channels), np.nan, dtype=np.float64)
    for bi in range(n_bins):
        lo, hi = int(boundaries[bi]), int(boundaries[bi + 1])
        bin_mask = derivation_mask[lo:hi]
        if not bin_mask.any():
            continue
        out[bi, :] = np.quantile(arr[lo:hi][bin_mask, :], quantile, axis=0)

    # Forward/backward fill empty bins with their nearest valid neighbor.
    # This keeps the threshold defined everywhere along the time axis;
    # bins with zero state-relevant samples otherwise produce NaN, which
    # would silently disable detection in those windows.
    out = _fill_nan_bins(out)

    return xr.DataArray(
        data=out,
        dims=("bin", "channel"),
        coords={
            "channel": da.channel,
            "y": ("channel", da.y.data),
            "bin_start_sample": (
                "bin",
                boundaries[:-1].astype(np.int64),
            ),
        },
        name="Detection threshold (per-bin)",
        attrs={
            "quantile": quantile,
            "threshold_method": threshold_method,
            "ndimage_filter_type": ndimage_filter_type,
            "ndimage_filter_kwargs": ndimage_filter_kwargs,
            "bin_boundaries": boundaries.tolist(),
        },
    )


def _fill_nan_bins(thresholds: np.ndarray) -> np.ndarray:
    """Fill all-NaN bin rows with the nearest valid-bin row."""
    n_bins = thresholds.shape[0]
    valid_idxs = np.flatnonzero(~np.isnan(thresholds[:, 0]))
    if len(valid_idxs) == 0:
        raise ValueError(
            "All bins are empty under the derivation mask; cannot "
            "compute thresholds."
        )
    if len(valid_idxs) == n_bins:
        return thresholds
    out = thresholds.copy()
    invalid_idxs = np.setdiff1d(np.arange(n_bins), valid_idxs)
    for bi in invalid_idxs:
        nearest = valid_idxs[np.argmin(np.abs(valid_idxs - bi))]
        out[bi, :] = thresholds[nearest, :]
    return out


# -------------------- Laminar area computation --------------------


def add_laminar_areas(
    offs: pd.DataFrame,
    da: xr.DataArray,
    lbl_ixs: dict[int, tuple[np.ndarray, np.ndarray]],
    subject: str,
    probe: str,
    structure: str,
) -> None:
    """Add supra/infra area columns to the OFF DataFrame in-place.

    For cortical structures, counts the number of (sample, channel)
    pixels in each OFF that fall within the supragranular and
    infragranular compartments. Also records the number of detection
    channels in each compartment.

    For non-cortical structures, all four columns are set to NaN.

    .. note::
        These areas are stored in geometric order: "supra" is always the
        top 45% band (higher y/depth) per :func:`channel_anatomy.get_layer_borders`.
        For a few combos brain curvature flips the structure vertically vs the
        probe (``sps_conf.get_flipped_laminar_combos``), so this geometric
        "supra" is actually the infragranular layer. That per-combo orientation
        is corrected downstream at consumption (in
        :func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations`),
        not here, to avoid re-running 48h detection.

        TODO(source-fix): the cleaner long-term fix is to relabel the bands at
        the source (swap supra/infra in :func:`channel_anatomy.get_layer_borders` for
        flipped combos so these columns are anatomically honest) and re-run
        detection + re-export. If you do that, you MUST drop the consumption-time
        swap in ``laminar_concentrations`` at the same time, or the two
        corrections compound into a silent double-flip.

    Args:
        offs: DataFrame of detected OFFs (modified in-place).
        da: Detection DataArray with y-coordinates on channels.
        lbl_ixs: Label indices mapping label -> (time_ixs, chan_ixs).
        subject: Subject identifier.
        probe: Probe identifier.
        structure: Brain structure name.
    """
    if atlas.get_clade(structure) != "Cx" or offs.empty:
        offs["supra_area"] = pd.array(
            [pd.NA] * len(offs), dtype=pd.Int64Dtype()
        )
        offs["infra_area"] = pd.array(
            [pd.NA] * len(offs), dtype=pd.Int64Dtype()
        )
        offs["max_supra_nchans"] = pd.array(
            [pd.NA] * len(offs), dtype=pd.Int64Dtype()
        )
        offs["max_infra_nchans"] = pd.array(
            [pd.NA] * len(offs), dtype=pd.Int64Dtype()
        )
        return

    layer_borders = channel_anatomy.get_layer_borders(subject, probe, structure)
    y_coords = da.y.values

    supra_row = layer_borders[layer_borders["layer"] == "supra"].iloc[0]
    infra_row = layer_borders[layer_borders["layer"] == "infra"].iloc[0]

    supra_mask = (y_coords >= supra_row["lo"]) & (
        y_coords <= supra_row["hi"]
    )
    infra_mask = (y_coords >= infra_row["lo"]) & (
        y_coords <= infra_row["hi"]
    )

    max_supra_nchans = int(supra_mask.sum())
    max_infra_nchans = int(infra_mask.sum())

    supra_areas = []
    infra_areas = []
    for label in offs["label"]:
        if label not in lbl_ixs:
            supra_areas.append(0)
            infra_areas.append(0)
            continue
        _, chan_ixs = lbl_ixs[label]
        pixel_y = y_coords[chan_ixs]
        supra_areas.append(
            int(
                (
                    (pixel_y >= supra_row["lo"])
                    & (pixel_y <= supra_row["hi"])
                ).sum()
            )
        )
        infra_areas.append(
            int(
                (
                    (pixel_y >= infra_row["lo"])
                    & (pixel_y <= infra_row["hi"])
                ).sum()
            )
        )

    offs["supra_area"] = supra_areas
    offs["infra_area"] = infra_areas
    offs["max_supra_nchans"] = max_supra_nchans
    offs["max_infra_nchans"] = max_infra_nchans


# -------------------- Main detection function --------------------


def detect_offs(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    opts: DetectionOpts,
    source_config: MorphologicalSourceConfig,
    *,
    threshold_group: str | None = None,
) -> None:
    """Detect OFF periods and save results to disk.

    This function performs the complete OFF detection workflow:
    1. Loads trace data via ``source_config.open_traces_as_xarray``
    2. Computes channel thresholds based on quantile
    3. Detects OFF periods using threshold crossing and morphological
       operations
    4. Writes OFF results to parquet files via
       ``source_config.files_module``
    5. Writes channel thresholds to zarr (only after OFFs written
       successfully)

    Args:
        subject: Subject identifier
        probe: Probe identifier
        structure: Brain structure name
        condition: Experimental condition
        opts: Detection algorithm configuration. See ``DetectionOpts``.
        source_config: Variant plumbing. Determines both the trace
            reader and the on-disk write paths. Pass
            ``cnpix_local_sleep.morphological.mua.SOURCE_CONFIG``, the only variant left since
            tom-bugnon was deleted.
        threshold_group: Threshold group name, or None if not using
            threshold groups. If provided, uses minimum quantile across
            all conditions in the group.
    """
    cfg = source_config

    # Unpack opts
    threshold_method = opts["threshold_method"]
    ndimage_filter_type = opts.get("ndimage_filter_type")
    ndimage_filter_kwargs = opts.get("ndimage_filter_kwargs")

    # -------------------- Step 1: Load trace data into memory --------------------
    # Morphological uses per-bin thresholds; bin boundaries come from the
    # zarr's native dask chunks and must be captured *before*
    # ``.compute()`` (which clears ``da.chunks``). Tom-Bugnon stays on
    # whole-recording thresholds and ignores ``bin_boundaries``.
    da_lazy = cfg.open_traces_as_xarray(
        subject,
        probe,
        structure,
        condition,
        apply_detection_channel_mask=True,
        ndimage_filter_type=ndimage_filter_type,
        ndimage_filter_kwargs=ndimage_filter_kwargs,
    )
    if cfg.variant == "morphological":
        from cnpix_local_sleep.morphological.mua.readers import bin_boundaries_from_chunks

        bin_boundaries = bin_boundaries_from_chunks(da_lazy)
    else:
        bin_boundaries = None
    da = da_lazy.compute()
    del da_lazy

    # -------------------- Step 2: Compute channel thresholds --------------------
    # Get each unique condition in the current contrast
    if threshold_group is None:
        unique_conditions = [condition]
    else:
        contrast_conditions = const.CONTRASTS[threshold_group]
        assert condition in contrast_conditions, (
            "Condition must be part of contrast"
        )
        unique_conditions = list(contrast_conditions)

    # Get quantile from each of the conditions in the current contrast
    quantile_thresholds = [
        _get_quantile_threshold_for_condition(
            subject, probe, structure, cond, threshold_method, cfg
        )
        for cond in unique_conditions
    ]

    # Use the lowest quantile across all conditions in contrast
    quantile_threshold = np.min(quantile_thresholds)

    if cfg.variant == "morphological":
        assert bin_boundaries is not None, (
            "morphological must capture bin_boundaries before .compute()"
        )
        channel_thresholds = compute_per_bin_thresholds(
            da,
            quantile_threshold,
            bin_boundaries,
            threshold_method=threshold_method,
            ndimage_filter_type=ndimage_filter_type,
            ndimage_filter_kwargs=dict(ndimage_filter_kwargs)
            if ndimage_filter_kwargs is not None
            else None,
        )
    else:
        channel_thresholds = get_channel_threshold_values_from_quantile(
            da,
            quantile_threshold,
            threshold_method,
            ndimage_filter_type,
            dict(ndimage_filter_kwargs)
            if ndimage_filter_kwargs is not None
            else None,
        )

    # -------------------- Step 3: Detect OFF periods --------------------
    if not len(da.channel):
        # If there are no valid detection channels, return empty results
        lbl_ixs = {}
        offs = pd.DataFrame(columns=list(Off.__annotations__.keys()))
    else:
        # Otherwise, detect
        offs, _, lbl_ixs = morphology.detect_offs(
            da,
            channel_thresholds,
            do_clean_binary_mask=opts.get("clean_binary_mask", True),
            n_samples_connect=opts["n_samples_connect"],
            n_samples_clean=opts["n_samples_clean"],
            n_channels_clean=opts["n_channels_clean"],
            n_channels_connect=opts["n_channels_connect"],
        )

    # NB: The convention here is that the span (and max span) are based
    # on actual y-coordinates (in microns), not channel counts. So if 4
    # channels were used for detection, you cannot assume a span of 80
    # microns, because those channels may not be contiguous in y (if bad
    # channels were dropped).
    offs["max_span"] = da.y.max().item() - da.y.min().item()

    # Step 3b: Compute per-compartment areas (cortical structures only)
    add_laminar_areas(offs, da, lbl_ixs, subject, probe, structure)

    # Step 4: Write OFF detection results to disk
    lbls_path = cfg.files_module.get_off_label_indices_path(
        subject=subject,
        probe=probe,
        structure=structure,
        condition=condition,
        threshold_group=threshold_group,
    )
    offs_path = cfg.files_module.get_offs_path(
        subject=subject,
        probe=probe,
        structure=structure,
        condition=condition,
        threshold_group=threshold_group,
    )

    # Save label indices as parquet with nested arrays
    if lbl_ixs:
        lbl_ixs_df = pd.DataFrame(
            [
                {
                    "label": label,
                    "time_ixs": time_ixs.tolist(),
                    "chan_ixs": chan_ixs.tolist(),
                }
                for label, (time_ixs, chan_ixs) in lbl_ixs.items()
            ]
        )
    else:
        # Empty DataFrame with correct schema
        lbl_ixs_df = pd.DataFrame(
            columns=["label", "time_ixs", "chan_ixs"]
        )

    # Ensure parent directory exists
    lbls_path.parent.mkdir(parents=True, exist_ok=True)
    offs_path.parent.mkdir(parents=True, exist_ok=True)

    # Write OFFs to parquet
    lbl_ixs_df.to_parquet(lbls_path, index=False)
    offs.to_parquet(offs_path, index=False)

    # -------------------- Step 5: Write channel thresholds to disk --------------------
    channel_thresholds_filepath = cfg.files_module.get_channel_thresholds_path(
        subject=subject,
        probe=probe,
        structure=structure,
        condition=condition,
        threshold_group=threshold_group,
    )
    channel_thresholds.to_zarr(channel_thresholds_filepath, mode="w")


# -------------------- Structure-level orchestration --------------------


def do_structure(
    subject: str,
    probe: str,
    structure: str,
    opts: DetectionOpts,
    source_config: MorphologicalSourceConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Run OFF detection for all conditions in a structure.

    Args:
        subject: Subject identifier
        probe: Probe identifier
        structure: Brain structure name
        opts: Detection algorithm configuration. See ``DetectionOpts``.
        source_config: Variant plumbing. See ``detect_offs()``.
        overwrite: Whether to overwrite existing results
    """
    cfg = source_config
    threshold_method = opts["threshold_method"]
    for condition in const.CORE_CONDITIONS:
        threshold_groups = utils.get_threshold_groups_to_run(
            condition, threshold_method
        )
        for threshold_group in [None] + threshold_groups:
            run_detect = overwrite or not utils.detection_outputs_exist(
                subject,
                probe,
                structure,
                condition,
                threshold_group=threshold_group,
                files_module=cfg.files_module,
            )
            if run_detect:
                utils.log_step(
                    "Detecting OFFs",
                    condition=condition,
                    threshold_group=threshold_group,
                )
                detect_offs(
                    subject,
                    probe,
                    structure,
                    condition,
                    opts,
                    threshold_group=threshold_group,
                    source_config=cfg,
                )