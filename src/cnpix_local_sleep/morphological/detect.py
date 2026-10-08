"""Shared kernels of the morphological detector.

Per-bin quantile thresholds and the laminar (supra/infra) area columns, used by
``detect_full`` and ``laminar_null``.
"""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from cnpix_local_sleep import atlas
from cnpix_local_sleep import channel_anatomy


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
