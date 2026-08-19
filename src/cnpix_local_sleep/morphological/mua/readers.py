"""Readers for loading trace data for morphological OFF detection."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cnpix_local_sleep import hyp
from cnpix.mua import files as mua_files
from cnpix_local_sleep import channel_anatomy
from cnpix_local_sleep import trace_io


def open_mua_traces_as_xarray(
    subject: str,
    probe: str,
    structure: str | None = None,
    condition: str | None = None,
    layer: str | None = None,
    apply_detection_channel_mask: bool = True,
    gaussian_freq_max: float = 20.0,
    ndimage_filter_type: str | None = None,
    ndimage_filter_kwargs: dict | None = None,
) -> xr.DataArray:
    """Open MUA traces for morphological threshold detection.

    Loads pre-saved MUA traces (typically 500Hz), applies Gaussian
    smoothing equivalent to the Bugnon preprocessing chain
    (``freq_max=20Hz``), and optionally applies structure/channel
    filtering and detection-time ndimage filtering.

    This replaces ``open_preprocessed_traces_as_xarray()`` for new
    detection runs, avoiding the need to maintain separate preprocessed
    AP zarr files.

    Args:
        subject: Subject identifier.
        probe: Probe identifier.
        structure: Optional structure acronym to filter channels by.
        condition: Optional condition name to filter timepoints by.
        layer: Optional layer name ('supra' or 'infra') to filter
            channels by (only used if ``apply_detection_channel_mask``
            is True and ``structure`` is not None).
        apply_detection_channel_mask: Whether to apply the detection
            channel mask after structure filtering (only if
            ``structure`` is not None).
        gaussian_freq_max: Cutoff frequency (Hz) for the Gaussian
            low-pass smoothing applied to the MUA envelope. Matches
            the Bugnon preprocessing ``sp.gaussian_filter(freq_max=20)``.
        ndimage_filter_type: Optional filtering type ('median' or
            'gaussian') applied at detection time.
        ndimage_filter_kwargs: Optional keyword arguments for the
            detection-time filter.

    Returns:
        Dask-backed xarray DataArray with dimensions (time, channel)
        and coordinates y (depth), struct (structure acronym).
    """
    path = mua_files.get_mua_traces_path(subject, probe)
    if not path.exists():
        raise FileNotFoundError(
            f"MUA traces not found for {subject}, probe {probe} at "
            f"{path}. Run `cnpix-mua write-mua-traces {subject} {probe}` first."
        )

    da = trace_io.open_si_zarr_recording_as_xarray(path)
    da = trace_io.scale_to_uV(da)

    # Apply Gaussian smoothing equivalent to Bugnon preprocessing.
    # sigma = fs / (2 * pi * freq_max), computed from the recording's
    # own sampling rate so it adapts if the MUA rate changes.
    if gaussian_freq_max is not None:
        fs = da.attrs["fs"]
        sigma_samples = fs / (2 * np.pi * gaussian_freq_max)
        da.data = trace_io.smooth_dask_image(
            da.data, "gaussian", {"sigma": [sigma_samples, 0]}
        )

    # Annotate with structure information.
    structures = channel_anatomy.load_structures(subject, probe)
    da = channel_anatomy.assign_structures(da, structures)

    # Filter to requested structure and apply channel mask.
    if structure is not None:
        da = da.sel(channel=(da["struct"] == structure))
        if apply_detection_channel_mask:
            keep_chans = channel_anatomy.compute_channel_mask(
                da.y, subject, probe, structure, layer=layer
            )
            da = da.sel(channel=keep_chans)

    # Apply detection-time ndimage filter (e.g., median).
    if ndimage_filter_type is not None:
        da.data = trace_io.smooth_dask_image(
            da.data, ndimage_filter_type, ndimage_filter_kwargs
        )

    # Filter to condition timepoints.
    if condition is None:
        return da

    hg = hyp.load_statistical_condition_hypnograms(
        subject,
        probe,
    )[condition]
    mask = hg.covers_time(da.time)
    return da.sel(time=mask)


# Canonical name used by ``MorphologicalSourceConfig.open_traces_as_xarray``.
open_traces_as_xarray = open_mua_traces_as_xarray


def bin_boundaries_from_chunks(da: xr.DataArray) -> np.ndarray:
    """Return per-bin time-axis sample boundaries from a DataArray's dask chunks.

    Bins are defined as the zarr's native dask chunks (~5.5 min /
    165 000 samples for production MUA traces). Used to derive the
    bin partition for per-bin threshold fitting.

    Must be called *before* ``.compute()`` / ``.load()`` on ``da``:
    materializing replaces the dask-backed data with numpy and clears
    ``da.chunks``.

    Args:
        da: Dask-backed DataArray with dimension ``(time, channel)``.

    Returns:
        Array of shape ``(n_bins+1,)`` holding the cumulative sample
        index at each chunk boundary; ``boundaries[0] == 0`` and
        ``boundaries[-1] == n_time``. Bin ``i`` covers samples
        ``[boundaries[i], boundaries[i+1])``.

    Raises:
        TypeError: If ``da`` has been materialized (``da.chunks is None``).
    """
    if da.chunks is None:
        raise TypeError(
            "bin_boundaries_from_chunks requires a dask-backed "
            "DataArray. Call this before da.compute() / da.load()."
        )
    sizes = np.asarray(da.chunks[0], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(sizes)])


__all__ = [
    "open_mua_traces_as_xarray",
    "open_traces_as_xarray",
    "bin_boundaries_from_chunks",
]
