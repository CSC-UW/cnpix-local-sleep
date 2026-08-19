"""Trace I/O: open, scale and smooth the zarr/xarray recordings.

Covers the SpikeInterface zarr openers, the µV scaling, the dask-image smoothing, the
LFP opener, and :func:`open_preprocessed_traces_as_xarray`, the reader for the
annotation grid every manual OFF label is pinned to (see
:func:`cnpix_local_sleep.files.get_preprocessed_ap_path`).
"""

from __future__ import annotations

import warnings
from typing import Literal

import dask.array
import ecephys.utils
import numpy as np
import spikeinterface as si
import wisc_ecephys_tools as wet
import xarray as xr
import zarr

from cnpix_local_sleep import const, files as _op_files, hyp

import ecephys.wne.utils

from cnpix_local_sleep import channel_anatomy

def smooth_dask_image(
    image: dask.array.Array,
    ndimage_filter_type: Literal["median", "gaussian"],
    ndimage_filter_kwargs: dict | None = None,
):
    import dask_image.ndfilters

    func = {
        "median": dask_image.ndfilters.median_filter,
        "gaussian": dask_image.ndfilters.gaussian_filter,
    }[ndimage_filter_type]
    if ndimage_filter_kwargs is None:
        ndimage_filter_kwargs = {}
    return func(image, **ndimage_filter_kwargs)

def get_increasing_segments_mask(times):
    keep = np.ones((len(times),), dtype=bool)
    gap_ixs = np.where(np.diff(times) < 0)[0]
    for gap_ix in gap_ixs:
        pre_gap_values = times[gap_ix]
        next_ix = np.where(times[gap_ix:] > pre_gap_values)[0][0]
        keep[gap_ix + 1 : gap_ix + next_ix] = False
    return keep

def open_si_zarr_recording_as_xarray(fpath, hotfix_times: bool = True) -> xr.DataArray:
    """Load SI-saved zarr as dask-based xarray for OFF detection.

    Uses ``spikeinterface.load()`` to extract recording metadata (channel
    names, locations, gains, offsets, timestamps) and ``dask.array.from_zarr``
    for lazy trace loading.

    Args:
        fpath: Path to the zarr directory containing preprocessed data.
        hotfix_times: If True, removes non-monotonically increasing timestamps
            (default: True).

    Returns:
        xr.DataArray with dimensions (time, channel) and coordinates:
        - time: Timestamps in seconds
        - channel: Channel names
        - x: Lateral position of each channel (in µm)
        - y: Depth of each channel (in µm)
        - gain_to_uV: Per-channel gain (present only if the recording has it)
        - offset_to_uV: Per-channel offset (present only if the recording
          has it)
        Additional attributes:
        - fs: Sampling frequency in Hz
        - dtype: Original trace dtype as a string
        - has_scaleable_traces: Whether gain and offset are available

    Note:
        Non-monotonically increasing timestamps are dismissed when
        hotfix_times=True.  This handles cases where timestamps may have
        discontinuities or resets (e.g. all-zero ``times_seg0`` in v1 zarr
        files).
    """
    recording = si.load(fpath)
    fs = recording.get_sampling_frequency()
    n_samples = recording.get_num_samples()

    # Channel names: the property key varies between SI versions.
    prop_keys = recording.get_property_keys()
    channel_name_key = (
        "channel_name" if "channel_name" in prop_keys else "channel_names"
    )
    channel_names = recording.get_property(channel_name_key)

    # Channel locations.
    locations = recording.get_channel_locations()
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]

    # Gains and offsets (may be None for preprocessed float32 data).
    gains = recording.get_channel_gains()
    offsets = recording.get_channel_offsets()

    # Timestamps: SI handles times_seg0 / t_starts / arange fallback.
    # Still need broken-times check for v1 zarr files (all-zero times_seg0).
    times = recording.get_times(segment_index=0)
    if len(times) > 1 and times[0] == times[-1]:
        times = np.arange(n_samples, dtype=np.float64) / fs

    # Trace data via dask for lazy loading.
    zg = zarr.open(fpath)
    traces = dask.array.from_zarr(zg.traces_seg0).rechunk()

    coords: dict = {
        "time": times,
        "channel": channel_names,
        "x": ("channel", x_coords),
        "y": ("channel", y_coords),
    }
    if gains is not None:
        coords["gain_to_uV"] = ("channel", gains)
    if offsets is not None:
        coords["offset_to_uV"] = ("channel", offsets)

    da = xr.DataArray(
        data=traces,
        dims=("time", "channel"),
        coords=coords,
        attrs={
            "fs": fs,
            "dtype": str(recording.get_dtype()),
            "has_scaleable_traces": recording.has_scaleable_traces(),
        },
    )

    if hotfix_times:
        keep = get_increasing_segments_mask(da.time.data)
        da = da.sel(time=keep).copy()

    return da

def scale_to_uV(da: xr.DataArray) -> xr.DataArray:
    """Scale traces to microvolts using stored gain and offset.

    Applies the transformation ``traces_uV = traces * gain_to_uV +
    offset_to_uV``, matching the scaling behavior of SpikeInterface's
    ``recording.get_traces(return_in_uV=True)``.

    If the required coordinates are missing or the scaling is trivial
    (all gains are 1 and all offsets are 0), a warning is logged and
    the data is returned unchanged.

    Args:
        da: DataArray returned by ``open_si_zarr_recording_as_xarray``.
            Expected to have ``gain_to_uV`` and ``offset_to_uV``
            coordinates on the ``channel`` dimension.

    Returns:
        Scaled DataArray with the same coordinates and dimensions.
    """
    if "gain_to_uV" not in da.coords or "offset_to_uV" not in da.coords:
        warnings.warn(
            "DataArray does not have gain_to_uV and offset_to_uV coordinates. "
            "Returning data unchanged.",
            stacklevel=2,
        )
        return da

    gains = da.gain_to_uV
    offsets = da.offset_to_uV
    if np.all(gains.values == 1.0) and np.all(offsets.values == 0.0):
        warnings.warn(
            "Gains are all 1.0 and offsets are all 0.0; scaling is trivial. "
            "Returning data unchanged.",
            stacklevel=2,
        )
        return da

    return da * gains + offsets

def open_preprocessed_traces_as_xarray(
    subject: str,
    probe: str,
    structure: str | None = None,
    condition: str | None = None,
    layer: str | None = None,
    apply_detection_channel_mask: bool = True,
    ndimage_filter_type: str | None = None,
    ndimage_filter_kwargs: dict | None = None,
) -> xr.DataArray:
    """Open legacy preprocessed AP-band data (Tom-Bugnon) as a lazy xarray.

    Reads ``processed_ap.zarr`` (300 Hz) from
    ``<offproj>/<subject>/method=tom-bugnon/probe=<probe>/processed_ap.zarr``
    and applies optional structure/channel filtering + detection-time
    ndimage filtering.

    .. deprecated::
        For new *detection* runs, use
        ``cnpix_local_sleep.morphological.mua.readers.open_mua_traces_as_xarray()`` instead,
        which loads MUA traces (500 Hz) with equivalent smoothing.

        This function outlived the tom detection pipeline (deleted 2026-08-11):
        the napari annotation stacks and every manual OFF label are pinned to
        the y-coordinates of this grid, so Table 1's scoring needs it. It
        retires only when the stacks are re-rendered from ``mua_traces.zarr``.

    Args:
        subject: Subject identifier
        probe: Probe identifier
        structure: Optional structure acronym to filter channels by
        condition: Optional condition name to filter timepoints by
        layer: Optional layer name ('supra' or 'infra') to filter channels by
            (only used if apply_detection_channel_mask is True and structure
            is not None)
        apply_detection_channel_mask: Whether to apply the detection channel mask
            after structure filtering (only if structure is not None)
        ndimage_filter_type: Optional filtering type ('median' or 'gaussian') to
            apply to the data
        ndimage_filter_kwargs: Optional keyword arguments for the filtering function

    Returns:
        The preprocessed data as an xarray DataArray.
        The DataArray has dimensions (time, channel) with additional coordinates:
        - y: Depth of each channel
        - struct: Structure acronym of each channel
    """
    path = _op_files.get_preprocessed_ap_path(subject=subject, probe=probe)
    if not path.exists():
        raise FileNotFoundError(
            f"No processed recording for {subject}, probe {probe} at {path}"
        )

    da = open_si_zarr_recording_as_xarray(path)

    structures = channel_anatomy.load_structures(subject, probe)
    da = channel_anatomy.assign_structures(da, structures)

    if structure is not None:
        da = da.sel(channel=(da["struct"] == structure))
        if apply_detection_channel_mask:
            keep_chans = channel_anatomy.compute_channel_mask(
                da.y, subject, probe, structure, layer=layer
            )
            da = da.sel(channel=keep_chans)
            # TODO: channel_anatomy.load_structures() above uses s3 structures but
            # compute_channel_mask() uses detection borders, so the former must be a
            # superset of the latter. Not explicitly checked.

    if ndimage_filter_type is not None:
        da.data = smooth_dask_image(
            da.data, ndimage_filter_type, ndimage_filter_kwargs
        )

    if condition is None:
        return da

    hg = hyp.load_statistical_condition_hypnograms(subject, probe)[condition]
    mask = hg.covers_time(da.time)
    return da.sel(time=mask)


def open_lfps(
    subject: str, probe: str, drop_bad_channels: bool = True, **kwargs
) -> xr.DataArray:
    lfp = ecephys.wne.utils.open_lfps(
        wet.get_sglx_project("shared_nobak"),
        subject,
        const.EXPERIMENT,
        probe,
        anatomy_proj=wet.get_sglx_project("shared"),
        **kwargs,
    )
    if drop_bad_channels:
        params = wet.get_sglx_project("shared").load_experiment_subject_params(
            const.EXPERIMENT, subject
        )
        bad_channels = params["probes"][probe]["badChannels"]
        lfp = lfp.drop_sel({"channel": bad_channels})
    return lfp
