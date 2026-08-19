"""Write image stacks for OFF period annotation and SAM3 training.

This module generates image stacks from AP-band, LFP, and spike data for use
in napari annotation and SAM3 instance segmentation model training/inference.

Supports output in PNG/JPEG image format or OME-Zarr v0.4 format.
"""

import shutil
from io import BytesIO
from pathlib import Path
from typing import Literal

import ecephys.utils as eu
import ecephys.wne.utils as wne_utils
import ecephys.xrsig.core as xrc
import ecephys.xrsig.plt as xrp
import matplotlib.pyplot as plt
import numba

import numpy as np
import numpy.typing as npt
import wisc_ecephys_tools as wet
import xarray as xr
import zarr
from ecephys import plot as eplt
from ecephys import units
from numba.typed import Dict as NumbaDict
from ome_zarr.format import FormatV04
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from PIL import Image

import cnpix_local_sleep as op
import cnpix_local_sleep.hyp as oph
import cnpix_local_sleep.stacks.files as stk_files
import cnpix_local_sleep.units as opu
from cnpix_local_sleep import channel_anatomy
from cnpix_local_sleep import trace_io


def _open_ap_for_stacks(
    subject: str,
    probe: str,
    condition: str,
    structure: str | None = None,
    layer: str | None = None,
    apply_detection_channel_mask: bool = True,
) -> xr.DataArray:
    """Open preprocessed AP data for writing image stacks.

    This mirrors `cnpix_local_sleep.trace_io.open_preprocessed_traces_as_xarray` (which grew
    out of the same pre-migration helper).
    We don't need this if we're just reading v1 preprocessed data.

    Parameters
    ----------
    subject
        Subject identifier (e.g. "CNPIX12-Santiago").
    probe
        Probe identifier (e.g. "imec0").
    condition
        Experimental condition (e.g. "Early.REC.NREM").
    structure
        Brain structure acronym. If None, uses all channels.
    layer
        Cortical layer specification ("supra", "infra", or None).
    apply_detection_channel_mask
        Whether to restrict to detection channel mask.

    Returns
    -------
    xr.DataArray
        Preprocessed AP data with structure annotations.
    """
    # Open zarr, convert to dask+xarray.
    pp_path = op.files.get_preprocessed_ap_path(
        subject=subject, probe=probe, condition=condition, style="v3"
    )
    da = trace_io.open_si_zarr_recording_as_xarray(pp_path)

    # Annotate the data with anatomical structure information.
    s3 = wet.get_sglx_project("shared")
    structures_path = s3.get_experiment_subject_file(
        op.EXPERIMENT, subject, f"{probe}.structures.htsv"
    )
    structures = eu.read_htsv(structures_path)
    da = channel_anatomy.assign_structures(da, structures)

    # Subselect region of interest.
    if structure is not None:
        da = da.sel(channel=(da["struct"] == structure))

    # Further restrict to old-school detection borders.
    if structure is not None and apply_detection_channel_mask:
        keep_chans = channel_anatomy.compute_channel_mask(
            da.y, subject, probe, structure, layer=layer
        )
        da = da.sel(channel=keep_chans)

    return da


@numba.njit(cache=True)
def _bin_spike_train(
    spike_times: npt.NDArray[np.float64], bin_edges: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Bin spike times according to bin edges.

    Parameters
    ----------
    spike_times
        Monotonically increasing array of spike times.
    bin_edges
        Monotonically increasing array of bin edges.

    Returns
    -------
    npt.NDArray[np.int64]
        Number of spikes in each bin.
    """
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)

    i_spike = 0
    i_bin = 0

    # Early exit for empty arrays
    if len(spike_times) == 0 or len(bin_edges) <= 1:
        return counts

    # Iterate through spikes and bins simultaneously
    while i_spike < len(spike_times) and i_bin < len(bin_edges) - 1:
        if spike_times[i_spike] < bin_edges[i_bin]:
            i_spike += 1
        elif spike_times[i_spike] >= bin_edges[i_bin + 1]:
            i_bin += 1
        else:
            # Spike is in the current bin
            counts[i_bin] += 1
            i_spike += 1

    return counts


@numba.njit(parallel=True, cache=True)
def _bin_spike_trains(
    trains: numba.typed.Dict, bin_edges: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int64], list]:
    """Bin multiple spike trains according to bin edges.

    Parameters
    ----------
    trains
        Numba typed dictionary of spike trains, where keys are int64 identifiers
        and values are arrays of spike times.
    bin_edges
        Monotonically increasing array of bin edges.

    Returns
    -------
    tuple[npt.NDArray[np.int64], list]
        Tuple of (binned spike counts array, list of train IDs).
    """
    train_ids = list(trains.keys())
    binned = np.zeros(shape=(len(train_ids), len(bin_edges) - 1), dtype=np.int64)
    for i_unit in numba.prange(len(train_ids)):
        id = train_ids[i_unit]
        unit_spike_times = trains[id]
        binned[i_unit, :] = _bin_spike_train(unit_spike_times, bin_edges)

    return binned, train_ids


def bin_spike_trains(
    trains: units.dtypes.SpikeTrainDict_Secs,
    bin_edges: npt.NDArray[np.float64],
    train_keys: str = "spike_train",
) -> xr.DataArray:
    """Bin spike trains into time bins.

    Parameters
    ----------
    trains
        Dictionary mapping unit identifiers to spike time arrays.
    bin_edges
        Monotonically increasing array of bin edges (in seconds).
    train_keys
        Name for the dimension containing train identifiers.

    Returns
    -------
    xr.DataArray
        Binned spike counts with dimensions (train_keys, time).
    """
    # Create a numba-friendly dictionary of spike trains that uses int64 keys.
    # We will convert back to the original keys before returning the data.
    train_ids = np.array(list(trains.keys()))
    _trains = NumbaDict.empty(numba.types.int64, numba.types.float64[:])
    for numba_id, train_id in enumerate(train_ids):
        _trains[numba_id] = trains[train_id]

    # Bin the spike trains
    binned, _ids = _bin_spike_trains(_trains, bin_edges)

    # Create DataArray for convenience
    binned = xr.DataArray(
        binned,
        dims=(train_keys, "time"),
        coords={train_keys: train_ids[np.array(_ids)], "time": bin_edges[:-1]},
        name="spikes",
    )
    return binned


# -------------------- OME-Zarr stack preparation functions --------------------


def _minmax_normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 range [0, 255] using min-max scaling.

    Parameters
    ----------
    data
        Input array of any numeric dtype.

    Returns
    -------
    np.ndarray
        Uint8 array with same shape, values scaled to [0, 255].
    """
    dmin = data.min()
    dmax = data.max()
    if dmax > dmin:
        return ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        return np.zeros_like(data, dtype=np.uint8)


def _render_ap_stack(
    ap: xr.DataArray,
    chunk_ixs: npt.NDArray,
    img_width_pixels: int,
    img_height_pixels: int,
    pixels_per_channel: int,
) -> np.ndarray:
    """Render AP data as 3D grayscale stack (t, y, x) for OME-Zarr.

    Directly converts AP data to uint8 using per-chunk min-max normalization,
    without matplotlib rendering. Each channel is replicated vertically
    by pixels_per_channel to match expected image dimensions.

    Parameters
    ----------
    ap
        AP-band data with dims (time, channel).
    chunk_ixs
        Array of chunk start indices.
    img_width_pixels
        Expected image width in pixels (should equal samples_per_chunk).
    img_height_pixels
        Expected image height in pixels (should equal n_channels * pixels_per_channel).
    pixels_per_channel
        Number of vertical pixels per channel for replication.

    Returns
    -------
    np.ndarray
        Uint8 array with shape (n_chunks, height, width).
    """
    n_chunks = len(chunk_ixs) - 1
    n_channels = ap.sizes["channel"]

    assert img_height_pixels == n_channels * pixels_per_channel, (
        f"img_height_pixels ({img_height_pixels}) != "
        f"n_channels ({n_channels}) * pixels_per_channel ({pixels_per_channel})"
    )

    # Preallocate output array
    stack = np.zeros((n_chunks, img_height_pixels, img_width_pixels), dtype=np.uint8)

    for i in range(n_chunks):
        ap_snippet = ap.isel(time=slice(chunk_ixs[i], chunk_ixs[i + 1])).compute()
        data = ap_snippet.values  # Shape: (time, channel)

        # Per-chunk min-max normalize to 0-255 range
        normalized = _minmax_normalize_to_uint8(data)

        # Transpose to (channel, time) for image layout
        img = normalized.T  # Shape: (channel, time)

        # Flip vertically so lower y values are at bottom (matching matplotlib convention)
        img = img[::-1, :]

        # Replicate each channel row by pixels_per_channel
        img_expanded = np.repeat(img, pixels_per_channel, axis=0)

        stack[i] = img_expanded

    return stack


def _render_spike_stack(
    spikes: xr.DataArray,
    chunk_ixs: npt.NDArray,
    img_width_pixels: int,
    img_height_pixels: int,
    pixels_per_channel: int,
) -> np.ndarray:
    """Render spike data as 3D binary stack (t, y, x) for OME-Zarr.

    Directly converts spike counts to binary uint8 (0 or 255),
    without matplotlib rendering. Each y-position is replicated vertically
    by pixels_per_channel to match expected image dimensions.

    Parameters
    ----------
    spikes
        Binned spike data with dims (y, time). Values are spike counts.
    chunk_ixs
        Array of chunk start indices.
    img_width_pixels
        Expected image width in pixels (should equal samples_per_chunk).
    img_height_pixels
        Expected image height in pixels (should equal n_y * pixels_per_channel).
    pixels_per_channel
        Number of vertical pixels per y-position for replication.

    Returns
    -------
    np.ndarray
        Uint8 array with shape (n_chunks, height, width).
        Values are 0 (no spike) or 255 (spike present).
    """
    n_chunks = len(chunk_ixs) - 1
    n_y = spikes.sizes["y"]

    assert img_height_pixels == n_y * pixels_per_channel, (
        f"img_height_pixels ({img_height_pixels}) != "
        f"n_y ({n_y}) * pixels_per_channel ({pixels_per_channel})"
    )

    # Preallocate output array
    stack = np.zeros((n_chunks, img_height_pixels, img_width_pixels), dtype=np.uint8)

    for i in range(n_chunks):
        sp_snippet = spikes.isel(time=slice(chunk_ixs[i], chunk_ixs[i + 1]))
        data = sp_snippet.values  # Shape: (y, time)

        # Convert to binary: any spike count > 0 becomes 255
        binary = (data > 0).astype(np.uint8) * 255

        # Flip vertically so lower y values are at bottom
        binary = binary[::-1, :]

        # Replicate each y-position row by pixels_per_channel
        binary_expanded = np.repeat(binary, pixels_per_channel, axis=0)

        stack[i] = binary_expanded

    return stack


def _fig_to_rgba_array(fig: plt.Figure, dpi: int) -> np.ndarray:
    """Convert matplotlib figure to RGBA numpy array.

    Parameters
    ----------
    fig
        Matplotlib figure to convert.
    dpi
        Dots per inch for rendering.

    Returns
    -------
    np.ndarray
        Uint8 array with shape (height, width, 4) in RGBA format.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    return rgba


def _render_lf_stack(
    lf: xr.DataArray,
    chunk_times: npt.NDArray,
    img_width_inches: float,
    img_height_inches: float,
    img_width_pixels: int,
    img_height_pixels: int,
    dpi: int,
    channel_step: int = 10,
    vspace: int = 500,
) -> np.ndarray:
    """Render LFP data as 3D binary stack (t, y, x) for OME-Zarr.

    Renders LFP traces using matplotlib and captures as binary uint8.
    This produces images at the exact same dimensions as the AP output,
    suitable for overlaying in napari.

    Parameters
    ----------
    lf
        LFP data with dims (time, channel).
    chunk_times
        Array of chunk start times.
    img_width_inches
        Figure width in inches.
    img_height_inches
        Figure height in inches.
    img_width_pixels
        Expected image width in pixels.
    img_height_pixels
        Expected image height in pixels.
    dpi
        Dots per inch.
    channel_step
        Step size for subsampling channels.
    vspace
        Vertical spacing between traces.

    Returns
    -------
    np.ndarray
        Uint8 array with shape (n_chunks, height, width).
        Values are 0 (transparent/no trace) or 255 (trace present).
    """
    n_chunks = len(chunk_times) - 1

    # Preallocate output array
    stack = np.zeros((n_chunks, img_height_pixels, img_width_pixels), dtype=np.uint8)

    for i in range(n_chunks):
        lf_snippet = lf.sel(time=slice(chunk_times[i], chunk_times[i + 1])).compute()

        fig, ax = plt.subplots(
            figsize=(img_width_inches, img_height_inches), dpi=dpi, facecolor="none"
        )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        eplt.lfp_explorer(
            np.arange(lf_snippet["time"].size),
            lf_snippet[:, ::channel_step].values,
            ax,
            vspace=vspace,
            zero_mean=False,
            flip_dv=True,
            tight_ylim=True,
            linewidth=0.05,
        )
        ax.axis("off")

        # Convert to binary uint8: use alpha channel to detect trace lines
        rgba = _fig_to_rgba_array(fig, dpi)
        alpha = rgba[:, :, 3]
        # Handle potential size mismatch due to matplotlib rounding
        h, w = alpha.shape
        h_out = min(h, img_height_pixels)
        w_out = min(w, img_width_pixels)
        stack[i, :h_out, :w_out] = (alpha[:h_out, :w_out] > 0).astype(np.uint8) * 255

    return stack


def _render_structure_borders(
    ap: xr.DataArray,
    img_width_inches: float,
    img_height_inches: float,
    img_width_pixels: int,
    img_height_pixels: int,
    dpi: int,
) -> np.ndarray:
    """Render structure borders as 2D binary uint8 image.

    Parameters
    ----------
    ap
        AP data with structure annotations in 'struct' coordinate.
    img_width_inches
        Figure width in inches.
    img_height_inches
        Figure height in inches.
    img_width_pixels
        Expected image width in pixels.
    img_height_pixels
        Expected image height in pixels.
    dpi
        Dots per inch.

    Returns
    -------
    np.ndarray
        Uint8 array with shape (height, width).
        Values are 0 (no boundary) or 255 (boundary present).
    """
    fig, ax = plt.subplots(
        figsize=(img_width_inches, img_height_inches), dpi=dpi, facecolor="none"
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    xrp.add_structure_borders_to_laminar_plot(
        ap,
        ax,
        sigdim="channel",
        struct_coord="struct",
        lamdim="y",
        labels=False,
        line_kwargs=dict(color="r", linestyle="--", alpha=0.5, linewidth=0.1),
    )
    ax.axis("off")

    rgba = _fig_to_rgba_array(fig, dpi)
    # Convert to binary using alpha channel
    alpha = rgba[:, :, 3]
    # Handle potential size mismatch due to rounding
    h_out = min(alpha.shape[0], img_height_pixels)
    w_out = min(alpha.shape[1], img_width_pixels)
    binary = np.zeros((img_height_pixels, img_width_pixels), dtype=np.uint8)
    binary[:h_out, :w_out] = (alpha[:h_out, :w_out] > 0).astype(np.uint8) * 255
    return binary


def _prepare_timestamps_stack(
    ap: xr.DataArray,
    chunk_ixs: npt.NDArray,
) -> np.ndarray:
    """Prepare timestamps as 2D array (n_chunks, samples_per_chunk).

    Parameters
    ----------
    ap
        AP-band data with 'time' coordinate.
    chunk_ixs
        Array of chunk start indices.

    Returns
    -------
    np.ndarray
        Float64 array with shape (n_chunks, samples_per_chunk).
    """
    n_chunks = len(chunk_ixs) - 1
    samples_per_chunk = chunk_ixs[1] - chunk_ixs[0]

    # Preallocate output array
    timestamps = np.zeros((n_chunks, samples_per_chunk), dtype=np.float64)

    time_values = ap["time"].values
    for i in range(n_chunks):
        timestamps[i] = time_values[chunk_ixs[i] : chunk_ixs[i + 1]]

    return timestamps


def _write_timestamps(
    savedir: Path,
    ap: xr.DataArray,
    chunk_ixs: npt.NDArray,
) -> Path:
    """Write timestamps as zarr array with shape (n_chunks, samples_per_chunk).

    This function is called for all output formats (images or OME-Zarr) to store
    the actual float64 timestamps for each AP sample in each chunk.

    Parameters
    ----------
    savedir
        Directory to save the timestamps zarr store.
    ap
        AP-band data with 'time' coordinate.
    chunk_ixs
        Array of chunk start indices.

    Returns
    -------
    Path
        Path to the created timestamps.zarr store.
    """
    timestamps = _prepare_timestamps_stack(ap, chunk_ixs)
    timestamps_path = savedir / "timestamps.zarr"

    # Write as simple zarr array
    z = zarr.open(
        str(timestamps_path),
        mode="w",
        shape=timestamps.shape,
        dtype=timestamps.dtype,
        chunks=(1, timestamps.shape[1]),  # One chunk per time slice
    )
    z[:] = timestamps

    # Add metadata
    z.attrs["description"] = "Timestamps for each AP sample in each chunk"
    z.attrs["units"] = "seconds"
    z.attrs["dims"] = ["chunk", "sample"]
    z.attrs["sampling_frequency"] = float(ap.fs)

    return timestamps_path


def write_ome_zarr_stacks(
    ap: xr.DataArray,
    lf: xr.DataArray,
    spikes: xr.DataArray,
    savedir: Path,
    chunk_ixs: npt.NDArray,
    chunk_times: npt.NDArray,
    img_width_inches: float,
    img_height_inches: float,
    img_width_pixels: int,
    img_height_pixels: int,
    dpi: int,
) -> Path:
    """Write image stacks to OME-Zarr v0.4 format.

    Creates an OME-Zarr store with:
    - Group 0: AP data (uint8, per-chunk min-max normalized)
    - Group 1: Spikes (binary uint8, 0/255)
    - Group 2: LFP (binary uint8, 0/255)
    - Group 3: Structure borders (binary uint8, 0/255)

    Parameters
    ----------
    ap
        AP-band data array with dims (time, channel).
    lf
        LFP data array with dims (time, channel).
    spikes
        Binned spike data array with dims (y, time).
    savedir
        Directory to save the OME-Zarr store.
    chunk_ixs
        Array of chunk start indices.
    chunk_times
        Array of chunk start times.
    img_width_inches
        Figure width in inches for rendered images.
    img_height_inches
        Figure height in inches for rendered images.
    img_width_pixels
        Image width in pixels.
    img_height_pixels
        Image height in pixels.
    dpi
        Dots per inch for rendered images.

    Returns
    -------
    Path
        Path to the created OME-Zarr store.
    """
    zarr_path = savedir / "off_stacks.ome.zarr"

    # Create the root zarr group with OME-Zarr v0.4 format
    store = parse_url(str(zarr_path), mode="w", fmt=FormatV04()).store
    root = zarr.group(store=store)

    # Compute pixels_per_channel for direct array conversion
    n_channels = ap.sizes["channel"]
    pixels_per_channel = img_height_pixels // n_channels
    assert img_height_pixels == n_channels * pixels_per_channel, (
        f"img_height_pixels ({img_height_pixels}) must be divisible by "
        f"n_channels ({n_channels})"
    )

    # Render AP stack using direct array conversion (no matplotlib)
    print("  Rendering AP stack...")
    ap_stack = _render_ap_stack(
        ap,
        chunk_ixs,
        img_width_pixels,
        img_height_pixels,
        pixels_per_channel,
    )

    # Render spike stack using direct binary conversion (no matplotlib)
    print("  Rendering spike stack...")
    spike_stack = _render_spike_stack(
        spikes,
        chunk_ixs,
        img_width_pixels,
        img_height_pixels,
        pixels_per_channel,
    )

    # Render LFP stack using matplotlib
    print("  Rendering LFP stack...")
    lf_stack = _render_lf_stack(
        lf,
        chunk_times,
        img_width_inches,
        img_height_inches,
        img_width_pixels,
        img_height_pixels,
        dpi,
    )

    print("  Rendering structure borders...")
    has_multiple_structures = not eu.all_equal(ap["struct"])
    if has_multiple_structures:
        borders = _render_structure_borders(
            ap,
            img_width_inches,
            img_height_inches,
            img_width_pixels,
            img_height_pixels,
            dpi,
        )
    else:
        borders = None

    # Write AP data (Group 0) - grayscale uint8
    print("  Writing AP to OME-Zarr...")
    ap_group = root.create_group("0")
    write_image(
        image=ap_stack,
        group=ap_group,
        axes=["t", "y", "x"],
        storage_options=dict(chunks=(1, ap_stack.shape[1], ap_stack.shape[2])),
        scaler=None,  # No multiscale pyramid
    )
    ap_group.attrs["name"] = "ap"
    ap_group.attrs["description"] = "AP-band data (uint8, per-chunk min-max normalized)"

    # Write spike data (Group 1) - binary uint8 (0 or 255)
    print("  Writing spikes to OME-Zarr...")
    spike_group = root.create_group("1")
    write_image(
        image=spike_stack,
        group=spike_group,
        axes=["t", "y", "x"],
        storage_options=dict(chunks=(1, spike_stack.shape[1], spike_stack.shape[2])),
        scaler=None,  # No multiscale pyramid
    )
    spike_group.attrs["name"] = "spikes"
    spike_group.attrs["description"] = "Binary spike presence (uint8, 0/255)"

    # Write LFP data (Group 2) - binary uint8 (0 or 255)
    print("  Writing LFP to OME-Zarr...")
    lf_group = root.create_group("2")
    write_image(
        image=lf_stack,
        group=lf_group,
        axes=["t", "y", "x"],
        storage_options=dict(chunks=(1, lf_stack.shape[1], lf_stack.shape[2])),
        scaler=None,  # No multiscale pyramid
    )
    lf_group.attrs["name"] = "lf"
    lf_group.attrs["description"] = "LFP traces (binary uint8, 0/255)"

    # Write structure borders (Group 3) if multiple structures
    # Binary uint8 with shape (y, x), same format as LFP and spikes
    if borders is not None:
        print("  Writing structure borders to OME-Zarr...")
        borders_group = root.create_group("3")
        write_image(
            image=borders,
            group=borders_group,
            axes=["y", "x"],
            storage_options=dict(chunks=(borders.shape[0], borders.shape[1])),
            scaler=None,  # No multiscale pyramid
        )
        borders_group.attrs["name"] = "structure_borders"
        borders_group.attrs["description"] = (
            "Structure boundary lines (binary uint8, 0/255)"
        )

    # Add root-level metadata
    root.attrs["offproj_version"] = "1.0"
    root.attrs["n_chunks"] = len(chunk_ixs) - 1
    root.attrs["samples_per_chunk"] = int(chunk_ixs[1] - chunk_ixs[0])
    root.attrs["sampling_frequency"] = float(ap.fs)
    root.attrs["n_channels"] = ap.sizes["channel"]

    print(f"  OME-Zarr written to {zarr_path}")
    return zarr_path


def load_data(
    subject: str,
    probe: str,
    condition: str,
    structure_acronym: str | None,
    ap_type: Literal["v1", "v3"] = "v1",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load AP, LFP, and spike data for stack generation.

    Parameters
    ----------
    subject
        Subject identifier.
    probe
        Probe identifier.
    condition
        Experimental condition.
    structure_acronym
        Brain structure acronym, or None for whole probe.
    ap_type
        AP data version to load ("v1" or "v3").

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        Tuple of (AP data, LFP data, binned spike data).
    """
    # Load AP data.
    apply_detection_channel_mask = False
    if ap_type == "v1":
        ap = trace_io.open_preprocessed_traces_as_xarray(
            subject,
            probe,
            condition=condition,
            structure=structure_acronym,
            apply_detection_channel_mask=apply_detection_channel_mask,
        )
    elif ap_type == "v3":
        ap = _open_ap_for_stacks(
            subject,
            probe,
            condition=condition,
            structure=structure_acronym,
            apply_detection_channel_mask=apply_detection_channel_mask,
        )
    else:
        raise ValueError(f"Unknown ap_type: {ap_type}")

    # Load LFP data.
    lf = wne_utils.open_lfps(
        wet.get_sglx_project("shared_nobak"),
        subject,
        op.EXPERIMENT,
        probe,
        hotfix_times=False,
        drop_duplicate_times=False,
        anatomy_proj=wet.get_sglx_project("shared"),
    )

    # Convert each AP channel name into an integer by extracting the number after "AP"
    channels = [int(ch.replace("AP", "")) for ch in ap["channel"].values]
    # Only load LFPs from the channels that were used in the AP detection
    lf = lf.sel({"channel": channels})

    # Select only the data from the time range of interest, plus extra for filter pad
    lf = lf.sel({"time": slice(ap["time"].min() - 10, ap["time"].max() + 10)})

    # A high-pass filter helps plotting
    lf = xrc.mne_filter(lf, 0.5, None).compute()

    # Select only the condition of interest
    # We do this after the filter to avoid edge effects
    # XXX: You are trusting that this is the same hypnogram used to select the AP data.
    hg = oph.load_statistical_condition_hypnograms(subject, probe)[condition]
    mask = hg.covers_time(lf["time"])
    lf = lf.sel({"time": mask})

    # Load spikes
    sorting = opu.load_sorting(subject, probe, unit_quality="all")
    spikes = sorting.get_trains_by_property(
        property_name="depth",
        values=ap["y"].values,
        return_times=True,
        start_time=ap["time"].min(),
        end_time=ap["time"].max(),
    )
    spikes = {k: np.unique(train) for k, train in sorted(spikes.items())}
    spikes = bin_spike_trains(spikes, ap["time"].values, train_keys="y")
    spikes = spikes.pad({"time": (0, 1)}, mode="constant", constant_values=0)

    return ap, lf, spikes


def get_stack_params(
    da: xr.DataArray,
    chunk_len_sec: float = 4,
    pixels_per_sample: int = 1,
    pixels_per_channel: int = 3,
    dpi: int = 300,
    verbose: bool = True,
) -> tuple[int, npt.NDArray, npt.NDArray, int, int, float, float]:
    """Calculate parameters for image stack generation.

    Parameters
    ----------
    da
        Data array with time and y dimensions.
    chunk_len_sec
        Duration of each chunk in seconds.
    pixels_per_sample
        Number of pixels per time sample.
    pixels_per_channel
        Number of pixels per channel.
    dpi
        Dots per inch for output images.
    verbose
        Whether to print parameter information.

    Returns
    -------
    tuple
        (chunk_len_samples, chunk_ixs, chunk_times, img_width_pixels,
         img_height_pixels, img_width_inches, img_height_inches)
    """
    assert np.all(np.diff(da["y"]) > 0), "y values are not sorted"

    chunk_len_samples = int(chunk_len_sec * da.fs)
    chunk_ixs = np.arange(0, da["time"].size, chunk_len_samples)
    chunk_times = da["time"].values[chunk_ixs]
    n_chunks = len(chunk_ixs) - 1
    if verbose:
        print(f"Samples per chunk: {chunk_len_samples}")
        print(f"Number of chunks: {n_chunks}")

    img_width_pixels = chunk_len_samples * pixels_per_sample
    img_height_pixels = pixels_per_channel * da["y"].size
    if verbose:
        print(f"Expected image size (pixels): {img_width_pixels} x {img_height_pixels}")
        print(f"Expected aspect ratio: {img_width_pixels / img_height_pixels:.2f}")

    img_width_inches = img_width_pixels / dpi
    img_height_inches = img_height_pixels / dpi
    if verbose:
        print(
            f"Expected image size (inches): {img_width_inches:.2f} x {img_height_inches:.2f}"
        )
    return (
        chunk_len_samples,
        chunk_ixs,
        chunk_times,
        img_width_pixels,
        img_height_pixels,
        img_width_inches,
        img_height_inches,
    )


def make_savedir(
    subject: str, probe: str, condition: str, structure_acronym: str | None
) -> Path:
    """Create and return the save directory for image stacks.

    Parameters
    ----------
    subject
        Subject identifier.
    probe
        Probe identifier.
    condition
        Experimental condition.
    structure_acronym
        Brain structure acronym, or None for whole probe.

    Returns
    -------
    Path
        Path to the save directory (``method=sam3/probe=<probe>/
        condition=<condition>``), matching where every stack reader
        (``cnpix_local_sleep.evaluation``, ``cnpix_local_sleep.stacks.files.get_sam3_*``) looks.
    """
    savedir = stk_files.get_sam3_savedir_path(
        subject, probe, condition, structure_acronym
    )
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)
    return savedir


def write_stacks(
    ap: xr.DataArray,
    lf: xr.DataArray,
    spikes: xr.DataArray,
    savedir: Path,
    chunk_ixs: npt.NDArray,
    chunk_times: npt.NDArray,
    dpi: int,
    max_chunks: int | None = None,
) -> None:
    """Write the OME-Zarr stack for AP, LFP and spike data.

    Parameters
    ----------
    ap
        AP-band data array.
    lf
        LFP data array.
    spikes
        Binned spike data array.
    savedir
        Directory to save images.
    chunk_ixs
        Array of chunk start indices.
    chunk_times
        Array of chunk start times.
    dpi
        Dots per inch. Sets the inches-per-pixel scale of the rendered stack.
    max_chunks
        Maximum number of chunks to process. If None, process all chunks.
        Useful for quick testing.
    """
    # Limit chunks if max_chunks is set
    if max_chunks is not None:
        # chunk_ixs has n_chunks+1 elements, chunk_times has n_chunks+1 elements
        # To get max_chunks chunks, we need max_chunks+1 boundary indices/times
        chunk_ixs = chunk_ixs[: max_chunks + 1]
        chunk_times = chunk_times[: max_chunks + 1]
        print(f"Limiting to {max_chunks} chunks for testing")

    # Always write timestamps (for all output formats)
    print("Writing timestamps...")
    _write_timestamps(savedir, ap, chunk_ixs)

    print("Writing OME-Zarr stacks...")
    # For OME-Zarr, always use 1 pixel per sample and 1 pixel per channel
    samples_per_chunk = chunk_ixs[1] - chunk_ixs[0]
    ome_zarr_width_pixels = samples_per_chunk
    ome_zarr_height_pixels = ap.sizes["channel"]
    ome_zarr_width_inches = ome_zarr_width_pixels / dpi
    ome_zarr_height_inches = ome_zarr_height_pixels / dpi
    write_ome_zarr_stacks(
        ap=ap,
        lf=lf,
        spikes=spikes,
        savedir=savedir,
        chunk_ixs=chunk_ixs,
        chunk_times=chunk_times,
        img_width_inches=ome_zarr_width_inches,
        img_height_inches=ome_zarr_height_inches,
        img_width_pixels=ome_zarr_width_pixels,
        img_height_pixels=ome_zarr_height_pixels,
        dpi=dpi,
    )


# TODO: Add option to omit LFP stack, since they can be slow to render. Add later?
def do_subject_probe(
    subject: str,
    probe: str,
    condition: str,
    structure_acronym: str | None = None,
    ap_type: Literal["v1", "v3"] = "v1",
    chunk_len_sec: float = 4,
    pixels_per_sample: int = 1,
    pixels_per_channel: int = 3,
    dpi: int = 300,
    max_chunks: int | None = None,
    overwrite: bool = False,
) -> None:
    """Generate the OME-Zarr stack for a subject/probe/condition.

    This is the main entry point for generating stacks for annotation
    and SAM3 training.

    Parameters
    ----------
    subject
        Subject identifier (e.g. "CNPIX12-Santiago").
    probe
        Probe identifier (e.g. "imec0").
    condition
        Experimental condition (e.g. "Early.REC.NREM").
    structure_acronym
        Brain structure acronym, or None for whole probe.
    ap_type
        AP data version to load ("v1" or "v3").
    chunk_len_sec
        Duration of each chunk in seconds.
    pixels_per_sample
        Number of pixels per time sample.
    pixels_per_channel
        Number of pixels per channel.
    dpi
        Dots per inch for output images.
    max_chunks
        Maximum number of chunks to process. If None, process all chunks.
        Useful for quick testing.
    overwrite
        If False (default) and the output directory already exists and is
        non-empty, skip processing.
    """
    label = structure_acronym if structure_acronym is not None else "whole probe"
    savedir = make_savedir(subject, probe, condition, structure_acronym)
    if savedir.exists() and any(savedir.iterdir()):
        if not overwrite:
            print(f"Skipping {label}: output already exists at {savedir}")
            return
        shutil.rmtree(savedir)
        savedir.mkdir(parents=True)

    print(f"Processing {label}")
    ap, lf, spikes = load_data(subject, probe, condition, structure_acronym, ap_type)
    # Only the chunk boundaries are used now; the OME-Zarr writer derives its
    # own pixel geometry (1 px per sample, 1 px per channel).
    (
        _chunk_len_samples,
        chunk_ixs,
        chunk_times,
        _img_width_pixels,
        _img_height_pixels,
        _img_width_inches,
        _img_height_inches,
    ) = get_stack_params(
        ap,
        chunk_len_sec=chunk_len_sec,
        pixels_per_sample=pixels_per_sample,
        pixels_per_channel=pixels_per_channel,
        dpi=dpi,
    )
    write_stacks(
        ap,
        lf,
        spikes,
        savedir,
        chunk_ixs,
        chunk_times,
        dpi,
        max_chunks=max_chunks,
    )


# -------------------- OFF labels for napari overlay --------------------


