"""Morphological OFF-detection kernel.

Turns a per-sample threshold comparison into discrete OFF *events*: build the
binary mask, clean it (binary closing/opening in the temporal and spatial
axes), label connected components, and measure each one.

This is the methodological centrepiece of the morphological detector --
:func:`clean_binary_mask`, :func:`get_off_properties`, :func:`detect_offs`. It
lived in ``cnpix_local_sleep.unit_free`` until 2026-08-11, a name from a pre-migration
layout; every consumer is now under ``cnpix_local_sleep.morphological``, so it lives here.

The standalone ``harding`` package carries a deliberate, self-contained copy of
the same morphology (``harding.morphology``) so it can depend on nothing
outside PyPI. If you change the algorithm here, check that file too.
"""

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats
import xarray as xr

from cnpix_local_sleep.off_tables import Off


def _below_threshold_mask(
    values: np.ndarray, thresholds: xr.DataArray
) -> np.ndarray:
    """Build a (n_time, n_channels) bool mask of ``values < thresholds``.

    Supports two threshold layouts:

    - 1-D ``(channel,)``: broadcast across time.
    - 2-D ``(bin, channel)``: applied per time-bin via the
      ``bin_boundaries`` attr (length n_bins+1 sample indices).
    """
    if thresholds.ndim == 1:
        return values < thresholds.values
    if thresholds.ndim != 2 or thresholds.dims != ("bin", "channel"):
        raise ValueError(
            f"thresholds must be (channel,) or (bin, channel); got "
            f"dims={thresholds.dims}"
        )
    boundaries = thresholds.attrs.get("bin_boundaries")
    if boundaries is None:
        raise ValueError(
            "Per-bin thresholds require attrs['bin_boundaries']"
        )
    boundaries = np.asarray(boundaries, dtype=np.int64)
    n_bins = thresholds.shape[0]
    if len(boundaries) != n_bins + 1:
        raise ValueError(
            f"bin_boundaries length {len(boundaries)} does not match "
            f"n_bins+1 = {n_bins + 1}"
        )
    if boundaries[0] != 0 or boundaries[-1] != values.shape[0]:
        raise ValueError(
            f"bin_boundaries must span [0, n_time={values.shape[0]}]; "
            f"got [{boundaries[0]}, {boundaries[-1]}]"
        )
    thresholds_arr = thresholds.values
    out = np.empty(values.shape, dtype=bool)
    for bi in range(n_bins):
        lo, hi = int(boundaries[bi]), int(boundaries[bi + 1])
        out[lo:hi] = values[lo:hi] < thresholds_arr[bi]
    return out


def clean_binary_mask(
    off_mask: np.ndarray,
    n_samples_connect: int | None = None,
    n_samples_clean: int | None = 10,
    n_channels_clean: int | None = 3,
    n_channels_connect: int | None = 5,
) -> np.ndarray:
    """Apply morphological operations to clean a binary OFF period detection mask.

    This function performs a sequence of binary morphological operations (closing and
    opening) to remove noise and connect spatially/temporally close detections. The
    operations are applied in both temporal (sample) and spatial (channel) dimensions.

    The cleaning pipeline follows this order:
    1. Temporal connection (closing) - Bridge short gaps in time
    2. Spatial connection (closing) - Connect across nearby/bad channels
    3. Temporal cleaning (opening) - Remove brief spurious detections
    4. Spatial cleaning (opening) - Remove detections on few channels
    5. Final spatial connection (closing) - Connect across larger spatial extent

    Morphological operations:
    - Binary closing (dilation -> erosion): Fills gaps and connects nearby regions
    - Binary opening (erosion -> dilation): Removes small objects and smooths borders

    Args:
        off_mask: Binary mask array with shape (time, channel) where True indicates
            detected OFF periods. Should be a numpy array (not dask-backed).
        n_samples_connect: Size of structuring element (in samples) for initial
            temporal connection. If None, this step is skipped. Connects OFF periods
            separated by gaps up to this duration.
        n_samples_clean: Size of structuring element (in samples) for temporal
            cleaning. If None, this step is skipped. Removes detected OFF periods
            shorter than this duration.
        n_channels_clean: Size of structuring element (in channels) for spatial
            operations. If None, spatial cleaning and the initial spatial connection
            are skipped. Used for both connecting across bad channels and removing
            detections on few channels.
        n_channels_connect: Size of structuring element (in channels) for final
            spatial connection. If None, this step is skipped. Connects OFF periods
            across larger vertical extents on the probe.

    Returns:
        The cleaned binary mask as a numpy array with the same shape as the input.

    Notes:
        - All morphological operations use `iterations=1`
        - Structuring elements are always rectangular (1D lines in either dimension)
        - The order of operations matters: closing before opening ensures genuine
          OFF periods are connected before small artifacts are removed

    Examples:
        >>> # Minimal cleaning (remove short events only)
        >>> cleaned = clean_binary_mask(
        ...     mask,
        ...     n_samples_clean=100,
        ...     n_channels_clean=None,
        ...     n_samples_connect=None,
        ...     n_channels_connect=None
        ... )
    """

    # Horizontal: Connect across samples
    if n_samples_connect is not None:
        struct = np.ones((n_samples_connect, 1))
        off_mask = scipy.ndimage.binary_closing(
            off_mask, structure=struct, iterations=1
        )

    # Vertical: Connect across bad channels
    if n_channels_clean is not None:
        struct = np.ones((1, n_channels_clean))
        off_mask = scipy.ndimage.binary_closing(
            off_mask, structure=struct, iterations=1
        )

    # Horizontal: Remove shorter blobs
    if n_samples_clean is not None:
        struct = np.ones((n_samples_clean, 1))
        off_mask = scipy.ndimage.binary_opening(
            off_mask, structure=struct, iterations=1
        )

    # Vertical: Remove few-channel epochs
    if n_channels_clean is not None:
        struct = np.ones((1, n_channels_clean))
        off_mask = scipy.ndimage.binary_opening(
            off_mask, structure=struct, iterations=1
        )

    # Vertical: Connect distant blobs vertically
    if n_channels_connect is not None:
        struct = np.ones((1, n_channels_connect))
        off_mask = scipy.ndimage.binary_closing(
            off_mask, structure=struct, iterations=1
        )

    return off_mask


def _edge_synchrony(
    depths: np.ndarray,
    times: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute linear-fit synchrony metrics for an onset or offset edge.

    Fits a line (time ~ depth) to the per-channel edge times and returns
    the slope, residual jitter, R², and MAD of the fit.

    Args:
        depths: Array of depth values (μm) for each channel on the edge.
        times: Array of onset/offset times (seconds) for each channel.

    Returns:
        Tuple of (slope, jitter, r2, mad):
        - slope: seconds/μm (positive = superficial->deep propagation)
        - jitter: std dev of residuals from the linear fit (seconds).
            Measures absolute temporal precision: how tightly
            per-channel times cluster around the fitted wavefront,
            independent of the depth span. Comparable across edges
            with similar time scales, but not normalized. Prefer
            jitter when you need a physical tolerance (e.g.
            "synchronized within 5 ms") or when the number of
            channels is small (where R² is unstable).
        - r2: coefficient of determination of the linear fit.
            Measures the fraction of timing variance explained by
            depth, a normalized, dimensionless summary of fit
            quality. Useful for comparing edges across probes with
            different channel spans, but sensitive to the spread of
            depths (narrow spans suppress R² even when jitter is
            small) and unreliable with very few channels.
        - mad: median absolute deviation of raw edge times (seconds).
            A model-free measure of temporal spread: how long the
            edge takes to complete, regardless of whether it follows
            a linear wavefront. Unlike jitter, MAD does not remove
            the linear trend, so it conflates propagation delay with
            disorder. More robust to outliers than std dev. Useful
            when you want a single number for "how spread out are
            these edge times?" without assuming a propagation model.
        For single-channel blobs, returns (NaN, 0.0, NaN, 0.0).
    """
    if len(depths) < 2:
        return (np.nan, 0.0, np.nan, 0.0)

    result = scipy.stats.linregress(depths, times)
    residuals = times - (result.slope * depths + result.intercept)
    jitter = np.std(residuals)
    r2 = result.rvalue**2
    mad = np.median(np.abs(times - np.median(times)))
    return (result.slope, jitter, r2, mad)


def get_off_properties(
    y_coords: np.ndarray,
    time_coords: np.ndarray,
    fs: float,
    lbl_ixs: dict[int, tuple[np.ndarray, np.ndarray]],
    values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Extract properties for each detected OFF period.

    Computes temporal, spatial, and geometric properties for labeled OFF periods
    including duration, spatial extent, area, and convexity metrics. When
    ``values`` is provided, also computes trace-value summaries, center of mass,
    and onset/offset edge synchrony metrics.

    Args:
        y_coords: Array of y-coordinates (depth in microns) for each channel.
        time_coords: Array of timestamps (in seconds) for each time sample.
        fs: Sampling frequency in Hz.
        lbl_ixs: Dictionary mapping each label ID to (time_indices, channel_indices)
            arrays indicating where that OFF period was detected.
        values: Optional 2D array of trace values with shape (time, channel),
            same shape as the label image. When provided, trace-value metrics,
            center of mass, and edge synchrony are computed.

    Returns:
        DataFrame with one row per OFF period, containing properties defined in
        the Off TypedDict schema. Returns empty DataFrame if no OFFs detected.

    Raises:
        NotImplementedError: If channels are not evenly spaced (required for
            accurate area and convexity calculations).
    """
    # -------------------- Early exit: No detections --------------------
    if not lbl_ixs:
        return pd.DataFrame(columns=list(Off.__annotations__.keys()))

    channel_spacings = np.diff(y_coords)
    if len(set(channel_spacings)) > 1:
        raise NotImplementedError("Evenly spaced, depth-ordered channels are required")

    # Build DataFrame with one row per labeled OFF period
    labels = np.sort(list(lbl_ixs.keys()))
    properties = []

    for label in labels:
        time_indices, channel_indices = lbl_ixs[label]

        # Per-channel frame statistics
        blob_df = pd.DataFrame(
            {
                "chan_idx": channel_indices,
                "time_idx": time_indices,
            }
        )
        per_chan = blob_df.groupby("chan_idx")["time_idx"]
        per_chan_min = per_chan.min()
        per_chan_max = per_chan.max()
        per_chan_count = per_chan.count()

        # Collect frame-based properties
        props = {
            "label": label,
            "area": time_indices.size,
            "start_frame": time_indices.min(),
            "end_frame": time_indices.max() + 1,  # Exclusive end
            "median_nframes": np.median(per_chan_count.values),
            "median_start_frame": int(np.median(per_chan_min.values)),
            "median_end_frame": int(np.median(per_chan_max.values)) + 1,
            "min_chan_idx": channel_indices.min(),
            "max_chan_idx": channel_indices.max(),
        }

        # Trace-value metrics, center of mass, and edge synchrony
        if values is not None:
            blob_values = values[time_indices, channel_indices]
            props["median_trace"] = np.median(blob_values)
            props["min_trace"] = np.min(blob_values)
            props["mad_trace"] = np.median(np.abs(blob_values - np.median(blob_values)))

            # Center of mass (unweighted centroid in physical units)
            props["center_of_mass_time"] = np.mean(time_coords[time_indices])
            props["center_of_mass_depth"] = np.mean(y_coords[channel_indices])

            # Edge synchrony: reuse per-channel onset/offset from above
            onset_idx = per_chan_min
            offset_idx = per_chan_max

            onset_depths = y_coords[onset_idx.index.values]
            onset_times = time_coords[onset_idx.values]
            offset_depths = y_coords[offset_idx.index.values]
            offset_times = time_coords[offset_idx.values]

            (
                props["onset_slope"],
                props["onset_jitter"],
                props["onset_r2"],
                props["onset_mad"],
            ) = _edge_synchrony(onset_depths, onset_times)
            (
                props["offset_slope"],
                props["offset_jitter"],
                props["offset_r2"],
                props["offset_mad"],
            ) = _edge_synchrony(offset_depths, offset_times)

        properties.append(props)

    df = pd.DataFrame(properties)

    # Add time-based properties (convert frames to seconds)
    df["start_time"] = time_coords[df["start_frame"]]
    df["end_time"] = time_coords[df["end_frame"] - 1] + (
        1 / fs
    )  # TODO: WHY?? Is this why duration is not always equal to median_duration
    # for collapsed detection?
    df["duration"] = df["end_time"] - df["start_time"]
    df["median_start_time"] = time_coords[df["median_start_frame"]]
    df["median_end_time"] = time_coords[df["median_end_frame"] - 1] + (1 / fs)
    df["median_duration"] = df["median_nframes"] / fs

    # Add spatial properties (vertical extent in microns)
    df["lo"] = y_coords[df["min_chan_idx"]]
    df["hi"] = y_coords[df["max_chan_idx"]]
    df["span"] = df["hi"] - df["lo"]

    # -------------------- Convert label to integer type --------------------
    df = df.astype({"label": np.int64})

    # Drop intermediate columns not needed downstream
    df = df.drop(
        columns=[
            "median_nframes",
            "median_start_frame",
            "median_end_frame",
            "min_chan_idx",
            "max_chan_idx",
            "start_frame",
            "end_frame",
        ]
    )

    return df


def detect_offs(
    da: xr.DataArray,
    thresholds: xr.DataArray,
    do_clean_binary_mask: bool = True,
    n_samples_connect: int | None = None,
    n_samples_clean: int | None = 10,
    n_channels_clean: int | None = 3,
    n_channels_connect: int | None = 5,
) -> tuple[pd.DataFrame, xr.DataArray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Detect OFF periods in preprocessed AP-band data.

    Creates a binary mask of below-threshold values, applies morphological cleaning,
    labels connected components, and extracts properties for each detected OFF period.

    Performance note: Previously used dask for lazy evaluation, but testing showed
    no benefit. For an input of 218 MB with 18 MB chunks in 10 graph layers, dask
    took ~8 minutes just to plan the computation. With pre-computed data using scipy,
    this takes ~0.1s for labeling and ~20s total for property extraction.

    Args:
        da: Input DataArray containing preprocessed AP-band data with dimensions
            (time, channel). Must be computed (numpy-backed), not dask-backed.
            Must have 'fs' (sampling frequency) in attrs and 'y' coordinate.
        thresholds: DataArray of threshold values. Either:

            - dim ``(channel,)``: one scalar threshold per channel
              (whole-recording / Tom-Bugnon).
            - dims ``(bin, channel)``: one threshold per (time bin,
              channel) (morphological per-bin). Must carry
              ``attrs["bin_boundaries"]`` as a length-(n_bins+1) sample
              index array.
        do_clean_binary_mask: Whether to apply morphological cleaning to the binary
            mask before labeling. Default True.
        n_samples_connect: Size of structuring element (in samples) for temporal
            connection. If None, this step is skipped.
        n_samples_clean: Size of structuring element (in samples) for temporal
            cleaning. If None, this step is skipped.
        n_channels_clean: Size of structuring element (in channels) for spatial
            operations.
        n_channels_connect: Size of structuring element (in channels) for final
            spatial connection.

    Returns:
        Tuple of:
        - offs: DataFrame with one row per OFF period containing temporal, spatial,
            and geometric properties
        - lbl_da: DataArray with same shape as input containing integer labels for
            each connected component (0 = background)
        - lbl_ixs: Dictionary mapping each label to (time_indices, channel_indices)
            arrays
    """
    off_mask = _below_threshold_mask(da.values, thresholds)

    # Morphological cleaning
    if do_clean_binary_mask:
        off_mask = clean_binary_mask(
            off_mask,
            n_samples_connect=n_samples_connect,
            n_samples_clean=n_samples_clean,
            n_channels_clean=n_channels_clean,
            n_channels_connect=n_channels_connect,
        )

    # Get labels for each contiguous blob
    # scipy.ndimage.label is ~0.5s, much faster than dask_image equivalent
    lbl_img, _ = scipy.ndimage.label(off_mask)
    del off_mask  # Free memory

    # Store labels in a DataArray
    lbl_da = da.copy()
    lbl_da.data = lbl_img
    lbl_da.name = "OFF label"

    # Get a dictionary mapping each unique label to its (time, channel) indices.
    # Example: {1: (array([100, 101, 102]), array([5, 5, 6]))}
    lbl_ixs = scipy.ndimage.value_indices(
        lbl_da.data,
        ignore_value=0,
    )  # Nearly instantaneous.

    offs = get_off_properties(
        da.y.values, da.time.values, da.attrs["fs"], lbl_ixs, values=da.values
    )  # ~20s

    return offs, lbl_da, lbl_ixs
