"""Time/depth geometry of the image-stack grid, for rasterizing detections.

Provides the absolute-time coordinate of every (chunk, sample) position and the
depth (µm) of every channel/row, so that OFF detections expressed in seconds and
micrometers (e.g. the full-48h morphological parquet) can be painted onto the same
``(n_chunks, n_rows, n_samples)`` grid as the manual labels, and labels read back
as time/depth events (:func:`label_events`).

The grid holds only the samples of the stack's condition window, concatenated, so
consecutive grid samples can straddle a gap between hypnogram bouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import zarr
from cnpix import stacks

from cnpix_local_sleep import trace_io

#: Consecutive grid samples further apart than this many sample periods straddle a gap.
GAP_FACTOR = 1.5


def load_stack_times_flat(subject: str, probe: str, condition: str) -> np.ndarray:
    """Flattened absolute timestamps (seconds) of the stack grid.

    Reads the stack ``timestamps.zarr`` (shape ``(n_chunks, samples_per_chunk)``)
    and flattens it so that index ``chunk * samples_per_chunk + sample`` gives the
    absolute time of that grid position.
    """
    ts_path = stacks.timestamps_path(subject, probe, condition)
    return zarr.open(str(ts_path), mode="r")[:].reshape(-1)


def channel_depths(subject: str, probe: str) -> np.ndarray:
    """Per-channel depth (µm), ascending, matching the stack's row count.

    Returns the ``y`` coordinate of the full-probe preprocessed (tom) traces the
    stacks were built from. ``y[i]`` is the depth of channel index ``i``; the
    stack flips this so stack row 0 is the deepest channel
    (``stack_row = (len(y) - 1) - i``). Length equals the number of label rows.

    Channel depths are condition-independent, so we open with ``condition=None``:
    passing a condition would trigger ``covers_time(da.time)``, which materializes
    the recording's full time vector (a costly SI pitfall) for no benefit here.
    """
    da = trace_io.open_preprocessed_traces_as_xarray(
        subject,
        probe,
        structure=None,
        condition=None,
        apply_detection_channel_mask=False,
    )
    return da.y.values


def sample_period(ts_flat: np.ndarray) -> float:
    """Median spacing (s) of consecutive grid samples."""
    return float(np.median(np.diff(ts_flat)))


def overlaps_grid(
    start_times: np.ndarray, end_times: np.ndarray, ts_flat: np.ndarray
) -> np.ndarray:
    """True for events ``[start, end]`` that contain at least one grid sample.

    Uses the inclusive bounds of :func:`cnpix_local_sleep.evaluation.rasterize.rasterize_offs`,
    so an event this rejects would paint nothing. ``ts_flat`` must be ascending.
    """
    lo = np.searchsorted(ts_flat, start_times, side="left")
    hi = np.searchsorted(ts_flat, end_times, side="right")
    return hi > lo


def times_to_grid_index(times: np.ndarray, ts_flat: np.ndarray) -> np.ndarray:
    """Flat grid index of the first grid sample at or after each time, or -1.

    A time gets -1 when that sample is more than ``GAP_FACTOR`` sample periods
    away, i.e. the time lies in a gap between bouts or outside the grid; a bare
    ``searchsorted`` would put it on the first sample after the gap.
    """
    times = np.asarray(times, dtype=float)
    ix = np.searchsorted(ts_flat, times, side="left")
    ok = ix < len(ts_flat)
    ok[ok] = ts_flat[ix[ok]] - times[ok] <= GAP_FACTOR * sample_period(ts_flat)
    return np.where(ok, ix, -1)


def label_events(
    label_array: np.ndarray, ts_flat: np.ndarray, y_coords: np.ndarray
) -> pd.DataFrame:
    """One row per labelled instance per image, as time and depth extents.

    ``label_array`` is a ``(n_chunks, n_rows, samples_per_chunk)`` instance-label
    array (manual or model labels, 0 = background); ``ts_flat`` and ``y_coords``
    come from :func:`load_stack_times_flat` and :func:`channel_depths` for the same
    stack. An instance cut by a chunk edge gives one row per chunk.

    Columns: ``chunk``, ``label``, ``start_time`` / ``end_time`` (times of the
    first and last labelled sample, inclusive, as
    :func:`cnpix_local_sleep.evaluation.rasterize.rasterize_offs` reads them),
    ``duration``, ``lo`` / ``hi`` (channel depth range, µm), ``n_pixels``, and
    ``crosses_gap`` (the instance's samples straddle a gap between bouts).
    """
    n_chunks, n_rows, spc = label_array.shape
    if len(y_coords) != n_rows:
        raise ValueError(f"{len(y_coords)} channel depths for {n_rows} label rows")
    if len(ts_flat) != n_chunks * spc:
        raise ValueError(f"{len(ts_flat)} timestamps for {n_chunks} x {spc} grid samples")
    ts = np.asarray(ts_flat).reshape(n_chunks, spc)
    row_depth = np.asarray(y_coords)[::-1]  # stack row r <-> channel (n_rows - 1 - r)

    chunk, row, sample = np.nonzero(label_array)
    px = pd.DataFrame(
        {"chunk": chunk, "label": label_array[chunk, row, sample], "row": row, "sample": sample}
    )
    ev = (
        px.groupby(["chunk", "label"], sort=True)
        .agg(
            s0=("sample", "min"),
            s1=("sample", "max"),
            r0=("row", "min"),
            r1=("row", "max"),
            n_pixels=("sample", "size"),
        )
        .reset_index()
    )
    c, s0, s1 = ev["chunk"].to_numpy(), ev["s0"].to_numpy(), ev["s1"].to_numpy()
    gaps = np.diff(ts, axis=1) > GAP_FACTOR * sample_period(ts_flat)
    n_gaps = np.concatenate([np.zeros((n_chunks, 1), dtype=int), np.cumsum(gaps, axis=1)], axis=1)
    return pd.DataFrame(
        {
            "chunk": c,
            "label": ev["label"].to_numpy(),
            "start_time": ts[c, s0],
            "end_time": ts[c, s1],
            "duration": ts[c, s1] - ts[c, s0],
            "lo": row_depth[ev["r1"].to_numpy()],
            "hi": row_depth[ev["r0"].to_numpy()],
            "n_pixels": ev["n_pixels"].to_numpy(),
            "crosses_gap": n_gaps[c, s1] > n_gaps[c, s0],
        }
    )
