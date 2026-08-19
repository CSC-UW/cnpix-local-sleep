"""Rasterize time/depth OFF detections onto the image-stack label grid.

Converts OFF events expressed as ``[start_time, end_time]`` (absolute seconds)
and ``[lo, hi]`` (depth µm), e.g. rows of the full-48h morphological parquet, into
a ``(n_chunks, n_rows, n_samples)`` instance-label array directly comparable to
the manual labels via ``cnpix_local_sleep.evaluation.metrics``.

Orientation (empirically verified against manual labels): the stack flips the
ascending channel axis, so stack ``row = (n_rows - 1) - channel_index`` where
``channel_index`` indexes the ascending ``y_coords``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rasterize_offs(
    offs: pd.DataFrame,
    ts_flat: np.ndarray,
    y_coords: np.ndarray,
    label_shape: tuple[int, int, int],
    *,
    eval_chunks: np.ndarray | None = None,
    start_col: str = "start_time",
    end_col: str = "end_time",
    lo_col: str = "lo",
    hi_col: str = "hi",
) -> np.ndarray:
    """Paint OFF events onto a ``(n_chunks, n_rows, n_samples)`` label grid.

    Parameters
    ----------
    offs
        OFF events with absolute-second time bounds (``start_col``/``end_col``)
        and depth bounds in µm (``lo_col``/``hi_col``).
    ts_flat
        Flattened, monotonic non-decreasing absolute timestamps of the stack
        grid (length ``n_chunks * n_samples``); see
        :func:`cnpix_local_sleep.evaluation.grid.load_stack_times_flat`.
    y_coords
        Ascending per-channel depth (µm); ``len(y_coords)`` must equal
        ``label_shape[1]`` (one row per channel). Row mapping is flipped.
    label_shape
        Target grid shape ``(n_chunks, n_rows, n_samples)``.
    eval_chunks
        Optional chunk indices that will actually be scored. Events falling
        entirely outside these chunks are skipped; they cannot affect a metric
        computed only over ``eval_chunks``, which avoids the per-event work for
        the (usually vast) majority of events outside the labeled chunks.

    Returns
    -------
    np.ndarray
        int32 instance-label array; each event gets a unique positive ID
        (its 1-based row position in ``offs``) so event-level metrics can run on
        the result.
    """
    n_chunks, n_rows, spc = label_shape
    n_full = len(y_coords)
    if n_full != n_rows:
        raise ValueError(
            f"y_coords length ({n_full}) must equal label rows ({n_rows}); the "
            "depth coordinate must correspond one-to-one to the stack rows."
        )

    out = np.zeros(label_shape, dtype=np.int32)
    if len(offs) == 0:
        return out

    start_times = offs[start_col].to_numpy()
    end_times = offs[end_col].to_numpy()
    los = offs[lo_col].to_numpy()
    his = offs[hi_col].to_numpy()

    # Time-pixel ranges (inclusive of both bounds) into the flattened grid.
    start_ix = np.searchsorted(ts_flat, start_times, side="left")
    end_ix = np.searchsorted(ts_flat, end_times, side="right")

    n_flat = n_chunks * spc
    if eval_chunks is not None:
        keep_chunk = np.zeros(n_chunks, dtype=bool)
        keep_chunk[eval_chunks] = True
        s_chunk = np.clip(start_ix, 0, n_flat - 1) // spc
        e_chunk = np.clip(end_ix - 1, 0, n_flat - 1) // spc
        event_iter = np.where(keep_chunk[s_chunk] | keep_chunk[e_chunk])[0]
    else:
        event_iter = range(len(offs))

    for k in event_iter:
        lo_i, hi_i = start_ix[k], min(end_ix[k], n_flat)
        if hi_i <= lo_i:
            continue  # event falls outside the inspected stack windows
        flat = np.arange(lo_i, hi_i)
        chunk_ids = flat // spc
        within = flat % spc

        chan_idx = np.where((y_coords >= los[k]) & (y_coords <= his[k]))[0]
        if chan_idx.size == 0:
            continue
        rows = (n_full - 1) - chan_idx  # flip (verified orientation)

        # Outer product over paired (chunk, within) time pixels and rows.
        n_t, n_r = flat.size, rows.size
        ci = np.repeat(chunk_ids, n_r)
        wi = np.repeat(within, n_r)
        ri = np.tile(rows, n_t)
        out[ci, ri, wi] = k + 1

    return out
