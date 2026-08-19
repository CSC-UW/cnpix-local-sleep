"""Time/depth geometry of the image-stack grid, for rasterizing detections.

Provides the absolute-time coordinate of every (chunk, sample) position and the
depth (µm) of every channel/row, so that OFF detections expressed in seconds and
micrometers (e.g. the full-48h morphological parquet) can be painted onto the same
``(n_chunks, n_rows, n_samples)`` grid as the manual labels.
"""

from __future__ import annotations

import numpy as np
import zarr

from cnpix_local_sleep import trace_io
from cnpix_local_sleep.evaluation import config
from cnpix_local_sleep.stacks import files as stk_files


def load_stack_times_flat(subject: str, probe: str, condition: str) -> np.ndarray:
    """Flattened absolute timestamps (seconds) of the stack grid.

    Reads the stack ``timestamps.zarr`` (shape ``(n_chunks, samples_per_chunk)``)
    and flattens it so that index ``chunk * samples_per_chunk + sample`` gives the
    absolute time of that grid position. ``condition`` is the *evaluation*
    condition; the truncated stack-directory name (e.g. ``Late.NOD`` for Wake) is
    resolved internally.
    """
    stack_cond = config.stack_condition(condition)
    ts_path = stk_files.get_sam3_off_stacks_timestamps_path(
        subject, probe, stack_cond, structure=None
    )
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
