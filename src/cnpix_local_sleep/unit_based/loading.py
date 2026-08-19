"""Loading and schema-mapping for unit-based OFF detection results.

Unit-based detection runs a pooled ON/OFF model (``on_off_detection``) per
(subject, probe, structure) and writes a single condition-agnostic full-recording
OFF dataframe in the shared :class:`cnpix_local_sleep.off_tables.Off` schema (parquet)
under ``method=unit_based/.../detection_mode=pooled-{algo}/``.

Because detection is structure-level *pooled* there is no spatial (channel x time)
OFF "blob": the spatial/morphology fields of the ``Off`` schema are filled from the
structure depth extent (span/area/lo/hi/max_span) or set to NaN (trace amplitudes,
onset/offset propagation, laminar areas). Downstream comparison is OFF-*rate*-focused,
so these spatial fields are not used analytically.
"""

import numpy as np
import pandas as pd

from cnpix_local_sleep.unit_based import files
from cnpix_local_sleep.off_tables import Off

# Column order of the Off schema, plus max_span (added at detection time by every
# method; used by postprocessing for span_rel2max / area_rel2span).
OFF_COLUMNS = list(Off.__annotations__.keys())
EXTRA_COLUMNS = ["max_span"]


def empty_off_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the full Off (+ max_span) column set."""
    cols = OFF_COLUMNS + EXTRA_COLUMNS
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})


def on_off_df_to_off_frame(
    off_rows: pd.DataFrame,
    *,
    depth_lo: float,
    depth_hi: float,
    binsize: float = 0.010,
) -> pd.DataFrame:
    """Map pooled ON/OFF detection output (OFF rows) to the ``Off`` schema.

    Args:
        off_rows: Rows with ``start_time``/``end_time``/``duration`` (the
            ``state == "off"`` subset of an ``on_off_detection`` result), in
            original recording time.
        depth_lo: Shallowest unit depth in the structure (um).
        depth_hi: Deepest unit depth in the structure (um).
        binsize: Detection bin size (s); used for the ``area`` proxy.

    Returns:
        DataFrame with the ``Off`` columns plus ``max_span``, sorted by
        ``start_time``.
    """
    n = len(off_rows)
    span = float(depth_hi - depth_lo)
    if n == 0:
        return empty_off_frame()

    start = off_rows["start_time"].to_numpy(dtype=float)
    end = off_rows["end_time"].to_numpy(dtype=float)
    dur = off_rows["duration"].to_numpy(dtype=float)
    nan = np.full(n, np.nan)

    frame = pd.DataFrame(
        {
            "label": np.arange(n, dtype=int),
            # Area proxy: number of detection bins the OFF spans (>=1, nonzero so
            # area_rel2span is finite).
            "area": np.maximum(1, np.round(dur / binsize)).astype(int),
            "start_time": start,
            "end_time": end,
            "duration": dur,
            # Pooled detection has no per-channel structure; per-channel medians
            # equal the scalar values.
            "median_start_time": start,
            "median_end_time": end,
            "median_duration": dur,
            "lo": np.full(n, depth_lo),
            "hi": np.full(n, depth_hi),
            "span": np.full(n, span),
            "median_trace": nan,
            "min_trace": nan,
            "mad_trace": nan,
            "center_of_mass_time": (start + end) / 2.0,
            "center_of_mass_depth": np.full(n, (depth_lo + depth_hi) / 2.0),
            "onset_slope": nan,
            "onset_jitter": nan,
            "onset_r2": nan,
            "onset_mad": nan,
            "offset_slope": nan,
            "offset_jitter": nan,
            "offset_r2": nan,
            "offset_mad": nan,
            "supra_area": nan,
            "infra_area": nan,
            "max_supra_nchans": nan,
            "max_infra_nchans": nan,
            # Constant per structure: every pooled OFF spans the full extent, so
            # span_rel2max == 1.0 downstream.
            "max_span": np.full(n, span),
        }
    )
    return frame.sort_values("start_time").reset_index(drop=True)


def load_full_offs(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pd.DataFrame:
    """Load the condition-agnostic full-recording OFF dataframe, if present."""
    path = files.get_full_offs_path(subject, probe, structure, algo)
    if not path.exists():
        return empty_off_frame()
    return pd.read_parquet(path)


def load_full_banded_offs(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pd.DataFrame:
    """Load the condition-agnostic full-recording *banded* OFF dataframe, if present.

    Sister to :func:`load_full_offs` for ``detection_mode=banded-{algo}`` (real per-OFF
    ``lo``/``hi``/``span`` depth footprints); see :mod:`cnpix_local_sleep.unit_based.banded`.
    """
    path = files.get_full_banded_offs_path(subject, probe, structure, algo)
    if not path.exists():
        return empty_off_frame()
    return pd.read_parquet(path)
