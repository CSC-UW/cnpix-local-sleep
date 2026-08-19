"""What an OFF row is, how to load tables of them, and the named LAS filters.

Holds the :class:`Off` row schema shared by every detection method, the
per-condition OFF-table loader, and the single point of truth for the
LLAS/CLAS/BLAS column-threshold filters.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from cnpix_local_sleep import const, sps_conf


class Off(TypedDict):
    """Schema for a single OFF period detection event.

    This defines the structure of rows in the OFF events DataFrame returned
    by detection algorithms and used throughout the OFF pipeline.

    Attributes:
        label: Unique identifier for the OFF event (used to index into lbl_ixs
            for visualization)
        area: Area of the OFF event in the spatiotemporal detection space
        start_time: Time (seconds) when OFF event begins
        end_time: Time (seconds) when OFF event ends
        duration: Duration of OFF event in seconds
        median_start_time: Median per-channel onset time (seconds)
        median_end_time: Median per-channel offset time (seconds)
        median_duration: Median duration in seconds across channels
        lo: Depth (um) of lowest channel involved
        hi: Depth (um) of highest channel involved
        span: Vertical span of OFF event (hi - lo) in um
        median_trace: Median trace value across all pixels in the blob
        min_trace: Minimum trace value across all pixels in the blob
        mad_trace: Median absolute deviation of trace values across all
            pixels in the blob. Robust measure of trace-value dispersion.
        center_of_mass_time: Centroid time coordinate (seconds)
        center_of_mass_depth: Centroid depth coordinate (um)
        onset_slope: Slope of linear fit to per-channel onset times vs depth
            (seconds/um). Positive means onset propagates superficial->deep.
        onset_jitter: Std dev of residuals from the onset linear fit (seconds)
        onset_r2: R-squared of the onset linear fit
        onset_mad: Median absolute deviation of raw onset times (seconds).
            Model-free measure of temporal spread, robust to outliers.
        offset_slope: Slope of linear fit to per-channel offset times vs depth
            (seconds/um)
        offset_jitter: Std dev of residuals from the offset linear fit
            (seconds)
        offset_r2: R-squared of the offset linear fit
        offset_mad: Median absolute deviation of raw offset times (seconds).
            Model-free measure of temporal spread, robust to outliers.
        supra_area: Number of (sample, channel) pixels in the supragranular
            compartment. NaN for non-cortical structures.
        infra_area: Number of (sample, channel) pixels in the infragranular
            compartment. NaN for non-cortical structures.
        max_supra_nchans: Number of detection channels in the supragranular
            compartment. NaN for non-cortical structures.
        max_infra_nchans: Number of detection channels in the infragranular
            compartment. NaN for non-cortical structures.
    """

    label: int
    area: int
    start_time: float
    end_time: float
    duration: float
    median_start_time: float
    median_end_time: float
    median_duration: float
    lo: float
    hi: float
    span: float
    median_trace: float
    min_trace: float
    mad_trace: float
    center_of_mass_time: float
    center_of_mass_depth: float
    onset_slope: float
    onset_jitter: float
    onset_r2: float
    onset_mad: float
    offset_slope: float
    offset_jitter: float
    offset_r2: float
    offset_mad: float
    supra_area: int
    infra_area: int
    max_supra_nchans: int
    max_infra_nchans: int

def load_subject_offs(
    subject: str,
    filter_name: str | None = None,
    *,
    files_module,
    with_label_indices: bool = False,
    convert_label_indices: bool = False,
) -> pd.DataFrame:
    """Load all cortical, from-value-threshold OFF periods for a subject.

    Discovers (probe, structure) combos via ``sps_conf`` and iterates over
    all ``CONDITIONS``, loading individual parquet files produced by the
    detection pipeline.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g. ``"CNPIX15-Claude"``).
    files_module
        Files module that provides ``get_offs_path`` and
        ``get_off_label_indices_path`` -- e.g. ``cnpix_local_sleep.morphological.mua.files``
        for morphological detections.
    filter_name : str | None
        Name of a filter preset to apply (e.g. ``"llas"``, ``"clas"``,
        ``"blas"``, ``"collapsed"``, ``"spatial_by_layer"``). See
        ``NAMED_FILTERS`` for available presets. If None, no filtering
        is applied.
    with_label_indices : bool
        If True, merge label-index columns (``time_ixs``, ``chan_ixs``)
        from the label-indices parquet files.
    convert_label_indices : bool
        If True (requires ``with_label_indices=True``), convert
        ``time_ixs`` to timestamps and ``chan_ixs`` to depths by
        indexing into the coordinate arrays of the preprocessed
        DataArray. The resulting columns are named ``times`` and
        ``depths``, and the original index columns are dropped.

    Returns
    -------
    pd.DataFrame
        Concatenated OFF periods with ``probe`` and ``condition`` as
        ordered categoricals. Empty DataFrame if no files are found.
    """
    fm = files_module

    spsl = sps_conf.get_subject_probe_structure_list(
        method=fm.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    spsl = [(s, p, st) for s, p, st in spsl if s == subject]

    offs = []
    for subj, probe, structure in spsl:
        for condition in const.CORE_CONDITIONS:
            pathspec = {
                "subject": subj,
                "probe": probe,
                "structure": structure,
                "threshold_group": None,
                "condition": condition,
            }
            offs_path = fm.get_offs_path(**pathspec)
            if offs_path.exists():
                _offs = (
                    pd.read_parquet(offs_path)
                    .assign(**pathspec)
                    .dropna(axis=1, how="all")
                )
                if with_label_indices:
                    lbls_path = fm.get_off_label_indices_path(**pathspec)
                    if not lbls_path.exists():
                        print(f"Warning: label indices not found at {lbls_path}")
                        continue
                    _lbls = pd.read_parquet(lbls_path)
                    _offs = _offs.merge(
                        _lbls[["label", "time_ixs", "chan_ixs"]],
                        on="label",
                        how="left",
                    )
                    if convert_label_indices:
                        from cnpix_local_sleep import trace_io

                        da = trace_io.open_preprocessed_traces_as_xarray(
                            subj,
                            probe,
                            structure,
                            condition,
                            apply_detection_channel_mask=True,
                        )
                        _time_coords = da.time.values
                        _y_coords = da.y.values
                        _offs["times"] = _offs["time_ixs"].map(
                            lambda ixs: _time_coords[np.asarray(ixs)]
                        )
                        _offs["depths"] = _offs["chan_ixs"].map(
                            lambda ixs: _y_coords[np.asarray(ixs)]
                        )
                        _offs = _offs.drop(columns=["time_ixs", "chan_ixs"])
                offs.append(_offs)

    if not offs:
        return pd.DataFrame()

    offs = pd.concat(offs, ignore_index=True)

    offs["probe"] = pd.Categorical(
        offs["probe"], categories=["imec0", "imec1"], ordered=True
    )
    offs["condition"] = pd.Categorical(
        offs["condition"], categories=list(const.CONDITIONS), ordered=True
    )

    if filter_name is not None:
        offs = filter_offs(offs, filter_name)

    return offs

IMPLAUSIBLE_OFF_DURATION_S: float = 0.6

"""Putative OFFs longer than this are likely fragmented OFFs or artifacts.

Not applied in LLAS/CLAS/BLAS filters. Analyses that compare median duration
between conditions should filter on this explicitly; analyses that care about
total area, counts, or threshold tuning should leave long OFFs in.
"""

llas_filters = {
    "span": (100, float("Inf")),
    "median_duration": (0.025, float("Inf")),
    "duration": (0.03, float("Inf")),
}

"""Liberal low-amplitude segment/spatial OFF filters."""

clas_filters = {
    "span": (200, float("Inf")),
    "median_duration": (0.05, float("Inf")),
    "duration": (0.06, float("Inf")),
}

"""Conservative low-amplitude segment/spatial OFF filters."""

blas_filters = {
    **clas_filters,
    "span_rel2max": (0.75, 1.0),
}

"""Broad/big low-amplitude segment/spatial OFF filters."""

collapsed_filters = {
    "duration": (0.06, 1.0),
}

"""Default OFF filters for collapsed (population-average) detection."""

spatial_by_layer_filters = {
    "median_duration": (0.05, 0.6),
    "duration": (0.06, 0.6),
}

"""Default OFF filters for spatial detection with by-layer analysis."""

NAMED_FILTERS = {
    "llas": llas_filters,
    "clas": clas_filters,
    "blas": blas_filters,
    "collapsed": collapsed_filters,
    "spatial_by_layer": spatial_by_layer_filters,
}

def off_filter_mask(offs: pd.DataFrame, filter_name: str) -> pd.Series:
    """Boolean mask (aligned to ``offs.index``) of OFFs passing a named LAS filter.

    Single point of truth for the column-threshold logic shared by
    :func:`filter_offs` and callers that need set membership rather than a
    filtered copy (e.g. building "exclusive" OFF sets as mask differences like
    ``llas & ~clas``). ``span_rel2max`` is derived from ``span``/``max_span`` when
    absent, so the BLAS filter works on either the per-condition or the
    full-recording schema.
    """
    filters = NAMED_FILTERS[filter_name]
    need_derived_span_rel2max = (
        "span_rel2max" in filters
        and "span_rel2max" not in offs.columns
        and {"span", "max_span"} <= set(offs.columns)
    )
    span_rel2max = (
        offs["span"] / offs["max_span"]
        if need_derived_span_rel2max
        else offs.get("span_rel2max")
    )
    mask = pd.Series(True, index=offs.index)
    for col, (lo, hi) in filters.items():
        series = span_rel2max if col == "span_rel2max" else offs[col]
        mask &= series.between(lo, hi)
    return mask

def filter_offs(offs: pd.DataFrame, filter_name: str | None) -> pd.DataFrame:
    """Apply a named LAS filter (``llas``/``clas``/``blas``/...) to an OFF frame.

    Single point of truth for the column-threshold OFF filters used by both the
    per-condition loader (:func:`load_subject_offs`) and the full-recording
    true-mask path (``morphological.manual_validation``). ``span_rel2max`` is derived from
    ``span``/``max_span`` when absent; the full-48h ``offs.parquet`` ships without it
    (see ``morphological.pipeline.postprocess_offs``), so the BLAS filter works on either the
    per-condition or the full-recording schema. Returns a new, index-reset frame;
    ``filter_name=None`` is a passthrough.
    """
    if filter_name is None:
        return offs.reset_index(drop=True)
    return offs.loc[off_filter_mask(offs, filter_name)].reset_index(drop=True)
