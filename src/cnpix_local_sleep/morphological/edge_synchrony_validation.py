"""Validation of the OFF-period edge-synchrony estimators (onset/offset MAD).

``onset_mad``/``offset_mad`` (:func:`cnpix_local_sleep.morphological.morphology._edge_synchrony`) are
used as estimators of a latent quantity: how synchronously a population entered or
left an OFF period. They are also strongly correlated with event size. That
correlation admits two very different readings:

Generative (mediator). Larger OFF periods genuinely have more dispersed edges.
Size then mediates any condition effect, and adjusting for it would remove real
signal, just as a condition effect on ``area`` is not adjusted for area's
correlation with span and duration.

Mechanical (measurement confound). The estimator's value changes with event size
even when the latent edge dispersion does not. Size then confounds the measurement,
and adjustment (or a better estimator) is required.

The correlation alone cannot distinguish these. This module provides the tools to
apportion the observed MAD-vs-size relation between them:

- :func:`mad_zero_forced_max_n` / :func:`edge_floor_curve`: the analytic floor
  imposed by the detector's spatial structuring element, and its empirical
  counterpart.
- :func:`simulate_detector_edges`: forward simulation of synthetic events with a
  *known, size-independent* latent edge dispersion through the real cleaning,
  labeling and property-extraction code. Whatever size dependence comes back is
  mechanical by construction.

and, for the downstream consequences,

- :func:`fit_shared_size_curve`: the one size-adjustment curve, estimated once and
  shared by every cell, with a free per-unit amplitude.
- :func:`standardize_by_regression`: marginal (regression) standardization of the
  cell means against that curve. This is what ships.
- :func:`standardize_cell_means`: common-support direct standardization.
- :func:`fit_event_level`: per-combo covariate-adjusted condition effects, pooled
  across combos by DerSimonian-Laird.
- :func:`add_floor_free_rvs`: the alternative edge statistics already exported.

Everything here reads the event-level OFF parquets, which are published as GitHub
Release assets rather than committed (one exceeds GitHub's 100 MB per-file limit).
:func:`cnpix_local_sleep.release_data.get_event_table_path` prefers a copy already in
``r-offp/inst/extdata`` and fetches to a cache otherwise. No NFS is required either
way.
"""

import dataclasses
import pathlib
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats
import statsmodels.api as sm

from cnpix_local_sleep import files, release_data
from cnpix_local_sleep.morphological import correlation_stats, detection_opts
from cnpix_local_sleep.morphological import morphology
from cnpix_local_sleep import off_tables


class CleanOpts(TypedDict):
    """The four morphological-cleaning arguments of ``morphology.clean_binary_mask``."""

    n_samples_connect: int | None
    n_samples_clean: int | None
    n_channels_clean: int | None
    n_channels_connect: int | None

# -------------------- Constants --------------------

R_OFFP_EXTDATA = files.get_r_offp_extdata_dir()
"""The R package's data directory: the committed summarized tables, and the
event-level tables too on any machine that has run the exporters."""

CHANNEL_PITCH_UM = 20.0
"""Neuropixels channel pitch used by the detection grid."""

FS = 500.0
"""Sampling rate (Hz) of the MUA traces the morphological detector runs on."""

PRODUCTION_CLEAN_OPTS: CleanOpts = {
    "n_samples_connect": None,
    "n_samples_clean": 15,
    "n_channels_clean": 4,
    "n_channels_connect": 3,
}
"""Image-morphology cleaning options used by full-48h morphological OFF detection.

Mirrors ``cnpix_local_sleep/morphological/mua/data/spatial_detection_opts.yml``; kept here so the
forward simulation drives the same morphology production does. Verified against
that file by :func:`assert_production_clean_opts`.
"""

GAUSSIAN_FREQ_MAX_HZ = 20.0
"""Cutoff of the Gaussian low-pass applied to the MUA envelope before detection.

Default of ``cnpix_local_sleep.morphological.mua.readers.open_mua_traces_as_xarray``; the smoothing
sigma is ``FS / (2 * pi * GAUSSIAN_FREQ_MAX_HZ)`` samples.
"""

MEDIAN_FILTER_SAMPLES = 15
"""Length of the detection-time temporal median filter, in samples (30 ms at 500 Hz).

From ``ndimage_filter_kwargs.size`` in ``spatial_detection_opts.yml``. Checked
against the shipped YAML by :func:`assert_production_trace_opts`."""

NREM_CONDITIONS = [
    "Early.BSL.NREM",
    "Early.REC.NREM.Match",
    "Early.REC.NREM",
    "Late.REC.NREM",
]

WAKE_CONDITIONS = ["Early.NOD.Wake", "Late.NOD.Wake"]

SIZE_CLASSES = ["Small", "Medium+Large"]

GROUP_COLS = ["subject", "probe", "structure"]


def assert_production_clean_opts() -> None:
    """Fail loudly if :data:`PRODUCTION_CLEAN_OPTS` drifts from the shipped YAML."""
    opts = detection_opts.get_mua_spatial_detection_opts()
    shipped: CleanOpts = {
        "n_samples_connect": opts["n_samples_connect"],
        "n_samples_clean": opts["n_samples_clean"],
        "n_channels_clean": opts["n_channels_clean"],
        "n_channels_connect": opts["n_channels_connect"],
    }
    mismatched = {
        key: (value, shipped[key])  # type: ignore[literal-required]
        for key, value in PRODUCTION_CLEAN_OPTS.items()
        if shipped[key] != value  # type: ignore[literal-required]
    }
    if mismatched:
        raise AssertionError(
            "PRODUCTION_CLEAN_OPTS is stale relative to "
            f"spatial_detection_opts.yml: {mismatched}"
        )


def assert_production_trace_opts() -> None:
    """Fail loudly if the modelled trace filter chain drifts from the shipped YAML."""
    opts = detection_opts.get_mua_spatial_detection_opts()
    if opts["ndimage_filter_type"] != "median":
        raise AssertionError(
            "The trace-level simulation models a temporal *median* filter, but "
            f"spatial_detection_opts.yml now specifies {opts['ndimage_filter_type']!r}."
        )
    kwargs = opts["ndimage_filter_kwargs"] or {}
    size = list(kwargs.get("size", []))
    if size != [MEDIAN_FILTER_SAMPLES, 1]:
        raise AssertionError(
            f"MEDIAN_FILTER_SAMPLES is stale: YAML size is {size}, this module "
            f"models [{MEDIAN_FILTER_SAMPLES}, 1]."
        )


# -------------------- Event loading --------------------


def load_events(
    dataset: str = "llas",
    columns: list[str] | None = None,
    extdata_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Load the full-48h event-level OFF parquet, with derived columns.

    Parameters
    ----------
    dataset
        One of ``"llas"``, ``"clas"``, ``"blas"``.
    columns
        Optional column subset to read. ``span`` and the columns needed to derive
        ``size_class`` are always included.
    extdata_dir
        Checkout location to look in first; defaults to :data:`R_OFFP_EXTDATA`. When
        the table is not there it is fetched from the Release and cached -- see
        :func:`cnpix_local_sleep.release_data.get_event_table_path`.

    Returns
    -------
    The event frame with three derived columns appended:

    ``n_channels``
        ``span / 20 + 1``. Because :func:`scipy.ndimage.label` uses
        four-connectivity, every channel between an event's deepest and most
        superficial channel necessarily carries at least one of its pixels, so this
        is exactly the number of values each MAD is computed over. The identity is
        asserted, not assumed.
    ``size_class``
        ``"Medium+Large"`` for events passing the CLAS filter, ``"Small"``
        otherwise (the manuscript's exhaustive, non-overlapping partition of the
        LLAS set). Uses :func:`cnpix_local_sleep.off_tables.off_filter_mask` so the
        thresholds stay a single point of truth.
    ``combo``
        ``subject|probe|structure``, the unit the mixed models treat as a
        subject-structure pair.
    """
    path = release_data.get_event_table_path(dataset, extdata_dir=extdata_dir)
    needed = {"span", "median_duration", "duration", "max_span", *GROUP_COLS}
    if columns is not None:
        columns = sorted(set(columns) | needed)
    events = pd.read_parquet(path, columns=columns)

    n_channels = events["span"] / CHANNEL_PITCH_UM + 1.0
    if not np.allclose(n_channels, n_channels.round()):
        raise AssertionError(
            "span is not an integer multiple of the channel pitch; the "
            "n_channels identity does not hold for this frame"
        )
    events["n_channels"] = n_channels.round().astype(int)

    is_clas = off_tables.off_filter_mask(events, "clas")
    events["size_class"] = pd.Categorical(
        np.where(is_clas, "Medium+Large", "Small"), categories=SIZE_CLASSES
    )
    events["combo"] = (
        events["subject"].astype(str)
        + "|"
        + events["probe"].astype(str)
        + "|"
        + events["structure"].astype(str)
    )
    return events


# -------------------- Part 1.1: the analytic floor --------------------


def mad_zero_min_ties(n_channels: np.ndarray | int) -> np.ndarray:
    """Number of channels that must share the median edge time to force MAD = 0.

    ``MAD = median(|t - median(t)|)`` is exactly zero iff the median of the absolute
    deviations is zero, i.e. iff enough deviations are exactly zero. For odd ``n``
    the median is the ``(n+1)/2``-th order statistic, so ``(n+1)/2`` zeros are
    needed; for even ``n`` it is the mean of the ``n/2``-th and ``(n/2+1)``-th, so
    both must vanish and ``n/2 + 1`` zeros are needed. Both cases reduce to
    ``n // 2 + 1``.
    """
    return np.asarray(n_channels) // 2 + 1


def mad_zero_forced_max_n(run_length: int = 4) -> int:
    """Largest ``n_channels`` at which MAD is *forced* to zero by the detector.

    The spatial opening in :func:`cnpix_local_sleep.morphological.morphology.clean_binary_mask` uses a
    ``(1, n_channels_clean)`` structuring element, so after cleaning the set of ON
    channels at any single time sample is a union of runs of at least
    ``n_channels_clean`` consecutive channels. The subsequent spatial closing is
    extensive and can only add pixels. Therefore at a blob's earliest time sample
    (which is by definition the earliest ON time of every channel it contains) at
    least ``run_length`` channels turn on simultaneously, and those channels are
    four-connected, hence part of the same blob.

    Those tied channels sit at the minimum of the onset-time distribution, so if
    they are at least :func:`mad_zero_min_ties` many, the median deviation is zero
    and ``onset_mad`` is identically zero regardless of how dispersed the remaining
    channels are. Solving ``run_length >= n // 2 + 1`` gives
    ``n <= 2 * run_length - 1``. The same argument applies to the latest time sample
    and ``offset_mad``.

    Parameters
    ----------
    run_length
        The detector's ``n_channels_clean``. Defaults to the production value of 4.
    """
    return 2 * run_length - 1


def edge_floor_curve(
    events: pd.DataFrame,
    edge: str = "onset",
    by: list[str] | None = None,
    min_events: int = 200,
) -> pd.DataFrame:
    """Empirical floor and scaling of the edge statistics against ``n_channels``.

    Parameters
    ----------
    events
        Event frame from :func:`load_events`.
    edge
        ``"onset"`` or ``"offset"``.
    by
        Extra grouping columns (e.g. ``["condition"]`` or ``["combo"]``) to check
        that the curve is not a composition artifact.
    min_events
        Groups with fewer events than this are dropped.

    Returns
    -------
    One row per ``n_channels`` (× ``by``) with ``p_mad_zero``, ``p_jitter_zero``,
    ``mean_mad``, ``mean_jitter``, ``mean_ramp`` (``|slope| * span``, the magnitude
    of the fitted linear depth ramp) and ``n_events``. All times in seconds.
    """
    mad = f"{edge}_mad"
    jitter = f"{edge}_jitter"
    slope = f"{edge}_slope"
    frame = events.assign(_ramp=events[slope].abs() * events["span"])
    keys = ["n_channels"] + list(by or [])
    out = frame.groupby(keys, observed=True).agg(
        p_mad_zero=(mad, lambda s: float((s == 0).mean())),
        p_jitter_zero=(jitter, lambda s: float((s == 0).mean())),
        mean_mad=(mad, "mean"),
        mean_jitter=(jitter, "mean"),
        mean_ramp=("_ramp", "mean"),
        n_events=(mad, "size"),
    )
    return out[out["n_events"] >= min_events].reset_index()


# Part 1.3: forward simulation through the real detector


def _draw_event_mask(
    rng: np.random.Generator,
    n_channels: int,
    n_samples: int,
    duration_samples: int,
    sigma_samples: float,
    slope_samples_per_channel: float,
) -> np.ndarray:
    """Build one synthetic event's (time, channel) mask with known edge dispersion.

    Per-channel onset ``t_i = t0 + slope * i + eps_i`` with ``eps_i ~ N(0, sigma)``;
    the offset is the onset plus a fixed per-event duration, so the latent onset and
    offset dispersions are both exactly ``sigma``, independent of ``n_channels``.
    """
    mask = np.zeros((n_samples, n_channels), dtype=bool)
    centre = (n_samples - duration_samples) // 2
    jitter = rng.normal(0.0, sigma_samples, size=n_channels)
    ramp = slope_samples_per_channel * np.arange(n_channels)
    onsets = np.rint(centre + ramp - ramp.mean() + jitter).astype(int)
    onsets = np.clip(onsets, 0, n_samples - duration_samples - 1)
    for chan, onset in enumerate(onsets):
        mask[onset : onset + duration_samples, chan] = True
    return mask


def simulate_detector_edges(
    n_channels_values: list[int],
    sigma_ms: float = 6.0,
    slope_us_per_um: float = 0.0,
    slope_sd_us_per_um: float = 0.0,
    duration_ms: float = 120.0,
    duration_sd_ms: float = 0.0,
    n_events: int = 400,
    seed: int = 0,
    clean_opts: CleanOpts | None = None,
) -> pd.DataFrame:
    """Push synthetic events with a fixed latent edge dispersion through the detector.

    Each simulated event carries the *same* latent per-channel onset/offset
    dispersion ``sigma_ms`` and the same propagation slope, regardless of how many
    channels it spans. Events are laid out side by side in one (time, channel)
    array with generous margins, cleaned with
    :func:`cnpix_local_sleep.morphological.morphology.clean_binary_mask` under the production options,
    labeled with :func:`scipy.ndimage.label` (four-connectivity, as in
    ``detect_full``), and measured with
    :func:`cnpix_local_sleep.morphological.morphology.get_off_properties`, the same code path
    production uses.

    Any dependence of the recovered MAD on ``n_channels`` is therefore mechanical by
    construction. Note this is a *lower bound* on the mechanical component: it does
    not model the 30 ms temporal median filter applied to the traces upstream of
    thresholding, which smooths edges further.

    Parameters
    ----------
    n_channels_values
        Channel counts to simulate.
    sigma_ms
        Latent standard deviation of per-channel edge times, held constant.
    slope_us_per_um
        Mean latent propagation slope; 0 means a flat (non-travelling) wavefront.
    slope_sd_us_per_um
        Per-event standard deviation of the propagation slope, so simulated events
        carry a realistic spread of wavefront speeds and directions.
    duration_ms
        Mean per-channel OFF duration.
    duration_sd_ms
        Per-event standard deviation of the OFF duration.
    n_events
        Events per channel count.
    seed
        RNG seed.
    clean_opts
        Override for :data:`PRODUCTION_CLEAN_OPTS`.

    Returns
    -------
    One row per recovered event with ``requested_n_channels``, ``n_channels``,
    ``span``, ``onset_mad``, ``offset_mad``, ``onset_jitter``, ``offset_jitter``,
    ``onset_slope``, ``median_duration``, and the simulation settings.
    """
    clean_opts = clean_opts or PRODUCTION_CLEAN_OPTS
    # A disabled (None) morphological step imposes no constraint of its own.
    channel_structure = max(
        clean_opts["n_channels_clean"] or 1, clean_opts["n_channels_connect"] or 1
    )
    min_duration_samples = (clean_opts["n_samples_clean"] or 0) + 1

    rng = np.random.default_rng(seed)
    sigma_samples = sigma_ms * 1e-3 * FS
    max_duration_samples = int(round((duration_ms + 4 * duration_sd_ms) * 1e-3 * FS))
    max_slope = abs(slope_us_per_um) + 4 * slope_sd_us_per_um
    max_n = max(n_channels_values)
    # Enough room for the event, several sigma of jitter, and the full ramp.
    margin = (
        int(round(6 * sigma_samples))
        + int(round(max_slope * 1e-6 * CHANNEL_PITCH_UM * FS * max_n))
        + 40
    )
    n_samples = max_duration_samples + 2 * margin

    frames = []
    for n_channels in n_channels_values:
        # Pad each event with dead channels so neighbouring events never merge,
        # using more than the widest structuring element.
        pad = channel_structure + 2
        block_channels = n_channels + 2 * pad
        canvas = np.zeros((n_samples * n_events, block_channels), dtype=bool)
        for event in range(n_events):
            duration_samples = int(
                round(
                    max(
                        rng.normal(duration_ms, duration_sd_ms) * 1e-3 * FS,
                        min_duration_samples,
                    )
                )
            )
            slope_samples_per_channel = (
                rng.normal(slope_us_per_um, slope_sd_us_per_um)
                * 1e-6
                * CHANNEL_PITCH_UM
                * FS
            )
            mask = _draw_event_mask(
                rng,
                n_channels,
                n_samples,
                duration_samples,
                sigma_samples,
                slope_samples_per_channel,
            )
            row = event * n_samples
            canvas[row : row + n_samples, pad : pad + n_channels] = mask

        cleaned = morphology.clean_binary_mask(canvas, **clean_opts)
        labels, _ = scipy.ndimage.label(cleaned)
        lbl_ixs = scipy.ndimage.value_indices(labels, ignore_value=0)
        y_coords = np.arange(block_channels) * CHANNEL_PITCH_UM
        time_coords = np.arange(canvas.shape[0]) / FS
        props = morphology.get_off_properties(
            y_coords,
            time_coords,
            FS,
            lbl_ixs,
            values=cleaned.astype(float),
        )
        props["requested_n_channels"] = n_channels
        frames.append(props)

    out = pd.concat(frames, ignore_index=True)
    out["n_channels"] = (out["span"] / CHANNEL_PITCH_UM + 1).round().astype(int)
    out["sigma_ms"] = sigma_ms
    out["slope_us_per_um"] = slope_us_per_um
    out["slope_sd_us_per_um"] = slope_sd_us_per_um
    out["duration_ms"] = duration_ms
    return out


# Part 1.3b: the same simulation, driven from synthetic *traces*


def _draw_event_trace(
    rng: np.random.Generator,
    n_channels: int,
    n_samples: int,
    duration_samples: int,
    sigma_samples: float,
    slope_samples_per_channel: float,
    baseline: float,
    off_level: float,
) -> np.ndarray:
    """Build one synthetic event's (time, channel) *envelope*, before any filtering.

    Same latent model as :func:`_draw_event_mask`: per-channel onset
    ``t_i = t0 + slope * i + eps_i`` with ``eps_i ~ N(0, sigma)`` and a common
    per-event duration, so the latent onset and offset dispersions are both exactly
    ``sigma``, independent of ``n_channels`` and of ``duration_samples``. The
    difference is that the event is written as a *drop in envelope amplitude* rather
    than as a mask, so that the caller can apply the production trace filters and
    threshold it the way detection does.
    """
    trace = np.full((n_samples, n_channels), baseline, dtype=np.float32)
    centre = (n_samples - duration_samples) // 2
    jitter = rng.normal(0.0, sigma_samples, size=n_channels)
    ramp = slope_samples_per_channel * np.arange(n_channels)
    onsets = np.rint(centre + ramp - ramp.mean() + jitter).astype(int)
    onsets = np.clip(onsets, 0, n_samples - duration_samples - 1)
    for chan, onset in enumerate(onsets):
        trace[onset : onset + duration_samples, chan] = off_level
    return trace


def simulate_trace_level_edges(
    n_channels_values: list[int],
    sigma_ms: float = 6.0,
    slope_us_per_um: float = 0.0,
    slope_sd_us_per_um: float = 0.0,
    duration_ms: float = 120.0,
    duration_sd_ms: float = 0.0,
    n_events: int = 200,
    baseline: float = 1.0,
    off_level: float = 0.15,
    noise_sd: float = 0.25,
    threshold: float = 0.5,
    seed: int = 0,
    clean_opts: CleanOpts | None = None,
    apply_gaussian: bool = True,
    apply_median: bool = True,
) -> pd.DataFrame:
    """Forward-simulate the whole detection chain, filters included.

    :func:`simulate_detector_edges` injects a mask directly and therefore models only
    the morphological cleaning. Production reaches that mask through three earlier
    steps that this function adds:

    1. a Gaussian low-pass of the MUA envelope at
       :data:`GAUSSIAN_FREQ_MAX_HZ` (sigma ``FS / (2*pi*f_max)`` samples),
    2. a temporal median filter of :data:`MEDIAN_FILTER_SAMPLES` samples, and
    3. thresholding, ``envelope < threshold`` (as in
       ``detect_full._build_state_aware_binary_mask``).

    Those steps are the reason a *lower bound* caveat attaches to the mask-level
    simulation: both filters act over a fixed time constant, so their effect on a
    recovered edge could in principle depend on how long the event lasts. Sweeping
    ``duration_ms`` here with every latent parameter held fixed is what settles
    whether a mechanical duration-to-MAD route exists.

    The latent edge dispersion is ``sigma_ms`` for every event regardless of
    ``n_channels`` and ``duration_ms``, so (exactly as in the mask-level
    simulation) any dependence of the recovered statistics on either is mechanical
    by construction.

    Parameters
    ----------
    n_channels_values
        Channel counts to simulate.
    sigma_ms
        Latent standard deviation of per-channel edge times, held constant.
    slope_us_per_um, slope_sd_us_per_um
        Mean and per-event SD of the latent propagation slope.
    duration_ms, duration_sd_ms
        Mean and per-event SD of the OFF duration. The axis under test.
    n_events
        Events per channel count.
    baseline, off_level
        Envelope amplitude outside and inside an OFF period, in arbitrary units.
        ``off_level / baseline`` sets how deep the dip is.
    noise_sd
        SD of white noise added to the unfiltered envelope.
    threshold
        Detection threshold applied to the filtered envelope. Production derives it
        as a per-bin per-channel quantile; here it is a fixed value between
        ``off_level`` and ``baseline``.
    seed
        RNG seed.
    clean_opts
        Override for :data:`PRODUCTION_CLEAN_OPTS`.
    apply_gaussian, apply_median
        Switch the two trace filters off, to isolate which one is responsible for
        any duration dependence that appears.

    Returns
    -------
    One row per recovered event, with the same columns as
    :func:`simulate_detector_edges` plus ``requested_duration_ms``.
    """
    clean_opts = clean_opts or PRODUCTION_CLEAN_OPTS
    channel_structure = max(
        clean_opts["n_channels_clean"] or 1, clean_opts["n_channels_connect"] or 1
    )
    min_duration_samples = (clean_opts["n_samples_clean"] or 0) + 1

    rng = np.random.default_rng(seed)
    sigma_samples = sigma_ms * 1e-3 * FS
    gaussian_sigma = FS / (2 * np.pi * GAUSSIAN_FREQ_MAX_HZ)
    max_duration_samples = int(round((duration_ms + 4 * duration_sd_ms) * 1e-3 * FS))
    max_slope = abs(slope_us_per_um) + 4 * slope_sd_us_per_um
    max_n = max(n_channels_values)
    # Room for the event, several sigma of jitter, the full ramp, and enough
    # baseline either side that the filters never see two events at once.
    margin = (
        int(round(6 * sigma_samples))
        + int(round(max_slope * 1e-6 * CHANNEL_PITCH_UM * FS * max_n))
        + 8 * MEDIAN_FILTER_SAMPLES
    )
    n_samples = max_duration_samples + 2 * margin

    frames = []
    for n_channels in n_channels_values:
        pad = channel_structure + 2
        block_channels = n_channels + 2 * pad
        canvas = np.full(
            (n_samples * n_events, block_channels), baseline, dtype=np.float32
        )
        for event in range(n_events):
            duration_samples = int(
                round(
                    max(
                        rng.normal(duration_ms, duration_sd_ms) * 1e-3 * FS,
                        min_duration_samples,
                    )
                )
            )
            slope_samples_per_channel = (
                rng.normal(slope_us_per_um, slope_sd_us_per_um)
                * 1e-6
                * CHANNEL_PITCH_UM
                * FS
            )
            trace = _draw_event_trace(
                rng,
                n_channels,
                n_samples,
                duration_samples,
                sigma_samples,
                slope_samples_per_channel,
                baseline,
                off_level,
            )
            row = event * n_samples
            canvas[row : row + n_samples, pad : pad + n_channels] = trace

        canvas += rng.normal(0.0, noise_sd, size=canvas.shape).astype(np.float32)
        np.clip(canvas, 0.0, None, out=canvas)
        # -------------------- the production trace chain --------------------
        if apply_gaussian:
            canvas = scipy.ndimage.gaussian_filter1d(
                canvas, gaussian_sigma, axis=0, mode="nearest"
            )
        if apply_median:
            canvas = scipy.ndimage.median_filter(
                canvas, size=(MEDIAN_FILTER_SAMPLES, 1), mode="nearest"
            )
        off_mask = canvas < threshold
        # and from here on, exactly as production
        cleaned = morphology.clean_binary_mask(off_mask, **clean_opts)
        labels, _ = scipy.ndimage.label(cleaned)
        lbl_ixs = scipy.ndimage.value_indices(labels, ignore_value=0)
        y_coords = np.arange(block_channels) * CHANNEL_PITCH_UM
        time_coords = np.arange(canvas.shape[0]) / FS
        props = morphology.get_off_properties(
            y_coords,
            time_coords,
            FS,
            lbl_ixs,
            values=cleaned.astype(float),
        )
        props["requested_n_channels"] = n_channels
        frames.append(props)

    out = pd.concat(frames, ignore_index=True)
    out["n_channels"] = (out["span"] / CHANNEL_PITCH_UM + 1).round().astype(int)
    out["sigma_ms"] = sigma_ms
    out["slope_us_per_um"] = slope_us_per_um
    out["slope_sd_us_per_um"] = slope_sd_us_per_um
    out["requested_duration_ms"] = duration_ms
    out["duration_ms"] = duration_ms
    return out


def fit_mechanical_surface(
    empirical: pd.DataFrame,
    sigma_grid: list[float],
    slope_sd_grid: list[float],
    edge: str = "onset",
    statistic: str = "mean_mad",
    min_n_channels: int = 8,
    min_simulated_events: int = 30,
    **simulate_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Fit one size-independent (dispersion, propagation) pair to the whole curve.

    The complete apportionment test. Rather than inverting the observed
    statistic separately at each ``n``, this asks whether a single latent
    dispersion *and* a single distribution of propagation slopes (both
    independent of event size) can reproduce the observed statistic across the
    whole range of ``n`` at once.

    If some pair fits well, no size-dependent change in the underlying edge
    behaviour is needed to explain the data: everything the estimator shows about
    size is a consequence of the measurement plus geometry. Residual structure that
    no pair can absorb is the generative component.

    Points below ``min_n_channels`` are excluded from the fit because the statistic
    is floored there and carries no information (see
    :func:`mad_zero_forced_max_n`); they are still returned for display. Morphology
    shifts some simulated events off their requested channel count, so simulated
    cells holding fewer than ``min_simulated_events`` events are dropped rather
    than compared against a well-sampled empirical cell.

    Returns
    -------
    ``(grid, best)``. ``grid`` has one row per (sigma, slope_sd) with the RMS error
    in seconds and the simulated curve; ``best`` describes the minimiser and holds
    its predicted curve under ``"predicted"``.
    """
    n_values = sorted(empirical["n_channels"].unique())
    column = {"mean_mad": f"{edge}_mad", "mean_jitter": f"{edge}_jitter"}[statistic]
    observed = empirical.set_index("n_channels")[statistic]
    fit_n = [n for n in n_values if n >= min_n_channels]

    rows = []
    predictions = {}
    for sigma in sigma_grid:
        for slope_sd in slope_sd_grid:
            sim = simulate_detector_edges(
                n_values,
                sigma_ms=sigma,
                slope_sd_us_per_um=slope_sd,
                **simulate_kwargs,
            )
            grouped = sim.groupby("n_channels")[column]
            curve = grouped.mean()[grouped.size() >= min_simulated_events]
            residual = (curve.reindex(fit_n) - observed.reindex(fit_n)).dropna()
            rmse = float(np.sqrt(np.mean(residual.to_numpy() ** 2)))
            rows.append(
                {
                    "sigma_ms": sigma,
                    "slope_sd_us_per_um": slope_sd,
                    "rmse": rmse,
                    "n_points": len(residual),
                }
            )
            predictions[(sigma, slope_sd)] = curve

    grid = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    winner = grid.iloc[0]
    key = (winner["sigma_ms"], winner["slope_sd_us_per_um"])
    best = {
        "sigma_ms": float(winner["sigma_ms"]),
        "slope_sd_us_per_um": float(winner["slope_sd_us_per_um"]),
        "rmse": float(winner["rmse"]),
        "predicted": predictions[key],
        "observed": observed,
        "fit_n_channels": fit_n,
    }
    return grid, best


# Part 2: common-support direct standardization


def standardize_cell_means(
    events: pd.DataFrame,
    value: str,
    strata: list[str],
    group_col: str = "combo",
    condition_col: str = "condition",
    common_support: bool = True,
) -> pd.DataFrame:
    """Per-group direct standardization of a cell mean over nuisance strata.

    For each group (a subject-probe-structure combo), every condition's events are
    reweighted to that group's *pooled* stratum distribution, so all conditions are
    compared at the same mix of ``strata``. This estimates the condition effect that
    would be seen if the size composition had not moved.

    With ``common_support=True`` the standardization is restricted to strata
    populated in every condition of that group. Renormalizing over each
    condition's own support instead would silently compare conditions on different
    stratum sets: tolerable within NREM, badly wrong across wake conditions, whose
    event-size support is much narrower.

    Parameters
    ----------
    events
        Event frame; must contain ``value``, ``strata``, ``group_col`` and
        ``condition_col``.
    value
        Column to average.
    strata
        Columns defining the strata to standardize over.
    group_col, condition_col
        Grouping and condition column names.
    common_support
        Restrict to strata present in every condition of the group.

    Returns
    -------
    One row per (group, condition) with ``raw`` (the unstandardized mean, i.e. the
    published statistic), ``standardized``, ``n_events``, ``n_events_used`` and
    ``frac_dropped``.
    """
    frame = events[[group_col, condition_col, value, *strata]].copy()
    frame["_stratum"] = list(
        zip(*(frame[col].astype(str) for col in strata), strict=True)
    )

    records = []
    for group, block in frame.groupby(group_col, observed=True):
        conditions = block[condition_col].unique()
        used = block
        if common_support:
            per_condition = [
                set(sub["_stratum"])
                for _, sub in block.groupby(condition_col, observed=True)
            ]
            shared = set.intersection(*per_condition) if per_condition else set()
            used = block[block["_stratum"].isin(shared)]
        if used.empty:
            continue
        weights = used["_stratum"].value_counts(normalize=True)
        cell = used.groupby([condition_col, "_stratum"], observed=True)[value].mean()
        for condition in conditions:
            if condition not in cell.index.get_level_values(0):
                continue
            means = cell.loc[condition]
            aligned = weights.reindex(means.index)
            raw_block = block[block[condition_col] == condition]
            used_block = used[used[condition_col] == condition]
            records.append(
                {
                    group_col: group,
                    condition_col: condition,
                    "raw": raw_block[value].mean(),
                    "standardized": float(
                        (means * aligned).sum() / aligned.sum()
                    ),
                    "n_events": len(raw_block),
                    "n_events_used": len(used_block),
                    "frac_dropped": 1.0 - len(used_block) / len(raw_block),
                }
            )
    return pd.DataFrame.from_records(records)


# -------------------- The shared size-adjustment curve --------------------

SIZE_TERMS = ("shared_curve", "per_combo_factor", "none")
"""How :func:`standardize_by_regression` may represent the size dependence.

``"none"`` fits no size term at all. It is not an adjustment; it recovers the
*unadjusted* contrast through the identical estimator, so adjusted and unadjusted
numbers differ only by the size term and not by the machinery, and it is also how an
arbitrary size coding is expressed: put the coding in ``covariates`` (a polynomial in
channel count, a spline basis) and take the size term out of the way.
"""


@dataclasses.dataclass(frozen=True)
class SharedSizeCurve:
    """One size-adjustment curve, shared by every cell, with per-unit amplitude.

    The mechanical dependence of MAD on event size is a property of the *detector*
    (the structuring element, the sampling grid, the arithmetic of a median), so it
    belongs to the measurement, not to the animal. Estimating it separately inside
    every subject-structure combo is therefore wasteful where events are plentiful
    and ruinous where they are not: a wake combo holding 29 events across 15 distinct
    channel counts cannot support a parameter per count.

    This object holds the curve estimated once, across all cells at once, by
    :func:`fit_shared_size_curve`. Cells differ only in how strongly it applies to
    them (``unit_lambda``), which is one parameter rather than one per channel count.

    Attributes
    ----------
    curve
        ``f(n)``, the *shape* of the size dependence, indexed by channel count and
        normalized to mean 0 and unit standard deviation over events. It is
        therefore dimensionless, and carries no information about how large the size
        effect is, only about its profile. Evaluate it at arbitrary counts with
        :meth:`evaluate`.
    unit_lambda
        The amplitude: the coefficient multiplying ``curve`` in one unit, where a
        unit is one (combo, state) block, the group a single standardization is run
        over. Because ``curve`` has unit standard deviation over events, an amplitude
        is in the units of the response and reads as *how much of the response a
        one-standard-deviation move along the size curve is worth here*. It scales the
        whole curve without changing its shape, which is what "amplitude" means: two
        units with the same amplitude have identically sized size effects, and one with
        twice the amplitude swings twice as far across the same span range.

        For onset MAD in Medium+Large NREM this runs 0.86-3.11 ms across the 29
        subject-structure pairs (median 1.96 ms), a 3.6-fold range, which is why it is
        fitted per unit while the shape is not. Units below
        ``fit_shared_size_curve(min_events_for_amplitude=...)`` carry
        :attr:`pooled_lambda` here instead of an estimate of their own.
    pooled_lambda
        The amplitude fitted with every unit forced to share it. Used for units too
        small to identify their own.
    unit_events, unit_shape_correlation
        Diagnostics: how many events each unit holds, and how well its own
        size profile correlates with the shared shape.
    n_iterations, converged
        Alternating-least-squares bookkeeping.
    """

    curve: pd.Series
    unit_lambda: pd.Series
    pooled_lambda: float
    unit_events: pd.Series
    unit_shape_correlation: pd.Series
    n_iterations: int
    converged: bool

    def evaluate(self, n_channels: np.ndarray | pd.Series) -> np.ndarray:
        """Evaluate the curve, interpolating between (and clamping outside) levels."""
        return np.interp(
            np.asarray(n_channels, dtype=float),
            self.curve.index.to_numpy(dtype=float),
            self.curve.to_numpy(dtype=float),
        )


def _demean_within(values: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Subtract the group mean of ``values`` from every element."""
    totals = np.bincount(codes, weights=values, minlength=len(counts))
    return values - (totals / counts)[codes]


def _cell_demeaned(
    response: np.ndarray,
    curve_values: np.ndarray,
    cell_codes: np.ndarray,
    cell_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Both series with their cell means removed.

    Removing the cell means is what keeps condition out of any amplitude fitted from
    the result: a cell is one (unit, condition) block, so anything the condition does
    to the level of the response is absorbed before the curve is scaled to it.
    """
    return (
        _demean_within(response, cell_codes, cell_counts),
        _demean_within(curve_values, cell_codes, cell_counts),
    )


def _pooled_amplitude_from_codes(
    response: np.ndarray,
    curve_values: np.ndarray,
    cell_codes: np.ndarray,
    cell_counts: np.ndarray,
) -> float:
    """The single least-squares amplitude of ``curve_values`` across all cells."""
    y, f = _cell_demeaned(response, curve_values, cell_codes, cell_counts)
    denominator = float(f @ f)
    return float(f @ y) / denominator if denominator > 0 else 0.0


def _unit_amplitudes(
    response: np.ndarray,
    curve_values: np.ndarray,
    cell_codes: np.ndarray,
    cell_counts: np.ndarray,
    unit_codes: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """One least-squares amplitude of ``curve_values`` per unit."""
    y, f = _cell_demeaned(response, curve_values, cell_codes, cell_counts)
    numerator = np.bincount(unit_codes, weights=f * y, minlength=n_units)
    denominator = np.bincount(unit_codes, weights=f * f, minlength=n_units)
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )


def _solve_curve(
    response: np.ndarray,
    lam: np.ndarray,
    level_codes: np.ndarray,
    n_levels: int,
    cell_codes: np.ndarray,
    cell_counts: np.ndarray,
) -> np.ndarray:
    """Solve ``response ~ cell_dummies + lam * f(level)`` for ``f``, exactly.

    The cell means are profiled out algebraically (Frisch-Waugh) rather than carried
    as columns, which turns a design with one column per cell *and* per level into an
    ``n_levels x n_levels`` dense system. Every cross-product is a
    :func:`numpy.bincount`, so the whole solve is linear in the number of events.
    """
    n_cells = len(cell_counts)
    wty = np.bincount(level_codes, weights=lam * response, minlength=n_levels)
    wtw = np.bincount(level_codes, weights=lam * lam, minlength=n_levels)
    cty = np.bincount(cell_codes, weights=response, minlength=n_cells)
    cross = np.bincount(
        level_codes * n_cells + cell_codes, weights=lam, minlength=n_levels * n_cells
    ).reshape(n_levels, n_cells)
    scaled = cross / cell_counts
    design = np.diag(wtw) - scaled @ cross.T
    target = wty - scaled @ cty
    # Adding a constant to f is absorbed by the cell means, so the system is rank
    # deficient by one; lstsq returns the minimum-norm solution, which the caller
    # then re-centres anyway.
    return np.linalg.lstsq(design, target, rcond=None)[0]


def fit_shared_size_curve(
    events: pd.DataFrame,
    value: str,
    unit_col: str,
    cell_col: str,
    n_channels_col: str = "n_channels",
    min_events_for_amplitude: int = 60,
    n_iter: int = 50,
    tol: float = 1e-9,
) -> SharedSizeCurve:
    """Estimate one size-adjustment curve shared across cells, by ALS.

    Fits

    ``value_i = alpha_{cell(i)} + lambda_{unit(i)} * f(n_i) + eps_i``

    by alternating least squares: solve for ``f`` with the amplitudes held fixed
    (:func:`_solve_curve`), then for the amplitudes with ``f`` held fixed
    (:func:`_amplitude`), and repeat. Both steps profile out the cell means, so the
    shape ``f`` and the amplitudes are estimated from *within-cell* variation only
    and cannot absorb a condition effect, which is the whole point, since the
    contrast being adjusted lives between cells of the same unit.

    ``f`` is normalized to mean zero and unit standard deviation over events, and its
    sign is fixed so that the pooled amplitude is positive (MAD rises with size).

    Parameters
    ----------
    events
        Event-level frame; needs ``value``, ``n_channels_col``, ``unit_col`` and
        ``cell_col``.
    unit_col
        Column identifying the block a single standardization is run over: one
        (combo, state) pair in production.
    cell_col
        Column identifying (unit, condition), the unit the mixed model treats as one
        observation.
    min_events_for_amplitude
        Units holding fewer events than this are given the pooled amplitude instead
        of their own. A unit with a handful of events cannot identify an amplitude
        (left free, some come back negative), and letting such an estimate weight the
        shared curve would be letting noise steer the thing every other cell relies
        on.
    n_iter, tol
        Maximum alternations and the convergence tolerance on ``f``. Convergence is
        geometric; the production fits reach ``1e-9`` in about 25 iterations.

    Returns
    -------
    A :class:`SharedSizeCurve`.
    """
    frame = events[[value, n_channels_col, unit_col, cell_col]].dropna()
    if frame.empty:
        raise ValueError(f"No events with a finite {value!r} to fit a size curve to")

    levels = frame[n_channels_col].astype(int).astype("category")
    cells = frame[cell_col].astype("category")
    units = frame[unit_col].astype("category")
    # Categorical codes come back as the narrowest integer dtype that fits, which
    # overflows when the level and cell codes are combined into one flat index.
    level_codes = levels.cat.codes.to_numpy(dtype=np.intp)
    cell_codes = cells.cat.codes.to_numpy(dtype=np.intp)
    unit_codes = units.cat.codes.to_numpy(dtype=np.intp)
    n_levels = len(levels.cat.categories)
    n_units = len(units.cat.categories)
    cell_counts = np.bincount(cell_codes, minlength=len(cells.cat.categories)).astype(float)
    response = frame[value].to_numpy(dtype=float)

    unit_events = np.bincount(unit_codes, minlength=n_units)
    identified = unit_events >= min_events_for_amplitude
    lam = np.ones(len(frame))
    unit_lambda = np.ones(n_units)
    curve = np.zeros(n_levels)
    converged = False
    iterations = 0
    for iterations in range(1, n_iter + 1):
        previous = curve
        curve = _solve_curve(
            response, lam, level_codes, n_levels, cell_codes, cell_counts
        )
        per_event = curve[level_codes]
        curve = curve - per_event.mean()
        spread = per_event.std()
        if spread > 0:
            curve = curve / spread
        per_event = curve[level_codes]
        unit_lambda = _unit_amplitudes(
            response, per_event, cell_codes, cell_counts, unit_codes, n_units
        )
        unit_lambda[~identified] = _pooled_amplitude_from_codes(
            response, per_event, cell_codes, cell_counts
        )
        lam = unit_lambda[unit_codes]
        if np.max(np.abs(curve - previous)) < tol:
            converged = True
            break

    pooled = _pooled_amplitude_from_codes(
        response, curve[level_codes], cell_codes, cell_counts
    )
    if pooled < 0:
        curve, unit_lambda, pooled = -curve, -unit_lambda, -pooled

    # Diagnostic: how much of each unit's own size profile the shared shape captures.
    shape = pd.Series(curve, index=levels.cat.categories.astype(int))
    profile = pd.DataFrame(
        {
            "unit": units.to_numpy(),
            "level": levels.to_numpy(),
            "residual": _demean_within(response, cell_codes, cell_counts),
        }
    )
    observed = (
        profile.groupby(["unit", "level"], observed=True)["residual"].mean().reset_index()
    )
    observed["shared"] = shape.reindex(observed["level"].to_numpy()).to_numpy()
    correlation = pd.Series(
        {
            unit: (
                block["residual"].corr(block["shared"])
                if len(block) > 2
                and block["residual"].std() > 0
                and block["shared"].std() > 0
                else np.nan
            )
            for unit, block in observed.groupby("unit", observed=True)
        },
        dtype=float,
    )

    return SharedSizeCurve(
        curve=shape,
        unit_lambda=pd.Series(unit_lambda, index=units.cat.categories),
        pooled_lambda=pooled,
        unit_events=pd.Series(unit_events, index=units.cat.categories),
        unit_shape_correlation=correlation.reindex(units.cat.categories),
        n_iterations=iterations,
        converged=converged,
    )


def pooled_amplitude(
    events: pd.DataFrame,
    value: str,
    curve: SharedSizeCurve,
    condition_col: str = "condition",
    group_col: str = "combo",
) -> float:
    """The one amplitude that best fits ``curve`` to ``events``, cell means removed.

    This is what a combo too small to fit its own amplitude is given. It is computed
    per state rather than once overall because wake MAD sits at a lower level than
    NREM MAD, so an amplitude borrowed across states over-corrects; the failure mode
    that makes a *fixed additive* shared curve flip the wake contrast's sign.
    """
    cells = (
        events[group_col].astype(str) + "@" + events[condition_col].astype(str)
    ).astype("category")
    codes = cells.cat.codes.to_numpy(dtype=np.intp)
    counts = np.bincount(codes, minlength=len(cells.cat.categories)).astype(float)
    return _pooled_amplitude_from_codes(
        events[value].to_numpy(dtype=float),
        curve.evaluate(events["n_channels"]),
        codes,
        counts,
    )


# -------------------- Regression standardization (g-computation) --------------------


def _size_bins(n_channels: pd.Series, n_bins: int, exact: bool = False) -> pd.Series:
    """Discretize channel count for use as a factor.

    ``exact=True`` gives one level per distinct channel count. That is the only
    coding fine enough to absorb the parity sawtooth in the MAD floor (see
    :func:`mad_zero_min_ties`), since the sawtooth alternates between *adjacent*
    integers and any coarser bin averages over it. It costs a parameter per
    observed channel count, so it needs a well-populated combo.
    """
    if exact:
        return n_channels.astype(int)
    binned = pd.qcut(n_channels.rank(method="first"), n_bins, labels=False,
                     duplicates="drop")
    return pd.Series(binned, index=n_channels.index).astype(int)


def _choose_size_coding(
    n_channels: pd.Series,
    n_size_bins: int,
    events_per_size_bin: int,
    prefer_exact: bool,
) -> tuple[pd.Series, str]:
    """Code channel count as finely as the combo's event count can support.

    Exact levels are preferred where they are affordable: they impose no shape, and
    they are the only coding that represents the parity sawtooth in
    :func:`mad_zero_min_ties`. They cost a parameter per level, though, and a wake
    combo holding a hundred events across forty distinct channel counts would be
    fitting noise. Combos that cannot afford exact levels fall back to quantile bins,
    whose number also adapts (``n_events // events_per_size_bin``, clipped to
    ``[2, n_size_bins]``). Which coding a combo received is reported per row so the
    mix is visible rather than assumed.

    This whole per-combo scheme is the sensitivity path, not the shipped one. A
    mechanical size effect is a property of the detector rather than of the animal, so
    re-estimating it inside every combo spends parameters to learn the same curve
    repeatedly, and breaks down entirely where events are scarce. What ships is
    :func:`fit_shared_size_curve`; see ``size_term`` on
    :func:`standardize_by_regression`.
    """
    n_distinct = int(n_channels.nunique())
    if prefer_exact and len(n_channels) >= events_per_size_bin * n_distinct:
        return _size_bins(n_channels, 0, exact=True), "exact"
    bins = int(np.clip(len(n_channels) // events_per_size_bin, 2, n_size_bins))
    return _size_bins(n_channels, bins, exact=False), f"quantile[{bins}]"


def _gcomp_design(
    block: pd.DataFrame,
    conditions: list[str],
    size_bins: list[int],
    covariates: list[str],
    condition_col: str,
    interaction: bool,
    size_column: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Design columns for the g-computation fit, in a fixed, reusable order.

    ``size_column`` names a single continuous size term (the shared curve evaluated
    at each event's channel count), and replaces the per-combo dummies with it.
    """
    columns: dict[str, np.ndarray] = {"const": np.ones(len(block))}
    for condition in conditions[1:]:
        columns[f"cond[{condition}]"] = (
            block[condition_col].to_numpy() == condition
        ).astype(float)
    size_terms: list[str] = []
    if size_column is not None:
        columns[size_column] = block[size_column].to_numpy(dtype=float)
        size_terms = [size_column]
    else:
        for size_bin in size_bins[1:]:
            name = f"size[{size_bin}]"
            columns[name] = (block["_size_bin"].to_numpy() == size_bin).astype(float)
            size_terms.append(name)
    if interaction:
        for condition in conditions[1:]:
            for size_term in size_terms:
                columns[f"cond[{condition}]:{size_term}"] = (
                    columns[f"cond[{condition}]"] * columns[size_term]
                )
    for covariate in covariates:
        columns[covariate] = block[covariate].to_numpy(dtype=float)
    names = list(columns)
    return np.column_stack([columns[name] for name in names]), names


def _gcomp_cell_means(
    block: pd.DataFrame,
    conditions: list[str],
    size_bins: list[int],
    covariates: list[str],
    condition_col: str,
    interaction: bool,
    size_column: str | None = None,
    offset: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit once, then predict every condition over the *whole* event population.

    ``offset`` is a per-event quantity held out of the fit and added back to every
    prediction: the shared curve at a *fixed* amplitude, for units too small to
    identify their own. Because g-computation averages each condition's predictions
    over the same events, a common offset shifts every condition's adjusted mean by
    the same amount and leaves the contrasts untouched; it is carried so the adjusted
    means stay on the response scale.
    """
    design, _ = _gcomp_design(
        block, conditions, size_bins, covariates, condition_col, interaction,
        size_column,
    )
    response = block[condition_col + "_response"].to_numpy(dtype=float)
    if offset is not None:
        response = response - offset
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    shift = float(offset.mean()) if offset is not None else 0.0

    means = {}
    for condition in conditions:
        counterfactual = block.assign(**{condition_col: condition})
        design_c, _ = _gcomp_design(
            counterfactual, conditions, size_bins, covariates, condition_col,
            interaction, size_column,
        )
        means[condition] = float((design_c @ coefficients).mean()) + shift
    return means


def standardize_by_regression(
    events: pd.DataFrame,
    value: str,
    condition_col: str = "condition",
    group_col: str = "combo",
    covariates: tuple[str, ...] = (),
    size_term: Literal["shared_curve", "per_combo_factor"] = "shared_curve",
    shared_curve: SharedSizeCurve | None = None,
    pooled_lambda: float | None = None,
    free_lambda_min_events: int = 60,
    n_size_bins: int = 6,
    events_per_size_bin: int = 60,
    exact_size_levels: bool = True,
    interaction: bool = False,
    min_events: int | None = None,
    n_boot: int = 0,
    block_seconds: float = 60.0,
    time_col: str = "start_time",
    seed: int = 0,
) -> pd.DataFrame:
    """Marginal (regression) standardization of each cell mean: the g-formula.

    For each subject-structure combo, fits

    ``value ~ condition + size``

    over that combo's individual OFF periods, then for each condition predicts
    the fitted value for *every* event in the combo with the condition set to that
    value, and averages. Each condition is therefore scored over one and the same
    population of event sizes (the combo's own), so the resulting per-cell numbers
    are directly comparable and can be modelled and plotted exactly like the raw
    cell means they replace.

    Why this rather than the stratified reweighting in
    :func:`standardize_cell_means`: reweighting needs every condition to populate
    every stratum, which holds comfortably in NREM but fails badly in wake (a wake
    cell can lose 45-79 % of its events to the common-support restriction). The
    regression form buys that support with an assumption about the shape of the
    size effect instead, which is why it is worth running both and comparing.

    What belongs on the right-hand side. Only variables whose relation to
    ``value`` is *mechanical*: a property of the estimator rather than of the
    events. Channel count qualifies: the detector's spatial structuring element
    forces ``MAD == 0`` below a 120 µm span and the estimator keeps climbing with
    span above it, and :func:`simulate_trace_level_edges` reproduces the whole
    observed curve from a single size-independent latent dispersion. Event
    *duration* does not qualify: swept through the same full production chain with
    the latent dispersion held fixed, recovered MAD does not move, so the strong
    empirical MAD-vs-duration relation is generative. Duration is therefore a
    mediator of the condition effect, and conditioning on it would remove real
    signal. It is available through ``covariates`` for sensitivity analysis, but it
    is deliberately not a default.

    How size enters (``size_term``):

    ``"shared_curve"`` (default, and what ships)
        One curve ``f(n)``, estimated once across all cells by
        :func:`fit_shared_size_curve` and passed in as ``shared_curve``, enters as a
        single continuous column whose coefficient (the combo's amplitude) is the
        only size parameter fitted here. The mechanical size effect is a property of
        the detector, not of the animal, so its *shape* is shared; only how strongly
        it bites is allowed to differ. Combos holding fewer than
        ``free_lambda_min_events`` events cannot identify even that one coefficient,
        so they are given ``pooled_lambda`` (held fixed, as an offset) rather than
        being dropped. No combo is dropped for being small: the adjusted panel
        therefore covers exactly the cells the unadjusted one does.
    ``"none"``
        No size term. Gives the unadjusted contrast through the identical estimator,
        and is the hook for any other coding: pass it through ``covariates``.
    ``"per_combo_factor"``
        The earlier scheme, kept for sensitivity analysis: channel count as a factor
        re-estimated inside every combo, at one level per distinct count where the
        combo can afford it and quantile bins where it cannot
        (:func:`_choose_size_coding`). A factor imposes no functional form and is the
        only coding that resolves the parity sawtooth in ``P(MAD == 0)`` (see
        :func:`mad_zero_min_ties`), which is substantial up to ``n`` around 20. But
        the sawtooth is a function of ``n`` alone, and g-computation scores every
        condition over one common ``n`` population, so it contributes equally to each
        condition's marginal mean and cancels in the contrast; a two-parameter
        quadratic reproduces the factor estimate to 0.001 ms. Where events are scarce
        the factor is actively harmful, which is why it is not the default.

    Parameters
    ----------
    events
        Event-level frame; needs ``value``, ``n_channels``, the covariates,
        ``group_col``, ``condition_col`` and (for bootstrapping) ``time_col``.
    covariates
        Extra continuous terms. Empty by default; see above.
    shared_curve
        The curve to use under ``size_term="shared_curve"``. Fitted from ``events``
        themselves when omitted, which is right for a standalone call but *not* for
        the production export, where the curve is fitted once across both states and
        passed in (see :func:`compute_adjusted_cell_means`).
    pooled_lambda
        Amplitude given to combos below ``free_lambda_min_events``. Defaults to the
        curve's own pooled amplitude; production passes the state-specific one,
        because wake MAD sits at a lower level than NREM MAD and an amplitude
        borrowed across states over-corrects.
    interaction
        Whether to fit ``condition × size``. Off by default: under the mechanical
        reading the size effect is a property of the estimator and so cannot depend
        on condition, and letting it do so gives the model a route to absorb the
        very condition signal being estimated. Retained as a sensitivity switch.
    min_events
        Combos below this are skipped entirely. Defaults to 1 under
        ``"shared_curve"`` (nothing is dropped) and to 60 under
        ``"per_combo_factor"``, which is the threshold the per-combo path needs and
        the value it historically used.
    n_boot
        Bootstrap replicates for the standard error. Resampling is over
        ``block_seconds``-long time blocks within each combo, so it respects the
        temporal clustering of OFF periods rather than treating events as
        independent. 0 disables it.

    Returns
    -------
    One row per (group, condition) with ``raw`` (the published statistic),
    ``standardized``, ``n_events``, ``size_coding``, ``size_lambda`` (the fitted
    amplitude, on the response scale), ``positivity`` (the share of the prediction
    population lying at sizes this condition actually populates; 1.0 is pure
    interpolation, lower values mean the rest is extrapolated from the fitted size
    term), and, when bootstrapping, ``se`` and a percentile interval.
    """
    if size_term not in SIZE_TERMS:
        raise ValueError(f"size_term must be one of {SIZE_TERMS}, got {size_term!r}")
    covariate_list = list(covariates)
    rng = np.random.default_rng(seed)
    records = []
    shared = size_term == "shared_curve"
    if min_events is None:
        min_events = 60 if size_term == "per_combo_factor" else 1
    if shared:
        if shared_curve is None:
            frame = events.assign(
                _unit=events[group_col].astype(str),
                _cell=events[group_col].astype(str)
                + "@"
                + events[condition_col].astype(str),
            )
            shared_curve = fit_shared_size_curve(frame, value, "_unit", "_cell")
        if pooled_lambda is None:
            pooled_lambda = shared_curve.pooled_lambda
    size_column = "_size_curve" if shared else None

    for group, block in events.groupby(group_col, observed=True):
        if len(block) < min_events:
            continue
        block = block.copy()
        block[condition_col] = block[condition_col].astype(str)
        block[condition_col + "_response"] = block[value].to_numpy(dtype=float)
        offset = None
        if size_term == "none":
            block["_size_bin"] = 0
            size_coding = "none"
            free_lambda = False
        elif shared:
            assert shared_curve is not None and pooled_lambda is not None
            block[size_column] = shared_curve.evaluate(block["n_channels"])
            block["_size_bin"] = 0
            free_lambda = len(block) >= free_lambda_min_events
            size_coding = f"shared_curve[{'free' if free_lambda else 'pooled'}]"
            if not free_lambda:
                offset = pooled_lambda * block[size_column].to_numpy(dtype=float)
        else:
            block["_size_bin"], size_coding = _choose_size_coding(
                block["n_channels"], n_size_bins, events_per_size_bin,
                exact_size_levels,
            )
            free_lambda = True

        conditions = sorted(block[condition_col].unique())
        size_bins = sorted(block["_size_bin"].unique())
        fit_kwargs = dict(size_column=size_column if free_lambda else None)

        means = _gcomp_cell_means(
            block, conditions, size_bins, covariate_list, condition_col, interaction,
            offset=offset, **fit_kwargs,
        )
        raw = block.groupby(condition_col, observed=True)[value].mean()
        size_lambda = float("nan")
        if shared and size_column is not None:
            size_lambda = float(pooled_lambda or 0.0)
            if free_lambda:
                design, names = _gcomp_design(
                    block, conditions, size_bins, covariate_list, condition_col,
                    interaction, size_column,
                )
                response = block[condition_col + "_response"].to_numpy(dtype=float)
                coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
                size_lambda = float(coefficients[names.index(size_column)])
        # Positivity: what share of the population each condition is predicted over
        # is *interpolation* (a size this condition actually populates) rather than
        # extrapolation from the fitted size term. The shared curve is continuous, so
        # anything inside a condition's own span of channel counts is interpolation;
        # the per-combo factor only ever sees the levels it observed.
        if shared:
            counts = block["n_channels"].to_numpy()
            spans = block.groupby(condition_col, observed=True)["n_channels"].agg(
                ["min", "max"]
            )
            support = {
                condition: float(
                    ((counts >= row["min"]) & (counts <= row["max"])).mean()
                )
                for condition, row in spans.iterrows()
            }
        else:
            occupied = block.groupby(condition_col, observed=True)["_size_bin"].agg(set)
            level_counts = block["_size_bin"].value_counts()
            support = {
                condition: float(level_counts[list(levels)].sum() / len(block))
                for condition, levels in occupied.items()
            }

        draws: dict[str, list[float]] = {c: [] for c in conditions}
        if n_boot:
            blocks = (block[time_col].to_numpy() // block_seconds).astype(np.int64)
            unique_blocks = np.unique(blocks)
            index_by_block = {b: np.flatnonzero(blocks == b) for b in unique_blocks}
            for _ in range(n_boot):
                picked = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
                rows = np.concatenate([index_by_block[b] for b in picked])
                resampled = block.iloc[rows]
                if resampled[condition_col].nunique() < len(conditions):
                    continue
                try:
                    boot_means = _gcomp_cell_means(
                        resampled, conditions, size_bins, covariate_list,
                        condition_col, interaction,
                        offset=None if offset is None else offset[rows],
                        **fit_kwargs,
                    )
                except np.linalg.LinAlgError:  # pragma: no cover - defensive
                    continue
                for condition, mean in boot_means.items():
                    draws[condition].append(mean)

        for condition in conditions:
            record = {
                group_col: group,
                condition_col: condition,
                "raw": float(raw[condition]),
                "standardized": means[condition],
                "n_events": int((block[condition_col] == condition).sum()),
                "interaction": interaction,
                "size_coding": size_coding,
                "n_size_bins": 1 if shared else len(size_bins),
                "size_lambda": float(size_lambda),
                "positivity": support[condition],
            }
            if n_boot and draws[condition]:
                sample = np.asarray(draws[condition])
                record["se"] = float(sample.std(ddof=1))
                record["ci_lo"] = float(np.percentile(sample, 2.5))
                record["ci_hi"] = float(np.percentile(sample, 97.5))
                record["n_boot"] = len(sample)
            records.append(record)

    return pd.DataFrame.from_records(records)


# Export of size-adjusted cell means for r-offp

ADJUSTED_DATASETS = ["llas", "clas", "llas_exclusive"]
"""OFF classes for which adjusted edge statistics are exported."""

ADJUSTED_PREFIX = "adj_"
"""Prefix distinguishing adjusted response variables from the published ones."""

ADJUSTED_SCALE = 1e3
"""Adjusted edge statistics are exported in milliseconds, not seconds.

The published `mean_*_mad` columns are in seconds, which puts the axis labels of
S2b at values like 0.0045. Since these columns are new, they are scaled once here
so figures and tables read in ms. Contrasts and confidence intervals scale by the
same constant; Cohen's d and f-squared are scale-invariant and are unaffected.
"""


def compute_adjusted_cell_means(
    events: pd.DataFrame,
    edges: tuple[str, ...] = ("onset", "offset"),
    n_boot: int = 0,
    size_term: Literal["shared_curve", "per_combo_factor"] = "shared_curve",
    **standardize_kwargs,
) -> pd.DataFrame:
    """Size-adjusted cell means for both edges, standardized within state.

    The g-computation is run separately over the NREM conditions and the wake
    conditions rather than over all six at once. Pooling them would standardize
    the wake cells to a size distribution dominated by NREM events, which are far
    larger; and every post-hoc contrast the manuscript draws is within-state
    anyway (the four NREM contrasts and `NOD.Incline`), so nothing is lost.

    The shared size curve, by contrast, is fitted once across both states. It
    describes the detector rather than the animal or the vigilance state, so there is
    no reason to estimate it twice, and every reason not to, since wake holds a few
    thousand events against NREM's several hundred thousand. What is allowed to
    differ by state is the *amplitude*: wake MAD sits at a lower level, so each
    state gets its own pooled amplitude for the combos too small to fit their own.

    Returns one row per (subject, probe, structure, condition) with
    ``adj_mean_{onset,offset}_mad`` alongside the unadjusted ``mean_*`` for
    comparison.
    """
    curves: dict[str, SharedSizeCurve] = {}
    if size_term == "shared_curve":
        keyed = events.assign(
            _unit=events["combo"].astype(str)
            + "@"
            + np.where(events["condition"].isin(NREM_CONDITIONS), "nrem", "wake"),
        )
        keyed["_cell"] = keyed["_unit"] + "@" + keyed["condition"].astype(str)
        for edge in edges:
            curves[edge] = fit_shared_size_curve(keyed, f"{edge}_mad", "_unit", "_cell")

    frames = []
    for conditions in (NREM_CONDITIONS, WAKE_CONDITIONS):
        state_events = events[events["condition"].isin(conditions)]
        if state_events.empty:
            continue
        per_edge = []
        for edge in edges:
            curve = curves.get(edge)
            adjusted = standardize_by_regression(
                state_events,
                f"{edge}_mad",
                n_boot=n_boot,
                size_term=size_term,
                shared_curve=curve,
                pooled_lambda=(
                    None
                    if curve is None
                    else pooled_amplitude(state_events, f"{edge}_mad", curve)
                ),
                **standardize_kwargs,
            )
            if adjusted.empty:
                continue
            adjusted = adjusted.rename(columns={
                "raw": f"mean_{edge}_mad",
                "standardized": f"{ADJUSTED_PREFIX}mean_{edge}_mad",
                "se": f"{ADJUSTED_PREFIX}mean_{edge}_mad_se",
                "ci_lo": f"{ADJUSTED_PREFIX}mean_{edge}_mad_ci_lo",
                "ci_hi": f"{ADJUSTED_PREFIX}mean_{edge}_mad_ci_hi",
                "interaction": f"{edge}_interaction",
                "size_coding": f"{edge}_size_coding",
                "n_size_bins": f"{edge}_n_size_levels",
                "size_lambda": f"{edge}_size_lambda",
                "positivity": f"{edge}_positivity",
            })
            per_edge.append(adjusted.set_index(["combo", "condition"]))
        if not per_edge:
            continue
        merged = per_edge[0]
        for extra in per_edge[1:]:
            merged = merged.join(
                extra.drop(columns=["n_events", "n_boot"], errors="ignore"),
                how="outer", rsuffix="_dup",
            )
        frames.append(merged.reset_index())

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_dup")])
    for column in out.columns:
        if column.startswith(ADJUSTED_PREFIX) or column.endswith("_size_lambda"):
            out[column] = out[column] * ADJUSTED_SCALE
    keys = out["combo"].str.split("|", expand=True)
    out["subject"], out["probe"], out["structure"] = keys[0], keys[1], keys[2]
    front = [*GROUP_COLS, "condition"]
    return out[front + [c for c in out.columns if c not in front and c != "combo"]]


def export_adjusted_edge_statistics(
    output_dir: pathlib.Path,
    datasets: tuple[str, ...] = tuple(ADJUSTED_DATASETS),
    n_boot: int = 0,
    extdata_dir: pathlib.Path | None = None,
    size_term: Literal["shared_curve", "per_combo_factor"] = "shared_curve",
    **standardize_kwargs,
) -> dict[str, pathlib.Path]:
    """Write ``summarized_full48h_<dataset>_edge_adjusted.parquet`` for r-offp.

    Additive companion to ``export-full48h-offs``: reads the event-level
    ``full48h_llas_offs.parquet`` -- from the checkout if it is there, otherwise
    fetched from the Release -- and writes one adjusted cell-mean table per OFF
    class. Requires no NFS and touches no existing artifact.

    The size curve is fitted independently for each dataset, because each OFF class
    covers a different range of channel counts (the Small class is largely below the
    hard floor, where the curve is flat at zero by construction).
    """
    events = load_events("llas", extdata_dir=extdata_dir)
    events["log_median_duration"] = np.log(events["median_duration"])

    written = {}
    for dataset in datasets:
        if dataset == "llas":
            subset = events
        elif dataset == "clas":
            subset = events[events["size_class"] == "Medium+Large"]
        elif dataset == "llas_exclusive":
            subset = events[events["size_class"] == "Small"]
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")

        table = compute_adjusted_cell_means(
            subset, n_boot=n_boot, size_term=size_term, **standardize_kwargs
        )
        path = output_dir / f"summarized_full48h_{dataset}_edge_adjusted.parquet"
        table.to_parquet(path, index=False)
        written[dataset] = path
    return written


# -------------------- Part 2: event-level covariate model --------------------


def _design_matrix(
    block: pd.DataFrame,
    condition_col: str,
    reference: str,
    covariates: list[str] | None,
    n_channel_factor: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the per-group design: condition dummies plus size covariates."""
    conditions = [c for c in block[condition_col].unique() if c != reference]
    design = pd.DataFrame(index=block.index)
    for condition in sorted(conditions):
        design[f"cond[{condition}]"] = (
            block[condition_col] == condition
        ).astype(float)
    condition_terms = list(design.columns)

    if n_channel_factor:
        # A dummy per observed channel count: the most flexible possible control
        # for the estimator's n-dependence, with no functional-form assumption.
        levels = sorted(block["n_channels"].unique())[1:]
        for level in levels:
            design[f"n_chan[{level}]"] = (block["n_channels"] == level).astype(float)
    for covariate in covariates or []:
        design[covariate] = block[covariate].to_numpy(dtype=float)
    design = sm.add_constant(design, has_constant="add")
    return design, condition_terms


def fit_event_level(
    events: pd.DataFrame,
    value: str,
    reference: str,
    condition_col: str = "condition",
    group_col: str = "combo",
    covariates: list[str] | None = None,
    n_channel_factor: bool = True,
    min_events: int = 500,
    cluster_seconds: float | None = None,
    time_col: str = "start_time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group covariate-adjusted condition effects, pooled across groups.

    Fits, within each group, ``value ~ condition + n_channels + covariates`` by OLS
    and extracts the condition contrasts against ``reference``. Group estimates are
    then combined with DerSimonian-Laird random effects
    (:func:`cnpix_local_sleep.morphological.correlation_stats._dersimonian_laird`), matching the
    two-stage approach used in the sequential-added-value
    notebook. This is deliberately not one 1.4M-row mixed model: the two-stage form
    keeps the per-group heterogeneity visible and is what the rest of the package
    already does.

    Passing ``n_channel_factor=False`` and ``covariates=None`` gives the unadjusted
    contrasts through the identical estimator, so adjusted and unadjusted numbers
    are directly comparable.

    Returns
    -------
    ``(per_group, pooled)``. ``per_group`` has one row per (group, contrast) with
    ``estimate``, ``se`` and ``n_events``; ``pooled`` has one row per contrast with
    ``estimate``, ``se``, ``ci_lo``, ``ci_hi``, ``p``, ``i_squared``, ``k``.
    """
    records = []
    for group, block in events.groupby(group_col, observed=True):
        if len(block) < min_events:
            continue
        if reference not in set(block[condition_col]):
            continue
        design, condition_terms = _design_matrix(
            block, condition_col, reference, covariates, n_channel_factor
        )
        response = block[value].to_numpy(dtype=float)
        keep = np.isfinite(response) & np.isfinite(design.to_numpy()).all(axis=1)
        if keep.sum() < min_events:
            continue
        if cluster_seconds:
            # Events within a bout are not independent draws; clustering the
            # sandwich estimator on time blocks stops stage 1 reporting
            # standard errors that assume they are.
            clusters = (block.loc[keep, time_col] // cluster_seconds).astype(
                np.int64
            )
            model = sm.OLS(response[keep], design[keep]).fit(
                cov_type="cluster", cov_kwds={"groups": clusters.to_numpy()}
            )
        else:
            model = sm.OLS(response[keep], design[keep]).fit()
        for term in condition_terms:
            if term not in model.params.index:
                continue
            records.append(
                {
                    group_col: group,
                    "contrast": term.removeprefix("cond[").removesuffix("]"),
                    "estimate": float(model.params[term]),
                    "se": float(model.bse[term]),
                    "n_events": int(keep.sum()),
                }
            )

    per_group = pd.DataFrame.from_records(records)
    if per_group.empty:
        return per_group, pd.DataFrame()

    pooled = []
    for contrast, block in per_group.groupby("contrast"):
        block = block[block["se"] > 0]
        if len(block) < 2:
            continue
        dl = correlation_stats._dersimonian_laird(
            block["estimate"].to_numpy(), block["se"].to_numpy() ** 2
        )
        estimate, se = dl["overall_z"], dl["overall_z_se"]
        pooled.append(
            {
                "contrast": contrast,
                "estimate": estimate,
                "se": se,
                "ci_lo": estimate - 1.96 * se,
                "ci_hi": estimate + 1.96 * se,
                "p": 2 * scipy.stats.norm.sf(abs(estimate / se)),
                "i_squared": dl["i_squared"],
                "k": len(block),
            }
        )
    return per_group, pd.DataFrame.from_records(pooled)


# Part 2: floor-free and scale-free alternatives


def add_floor_free_rvs(events: pd.DataFrame) -> pd.DataFrame:
    """Append the alternative edge statistics, in place, and return the frame.

    All of these are already computed per event by
    :func:`cnpix_local_sleep.morphological.morphology._edge_synchrony`; this only derives the
    convenience combinations.

    ``{edge}_abs_slope``
        ``|slope|`` in s/µm: inverse apparent propagation speed. Lower means a
        faster, more synchronous sweep. Its cell-level aggregate
        ``median_abs_{edge}_slope`` is already exported.
    ``{edge}_ramp``
        ``|slope| * span``, the traversal time implied by the fitted wavefront:
        the part of the raw edge spread that is pure geometry.
    ``{edge}_mad_per_um``
        ``mad / span``, the scale-free version of MAD.
    ``{edge}_mad_rel_duration``
        ``mad / median_duration``, edge spread relative to the event's own length.
    """
    for edge in ("onset", "offset"):
        events[f"{edge}_abs_slope"] = events[f"{edge}_slope"].abs()
        events[f"{edge}_ramp"] = events[f"{edge}_abs_slope"] * events["span"]
        events[f"{edge}_mad_per_um"] = events[f"{edge}_mad"] / events["span"]
        events[f"{edge}_mad_rel_duration"] = (
            events[f"{edge}_mad"] / events["median_duration"]
        )
    return events


# Mixture decomposition of the "All OFFs" panel


def decompose_mixture(
    events: pd.DataFrame,
    value: str,
    class_col: str = "size_class",
    group_col: str = "combo",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Split a pooled cell mean into within-class and composition components.

    The "All OFFs" statistic averages two populations whose means differ several
    fold, with mixing weights that move across conditions. For each (group,
    condition) this returns the observed mean alongside two counterfactuals: the
    mean the cell would show with its own class means but the group's pooled class
    weights (``fixed_weights``), and with the group's pooled class means but its own
    weights (``fixed_class_means``).

    This is a decomposition, not a correction. Whether the composition component
    should be treated as signal or nuisance depends on whether the class difference
    in ``value`` is generative or mechanical.
    """
    records = []
    for group, block in events.groupby(group_col, observed=True):
        pooled_weights = block[class_col].value_counts(normalize=True)
        pooled_means = block.groupby(class_col, observed=True)[value].mean()
        for condition, cell in block.groupby(condition_col, observed=True):
            weights = cell[class_col].value_counts(normalize=True)
            means = cell.groupby(class_col, observed=True)[value].mean()
            classes = means.dropna().index
            records.append(
                {
                    group_col: group,
                    condition_col: condition,
                    "observed": cell[value].mean(),
                    "fixed_weights": float(
                        (means[classes] * pooled_weights[classes]).sum()
                        / pooled_weights[classes].sum()
                    ),
                    "fixed_class_means": float(
                        (pooled_means[classes] * weights[classes]).sum()
                        / weights[classes].sum()
                    ),
                    "n_events": len(cell),
                    **{
                        f"frac_{cls}": float(weights.get(cls, 0.0))
                        for cls in SIZE_CLASSES
                    },
                }
            )
    return pd.DataFrame.from_records(records)
