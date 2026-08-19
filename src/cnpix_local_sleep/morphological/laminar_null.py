"""Mechanical null for the laminar-dominance trimodality.

Every subject/probe/structure shows a striking trimodality in two per-OFF
quantities:

- ``supra_concentration = supra_area / (supra_area + infra_area)``: sharp mass at
  0 and 1 plus a central ~0.5 peak.
- ``center_of_mass_depth`` (COM): a milder version of the same.

Both are lossy projections of one object: each OFF's per-pixel depth marginal.
This module builds the decisive shape-preserving, depth-randomized null that
isolates the single variable of interest (the OFF's *true depth position*) while
holding everything else (footprint size, shape, per-channel time-occupancy, the
band geometry, the channel grid) exactly as in production.

The null reuses the real measurement code verbatim:

- supra/infra pixel counts via :func:`cnpix_local_sleep.morphological.detect.add_laminar_areas`;
- the flip-aware concentration via
  :func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations`;
- COM via the one-line centroid of :mod:`cnpix_local_sleep.morphological.morphology` (line ref in
  :func:`center_of_mass_depths`).

So the *only* thing the null destroys relative to reality is where each OFF sits
along depth. If a measure's empirical trimodality is reproduced by the null, that
trimodality carries no depth-occurrence information beyond size + geometry; the
residual is what demands a biological (or detection-rate-vs-depth) explanation.

All inputs come from the whole-recording (full-48h) ``morphological`` detection:
``offs.parquet`` and ``off_label_indices.parquet`` (per-OFF footprints). Requires
NFS.
"""

from __future__ import annotations

from types import ModuleType
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance

from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.morphological import detect as morphological_detect
from cnpix_local_sleep.morphological import mua
from cnpix_local_sleep.morphological.common import MorphologicalSourceConfig
from cnpix_local_sleep.morphological.pipeline import postprocess_offs as ppo
from cnpix_local_sleep import channel_anatomy

Footprints = dict[int, tuple[np.ndarray, np.ndarray]]


# -------------------- Loading --------------------
def load_channel_depths(
    subject: str,
    probe: str,
    structure: str,
    *,
    source_config: MorphologicalSourceConfig | None = None,
) -> np.ndarray:
    """Return the detection-channel depths (um), in the same order detection used.

    Opens the lazy (dask-backed) MUA traces with ``apply_detection_channel_mask=True``
    (identical to the whole-recording detection in
    :func:`cnpix_local_sleep.morphological.detect_full.detect_offs_full`) and reads only the ``y``
    coordinate, so no trace chunks are materialized. The returned array indexes
    exactly as the ``chan_ixs`` stored in ``off_label_indices.parquet``.
    """
    cfg = source_config or mua.SOURCE_CONFIG
    da = cfg.open_traces_as_xarray(
        subject,
        probe,
        structure,
        condition=None,
        apply_detection_channel_mask=True,
    )
    return np.asarray(da.y.values, dtype=float)


def load_footprints(
    subject: str,
    probe: str,
    structure: str,
    *,
    files_module: ModuleType | None = None,
) -> Footprints:
    """Load per-OFF footprints ``label -> (time_ixs, chan_ixs)``.

    Mirrors the read idiom in :mod:`cnpix_local_sleep.morphological.manual_validation`. Indices are
    per-pixel arrays (one entry per ``(time, channel)`` pixel), so a channel that
    is OFF for more samples appears more often; this is what makes COM and the
    band areas time-occupancy weighted.
    """
    fm = files_module or mua.files
    lbls = pd.read_parquet(
        fm.get_full_off_label_indices_path(subject, probe, structure),
        columns=["label", "time_ixs", "chan_ixs"],
    )
    return {
        int(row.label): (
            np.asarray(row.time_ixs, dtype=np.int64),
            np.asarray(row.chan_ixs, dtype=np.int64),
        )
        for row in lbls.itertuples(index=False)
    }


def load_structure_data(
    subject: str,
    probe: str,
    structure: str,
    *,
    source_config: MorphologicalSourceConfig | None = None,
    files_module: ModuleType | None = None,
) -> tuple[pd.DataFrame, Footprints, np.ndarray]:
    """Convenience loader: ``(offs_df, footprints, y_coords)`` for one combo."""
    cfg = source_config or mua.SOURCE_CONFIG
    fm = files_module or mua.files
    offs = (
        pd.read_parquet(fm.get_full_offs_path(subject, probe, structure))
        .sort_values("start_time")
        .reset_index(drop=True)
    )
    lbl_ixs = load_footprints(subject, probe, structure, files_module=fm)
    y_coords = load_channel_depths(
        subject, probe, structure, source_config=cfg
    )
    return offs, lbl_ixs, y_coords


# Per-footprint measurement (reuses the real code)
def _y_da(y_coords: np.ndarray) -> xr.DataArray:
    """Minimal DataArray carrying only the ``y`` channel coordinate.

    :func:`cnpix_local_sleep.morphological.detect.add_laminar_areas` reads nothing from the trace
    DataArray except ``da.y.values``, so this stand-in lets the null call that
    function verbatim without materializing traces.
    """
    n = y_coords.size
    return xr.DataArray(
        np.zeros(n, dtype=np.int8),
        dims="channel",
        coords={"y": ("channel", y_coords)},
    )


def laminar_areas_for_footprints(
    lbl_ixs: Footprints,
    y_coords: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
) -> pd.DataFrame:
    """``supra_area``/``infra_area`` per footprint, via the real band code.

    Builds a minimal OFF frame keyed by the footprint labels and delegates the
    actual pixel counting to :func:`cnpix_local_sleep.morphological.detect.add_laminar_areas` (the
    single point of truth), so band geometry, the 45/45/10 split, and the layer
    borders are exactly the production ones.
    """
    offs = pd.DataFrame({"label": list(lbl_ixs.keys())})
    morphological_detect.add_laminar_areas(
        offs, _y_da(y_coords), lbl_ixs, subject, probe, structure
    )
    return offs


def center_of_mass_depths(
    lbl_ixs: Footprints,
    y_coords: np.ndarray,
) -> pd.Series:
    """Per-footprint center-of-mass depth, the unweighted-over-pixels centroid.

    Identical formula to detection (``cnpix_local_sleep/morphological/morphology.py``:
    ``center_of_mass_depth = np.mean(y_coords[channel_indices])`` where
    ``channel_indices`` is the *per-pixel* channel array), so deeper/longer
    channels pull the centroid down exactly as in production.
    """
    return pd.Series(
        {
            label: float(np.mean(y_coords[chan_ixs]))
            for label, (_t, chan_ixs) in lbl_ixs.items()
        },
        name="center_of_mass_depth",
    )


def measure_footprints(
    lbl_ixs: Footprints,
    y_coords: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
) -> pd.DataFrame:
    """All per-footprint laminar measures (areas, concentrations, COM).

    Used both for the zero-shift identity check (against ``offs.parquet``) and as
    the inner kernel of the depth-randomized null. Concentrations come from the
    flip-aware SPOT
    :func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations`.
    """
    out = laminar_areas_for_footprints(
        lbl_ixs, y_coords, subject, probe, structure
    )
    supra_conc, infra_conc = ppo.laminar_concentrations(
        out, subject=subject, probe=probe, structure=structure
    )
    out["supra_concentration"] = supra_conc
    out["infra_concentration"] = infra_conc
    com = center_of_mass_depths(lbl_ixs, y_coords)
    out["center_of_mass_depth"] = out["label"].map(com)
    return out


# -------------------- The depth-randomized null --------------------
def structure_index_bounds(
    y_coords: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
) -> tuple[float, float]:
    """The anatomical structure extent in detection-channel index coordinates.

    The detection channels (``y_coords``) generally cover only a *fraction* of the
    full cortical structure: the structure typically continues for tens of channels
    above and/or below the channels OFF detection actually ran on. This returns the
    structure's full anatomical depth extent, from the registration
    ``<probe>.structures.htsv`` (:func:`cnpix_local_sleep.channel_anatomy.load_structures`),
    *not* the detection-derived :func:`cnpix_local_sleep.channel_anatomy.get_layer_borders`
    (which is defined from the detection channels and so merely reproduces the
    detection span), expressed in the same fractional index frame as ``y_coords``
    (channel ``0`` at depth ``y_coords[0]``, unit step = one channel pitch).

    The returned ``(lo, hi)`` may lie outside ``[0, n - 1]`` (the structure extends
    past the detection window). It is unioned with the detection window, since the
    detection channels are by definition part of the structure, so the structure
    must cover at least ``[0, n - 1]``; the union only guards border/rounding
    mismatches. Used by the ``uniform`` null to place same-size OFFs uniformly over
    the *whole structure* and then observe only the detection window.
    """
    structs = channel_anatomy.load_structures(subject, probe)
    row = structs[structs["acronym"] == structure]
    if row.empty:
        row = structs[structs["acronym"].astype(str).str.startswith(structure)]
    if row.empty:
        raise ValueError(
            f"structure {structure!r} not found in {subject}/{probe} "
            "structures.htsv"
        )
    s_lo = float(row["lo"].min())
    s_hi = float(row["hi"].max())

    y = np.asarray(y_coords, dtype=float)
    n = y.size
    step = (y[-1] - y[0]) / (n - 1)
    a = (s_lo - y[0]) / step
    b = (s_hi - y[0]) / step
    lo_idx, hi_idx = (a, b) if a <= b else (b, a)
    return (min(lo_idx, 0.0), max(hi_idx, float(n - 1)))


def _placement_delta_bounds(
    mins: np.ndarray, maxs: np.ndarray, pos_lo: float, pos_hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive integer-shift bounds ``[lo, hi]`` keeping each footprint in ``[pos_lo, pos_hi]``.

    A footprint spanning channels ``[min, max]`` can be translated by an integer
    ``delta`` so it stays within the (possibly fractional, possibly out-of-probe)
    position window ``[pos_lo, pos_hi]`` iff ``min + delta >= pos_lo`` and
    ``max + delta <= pos_hi``, i.e. ``delta in [ceil(pos_lo - min), floor(pos_hi -
    max)]``. A footprint wider than the window has ``hi < lo`` (no slack); the empty
    range is collapsed to ``{0}`` (no shift).

    With ``[pos_lo, pos_hi] = [0, n - 1]`` this is the in-detection feasible range
    (:func:`_feasible_delta_bounds`); with the anatomical structure window
    (:func:`structure_index_bounds`, which can extend past the probe ends) it is the
    in-structure range whose overhang past the detection window is clipped by the
    caller.
    """
    mins = np.asarray(mins, dtype=np.int64)
    maxs = np.asarray(maxs, dtype=np.int64)
    lo = np.ceil(pos_lo - mins).astype(np.int64)
    hi = np.floor(pos_hi - maxs).astype(np.int64)
    full = hi < lo
    lo = np.where(full, 0, lo)
    hi = np.where(full, 0, hi)
    return lo, hi


def _feasible_delta_bounds(
    mins: np.ndarray, maxs: np.ndarray, n_chans: int
) -> tuple[np.ndarray, np.ndarray]:
    """In-detection feasible shift bounds: :func:`_placement_delta_bounds` over ``[0, n-1]``."""
    return _placement_delta_bounds(mins, maxs, 0.0, float(n_chans - 1))


def _draw_deltas(
    placement: str,
    n_chans: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    rng: np.random.Generator,
    struct_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Per-OFF integer channel shift under one of the two null placements.

    Both placements draw the shift uniformly over a footprint's admissible range
    (so the footprint is translated rigidly, size/shape preserved); they differ
    only in the position window the footprint must fit inside, and hence in whether
    any pixel ends up outside the detection probe and is clipped by the caller:

    - ``"feasible"`` (the size-preserving, no-clip null): the window is the
      detection probe ``[0, n_chans - 1]`` (:func:`_feasible_delta_bounds`), so the
      footprint always stays in-bounds and no pixel is ever lost. A full-span OFF
      has a single feasible placement (``delta = 0``) and so no positional freedom:
      the honest baseline for the centroid-contraction question.
    - ``"uniform"`` (the whole-structure null): the window is the *anatomical
      structure* ``struct_bounds`` (:func:`structure_index_bounds`), which typically
      extends past the detection probe ends. Each same-size OFF is placed uniformly
      over the whole structure; placements overhanging the detection window have
      their out-of-window pixels clipped by the caller (``edge_clip``), so an OFF
      centered beyond the detection edge is seen only partially (or not at all).
      This models "structure OFFs share the observed size/shape distribution, but a
      limited detection span lets us see only part of them." When ``struct_bounds``
      is ``None`` it falls back to the detection window, degenerating to ``feasible``
      placement plus a no-op clip.
    """
    if placement == "uniform":
        if struct_bounds is None:
            pos_lo, pos_hi = 0.0, float(n_chans - 1)
        else:
            pos_lo, pos_hi = float(struct_bounds[0]), float(struct_bounds[1])
        lo, hi = _placement_delta_bounds(mins, maxs, pos_lo, pos_hi)
        return rng.integers(lo, hi + 1).astype(np.int64)
    if placement == "feasible":
        lo, hi = _feasible_delta_bounds(mins, maxs, n_chans)
        return rng.integers(lo, hi + 1).astype(np.int64)
    raise ValueError(
        f"placement must be 'uniform' or 'feasible', got {placement!r}"
    )


def _shift_footprints(
    lbl_ixs: Footprints,
    n_chans: int,
    rng: np.random.Generator,
    *,
    edge_clip: bool,
    placement: str = "uniform",
    struct_bounds: tuple[float, float] | None = None,
) -> Footprints:
    """Translate each footprint to a random depth (see :func:`_draw_deltas`).

    With ``placement="feasible"`` the footprint is dropped uniformly among its
    in-detection positions, so no pixel is ever clipped and the size/shape is
    preserved exactly (``edge_clip`` is then a no-op). With ``placement="uniform"``
    the footprint is dropped uniformly over the whole anatomical structure
    (``struct_bounds``, which extends past the probe ends); ``edge_clip`` then drops
    the pixels that overhang the detection window, so an OFF placed past the
    detection edge is seen only partially. With ``struct_bounds=None`` the uniform
    placement falls back to the detection window.

    Channels are assumed depth-ordered with near-uniform pitch (true for a single
    shank within a structure), so an integer index shift is a depth translation.
    """
    items = list(lbl_ixs.items())
    mins = np.array([c.min() for _l, (_t, c) in items], dtype=np.int64)
    maxs = np.array([c.max() for _l, (_t, c) in items], dtype=np.int64)
    deltas = _draw_deltas(
        placement, n_chans, mins, maxs, rng, struct_bounds=struct_bounds
    )

    shifted: Footprints = {}
    for (label, (time_ixs, chan_ixs)), delta in zip(items, deltas, strict=True):
        new_chan = chan_ixs + int(delta)
        if placement == "feasible":
            shifted[label] = (time_ixs, new_chan)  # in-bounds by construction
        elif edge_clip:
            keep = (new_chan >= 0) & (new_chan < n_chans)
            shifted[label] = (time_ixs[keep], new_chan[keep])
        else:
            shifted[label] = (time_ixs, np.clip(new_chan, 0, n_chans - 1))
    return shifted


def _collapse_footprints(
    lbl_ixs: Footprints,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten footprints to distinct channels: ``(ch, w, sizes, means, seg)``.

    ``ch``/``w`` are per-distinct-channel index/pixel-weight (concatenated over
    OFFs), ``sizes`` the per-OFF distinct-channel count, ``means`` the per-OFF
    pixel-weighted mean channel (the index-space COM that the shift centers), and
    ``seg`` the per-OFF start offsets for :func:`numpy.add.reduceat` segment sums.
    """
    uch, wgt, sizes, means = [], [], [], []
    for _t, chan in lbl_ixs.values():
        u, c = np.unique(chan, return_counts=True)
        uch.append(u.astype(np.int32))
        wgt.append(c.astype(np.float64))
        sizes.append(u.size)
        means.append(float((u * c).sum() / c.sum()))
    ch = np.concatenate(uch) if uch else np.empty(0, dtype=np.int32)
    w = np.concatenate(wgt) if wgt else np.empty(0, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.int64)
    means = np.asarray(means, dtype=np.float64)
    seg = (
        np.concatenate([[0], np.cumsum(sizes)[:-1]])
        if sizes.size
        else np.empty(0, dtype=np.int64)
    )
    return ch, w, sizes, means, seg


class CollapsedFootprints(NamedTuple):
    """Distinct-channel representation of a structure's OFF footprints.

    The collapse (:func:`_collapse_footprints`, a per-OFF :func:`numpy.unique`
    loop over ~10^5 OFFs) is the dominant cost of the null. Computing it once
    and passing this bundle into :func:`null_measures_per_off`,
    :func:`null_measure_bands`, and :func:`occupancy_null_test` (via their
    ``collapsed=`` argument) avoids re-deriving it for every placement/readout.

    Fields: ``labels`` (per-OFF label, in footprint order), ``ch``/``w``
    (per-distinct-channel index/pixel-weight, concatenated over OFFs), ``sizes``
    (per-OFF distinct-channel count), ``means`` (per-OFF pixel-weighted mean
    channel, the index-space COM the shift centers), ``seg`` (per-OFF
    :func:`numpy.add.reduceat` start offsets), and ``mins``/``maxs`` (per-OFF
    channel extent, for the feasible-placement bounds).
    """

    labels: list[int]
    ch: np.ndarray
    w: np.ndarray
    sizes: np.ndarray
    means: np.ndarray
    seg: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray


def collapse_footprints(lbl_ixs: Footprints) -> CollapsedFootprints:
    """Collapse footprints to the reusable :class:`CollapsedFootprints` bundle.

    Wraps :func:`_collapse_footprints` and additionally records the per-OFF
    ``labels`` and channel extent (``mins``/``maxs``), so a caller can run the
    expensive per-OFF collapse a single time and feed the result to every null
    helper. Segment-wise min/max reuse the same ``seg`` offsets as the band sums.
    """
    ch, w, sizes, means, seg = _collapse_footprints(lbl_ixs)
    if ch.size:
        mins = np.minimum.reduceat(ch, seg)
        maxs = np.maximum.reduceat(ch, seg)
    else:
        mins = np.empty(0, dtype=ch.dtype)
        maxs = np.empty(0, dtype=ch.dtype)
    return CollapsedFootprints(
        list(lbl_ixs.keys()), ch, w, sizes, means, seg, mins, maxs
    )


def _shifted_measures(
    ch: np.ndarray,
    w: np.ndarray,
    sizes: np.ndarray,
    seg: np.ndarray,
    delta: np.ndarray,
    y_coords: np.ndarray,
    supra_mask: np.ndarray,
    infra_mask: np.ndarray,
    *,
    flipped: bool,
    edge_clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-OFF ``(supra_concentration, center_of_mass_depth)`` for one shift.

    Vectorized collapsed-channel form of the production measurement; at
    ``delta == 0`` it reproduces :func:`cnpix_local_sleep.morphological.detect.add_laminar_areas`
    and :func:`center_of_mass_depths` exactly (asserted in the tests). ``flipped``
    applies the per-combo supra/infra orientation swap (see
    :func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations`).
    """
    n = y_coords.size
    new = ch + np.repeat(delta, sizes)
    if edge_clip:
        valid = (new >= 0) & (new < n)
    else:
        new = np.clip(new, 0, n - 1)
        valid = np.ones(new.shape, dtype=bool)
    nv = np.where(valid, new, 0)
    wv = np.where(valid, w, 0.0)
    supra = np.add.reduceat(wv * supra_mask[nv], seg)
    infra = np.add.reduceat(wv * infra_mask[nv], seg)
    wsum = np.add.reduceat(wv, seg)
    comnum = np.add.reduceat(wv * y_coords[nv], seg)
    tot = supra + infra
    dominant = infra if flipped else supra
    with np.errstate(invalid="ignore", divide="ignore"):
        conc = np.where(tot > 0, dominant / tot, np.nan)
        com = np.where(wsum > 0, comnum / wsum, np.nan)
    return conc, com


def null_measure_bands(
    lbl_ixs: Footprints,
    y_coords: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
    rng: np.random.Generator,
    *,
    n_reps: int,
    conc_bins: np.ndarray,
    depth_bins: np.ndarray,
    edge_clip: bool = True,
    placement: str = "uniform",
    collapsed: CollapsedFootprints | None = None,
) -> dict:
    """Per-bin null sampling band (mean ± sd) for the concentration and COM histograms.

    The vectorized null: for each of
    *n_reps* uniform-depth placements it histograms the null
    ``supra_concentration`` and ``center_of_mass_depth`` (densities) and returns
    the across-rep mean and sd per bin. Used to draw the same null sampling band
    on the concentration and COM panels as the occupancy panels carry, so all
    emp-vs-null panels are read on equal footing. The per-OFF arithmetic
    (:func:`_shifted_measures`) is validated to match the production code at zero
    shift.

    Pass a pre-built ``collapsed`` (:func:`collapse_footprints`) to reuse a single
    footprint collapse across placements/readouts; otherwise it is computed here.
    """
    c = collapsed if collapsed is not None else collapse_footprints(lbl_ixs)
    ch, w, sizes, seg, mins, maxs = (
        c.ch, c.w, c.sizes, c.seg, c.mins, c.maxs
    )
    n = y_coords.size
    borders = channel_anatomy.get_layer_borders(subject, probe, structure)
    supra_row = borders[borders["layer"] == "supra"].iloc[0]
    infra_row = borders[borders["layer"] == "infra"].iloc[0]
    supra_mask = (y_coords >= supra_row["lo"]) & (y_coords <= supra_row["hi"])
    infra_mask = (y_coords >= infra_row["lo"]) & (y_coords <= infra_row["hi"])
    flipped = (subject, probe, structure) in sps_conf.get_flipped_laminar_combos()
    sb = (
        structure_index_bounds(y_coords, subject, probe, structure)
        if placement == "uniform"
        else None
    )

    conc_h = np.empty((n_reps, len(conc_bins) - 1))
    com_h = np.empty((n_reps, len(depth_bins) - 1))
    for r in range(n_reps):
        delta = _draw_deltas(placement, n, mins, maxs, rng, struct_bounds=sb)
        conc, com = _shifted_measures(
            ch, w, sizes, seg, delta, y_coords, supra_mask, infra_mask,
            flipped=flipped, edge_clip=edge_clip,
        )
        conc_h[r], _ = np.histogram(
            conc[np.isfinite(conc)], bins=conc_bins, density=True
        )
        com_h[r], _ = np.histogram(
            com[np.isfinite(com)], bins=depth_bins, density=True
        )
    return {
        "conc_mean": conc_h.mean(0),
        "conc_sd": conc_h.std(0, ddof=1),
        "com_mean": com_h.mean(0),
        "com_sd": com_h.std(0, ddof=1),
        "conc_bins": conc_bins,
        "depth_bins": depth_bins,
    }


def null_measures_per_off(
    lbl_ixs: Footprints,
    y_coords: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
    rng: np.random.Generator,
    *,
    placement: str = "uniform",
    edge_clip: bool = True,
    collapsed: CollapsedFootprints | None = None,
) -> pd.DataFrame:
    """Per-OFF ``(supra_concentration, center_of_mass_depth)`` under one null draw.

    The fast, vectorized single-replicate kernel behind
    :func:`_shifted_measures` (same collapsed-channel arithmetic as
    :func:`null_measure_bands`, validated to reproduce the production measurement
    at zero shift). Returns one row per footprint, label-aligned to ``lbl_ixs``,
    with columns ``label``, ``supra_concentration``, ``center_of_mass_depth``.

    ``placement="feasible"`` uses the no-clip, size-preserving null, the clean
    baseline for the COM (centroid-contraction) question. Pass a pre-built
    ``collapsed`` (:func:`collapse_footprints`) to reuse a single footprint
    collapse across placements; otherwise it is computed here.
    """
    c = collapsed if collapsed is not None else collapse_footprints(lbl_ixs)
    ch, w, sizes, seg, mins, maxs = (
        c.ch, c.w, c.sizes, c.seg, c.mins, c.maxs
    )
    n = y_coords.size
    borders = channel_anatomy.get_layer_borders(subject, probe, structure)
    supra_row = borders[borders["layer"] == "supra"].iloc[0]
    infra_row = borders[borders["layer"] == "infra"].iloc[0]
    supra_mask = (y_coords >= supra_row["lo"]) & (y_coords <= supra_row["hi"])
    infra_mask = (y_coords >= infra_row["lo"]) & (y_coords <= infra_row["hi"])
    flipped = (subject, probe, structure) in sps_conf.get_flipped_laminar_combos()
    sb = (
        structure_index_bounds(y_coords, subject, probe, structure)
        if placement == "uniform"
        else None
    )

    delta = _draw_deltas(placement, n, mins, maxs, rng, struct_bounds=sb)
    conc, com = _shifted_measures(
        ch, w, sizes, seg, delta, y_coords, supra_mask, infra_mask,
        flipped=flipped, edge_clip=edge_clip,
    )
    return pd.DataFrame(
        {
            "label": c.labels,
            "supra_concentration": conc,
            "center_of_mass_depth": com,
        }
    )


def occupancy_null_test(
    lbl_ixs: Footprints,
    y_coords: np.ndarray,
    rng: np.random.Generator,
    *,
    n_perm: int = 200,
    edge_clip: bool = True,
    weighting: str = "time",
    placement: str = "uniform",
    struct_bounds: tuple[float, float] | None = None,
    collapsed: CollapsedFootprints | None = None,
) -> dict:
    """Test whether the per-channel OFF occupancy departs from the depth null.

    Two readouts (``weighting``). Both pool a per-channel depth marginal over
    all OFFs and place each OFF at uniform-random depth under the *same* null; they
    differ only in what each OFF contributes to a channel it touches:

    - ``"time"`` (default): the channel's pixel count, i.e. total time spent OFF
      at that depth. Dominated by long, tall OFFs; the honest "fraction of OFF
      time vs depth" marginal.
    - ``"count"``: 1 per distinct channel the OFF touches, regardless of
      duration. Counts how many OFF *events* reach each depth: duration-blind and
      more sensitive to where events occur, without the centroid contraction of
      COM.

    The null (H0). Each detected OFF's *depth position is independent of its
    size/shape and uniformly distributed* over the structure, with only the
    detection window observed, i.e. the observed multiset of OFF footprints is
    placed along depth by iid uniform translation (exactly :func:`_shift_footprints`).
    Two placements realize this with different position windows:

    - ``"feasible"``: depth is uniform over the *detection* channels, so every OFF
      stays fully in-window. The expected occupancy is not flat: convolving the
      footprint-size distribution with feasible-in-window placement yields an
      edge-tapered envelope (large OFFs cannot center near the edge). That envelope
      (``null_mean``), not a flat line, is the baseline.
    - ``"uniform"``: depth is uniform over the *whole anatomical structure*
      (``struct_bounds``, from :func:`structure_index_bounds`), which extends past
      the detection window; placements overhanging the window are clipped, so OFFs
      whose true center sits beyond the detected channels are seen only partially.
      This removes the feasible envelope's detection-edge taper (the structure
      continues past the window, so there is no special edge there) and replaces it
      with the partial-visibility leak-in shape. The clipped overhang is divided out
      by unit-mass renormalization (see below), so it does not bias the effect size.
      When ``struct_bounds`` is ``None`` it degenerates to the ``feasible`` window.

    Shape comparison (the null is renormalized to unit mass). The data and every
    null draw are each normalized to sum to 1, so the comparison is purely about
    *where within the detection window the occupancy sits*, not how much total mass
    survives. This matters for the ``uniform`` placement: OFFs placed with their
    center beyond the detection window are clipped, so the surviving mass is a
    fraction ``f`` ~ (detection span)/(structure span) of the placed mass. That
    ``f``-dependent deficit is an uninformative geometric nuisance and, left in,
    would put a mechanical floor ``tv >= 0.5*(1 - f)`` under ``tv`` that varies by
    structure and destroys cross-structure comparability, so it is divided out by
    renormalizing each draw. What it does *not* divide out is the informative
    asymmetric leak-in shape: OFFs centered just outside a detection edge that
    abuts more structure ramp occupancy toward that edge, while an edge at the
    structure boundary has no ramp, a genuine shape feature of partial visibility,
    preserved here. Both ``tv`` and ``w1_um`` are therefore deficit-free shape
    distances. The ``feasible`` placement never clips (each draw already sums to the
    full mass), so the renormalization is a no-op there.

    Why a permutation test. Pixels within one OFF are massively correlated
    (contiguous channels × samples), so the independence unit is the OFF, not the
    pixel; a pixel-level chi-square would be pseudoreplicated by ~6 orders of
    magnitude. Permuting whole footprints (each kept rigid, only its depth
    randomized) preserves the within-OFF dependence and the OFF count, and
    realizes H0 directly, so the Monte-Carlo p-value needs no distributional
    assumptions.

    Statistics returned. A global effect size: ``w1_um`` (1-Wasserstein
    distance, in microns, between observed and null-mean occupancy) and ``tv``
    (fraction of occupancy mass that must move), plus its permutation p-value
    ``p_global`` (W1-to-null-mean statistic vs the realizations). Localization via
    per-channel standardized residuals ``z`` with an FWER-controlled threshold
    ``z_thresh`` (95th percentile of the per-realization max-\\|z\\|); ``sig`` flags
    channels whose occupancy is higher/lower than uniform placement predicts.

    Large-n caveat. With ~10^5 OFFs ``p_global`` is essentially 0 for any real
    departure; significance is cheap. The scientific content is the *effect size*
    (``w1_um``/``tv``) and *where* the residuals concentrate (e.g. at the
    supra/infra centroids); read those, not the p-value alone.
    """
    n = y_coords.size
    # Collapse each OFF to its DISTINCT channels with per-channel pixel weights:
    # occupancy is additive over pixels, so grouping by channel first is exact and
    # ~(samples-per-channel)x cheaper than carrying every pixel. Each OFF is then
    # translated rigidly (size/shape preserved) to a random in-window depth.
    c = collapsed if collapsed is not None else collapse_footprints(lbl_ixs)
    ch_flat, w_flat, sizes, mins, maxs = (
        c.ch, c.w, c.sizes, c.mins, c.maxs
    )
    n_off = sizes.size
    if weighting == "count":
        # Duration-blind: 1 per distinct channel an OFF touches. The placement
        # (the rigid translation) is unchanged, so only the readout differs.
        w_flat = np.ones_like(w_flat)
    elif weighting != "time":
        raise ValueError(f"weighting must be 'time' or 'count', got {weighting!r}")

    # Normalize the data and every null draw to UNIT mass, so the test compares the
    # depth *shape* within the detection window (see "Shape comparison" above).
    obs = np.bincount(ch_flat, weights=w_flat, minlength=n)
    obs_p = obs / obs.sum()

    null_curves = np.empty((n_perm, n), dtype=float)
    for b in range(n_perm):
        delta = _draw_deltas(placement, n, mins, maxs, rng, struct_bounds=struct_bounds)
        new = ch_flat + np.repeat(delta, sizes)
        if edge_clip:
            keep = (new >= 0) & (new < n)
            counts = np.bincount(new[keep], weights=w_flat[keep], minlength=n)
        else:
            np.clip(new, 0, n - 1, out=new)
            counts = np.bincount(new, weights=w_flat, minlength=n)
        # Renormalize this draw to unit mass: divide out the f-dependent
        # partial-visibility deficit (uniform placement clips out-of-window
        # overhang), keeping only the depth shape. No-op for feasible (no clip).
        total = counts.sum()
        null_curves[b] = counts / total if total > 0 else np.full(n, 1.0 / n)

    null_mean = null_curves.mean(0)
    null_sd = null_curves.std(0, ddof=1)
    safe_sd = np.where(null_sd > 0, null_sd, np.nan)

    tv = float(0.5 * np.abs(obs_p - null_mean).sum())
    w1_um = float(wasserstein_distance(y_coords, y_coords, obs_p, null_mean))
    t_null = np.array(
        [
            wasserstein_distance(y_coords, y_coords, null_curves[b], null_mean)
            for b in range(n_perm)
        ]
    )
    p_global = (1 + int((t_null >= w1_um).sum())) / (1 + n_perm)

    z = (obs_p - null_mean) / safe_sd
    z_null_max = np.array(
        [np.nanmax(np.abs((null_curves[b] - null_mean) / safe_sd)) for b in range(n_perm)]
    )
    z_thresh = float(np.nanpercentile(z_null_max, 95))

    return {
        "y": y_coords,
        "obs_p": obs_p,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z": z,
        "z_thresh": z_thresh,
        "sig": np.abs(z) > z_thresh,
        "tv": tv,
        "w1_um": w1_um,
        "p_global": p_global,
        "n_off": int(n_off),
        "weighting": weighting,
    }


# Superficial-vs-deep asymmetry of the occupancy excess
def superficial_half_mask(
    y_coords: np.ndarray, subject: str, probe: str, structure: str
) -> np.ndarray:
    """Boolean mask of the superficial half of the channels (50/50 depth split).

    Splits the detection channels at the depth midpoint ``m = (y_min + y_max)/2``
    into a superficial (toward the cortical surface) and a deep half, a plain
    50/50 split of the depth extent, with no excluded middle. Orientation (which
    end of the probe is superficial) is taken from the layer geometry, using the
    *same* flip-corrected convention as
    :func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations`: the
    superficial end is the supragranular side, or the infragranular side for combos
    flagged in :func:`cnpix_local_sleep.sps_conf.get_flipped_laminar_combos`. Returns the mask
    of channels on the superficial side of ``m``.
    """
    borders = channel_anatomy.get_layer_borders(subject, probe, structure)
    supra_c = float(
        borders[borders["layer"] == "supra"].iloc[0][["lo", "hi"]].mean()
    )
    infra_c = float(
        borders[borders["layer"] == "infra"].iloc[0][["lo", "hi"]].mean()
    )
    flipped = (subject, probe, structure) in sps_conf.get_flipped_laminar_combos()
    superficial_c = infra_c if flipped else supra_c
    deep_c = supra_c if flipped else infra_c
    m = 0.5 * (float(y_coords.min()) + float(y_coords.max()))
    return y_coords > m if superficial_c >= deep_c else y_coords < m


def occupancy_asymmetry(
    obs_p: np.ndarray,
    null_mean: np.ndarray,
    y_coords: np.ndarray,
    subject: str,
    probe: str,
    structure: str,
) -> dict[str, float]:
    """Signed superficial-vs-deep asymmetry of the empirical occupancy *excess*.

    A signed, directional companion to the unsigned ``w1_um`` / ``tv`` from
    :func:`occupancy_null_test`. Both ``obs_p`` and ``null_mean`` are normalized
    occupancy distributions over depth (each sums to 1), so the per-channel excess
    ``obs_p - null_mean`` sums to 0. Splitting the channels 50/50 at the depth
    midpoint (:func:`superficial_half_mask`),

        ``asym = sum_superficial(obs_p) - sum_superficial(null_mean)``

    is how much more empirical occupancy than the null sits on the superficial
    half. Because the distributions are normalized, the deep-side excess is
    exactly ``-asym``, so this one signed number is the whole "superficial - deep"
    story (the literal difference is ``2 * asym``).

    - ``asym > 0`` means the empirical occupancy exceeds the feasible null toward the
      superficial half; ``< 0`` means toward the deep half.
    - It is a difference of mass fractions, hence dimensionless and comparable
      across structures, and it lies in ``[-tv, tv]``.
    - ``asym_norm = asym / tv`` lies in ``[-1, 1]``: the share of the *displaced*
      occupancy mass that is one-sided (``+/-1`` = the entire departure is on one
      side of the midpoint, ``0`` = a departure symmetric about the midpoint).

    Orientation is flip-corrected, so the sign means the same thing across
    structures regardless of probe insertion direction.
    """
    sup = superficial_half_mask(y_coords, subject, probe, structure)
    obs_p = np.asarray(obs_p, dtype=float)
    null_mean = np.asarray(null_mean, dtype=float)
    asym = float(obs_p[sup].sum() - null_mean[sup].sum())
    tv = float(0.5 * np.abs(obs_p - null_mean).sum())
    return {"asym": asym, "asym_norm": asym / tv if tv > 0 else float("nan")}


def _w1_to_uniform(values: np.ndarray, lo: float, hi: float, n_ref: int = 20_000) -> float:
    """1-Wasserstein distance from *values* to ``Uniform[lo, hi]``.

    The uniform reference is a deterministic dense grid, so the result is
    reproducible and free of Monte-Carlo jitter.
    """
    ref = np.linspace(lo, hi, n_ref)
    return float(wasserstein_distance(values, ref))


def mechanical_attribution(
    empirical: np.ndarray,
    null: np.ndarray,
    *,
    support: tuple[float, float] | None = None,
    n_floor: int = 25,
    floor_cap: int = 50_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float | bool | tuple[float, float]]:
    """Principled, bin-free attribution of a measure's structure to the null.

    A Wasserstein (earth-mover) skill score, applied identically to
    ``supra_concentration`` and ``center_of_mass_depth``::

        attribution = 1 - W1(emp, null) / W1(emp, flat)

    where ``flat`` is the uniform distribution over *support* (the structureless
    baseline) and ``W1`` is the 1-Wasserstein distance. Unlike the histogram TV
    version this replaced, W1 needs no bins (important for the
    continuous depth (COM) axis), and is geometry-aware (it credits the null for
    putting mass *near* the empirical, not only in the exact same bin), while the
    ratio stays scale-free (units cancel).

    Interpretation:

    - ``attribution`` ~ 1: the depth-randomized null reproduces the empirical, so
      the measure's structure is mechanical (OFF size + band geometry under random
      depth).
    - ``attribution`` ~ 0: the null is no better than a featureless uniform; the
      empirical structure is a real depth-occurrence residual. (A small negative
      value means the null is marginally *worse* than uniform; it is a skill
      score, so negatives are meaningful, not a numerical defect.)

    ``floor`` is the sampling-noise floor: the expected ``W1`` between two
    independent same-size draws of the empirical (rescaled to the empirical n via
    the ``1/sqrt(n)`` law of W1 sampling error). ``resolvable`` is ``True`` when
    ``W1(emp, flat)`` clears that floor, i.e. when there is structure to attribute
    at all; if it is ``False`` the scalar should not be over-interpreted.

    *empirical*/*null* are per-OFF measure values; *support* defaults to the
    pooled min/max.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    emp = np.asarray(empirical, dtype=float)
    emp = emp[np.isfinite(emp)]
    nul = np.asarray(null, dtype=float)
    nul = nul[np.isfinite(nul)]

    nan = float("nan")
    if emp.size == 0 or nul.size == 0:
        return {
            "attribution": nan, "w_null": nan, "w_flat": nan,
            "floor": nan, "support": (nan, nan), "resolvable": False,
        }

    if support is None:
        lo = float(min(emp.min(), nul.min()))
        hi = float(max(emp.max(), nul.max()))
    else:
        lo, hi = float(support[0]), float(support[1])
    if hi <= lo:
        return {
            "attribution": nan, "w_null": nan, "w_flat": nan,
            "floor": nan, "support": (lo, hi), "resolvable": False,
        }

    w_null = float(wasserstein_distance(emp, nul))
    w_flat = _w1_to_uniform(emp, lo, hi)
    attribution = 1.0 - w_null / w_flat if w_flat > 0 else nan

    # Sampling-noise floor: W1 between two independent resamples of the empirical
    # (capped for speed, then rescaled to the empirical n via W1's 1/sqrt(n) rate).
    m = min(emp.size, floor_cap)
    floors = [
        float(
            wasserstein_distance(
                rng.choice(emp, m, replace=True),
                rng.choice(emp, m, replace=True),
            )
        )
        for _ in range(n_floor)
    ]
    floor = float(np.mean(floors)) * np.sqrt(m / emp.size)

    return {
        "attribution": attribution,
        "w_null": w_null,
        "w_flat": w_flat,
        "floor": floor,
        "support": (lo, hi),
        "resolvable": bool(w_flat > 3.0 * floor),
    }
