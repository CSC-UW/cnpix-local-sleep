"""Full-48h morphological OFF analysis: timecourses, condition re-aggregation,
and the NOD intrusion sweep.

The full-recording detection (:mod:`cnpix_local_sleep.morphological.detect_full`) writes one
condition-agnostic ``offs.parquet`` per (subject, probe, structure), with a
per-OFF ``state`` column. This module consumes those files to:

- verify extraction coverage (:func:`verify_extraction`);
- re-derive condition-based summaries by subsetting the 48h OFFs and compare
  them to the canonical per-condition aggregation (see
  :mod:`cnpix_local_sleep.morphological.pipeline.aggregate_experiment_offs`);
- sweep "intrusion" bouts (non-Wake bouts during the NOD deprivation window)
  to quantify how counting their OFFs changes the NOD incline. The swept
  metric defaults to total OFF area but can be any *additive* OFF property
  (count, rate, total area; see :data:`ADDITIVE_METRICS`), under any of the
  ``llas``/``clas``/``blas`` filters (:func:`intrusion_sweep`).

Quality gating is via :mod:`cnpix_local_sleep.sps_conf`: timecourse plots use
``get_plottable_spsl`` (any combo that can be detected); advanced analyses use
``get_analysis_spsl`` (good-enough signal quality, with a flag to drop the
"maybe exclude" combos).

NOTE: This pipeline is ``morphological``-only, by code (it imports
``morphological.mua.files`` directly, with no ``source_config`` switch) and by data
(only ``morphological`` has the ``full48h_*`` aggregates; ``tom-bugnon`` has no
whole-recording OFFs at all). Generalizing it across variants is not a pure
refactor.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubplots as pp
import seaborn as sns
import wisc_ecephys_tools as wet
from wisc_ecephys_tools.rats import cnd_hgs

from cnpix_local_sleep import atlas, const, hyp, plots, sps_conf
from cnpix_local_sleep.morphological.mua import files as mua_files
from cnpix_local_sleep.morphological.pipeline.postprocess_offs import postprocess_offs_frame
from cnpix_local_sleep import off_tables

EXPERIMENT = const.EXPERIMENT

# "All non-Wake except artifact": the states whose bouts during the NOD
# window count as intrusions (candidate mislabeled local sleep). Excludes
# Wake, Artifact, and NoData.
INTRUSION_STATES: tuple[str, ...] = ("NREM", "IS", "REM", "MA", "Other")

# Default location for generated figures. *.png and *.svg are gitignored, so
# these do not bloat the repo. Per notebooks/README.md every notebook writes
# beside itself into ``outputs/``, and the notebooks that call these plotters
# (incline_magnitudes, intrusion_sweep) live in ``notebooks/figures/``.
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[5]
    / "notebooks"
    / "figures"
    / "outputs"
    / "full48h"
)

# Publication intrusion-sweep figures live directly under the notebook's
# ``outputs/intrusion_sweep/`` (a sibling of ``outputs/full48h/``).
INTRUSION_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "intrusion_sweep"


def _load_full_offs(subject: str, probe: str, structure: str) -> pd.DataFrame:
    """Load the full-recording OFF dataframe for one (subject, probe, structure)."""
    path = mua_files.get_full_offs_path(subject, probe, structure)
    if not path.exists():
        raise FileNotFoundError(
            f"No full-48h offs.parquet for {subject} {probe} {structure} at "
            f"{path}. Run `morphological-offs detect-offs-full`."
        )
    return pd.read_parquet(path)


# -------------------- Extraction verification --------------------


def verify_extraction(
    spsl: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Check full-48h OFF extraction coverage and basic sanity.

    For each (subject, probe, structure) in ``spsl`` (default: every plottable
    combo), report whether ``offs.parquet`` exists, the OFF count, the time
    span in hours, and whether the ``state`` column is present. Combos missing
    from disk are flagged with ``exists=False`` and should be (re)run via
    ``morphological-offs detect-offs-full``.
    """
    if spsl is None:
        spsl = sps_conf.get_plottable_spsl()

    rows = []
    for subject, probe, structure in spsl:
        path = mua_files.get_full_offs_path(subject, probe, structure)
        rec: dict[str, object] = {
            "subject": subject,
            "probe": probe,
            "structure": structure,
            "exists": path.exists(),
            "n_offs": 0,
            "span_h": np.nan,
            "has_state": False,
        }
        if path.exists():
            df = pd.read_parquet(path)
            rec["n_offs"] = len(df)
            rec["has_state"] = "state" in df.columns
            if len(df):
                rec["span_h"] = round(
                    (df["end_time"].max() - df["start_time"].min()) / 3600, 1
                )
        rows.append(rec)

    out = pd.DataFrame(rows)
    n_missing = int((~out["exists"]).sum())
    n_empty = int((out["exists"] & (out["n_offs"] == 0)).sum())
    print(
        f"verify_extraction: {len(out)} combos, {n_missing} missing, {n_empty} empty."
    )
    return out


# -------------------- Continuous 48h OFF-area + SWA timecourse --------------------


def _normalized_off_area_trace(
    offs: pd.DataFrame, hg_clean, smoothing: str
) -> pd.DataFrame:
    """20s rolling-sum OFF-area trace, normalized to the all-NREM mean (× 100).

    Matches the normalization convention of
    :func:`cnpix_local_sleep.morphological.pipeline.plot_offs_vs_time._prepare_data` so this
    continuous view is comparable to the existing per-condition figures.
    """
    trace = plots.get_smoothed_trace(
        offs,
        ["area"],
        time_col="start_time",
        smoothing=smoothing,
        rolling_op="sum",
        fill_values={"area": np.nan},
    )
    nrem_mask = hg_clean.keep_states(["NREM"]).covers_time(trace["start_time"].values)
    nrem_mean = trace.loc[nrem_mask, "area"].mean()
    trace["area_norm"] = trace["area"] / nrem_mean * 100.0
    return trace


# -------------------- NOD intrusion sweep --------------------

DEFAULT_INTRUSION_THRESHOLDS: tuple[int, ...] = tuple(range(1, 21))

# Additive OFF metrics: those expressible as a sum of a per-OFF contribution
# (or a plain count), optionally divided by the FIXED canonical window duration.
# Only these can be swept by adding admitted-intrusion OFFs to precomputed
# per-bucket baselines, without re-aggregating over the whole OFF set at every
# threshold. Each entry maps the metric name to
# ``(per-OFF contribution column or _COUNT sentinel, normalize-by-duration?)``.
#
# Distributional metrics (median_span, mean span, median_duration, ...) are NOT
# additive -- the median/mean of (baseline + admitted intrusions) is not a
# function of the per-subset summaries -- so they are out of scope here and
# would require recomputing the metric over the row set per threshold.
_COUNT = "__count__"
ADDITIVE_METRICS: dict[str, tuple[str, bool]] = {
    "count": (_COUNT, False),
    "rate": (_COUNT, True),
    "total_area": ("area", False),
    "total_area_rel2span": ("area_rel2span", False),
    "total_area_norm": ("area_rel2span", True),
}


# "Exclusive" (adjacent-partition) LAS categories: OFFs admitted by the wider
# filter but rejected by the next-stricter one. Because BLAS nests inside CLAS inside LLAS, the
# three-way partition BLAS + CLAS-exclusive + LLAS-exclusive reconstructs LLAS.
EXCLUSIVE_FILTERS: dict[str, tuple[str, str]] = {
    "clas-exclusive": ("clas", "blas"),  # in CLAS, not in BLAS
    "llas-exclusive": ("llas", "clas"),  # in LLAS, not in CLAS
}

# Stacked-figure OFF-set orders (top -> bottom row), reused by the plotters.
INCLUSIVE_STACK: tuple[str, ...] = ("blas", "clas", "llas")
EXCLUSIVE_STACK: tuple[str, ...] = ("blas", "clas-exclusive", "llas-exclusive")


def _prepare_full_offs(
    subject: str,
    probe: str,
    structure: str,
    *,
    filter_name: str,
    off_filter: dict | None,
) -> pd.DataFrame:
    """Load full-48h OFFs, derive normalized columns, and apply a category filter.

    Adds the postprocessing-derived columns the filters and additive metrics
    need (``area_rel2span``, ``span_rel2max``, ...) via the canonical
    :func:`~cnpix_local_sleep.morphological.pipeline.postprocess_offs.postprocess_offs_frame`
    (the full-48h ``offs.parquet`` ships ``area``/``span``/``max_span`` but not
    their normalized forms), then filters to the requested LAS category, so the
    column definitions never drift from the aggregation pipeline.

    ``filter_name`` accepts the named LAS categories (``llas``/``clas``/``blas``)
    and the "exclusive" partition categories in :data:`EXCLUSIVE_FILTERS`
    (``clas-exclusive`` = CLAS\\BLAS, ``llas-exclusive`` = LLAS\\CLAS), computed
    as mask differences. An explicit ``off_filter`` (a ``column -> (lo, hi)``
    dict) overrides ``filter_name`` when provided.
    """
    offs = _load_full_offs(subject, probe, structure).copy()
    postprocess_offs_frame(offs, structure)
    if off_filter is not None:
        mask = pd.Series(True, index=offs.index)
        for col, (lo, hi) in off_filter.items():
            mask &= offs[col].between(lo, hi)
        return offs.loc[mask]
    if filter_name in EXCLUSIVE_FILTERS:
        include, exclude = EXCLUSIVE_FILTERS[filter_name]
        mask = off_tables.off_filter_mask(offs, include) & ~off_tables.off_filter_mask(
            offs, exclude
        )
        return offs.loc[mask]
    return offs.loc[off_tables.off_filter_mask(offs, filter_name)]


# Subjects whose Early.NOD.Wake window has missing/unreliable data and is
# dropped by the canonical aggregation
# (``aggregate_experiment_offs._apply_additional_exclusions``). Their NOD
# incline is undefined, so they are excluded from the incline sweep too.
NOD_INCLINE_EXCLUDED_SUBJECTS: frozenset[str] = frozenset(
    {"CNPIX7-Giuseppe", "CNPIX15-Claude"}
)


def intrusion_sweep(
    subject: str,
    probe: str,
    structure: str,
    *,
    metric: str = "total_area_norm",
    filter_name: str = "llas",
    thresholds: tuple[int, ...] = DEFAULT_INTRUSION_THRESHOLDS,
    off_filter: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep NOD intrusion-bout durations and measure the NOD incline in *metric*.

    "Intrusions" are non-Wake bouts (states in :data:`INTRUSION_STATES`) inside
    the novel-objects deprivation (NOD) window: candidate mislabeled local
    sleep. For each duration threshold ``T`` we admit intrusion bouts of
    duration <= ``T`` and add their OFFs to the canonical Early/Late.NOD.Wake
    buckets (fixed windows; denominators unchanged). ``T = 0`` admits no
    intrusions and recovers the canonical NOD incline.

    *metric* must be an :data:`ADDITIVE_METRICS` key, a metric expressible as a
    per-OFF sum (or count), optionally divided by the fixed canonical window
    duration, so admitting intrusions only adds to precomputed baselines.
    ``total_area_norm`` (the default) sums ``area_rel2span`` and normalizes by
    duration; ``count``/``rate`` count OFFs; ``total_area`` sums raw ``area``.
    Distributional metrics (median span, mean duration, ...) are not additive
    and are unsupported here.

    *filter_name* selects the LAS category (``llas``/``clas``/``blas``); pass an
    explicit ``off_filter`` dict to override it.

    Returns ``(sweep_df, intrusions_df)``:
    - ``sweep_df``: one row per threshold with early/late *metric* values, the
      incline, and the cumulative NOD intrusion contribution.
    - ``intrusions_df``: one row per intrusion bout (state, duration, bucket,
      value) for inspection, where ``value`` is the bout's contribution to the
      (un-normalized) metric numerator.
    """
    if metric not in ADDITIVE_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}. The intrusion sweep currently "
            f"supports only additive metrics: {sorted(ADDITIVE_METRICS)}. "
            "Distributional metrics (e.g. median_span) are not additive and "
            "would require per-threshold re-aggregation."
        )
    contrib_col, normalize = ADDITIVE_METRICS[metric]

    offs = _prepare_full_offs(
        subject, probe, structure, filter_name=filter_name, off_filter=off_filter
    )

    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    full_hg = hgs["Full.Conservative"]
    nod_start, nod_end = cnd_hgs.get_novel_objects_period(
        EXPERIMENT, wet.get_sglx_subject(subject)
    )

    early_hg = hgs.get("Early.NOD.Wake")
    late_hg = hgs.get("Late.NOD.Wake")
    if early_hg is None or late_hg is None or early_hg.empty or late_hg.empty:
        raise ValueError(
            f"Missing Early/Late.NOD.Wake hypnogram for "
            f"{subject} {probe} {structure}; cannot run intrusion sweep."
        )

    starts = offs["start_time"].to_numpy()
    # Per-OFF contribution to the metric numerator: 1 per OFF for count/rate,
    # else the named per-OFF column (area, area_rel2span, ...).
    if contrib_col is _COUNT:
        contrib = np.ones(len(offs), dtype="float64")
    else:
        contrib = offs[contrib_col].to_numpy(dtype="float64")

    early_mask = early_hg.covers_time(starts)
    late_mask = late_hg.covers_time(starts)
    early_dur = float(early_hg["duration"].sum())
    late_dur = float(late_hg["duration"].sum())
    base_early = float(contrib[early_mask].sum())
    base_late = float(contrib[late_mask].sum())
    # Normalized metrics (rate, total_area_norm) divide by the fixed canonical
    # window duration; non-normalized ones (count, total_area) do not.
    early_denom = early_dur if normalize else 1.0
    late_denom = late_dur if normalize else 1.0

    e_a, e_b = float(early_hg["start_time"].min()), float(early_hg["end_time"].max())
    l_a, l_b = float(late_hg["start_time"].min()), float(late_hg["end_time"].max())

    # Intrusion bouts within the NOD window.
    intrusions_hg = full_hg.trim(nod_start, nod_end).keep_states(list(INTRUSION_STATES))
    recs = []
    for bout in intrusions_hg.itertuples():
        mid = 0.5 * (bout.start_time + bout.end_time)
        if e_a <= mid <= e_b:
            bucket = "early"
        elif l_a <= mid <= l_b:
            bucket = "late"
        else:
            bucket = "middle"
        in_bout = (starts >= bout.start_time) & (starts < bout.end_time)
        recs.append(
            {
                "state": bout.state,
                "duration": float(bout.duration),
                "bucket": bucket,
                "value": float(contrib[in_bout].sum()),
            }
        )
    intrusions = pd.DataFrame(recs, columns=["state", "duration", "bucket", "value"])

    rows = []
    for thr in (0, *thresholds):
        sel = intrusions[intrusions["duration"] <= thr]
        early_extra = float(sel.loc[sel["bucket"] == "early", "value"].sum())
        late_extra = float(sel.loc[sel["bucket"] == "late", "value"].sum())
        e_val = (base_early + early_extra) / early_denom
        l_val = (base_late + late_extra) / late_denom
        rows.append(
            {
                "subject": subject,
                "probe": probe,
                "structure": structure,
                "metric": metric,
                "filter": filter_name,
                "threshold_s": thr,
                "early_value": e_val,
                "late_value": l_val,
                "incline": l_val - e_val,
                "incline_ratio": (l_val / e_val) if e_val > 0 else np.nan,
                "n_intrusions_le_thr": int((intrusions["duration"] <= thr).sum()),
                "nod_intrusion_value": float(sel["value"].sum()),
            }
        )
    sweep = pd.DataFrame(rows)
    return sweep, intrusions


# Curve line weights / marker sizes, in real points. They are passed through
# ``pp.scale`` inside the pubplots figma context so they keep their intended
# physical weight after the destination's figure-size scaling.
_MAIN_LW = 1.0  # main traces (early/late, incline, hidden-area)
_MEAN_LW = 2.0  # mean-across-combos trace in the summary
_COMBO_LW = 1.0  # individual per-combo traces in the summary
_COMBO_ALPHA = 0.6  # opacity of per-combo summary traces (they often overlap)
_MARKER_MS = 3.0  # markers on the per-structure traces


def _intrusion_sweep_dir(root: Path, metric: str, filter_name: str) -> Path:
    """Directory for a (metric, filter) intrusion-sweep run.

    Parameters are encoded as a directory hierarchy
    (``intrusion_sweep/<metric>/<filter>/``) rather than packed into filenames,
    with per-structure figures under a ``by_structure/`` leaf and the
    cross-structure summary at the top of the (metric, filter) directory.
    """
    return root / "intrusion_sweep" / metric / filter_name


def _draw_sweep_triptych(
    axes: "np.ndarray", sweep: pd.DataFrame, metric: str, *, titles: bool = True
) -> None:
    """Draw the three intrusion-sweep panels for one combo onto ``axes`` (len 3).

    Panels: Early/Late burden, NOD incline (with the T=0 baseline), and hidden
    OFF across NOD (cumulative intrusion value + bout count on a twin axis).
    Must be called inside a ``pubplots`` context (line weights use ``pp.scale``).
    """
    lw = pp.scale(_MAIN_LW)
    ms = pp.scale(_MARKER_MS)
    base = sweep.loc[sweep["threshold_s"] == 0].iloc[0]

    ax = axes[0]
    ax.plot(sweep["threshold_s"], sweep["early_value"], "o-", lw=lw, ms=ms, label="Early")
    ax.plot(sweep["threshold_s"], sweep["late_value"], "s-", lw=lw, ms=ms, label="Late")
    ax.set_ylabel(f"OFF {metric}")
    ax.legend()
    if titles:
        ax.set_title("Early/Late burden")

    ax = axes[1]
    ax.plot(
        sweep["threshold_s"], sweep["incline"], "o-", lw=lw, ms=ms, color="firebrick"
    )
    ax.axhline(base["incline"], ls="--", color="0.5", lw=pp.scale(0.9), label="T=0")
    ax.set_ylabel("incline (Late - Early)")
    ax.legend()
    if titles:
        ax.set_title("NOD incline")

    ax = axes[2]
    ax.plot(
        sweep["threshold_s"],
        sweep["nod_intrusion_value"],
        "o-",
        lw=lw,
        ms=ms,
        color="seagreen",
    )
    ax.set_ylabel(f"cumul. intrusion {metric}")
    ax2 = ax.twinx()
    ax2.plot(
        sweep["threshold_s"],
        sweep["n_intrusions_le_thr"],
        "x:",
        lw=pp.scale(1.6),
        ms=ms,
        color="0.5",
    )
    ax2.set_ylabel("# bouts <= T", color="0.5")
    if titles:
        ax.set_title("Hidden OFF across NOD")


def plot_intrusion_sweep(
    subject: str,
    probe: str,
    structure: str,
    sweep: pd.DataFrame,
    *,
    save_dir: Path | str | None = None,
    destination: str = "figma",
) -> Path:
    """Three-panel intrusion-sweep figure for one structure (exploratory SVG)."""
    root = Path(save_dir) if save_dir is not None else DEFAULT_OUTPUT_DIR
    metric = str(sweep["metric"].iloc[0])
    filter_name = str(sweep["filter"].iloc[0])
    out_dir = _intrusion_sweep_dir(root, metric, filter_name) / "by_structure"
    out_dir.mkdir(parents=True, exist_ok=True)

    xlabel = "intrusion <= T (s)"
    with pp.destination(destination):
        fig, axes = plt.subplots(1, 3, figsize=pp.scale(5.6, 1.65))
        _draw_sweep_triptych(axes, sweep, metric)
        for ax in axes:
            ax.set_xlabel(xlabel)
        fig.suptitle(f"{subject} {probe} {structure}: {metric}, {filter_name}")
        out_path = out_dir / f"{subject}_{probe}_{structure}.svg"
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# Human-readable OFF-set row labels (used only in non-pub titles) and y-axis
# measure labels for the stacked publication figures.
_FILTER_LABELS: dict[str, str] = {
    "blas": "BLAS",
    "clas": "CLAS",
    "llas": "LLAS",
    "clas-exclusive": "CLAS-exclusive",
    "llas-exclusive": "LLAS-exclusive",
}
_STACK_YLABELS: dict[str, str] = {
    "count": "Δ Late - Early SD count",
    "total_area_norm": "Δ Late - Early norm. area",
}


def _stack_ylabel(metric: str) -> str:
    """Y-axis label for the stacked change-vs-T=0 figures, per swept measure."""
    return _STACK_YLABELS.get(metric, f"Δ Late - Early {metric}")


def compute_intrusion_sweeps(
    spsl: list[tuple[str, str, str]] | None = None,
    *,
    include_maybe_exclude: bool = True,
    metrics: tuple[str, ...] = ("count", "total_area_norm"),
    thresholds: tuple[int, ...] = DEFAULT_INTRUSION_THRESHOLDS,
) -> pd.DataFrame:
    """Compute (no plotting) the NOD intrusion sweep table -- the slow part.

    For each *metric* (any :data:`ADDITIVE_METRICS` key) the sweep is run over
    the five LAS categories needed by the two stacked figures
    (:data:`INCLUSIVE_STACK` = BLAS/CLAS/LLAS and :data:`EXCLUSIVE_STACK` =
    BLAS/CLAS-exclusive/LLAS-exclusive) across the advanced-analysis combos. By
    default uses :func:`sps_conf.get_analysis_spsl` (pass
    ``include_maybe_exclude=False`` to drop borderline-quality combos).

    Each combo loads a full-48h ``offs.parquet`` from NFS, so this is the ~minutes
    step. The returned table (one row per subject/probe/structure/metric/filter/
    threshold) is small and fully drives :func:`plot_intrusion_sweeps`; cache it
    (e.g. via :func:`load_or_compute_intrusion_sweeps`) to re-plot without
    recomputing. Returns the concatenated sweep table (empty if nothing ran).
    """
    unsupported = [m for m in metrics if m not in ADDITIVE_METRICS]
    if unsupported:
        raise ValueError(
            f"Unsupported metric(s) {unsupported}. The intrusion sweep supports "
            f"only additive metrics: {sorted(ADDITIVE_METRICS)}."
        )

    if spsl is None:
        spsl = sps_conf.get_analysis_spsl(include_maybe_exclude=include_maybe_exclude)

    excluded = [s for s in spsl if s[0] in NOD_INCLINE_EXCLUDED_SUBJECTS]
    if excluded:
        print(
            "Excluding from NOD incline sweep (unreliable Early.NOD.Wake): "
            + ", ".join(f"{s}/{p}/{st}" for s, p, st in excluded)
        )
    spsl = [s for s in spsl if s[0] not in NOD_INCLINE_EXCLUDED_SUBJECTS]

    # The union of both stacks' OFF sets (order-preserving, de-duplicated).
    filters = tuple(dict.fromkeys((*INCLUSIVE_STACK, *EXCLUSIVE_STACK)))

    all_sweeps = []
    for metric in metrics:
        for filter_name in filters:
            for subject, probe, structure in spsl:
                try:
                    sweep, _ = intrusion_sweep(
                        subject,
                        probe,
                        structure,
                        metric=metric,
                        filter_name=filter_name,
                        thresholds=thresholds,
                    )
                except Exception as exc:  # noqa: BLE001 - keep going across combos
                    warnings.warn(
                        f"Intrusion sweep failed for {subject} {probe} "
                        f"{structure} ({metric}, {filter_name}): {exc}",
                        stacklevel=2,
                    )
                    continue
                all_sweeps.append(sweep)

    if not all_sweeps:
        return pd.DataFrame()
    return pd.concat(all_sweeps, ignore_index=True)


def load_or_compute_intrusion_sweeps(
    cache_path: Path | str,
    *,
    refresh: bool = False,
    metrics: tuple[str, ...] = ("count", "total_area_norm"),
    **compute_kwargs,
) -> pd.DataFrame:
    """Load the cached intrusion-sweep table, or compute + cache it (parquet).

    The cache is keyed only by ``cache_path``; it is reused when it exists,
    ``refresh`` is False, and it already contains every requested *metric*
    (otherwise it is recomputed). It does NOT detect changes to ``spsl`` /
    ``thresholds`` -- pass ``refresh=True`` after changing those. Lets you re-plot
    with :func:`plot_intrusion_sweeps` in seconds instead of re-running the
    ~minutes sweep. ``compute_kwargs`` are forwarded to
    :func:`compute_intrusion_sweeps` (``spsl``, ``include_maybe_exclude``,
    ``thresholds``).
    """
    cache_path = Path(cache_path)
    if not refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        have = set(cached["metric"].unique()) if "metric" in cached else set()
        missing = set(metrics) - have
        if not missing:
            return cached
        warnings.warn(
            f"Cached intrusion sweeps at {cache_path} miss metric(s) "
            f"{sorted(missing)}; recomputing.",
            stacklevel=2,
        )
    results = compute_intrusion_sweeps(metrics=metrics, **compute_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        results.to_parquet(cache_path)
    return results


def plot_intrusion_sweeps(
    results: pd.DataFrame,
    *,
    save_dir: Path | str | None = None,
    plot_for_pub: bool = True,
    log_y: bool = False,
    ylims: dict[str, dict[str, tuple[float | None, float | None]]] | None = None,
    by_structure: bool = True,
    destination: str = "figma",
) -> None:
    """Emit all intrusion-sweep figures from a precomputed *results* table.

    Fast (no NFS reads) -- drives entirely off :func:`compute_intrusion_sweeps`
    output, so re-run this after changing a plot parameter without recomputing
    the sweep. Writes per metric two 3-row stacked SVGs (``<metric>/inclusive.svg``
    and ``<metric>/exclusive.svg``) under ``save_dir`` (default
    :data:`INTRUSION_OUTPUT_DIR`), plus, if ``by_structure``, one 3×3 per-combo
    detail figure per combo under ``<metric>/by_structure/``. ``plot_for_pub``
    (default) strips titles/legends and keeps x ticks/labels on the bottom row
    only. ``log_y`` symlog-scales the stacked figures' y axis (see
    :func:`plot_intrusion_sweep_stack`; the per-combo detail figures are
    unaffected).

    ``ylims`` sets per-row y-limits, nested ``{metric: {filter_name: (low,
    high)}}`` (either bound may be ``None`` to autoscale that end); rows not
    listed autoscale. E.g. ``{"count": {"blas": (None, 100)}}`` caps the BLAS
    count row at 100 on a linear axis to tame a single outlier combo.
    """
    if results.empty:
        return
    root = Path(save_dir) if save_dir is not None else INTRUSION_OUTPUT_DIR
    for metric, mdf in results.groupby("metric"):
        metric = str(metric)
        metric_ylims = ylims.get(metric) if ylims else None
        plot_intrusion_sweep_stack(
            mdf,
            metric,
            INCLUSIVE_STACK,
            out_path=root / metric / "inclusive.svg",
            plot_for_pub=plot_for_pub,
            log_y=log_y,
            ylims=metric_ylims,
            destination=destination,
        )
        plot_intrusion_sweep_stack(
            mdf,
            metric,
            EXCLUSIVE_STACK,
            out_path=root / metric / "exclusive.svg",
            plot_for_pub=plot_for_pub,
            log_y=log_y,
            ylims=metric_ylims,
            destination=destination,
        )
        if by_structure:
            for (subject, probe, structure), cdf in mdf.groupby(
                ["subject", "probe", "structure"]
            ):
                by_filter = {
                    fn: cdf[cdf["filter"] == fn]
                    for fn in INCLUSIVE_STACK
                    if not cdf[cdf["filter"] == fn].empty
                }
                plot_intrusion_sweep_detail_stack(
                    str(subject),
                    str(probe),
                    str(structure),
                    by_filter,
                    metric,
                    INCLUSIVE_STACK,
                    out_path=(
                        root
                        / metric
                        / "by_structure"
                        / f"{subject}_{probe}_{structure}.svg"
                    ),
                    destination=destination,
                )


def _sps_structure_palette(combined: pd.DataFrame) -> dict[str, tuple]:
    """Color each ``combo`` by its structure, matching ``incline_magnitudes.ipynb``.

    Structures are ordered anterior->posterior
    (:func:`cnpix_local_sleep.atlas.sort_structures_by_anterior_posterior`) and assigned
    ``tab20`` colors; every "subject probe structure" combo inherits its
    structure's color.
    """
    structures = pd.unique(combined["structure"].astype(str))
    sorted_structures = atlas.sort_structures_by_anterior_posterior(structures)
    structure_palette = dict(
        zip(
            sorted_structures,
            sns.color_palette("tab20", n_colors=len(sorted_structures)),
        )
    )
    combo_structure = (
        combined.drop_duplicates("combo")
        .set_index("combo")["structure"]
        .astype(str)
        .to_dict()
    )
    return {
        combo: structure_palette[struct] for combo, struct in combo_structure.items()
    }


def _add_combo_and_incline_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``combo`` label and ``incline_delta`` (incline - its T=0 incline).

    ``incline_delta`` is computed per combo against that combo's own T=0 incline,
    so the input must be a single-filter subset (each combo appears once per
    threshold).
    """
    df = df.copy()
    df["combo"] = (
        df["subject"].astype(str)
        + " "
        + df["probe"].astype(str)
        + " "
        + df["structure"].astype(str)
    )
    base = df.loc[df["threshold_s"] == 0].set_index("combo")["incline"].to_dict()
    df["incline_delta"] = df["incline"] - df["combo"].map(base)
    return df


def plot_intrusion_sweep_stack(
    results: pd.DataFrame,
    metric: str,
    filter_order: tuple[str, ...],
    *,
    out_path: Path | str,
    plot_for_pub: bool = True,
    log_y: bool = False,
    ylims: dict[str, tuple[float | None, float | None]] | None = None,
    destination: str = "figma",
    width_in: float = 1.5,
) -> Path:
    """Stacked change-in-incline figure: one row per OFF set, aligned for pub.

    Each row shows, for one LAS category in *filter_order* (top -> bottom), every
    combo's change in NOD incline vs the intrusion threshold (``incline_delta``,
    starting at 0 at T=0), colored by structure, plus the across-combo mean. The
    figure uses :func:`cnpix_local_sleep.plots.stacked_rows` (shared vertical geometry) so
    it lines up row-for-row with the incline-magnitude figures when placed side
    by side. With ``plot_for_pub`` (default): no titles/suptitle, no legend, and
    x ticks/labels on the bottom row only.

    ``log_y`` log-scales the y axis. ``incline_delta`` is exactly 0 at T=0 for
    every combo and can be negative, so a plain log scale would drop those
    points; this uses symlog (log outside a small linear band through 0),
    with a per-row ``linthresh`` set to the smallest nonzero ``|incline_delta|``.

    ``ylims`` sets per-row y-limits: a ``{filter_name: (low, high)}`` dict; either
    bound may be ``None`` to autoscale that end. Applied after ``log_y``. This is
    the clean way to tame a single wildly-scaled combo on a linear axis (e.g.
    ``{"blas": (None, 100)}`` clips the outlier while keeping the near-zero and
    negative region at honest linear proportions).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = results[results["metric"] == metric]
    ylabel = _stack_ylabel(metric)
    xlabel = "Including intrusions <= T (s)"
    n = len(filter_order)

    with pp.destination(destination):
        combo_lw = pp.scale(_COMBO_LW)
        mean_lw = pp.scale(_MEAN_LW)
        fig, axes = plots.stacked_rows(
            width_in, ncols=1, nrows=n, sharex=True, left=0.26, right=0.97
        )
        for i, filter_name in enumerate(filter_order):
            ax = axes[i, 0]
            fdf = sub[sub["filter"] == filter_name]
            if not fdf.empty:
                fdf = _add_combo_and_incline_delta(fdf)
                palette = _sps_structure_palette(fdf)
                for combo, grp in fdf.groupby("combo"):
                    ax.plot(
                        grp["threshold_s"],
                        grp["incline_delta"],
                        "-",
                        lw=combo_lw,
                        alpha=_COMBO_ALPHA,
                        color=palette[combo],
                    )
                mean_by_thr = fdf.groupby("threshold_s")["incline_delta"].mean()
                ax.plot(
                    mean_by_thr.index,
                    mean_by_thr.to_numpy(),
                    "k-",
                    lw=mean_lw,
                    label="mean",
                )
                ax.axhline(0, ls="--", color="0.5", lw=pp.scale(0.9))
                if log_y:
                    nonzero = fdf["incline_delta"].abs()
                    nonzero = nonzero[nonzero > 0]
                    linthresh = float(nonzero.min()) if not nonzero.empty else 1.0
                    ax.set_yscale("symlog", linthresh=linthresh)
            if ylims and filter_name in ylims:
                lo, hi = ylims[filter_name]
                ax.set_ylim(bottom=lo, top=hi)
            is_bottom = i == n - 1
            if is_bottom:
                ax.set_xlabel(xlabel)
            if plot_for_pub:
                if not is_bottom:
                    ax.tick_params(bottom=False, labelbottom=False)
            else:
                ax.set_title(_FILTER_LABELS.get(filter_name, filter_name))
                if i == 0 and not fdf.empty:
                    ax.legend()
        # One figure-level y-label (the per-row label is long and would overflow
        # a single row's height); it spans the full figure so it never clips.
        fig.supylabel(ylabel)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_intrusion_sweep_detail_stack(
    subject: str,
    probe: str,
    structure: str,
    by_filter: dict[str, pd.DataFrame],
    metric: str,
    filter_order: tuple[str, ...],
    *,
    out_path: Path | str,
    destination: str = "figma",
) -> Path:
    """Per-combo 3×3 detail: one triptych row per OFF set (the old 1×3, stacked).

    ``by_filter`` maps each LAS category in *filter_order* to that combo's sweep
    table. Rows are OFF sets (top -> bottom); columns are the three triptych
    panels (Early/Late burden, NOD incline, hidden OFF across NOD). Exploratory,
    so it is not held to the cross-figure row geometry.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(filter_order)
    xlabel = "intrusion <= T (s)"

    with pp.destination(destination):
        fig, axes = plt.subplots(
            n, 3, figsize=pp.scale(5.6, 1.65 * n), squeeze=False
        )
        for i, filter_name in enumerate(filter_order):
            sweep = by_filter.get(filter_name)
            if sweep is None or sweep.empty:
                continue
            _draw_sweep_triptych(axes[i], sweep, metric, titles=(i == 0))
            axes[i, 0].set_ylabel(
                f"{_FILTER_LABELS.get(filter_name, filter_name)}\nOFF {metric}"
            )
        for ax in axes[-1]:
            ax.set_xlabel(xlabel)
        fig.suptitle(f"{subject} {probe} {structure}: {metric}")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path
