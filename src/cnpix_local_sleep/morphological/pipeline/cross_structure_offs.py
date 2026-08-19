"""Cross-structure OFF period relationship analysis.

Analyzes temporal relationships between OFF periods detected across brain
structures within a single multi-structure subject. Investigates local vs
global sleep, pairwise overlaps, event-locked OFF onset probability, OFF
property correlates of overlap, and jitter-based null distributions.

OFF-source-agnostic: every entry point accepts an ``off_source`` selecting one
of :data:`OFF_SOURCES`. ``"morphological"`` reads the per-condition
aggregated parquets produced by ``aggregate_experiment_offs.do_experiment()``;
``"morphological-full48h"`` (default) derives event-level OFFs in memory from the
whole-recording detection, subset to the six statistical conditions.
"""

import itertools
import pathlib

import numpy as np
import pandas as pd
from cnpix_local_sleep import const, hyp
from cnpix_local_sleep.morphological.mua import files as mua_files
from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.morphological.pipeline import utils


# -------------------- OFF source selection --------------------

OFF_SOURCES = ("morphological-full48h", "morphological")
DEFAULT_OFF_SOURCE = "morphological-full48h"

_LAS_MERGE_KEYS = ["subject", "probe", "structure", "start_time", "end_time"]


def _source_leaf(off_source: str) -> str:
    """Output-path leaf distinguishing whole-recording (48h) from per-condition."""
    return "full48h" if off_source == "morphological-full48h" else "per_condition"


def _check_off_source(off_source: str) -> None:
    if off_source not in OFF_SOURCES:
        raise ValueError(
            f"Unknown off_source {off_source!r}; expected one of {OFF_SOURCES}"
        )


# -------------------- Qualifying subjects --------------------


def get_multi_cortical_subjects(off_source: str = DEFAULT_OFF_SOURCE) -> list[str]:
    """Return subjects with OFFs in more than one cortical structure.

    Uses ``sps_conf.get_subject_probe_structure_list()`` filtered to cortical
    structures only, then groups by subject and keeps those with >1 unique
    structure. The per-method inclusion list is selected by ``off_source``.
    """
    _check_off_source(off_source)
    spsl = sps_conf.get_subject_probe_structure_list(
        method="morphological",
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    subject_structures: dict[str, set[str]] = {}
    for subj, _probe, struct in spsl:
        if subj not in subject_structures:
            subject_structures[subj] = set()
        subject_structures[subj].add(struct)
    return sorted(
        [subj for subj, structs in subject_structures.items() if len(structs) > 1],
        key=lambda s: int("".join(c for c in s if c.isdigit())),
    )


# -------------------- Output paths --------------------


def _get_output_dir(subject: str, off_source: str = DEFAULT_OFF_SOURCE) -> pathlib.Path:
    """Return the output directory for cross-structure analysis results.

    The detection is encoded by the ``method=`` segment of the chosen
    files module; ``off_source=<full48h|per_condition>`` distinguishes the
    whole-recording detection from the per-condition detection so the three
    sources never collide.
    """
    _check_off_source(off_source)
    return mua_files.get_path(
        "",
        subject=subject,
        enforce_in_schema=False,
        analysis="cross_structure_offs",
        off_source=_source_leaf(off_source),
    )


def _get_output_path(
    subject: str, filename: str, off_source: str = DEFAULT_OFF_SOURCE
) -> pathlib.Path:
    return _get_output_dir(subject, off_source) / filename


_OUTPUT_FILES = [
    "overlap_counts.parquet",
    "pairwise_overlaps.parquet",
    "peths.parquet",
    "jitter_null.parquet",
    "condition_comparison.parquet",
    "chance_baselines.parquet",
]


def _all_outputs_exist(subject: str, off_source: str = DEFAULT_OFF_SOURCE) -> bool:
    return all(
        _get_output_path(subject, f, off_source).exists() for f in _OUTPUT_FILES
    )


# -------------------- Pure computation functions --------------------


def compute_overlap_counts(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    condition: str,
) -> dict[str, np.ndarray]:
    """For each OFF in each structure, count how many OTHER structures have a
    temporally overlapping OFF.

    Returns dict mapping structure -> array of overlap counts
    (0 = purely local, up to n_structures-1 = fully global).

    Two intervals [a_start, a_end] and [b_start, b_end] overlap iff
    b_start < a_end AND b_end > a_start.

    Only searchsorted on other_starts (which is sorted) is used.
    other_ends is checked explicitly, since it may not be sorted if
    within-structure OFFs overlap.
    """
    result = {}
    for ref_struct in structure_labels:
        ref_offs = offs_dict[(ref_struct, condition)]
        if ref_offs.empty:
            result[ref_struct] = np.array([], dtype=int)
            continue

        ref_starts = ref_offs["start_time"].values
        ref_ends = ref_offs["end_time"].values
        overlap_count = np.zeros(len(ref_offs), dtype=int)

        for other_struct in structure_labels:
            if other_struct == ref_struct:
                continue
            other_offs = offs_dict[(other_struct, condition)]
            if other_offs.empty:
                continue

            other_starts = other_offs["start_time"].values
            other_ends = other_offs["end_time"].values

            for i in range(len(ref_starts)):
                # other_start < ref_end: all j in [0, j_end) satisfy this.
                j_end = np.searchsorted(other_starts, ref_ends[i], side="left")
                if j_end == 0:
                    continue
                # Among candidates, check if any has other_end > ref_start.
                j_mid = np.searchsorted(other_starts, ref_starts[i], side="right")
                if j_mid < j_end or np.any(other_ends[:j_end] > ref_starts[i]):
                    overlap_count[i] += 1

        result[ref_struct] = overlap_count
    return result


def compute_pairwise_overlaps(
    offs_a: pd.DataFrame,
    offs_b: pd.DataFrame,
) -> pd.DataFrame:
    """For each OFF in structure A, find all overlapping OFFs in structure B.

    Returns a DataFrame with one row per (OFF_A, OFF_B) overlap pair,
    containing overlap_duration, fraction_of_a, fraction_of_b,
    onset_lag (b.start - a.start), and offset_lag (b.end - a.end).

    Only searchsorted on b_starts (which is sorted) is used. b_ends is
    checked via vectorized comparison, since it may not be sorted if
    within-structure OFFs overlap.
    """
    if offs_a.empty or offs_b.empty:
        return pd.DataFrame(
            columns=[
                "index_a",
                "index_b",
                "overlap_duration",
                "fraction_of_a",
                "fraction_of_b",
                "onset_lag",
                "offset_lag",
            ]
        )

    a_starts = offs_a["start_time"].values
    a_ends = offs_a["end_time"].values
    b_starts = offs_b["start_time"].values
    b_ends = offs_b["end_time"].values

    records = []
    for i in range(len(a_starts)):
        # b_start < a_end: all j in [0, j_end) satisfy this.
        j_end = np.searchsorted(b_starts, a_ends[i], side="left")
        if j_end == 0:
            continue
        # Among candidates, filter to those with b_end > a_start.
        overlap_mask = b_ends[:j_end] > a_starts[i]
        for j in np.where(overlap_mask)[0]:
            overlap_start = max(a_starts[i], b_starts[j])
            overlap_end = min(a_ends[i], b_ends[j])
            overlap_dur = overlap_end - overlap_start
            dur_a = a_ends[i] - a_starts[i]
            dur_b = b_ends[j] - b_starts[j]
            records.append(
                {
                    "index_a": i,
                    "index_b": j,
                    "overlap_duration": overlap_dur,
                    "fraction_of_a": overlap_dur / dur_a if dur_a > 0 else 0,
                    "fraction_of_b": overlap_dur / dur_b if dur_b > 0 else 0,
                    "onset_lag": b_starts[j] - a_starts[i],
                    "offset_lag": b_ends[j] - a_ends[i],
                }
            )

    return pd.DataFrame(records)


def compute_event_peth(
    reference_times: np.ndarray,
    target_times: np.ndarray,
    window: float = 1.0,
    bin_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Peri-event time histogram of target events relative to reference events.

    Parameters
    ----------
    reference_times
        Times of reference events (e.g., OFF onsets in structure A).
    target_times
        Times of target events (e.g., OFF onsets in structure B).
    window
        Half-window size in seconds (histogram spans [-window, +window]).
    bin_size
        Bin width in seconds.

    Returns
    -------
    bin_centers
        Center of each time bin.
    counts
        Number of target events in each bin, summed across all
        reference events.
    """
    n_bins = int(2 * window / bin_size)
    bin_edges = np.linspace(-window, window, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)

    target_sorted = np.sort(target_times)
    for ref_t in reference_times:
        lo_idx = np.searchsorted(target_sorted, ref_t - window)
        hi_idx = np.searchsorted(target_sorted, ref_t + window)
        relative_times = target_sorted[lo_idx:hi_idx] - ref_t
        hist, _ = np.histogram(relative_times, bins=bin_edges)
        counts += hist

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts


def jitter_off_times(
    offs: pd.DataFrame,
    condition_duration: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Circularly shift all OFF times by a random offset.

    Preserves the temporal structure (inter-OFF intervals) while
    destroying the alignment with other structures.
    """
    if offs.empty:
        return offs.copy()

    shift = rng.uniform(10.0, condition_duration - 10.0)
    jittered = offs.copy()
    jittered["start_time"] = (offs["start_time"] + shift) % condition_duration
    jittered["end_time"] = (offs["end_time"] + shift) % condition_duration

    # Handle wrap-around: split events that cross the boundary.
    wrapped = jittered["end_time"] < jittered["start_time"]
    if wrapped.any():
        pre_wrap = jittered[wrapped].copy()
        post_wrap = jittered[wrapped].copy()
        pre_wrap["end_time"] = condition_duration
        post_wrap["start_time"] = 0.0
        jittered = pd.concat(
            [jittered[~wrapped], pre_wrap, post_wrap], ignore_index=True
        )

    jittered = jittered.sort_values("start_time").reset_index(drop=True)
    return jittered


# -------------------- Data loading --------------------


def _assign_las_category(
    llas: pd.DataFrame, clas: pd.DataFrame, blas: pd.DataFrame
) -> pd.Categorical:
    """Most-restrictive LAS label (BLAS>CLAS>LLAS) per LLAS row, by set membership.

    ``clas`` and ``blas`` are the (nested) subsets of ``llas`` produced by the
    tighter filters, so membership is tested on the shared merge keys
    (:data:`_LAS_MERGE_KEYS`). Valid for both the per-condition aggregates and
    the full-48h frames, since all three levels derive from the same base rows.
    """
    clas_keys = set(map(tuple, clas[_LAS_MERGE_KEYS].values))
    blas_keys = set(map(tuple, blas[_LAS_MERGE_KEYS].values))
    return pd.Categorical(
        [
            "BLAS" if t in blas_keys else "CLAS" if t in clas_keys else "LLAS"
            for t in map(tuple, llas[_LAS_MERGE_KEYS].values)
        ],
        categories=["LLAS", "CLAS", "BLAS"],
        ordered=True,
    )


def load_cross_structure_offs(
    off_source: str = DEFAULT_OFF_SOURCE,
    *,
    subject: str | None = None,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    """Condition-subset, event-level LLAS OFFs across cortical structures.

    Returns one row per OFF (all cortical structures, optionally restricted to a
    single ``subject``), tagged with its statistical ``condition`` and an ordered
    ``category`` column (BLAS>CLAS>LLAS). Works for every key in
    :data:`OFF_SOURCES`:

    - ``"morphological-full48h"`` (default): in-memory full-48h OFFs subset to the
      six conditions via ``aggregate_experiment_offs.load_subset_of_48h_offs``.
    - ``"morphological"``: the per-condition aggregated
      ``{llas,clas,blas}_offs.parquet`` read through the variant files module.
    """
    _check_off_source(off_source)
    if conditions is None:
        conditions = list(const.CORE_CONDITIONS)

    if off_source == "morphological-full48h":
        from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs as agg

        llas = agg.load_subset_of_48h_offs("llas").copy()
        clas = agg.load_subset_of_48h_offs("clas")
        blas = agg.load_subset_of_48h_offs("blas")
        llas["category"] = _assign_las_category(llas, clas, blas)
        offs = llas  # full-48h frames have no ``threshold_group`` to filter.
    else:
        fm = mua_files
        llas = pd.read_parquet(fm.get_path("llas_offs.parquet")).copy()
        clas = pd.read_parquet(fm.get_path("clas_offs.parquet"))
        blas = pd.read_parquet(fm.get_path("blas_offs.parquet"))
        llas["category"] = _assign_las_category(llas, clas, blas)
        offs = llas[llas["threshold_group"] == "None"]

    if subject is not None:
        offs = offs[offs["subject"] == subject]
    offs = offs[offs["condition"].isin(conditions)]
    return offs.reset_index(drop=True)


def load_whole_recording_offs(
    off_source: str = DEFAULT_OFF_SOURCE,
    *,
    subject: str | None = None,
    pseudo_condition: str = "Full48h",
) -> pd.DataFrame:
    """All cortical OFFs over the WHOLE recording (NOT condition-subset).

    Unlike :func:`load_cross_structure_offs`, OFFs are kept regardless of which
    statistical-condition window covers them. The result carries a single
    pseudo-condition (``pseudo_condition``) in its ``condition`` column and the
    same ordered ``category`` column. Only defined for the full-48h source,
    since per-condition detection has no whole-recording counterpart.
    """
    if off_source != "morphological-full48h":
        raise ValueError(
            "load_whole_recording_offs is only defined for "
            f"off_source='morphological-full48h', got {off_source!r}"
        )
    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs as agg
    from cnpix_local_sleep import off_tables

    offs = agg._collect_cortical_48h_whole()
    if subject is not None:
        offs = offs.loc[offs["subject"] == subject].reset_index(drop=True)

    llas = off_tables.filter_offs(offs, "llas")
    clas = off_tables.filter_offs(llas, "clas")
    blas = off_tables.filter_offs(clas, "blas")
    llas = llas.copy()
    llas["category"] = _assign_las_category(llas, clas, blas)
    llas["condition"] = pseudo_condition
    return llas.reset_index(drop=True)


# -------------------- Analysis steps --------------------


def _compute_condition_durations(
    subject: str,
    ref_probe: str,
    conditions: list[str],
) -> dict[str, float]:
    """Load condition durations from hypnograms."""
    hgs = hyp.load_statistical_condition_hypnograms(subject, ref_probe)
    return {cond: hgs[cond]["duration"].sum() for cond in conditions if cond in hgs}


def _build_offs_dict(
    offs: pd.DataFrame,
    structure_labels: list[str],
    conditions: list[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Build offs_dict keyed by (structure, condition)."""
    offs_dict: dict[tuple[str, str], pd.DataFrame] = {}
    for (struct, cond), group in offs.groupby(
        ["structure", "condition"], observed=True
    ):
        offs_dict[(struct, cond)] = group.sort_values("start_time").reset_index(
            drop=True
        )
    # Ensure all (structure, condition) pairs exist.
    for struct in structure_labels:
        for cond in conditions:
            if (struct, cond) not in offs_dict:
                offs_dict[(struct, cond)] = offs.iloc[:0]
    return offs_dict


def _compute_all_overlap_counts(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    conditions: list[str],
    subject: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Compute overlap counts for all conditions.

    Returns
    -------
    overlap_df
        DataFrame with columns: subject, structure, condition, off_index,
        start_time, end_time, n_overlapping_structures, is_local. The
        ``start_time``/``end_time`` keys let downstream consumers merge against a
        re-loaded OFF frame on stable interval bounds rather than the positional
        ``off_index`` (which is sort-order dependent).
    raw_counts
        ``{condition: {structure: np.ndarray}}`` for downstream use.
    """
    raw_counts: dict[str, dict[str, np.ndarray]] = {}
    records = []

    for cond in conditions:
        counts = compute_overlap_counts(offs_dict, structure_labels, cond)
        raw_counts[cond] = counts
        for struct in structure_labels:
            arr = counts[struct]
            group = offs_dict[(struct, cond)]
            starts = group["start_time"].to_numpy()
            ends = group["end_time"].to_numpy()
            for i, c in enumerate(arr):
                records.append(
                    {
                        "subject": subject,
                        "structure": struct,
                        "condition": cond,
                        "off_index": i,
                        "start_time": starts[i],
                        "end_time": ends[i],
                        "n_overlapping_structures": int(c),
                        "is_local": c == 0,
                    }
                )

    return pd.DataFrame(records), raw_counts


def _compute_chance_baselines(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    conditions: list[str],
    condition_durations: dict[str, float],
    raw_counts: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Compute expected vs observed local fractions under independence."""
    records = []
    for cond in conditions:
        T = condition_durations.get(cond, 0)
        if T == 0:
            continue
        for ref_struct in structure_labels:
            counts = raw_counts[cond][ref_struct]
            if len(counts) == 0:
                continue
            p_independent_no_overlap = 1.0
            for other_struct in structure_labels:
                if other_struct == ref_struct:
                    continue
                other_offs = offs_dict[(other_struct, cond)]
                p_off = other_offs["duration"].sum() / T if not other_offs.empty else 0
                p_independent_no_overlap *= 1 - p_off

            observed_local = np.mean(counts == 0)
            expected_local = p_independent_no_overlap
            records.append(
                {
                    "structure": ref_struct,
                    "condition": cond,
                    "observed_local_frac": observed_local,
                    "expected_local_frac": expected_local,
                    "ratio": (
                        observed_local / expected_local
                        if expected_local > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _compute_all_pairwise_overlaps(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    conditions: list[str],
) -> pd.DataFrame:
    """Compute pairwise overlaps for all structure pairs and conditions."""
    structure_pairs = list(itertools.combinations(structure_labels, 2))
    dfs = []
    for struct_a, struct_b in structure_pairs:
        for cond in conditions:
            overlaps = compute_pairwise_overlaps(
                offs_dict[(struct_a, cond)],
                offs_dict[(struct_b, cond)],
            )
            if not overlaps.empty:
                overlaps = overlaps.assign(
                    struct_a=struct_a,
                    struct_b=struct_b,
                    condition=cond,
                )
                dfs.append(overlaps)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def _compute_all_peths(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    conditions: list[str],
    window: float,
    bin_size: float,
) -> pd.DataFrame:
    """Compute PETHs for all structure-pair x condition combos."""
    records = []
    for cond in conditions:
        for ref_struct in structure_labels:
            ref_onsets = offs_dict[(ref_struct, cond)]["start_time"].values
            if len(ref_onsets) == 0:
                continue
            for tgt_struct in structure_labels:
                tgt_onsets = offs_dict[(tgt_struct, cond)]["start_time"].values
                if len(tgt_onsets) == 0:
                    continue
                bin_centers, counts = compute_event_peth(
                    ref_onsets, tgt_onsets, window=window, bin_size=bin_size
                )
                rate = counts / (len(ref_onsets) * bin_size)
                for bc, r, c in zip(bin_centers, rate, counts):
                    records.append(
                        {
                            "ref_structure": ref_struct,
                            "tgt_structure": tgt_struct,
                            "condition": cond,
                            "bin_center": bc,
                            "rate": r,
                            "count": c,
                            "n_ref": len(ref_onsets),
                        }
                    )
    return pd.DataFrame(records)


def _compute_jitter_null(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    conditions: list[str],
    condition_durations: dict[str, float],
    raw_counts: dict[str, dict[str, np.ndarray]],
    n_shuffles: int,
) -> pd.DataFrame:
    """Compute jitter null distributions for overlap fractions."""
    rng = np.random.default_rng(42)
    records = []

    for cond in conditions:
        T = condition_durations.get(cond, 0)
        if T == 0:
            continue

        # Observed overlap fractions.
        observed_fracs = {}
        for struct in structure_labels:
            counts = raw_counts[cond][struct]
            observed_fracs[struct] = (
                np.mean(counts > 0) if len(counts) > 0 else np.nan
            )

        # Null distribution by jittering.
        null_fracs: dict[str, list[float]] = {s: [] for s in structure_labels}
        for _ in range(n_shuffles):
            jittered_dict: dict[tuple[str, str], pd.DataFrame] = {}
            for struct in structure_labels:
                jittered_dict[(struct, cond)] = jitter_off_times(
                    offs_dict[(struct, cond)], T, rng
                )
            jittered_counts = compute_overlap_counts(
                jittered_dict, structure_labels, cond
            )
            for struct in structure_labels:
                c = jittered_counts[struct]
                if len(c) > 0:
                    null_fracs[struct].append(np.mean(c > 0))

        for struct in structure_labels:
            null_arr = np.array(null_fracs[struct])
            obs = observed_fracs[struct]
            if np.isnan(obs) or len(null_arr) == 0:
                continue
            p_value = np.mean(null_arr >= obs)
            for shuf_idx, nf in enumerate(null_arr):
                records.append(
                    {
                        "structure": struct,
                        "condition": cond,
                        "shuffle_index": shuf_idx,
                        "null_overlap_frac": nf,
                        "observed_overlap_frac": obs,
                        "p_value": p_value,
                    }
                )

    return pd.DataFrame(records)


def _compute_condition_comparison(
    structure_labels: list[str],
    conditions: list[str],
    raw_counts: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Compare cross-structure overlap patterns between conditions."""
    records = []
    for cond in conditions:
        for struct in structure_labels:
            counts = raw_counts[cond][struct]
            if len(counts) == 0:
                continue
            records.append(
                {
                    "condition": cond,
                    "structure": struct,
                    "n_offs": len(counts),
                    "frac_local": np.mean(counts == 0),
                    "frac_any_overlap": np.mean(counts > 0),
                    "mean_overlap_degree": np.mean(counts),
                }
            )
    return pd.DataFrame(records)


# Windowed local-shift null and per-OFF "excess globality"
#
# The global circular jitter above (``jitter_off_times``) is invalid for the
# absolute-time, whole-recording 48h OFFs: a global shift smears NREM-dense OFFs
# into wake, destroying the rate non-stationarity that *drives* chance overlap.
#
# This section builds a non-stationarity-respecting null instead. For each focal
# OFF we hold the focal structure FIXED and, on each shuffle, locally jitter the
# onset of every PARTNER structure's OFFs *within their own NREM bout* by
# +/- window/2 (window ~30-180s). This preserves (a) each OFF's duration and
# (b) the within-bout OFF density of every structure, while destroying
# fine-timescale cross-structure alignment. Comparing the observed overlap
# degree to this per-OFF null yields "excess globality": coordination beyond what
# the trivial collision geometry (longer/denser OFFs collide more) already buys.
#
# The null domain (which NREM, hence which OFFs are scored and where jitter is
# confined) is selectable: ``null_scope="whole_recording"`` uses every NREM bout
# across the 48h consolidated hypnogram; a condition name (e.g.
# ``"Early.REC.NREM"``) confines the null to that statistical condition's NREM.


def _overlaps_any(
    ref_starts: np.ndarray,
    ref_ends: np.ndarray,
    other_starts: np.ndarray,
    other_ends: np.ndarray,
) -> np.ndarray:
    """Vectorized: for each ref interval, does ANY ``other`` interval overlap it?

    Intervals ``[a_start, a_end]`` and ``[b_start, b_end]`` overlap iff
    ``b_start < a_end`` AND ``b_end > a_start``. ``other_starts`` must be sorted
    ascending; ``other_ends`` need not be (within-structure OFFs can nest), so the
    "any end exceeds ref_start" test uses the running maximum of the end times
    over the start-sorted prefix ``[0, j_end)`` of candidates with
    ``other_start < ref_end``.
    """
    n_ref = len(ref_starts)
    if n_ref == 0 or len(other_starts) == 0:
        return np.zeros(n_ref, dtype=bool)
    j_end = np.searchsorted(other_starts, ref_ends, side="left")
    running_max_end = np.maximum.accumulate(other_ends)
    has_candidate = j_end > 0
    idx = np.clip(j_end - 1, 0, None)
    return has_candidate & (running_max_end[idx] > ref_starts)


def compute_overlap_degree(
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    condition: str,
) -> dict[str, np.ndarray]:
    """Vectorized overlap degree per structure (count of OTHER structures with an
    overlapping OFF). Same definition as :func:`compute_overlap_counts`, computed
    via :func:`_overlaps_any` so it can be called once per shuffle cheaply.
    """
    result: dict[str, np.ndarray] = {}
    for ref_struct in structure_labels:
        ref = offs_dict[(ref_struct, condition)]
        if ref.empty:
            result[ref_struct] = np.zeros(0, dtype=int)
            continue
        rs = ref["start_time"].to_numpy()
        re = ref["end_time"].to_numpy()
        degree = np.zeros(len(ref), dtype=int)
        for other_struct in structure_labels:
            if other_struct == ref_struct:
                continue
            other = offs_dict[(other_struct, condition)]
            if other.empty:
                continue
            degree += _overlaps_any(
                rs, re, other["start_time"].to_numpy(), other["end_time"].to_numpy()
            ).astype(int)
        result[ref_struct] = degree
    return result


def get_nrem_bouts(
    subject: str,
    probe: str | None,
    null_scope: str,
) -> np.ndarray:
    """NREM bouts (``(N, 2)`` array of ``[start_time, end_time]``) for the null.

    ``null_scope``:

    - ``"whole_recording"``: NREM bouts across the 48h consolidated hypnogram.
    - a statistical-condition name (e.g. ``"Early.REC.NREM"``): NREM bouts within
      that condition's window only.
    """
    if null_scope == "whole_recording":
        hg = hyp.load_whole_recording_hypnogram(subject, probe)
    else:
        hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
        if null_scope not in hgs:
            raise ValueError(
                f"null_scope {null_scope!r} is neither 'whole_recording' nor a "
                f"condition hypnogram for {subject}: {sorted(hgs)}"
            )
        hg = hgs[null_scope]
    nrem = hg.keep_states(["NREM"])
    return np.column_stack(
        [nrem["start_time"].to_numpy(), nrem["end_time"].to_numpy()]
    )


def _assign_bouts(
    starts: np.ndarray, ends: np.ndarray, bouts: np.ndarray
) -> np.ndarray:
    """Index of the NREM bout fully containing each OFF, or ``-1`` if none.

    ``bouts`` must be sorted by start time. An OFF is assigned to bout ``b`` iff
    ``bouts[b, 0] <= start`` and ``end <= bouts[b, 1]`` (full containment, so the
    OFF can be validly jittered without leaving NREM).
    """
    if len(starts) == 0:
        return np.zeros(0, dtype=int)
    bstart = bouts[:, 0]
    bend = bouts[:, 1]
    idx = np.searchsorted(bstart, starts, side="right") - 1
    cand = np.clip(idx, 0, None)
    contained = (idx >= 0) & (starts >= bstart[cand]) & (ends <= bend[cand])
    return np.where(contained, idx, -1)


def windowed_jitter_off_times(
    offs: pd.DataFrame,
    bout_idx: np.ndarray,
    bouts: np.ndarray,
    window: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Locally shift each OFF's onset within its own NREM bout.

    Each onset is shifted by ``U(-window/2, +window/2)`` and reflected (folded)
    back into the valid onset range ``[bout_start, bout_end - duration]`` so the
    OFF stays entirely inside its assigned bout. Reflecting (rather than clipping)
    avoids piling probability mass at the bout edges, preserving duration and
    within-bout OFF density at scales ``>> window`` while randomizing
    fine-timescale alignment. OFFs with ``bout_idx < 0`` are out of domain and
    dropped. Returns a start-sorted copy.
    """
    if offs.empty:
        return offs.copy()
    keep = bout_idx >= 0
    o = offs.loc[keep].copy()
    b = bout_idx[keep]
    starts = o["start_time"].to_numpy()
    durations = o["end_time"].to_numpy() - starts
    lo = bouts[b, 0]
    hi = bouts[b, 1] - durations  # latest valid onset (>= lo for contained OFFs)
    span = hi - lo
    proposed = starts + rng.uniform(-window / 2.0, window / 2.0, size=len(o))
    # Triangle-wave fold of ``proposed`` into [lo, hi]. Degenerate bouts that
    # exactly fit the OFF (span == 0) pin to ``lo``.
    new_start = lo.copy()
    good = span > 0
    period = 2.0 * span[good]
    t = np.mod(proposed[good] - lo[good], period)
    t = np.where(t > span[good], period - t, t)
    new_start[good] = lo[good] + t
    o["start_time"] = new_start
    o["end_time"] = new_start + durations
    return o.sort_values("start_time").reset_index(drop=True)


_EXCESS_PROPERTY_COLS = ("span", "area", "duration", "span_rel2max", "area_rel2span")


def compute_windowed_excess_globality(
    offs: pd.DataFrame,
    structure_labels: list[str],
    *,
    bouts: np.ndarray,
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-OFF "excess globality" under the windowed local-shift null.

    Only OFFs fully contained in a ``bouts`` NREM bout are scored (others are out
    of the null's domain). For each such focal OFF, the focal structure is held
    fixed while every partner structure is locally jittered within its bouts
    (:func:`windowed_jitter_off_times`); the focal overlap degree is recomputed
    ``n_shuffles`` times to form a per-OFF null.

    Returns one row per scored focal OFF, keyed by
    ``(subject, structure, start_time, end_time)`` for merging back onto the OFF
    frame, with:

    - ``observed_degree``: overlap degree on the in-domain OFFs.
    - ``null_mean`` / ``null_std``: mean / SD of the null degree.
    - ``excess_globality`` = ``observed_degree - null_mean`` (duration-corrected
      by construction: the null already contains the collision-geometry effect).
    - ``excess_z`` = ``excess_globality / null_std`` (NaN where ``null_std == 0``).
    - ``p_greater`` = fraction of shuffles with ``null_degree >= observed_degree``
      (one-sided "more global than chance"; small = significant excess).
    - the OFF property columns present in ``offs`` (``span``, ``area`` ...), to
      regress excess against without a second merge.
    """
    rng = np.random.default_rng(seed)
    cond = "__null__"

    # Restrict every structure to in-domain (bout-contained) OFFs; the same
    # restricted set backs both the observed degree and the null.
    offs_dict: dict[tuple[str, str], pd.DataFrame] = {}
    bout_idx: dict[str, np.ndarray] = {}
    for struct in structure_labels:
        s = (
            offs.loc[offs["structure"] == struct]
            .sort_values("start_time")
            .reset_index(drop=True)
        )
        bidx = _assign_bouts(
            s["start_time"].to_numpy(), s["end_time"].to_numpy(), bouts
        )
        s = s.loc[bidx >= 0].reset_index(drop=True)
        offs_dict[(struct, cond)] = s
        bout_idx[struct] = _assign_bouts(
            s["start_time"].to_numpy(), s["end_time"].to_numpy(), bouts
        )

    observed = compute_overlap_degree(offs_dict, structure_labels, cond)
    present = [s for s in structure_labels if not offs_dict[(s, cond)].empty]

    # Analytic collision expectation (independence cross-check on the simulation).
    # For a focal OFF of duration d, the chance it overlaps >=1 OFF in partner s,
    # treating s as a Poisson OFF process of rate lambda_s and mean duration mu_s
    # over the NREM domain, is 1 - exp(-lambda_s * (d + mu_s)) (the "extended
    # interval" argument). E[degree | d] sums this over present partners.
    T_nrem = float(np.sum(bouts[:, 1] - bouts[:, 0]))
    partner_rate: dict[str, float] = {}
    partner_mu: dict[str, float] = {}
    for s in present:
        durs = (
            offs_dict[(s, cond)]["end_time"].to_numpy()
            - offs_dict[(s, cond)]["start_time"].to_numpy()
        )
        partner_rate[s] = len(durs) / T_nrem if T_nrem > 0 else 0.0
        partner_mu[s] = float(durs.mean()) if len(durs) else 0.0

    null_sum = {s: np.zeros(len(offs_dict[(s, cond)])) for s in present}
    null_sumsq = {s: np.zeros(len(offs_dict[(s, cond)])) for s in present}
    null_ge = {s: np.zeros(len(offs_dict[(s, cond)])) for s in present}

    for _ in range(n_shuffles):
        jittered = dict(offs_dict)
        for s in present:
            jittered[(s, cond)] = windowed_jitter_off_times(
                offs_dict[(s, cond)], bout_idx[s], bouts, window, rng
            )
        for focal in present:
            eval_dict = dict(jittered)
            eval_dict[(focal, cond)] = offs_dict[(focal, cond)]  # focal fixed
            deg = compute_overlap_degree(eval_dict, structure_labels, cond)[focal]
            null_sum[focal] += deg
            null_sumsq[focal] += deg.astype(float) ** 2
            null_ge[focal] += deg >= observed[focal]

    records = []
    for focal in present:
        s = offs_dict[(focal, cond)]
        mean = null_sum[focal] / n_shuffles
        var = np.clip(null_sumsq[focal] / n_shuffles - mean**2, 0.0, None)
        std = np.sqrt(var)
        obs = observed[focal].astype(float)
        excess = obs - mean
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 0, excess / std, np.nan)
        d_focal = s["end_time"].to_numpy() - s["start_time"].to_numpy()
        analytic = np.zeros(len(s))
        for other in present:
            if other == focal:
                continue
            analytic += 1.0 - np.exp(
                -partner_rate[other] * (d_focal + partner_mu[other])
            )
        rec = pd.DataFrame(
            {
                "subject": s["subject"].to_numpy()
                if "subject" in s.columns
                else np.nan,
                "structure": focal,
                "start_time": s["start_time"].to_numpy(),
                "end_time": s["end_time"].to_numpy(),
                "observed_degree": observed[focal],
                "null_mean": mean,
                "null_std": std,
                "analytic_expected_degree": analytic,
                "excess_globality": excess,
                "excess_z": z,
                "p_greater": null_ge[focal] / n_shuffles,
            }
        )
        for col in _EXCESS_PROPERTY_COLS:
            if col in s.columns and col not in rec.columns:
                rec[col] = s[col].to_numpy()
        records.append(rec)

    return (
        pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    )


def do_subject_excess_globality(
    subject: str,
    *,
    off_source: str = DEFAULT_OFF_SOURCE,
    null_scope: str = "whole_recording",
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
    save: bool = False,
) -> pd.DataFrame:
    """Per-OFF excess-globality for one subject (whole-recording 48h OFFs).

    Loads the whole-recording cortical OFFs, restricts them to the NREM of
    ``null_scope`` (``"whole_recording"`` or a condition name such as
    ``"Early.REC.NREM"``), and scores each OFF against the windowed local-shift
    null. Requires ``off_source="morphological-full48h"`` (the regime the global
    jitter cannot serve). With ``save=True`` writes
    ``excess_globality_{null_scope}_w{window}.parquet`` next to the other
    cross-structure outputs.
    """
    if off_source != "morphological-full48h":
        raise ValueError(
            "excess-globality is defined only for off_source='morphological-full48h'"
        )
    if not 30.0 <= window <= 180.0:
        utils.log_step(
            "WARNING: window outside recommended 30-180s range",
            subject=subject,
            window=window,
        )

    offs = load_whole_recording_offs(off_source, subject=subject)
    structure_labels = sorted(offs["structure"].dropna().unique())
    if len(structure_labels) < 2:
        utils.log_step("SKIPPING (fewer than 2 structures)", subject=subject)
        return pd.DataFrame()

    ref_probe = offs["probe"].iloc[0]
    bouts = get_nrem_bouts(subject, ref_probe, null_scope)
    utils.log_step(
        "Computing windowed excess globality",
        subject=subject,
        null_scope=null_scope,
        window=window,
        n_shuffles=n_shuffles,
        n_bouts=len(bouts),
    )
    result = compute_windowed_excess_globality(
        offs,
        structure_labels,
        bouts=bouts,
        window=window,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    result["null_scope"] = null_scope
    result["window"] = window

    if save:
        out = _get_output_path(
            subject,
            f"excess_globality_{null_scope}_w{int(window)}.parquet",
            off_source,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out)
        utils.log_step("Saved excess globality", subject=subject, path=str(out))

    return result


def test_excess_above_chance(
    excess_df: pd.DataFrame,
    *,
    alternative: str = "greater",
) -> dict:
    """Group-level test that observed overlap degree exceeds the windowed null.

    Asserts the *location* claim "mean excess globality > 0" (i.e. OFFs are more
    cross-structure-coordinated than the windowed local-shift null predicts) at
    the subject level, never per-OFF -- per-OFF ``p_greater`` values must NOT be
    pooled (that re-introduces pseudoreplication). Two complementary tests:

    1. Paired subject-level (headline, most conservative): per subject, the
       mean ``observed_degree`` vs the mean ``null_mean``, compared with a paired
       Wilcoxon signed-rank test and a paired t-test across the ``n`` subjects
       (one paired value per subject -- immune to within-subject correlation).
    2. Mixed model: per-``(subject, structure)`` mean ``excess_globality``
       with a subject random intercept (``MixedLM``, structures as replicates
       nested in subjects); tests the fixed intercept > 0.

    ``alternative`` is passed to the paired tests (``"greater"`` tests
    observed > null) and used to one-side the mixed-model intercept.

    Returns a dict with ``per_subject`` (DataFrame), ``paired_wilcoxon``,
    ``paired_t``, ``mixedlm`` (each a small result dict), and ``n_subjects``.
    """
    from scipy import stats

    if not {"subject", "structure", "observed_degree", "null_mean",
            "excess_globality"}.issubset(excess_df.columns):
        raise ValueError("excess_df is missing required columns")

    # 1) Paired subject-level (one observed/null pair per subject).
    per_subject = (
        excess_df.groupby("subject", observed=True)
        .agg(
            mean_observed=("observed_degree", "mean"),
            mean_null=("null_mean", "mean"),
            n_offs=("observed_degree", "size"),
        )
        .reset_index()
    )
    per_subject["mean_excess"] = (
        per_subject["mean_observed"] - per_subject["mean_null"]
    )
    n_subjects = len(per_subject)

    w = stats.wilcoxon(
        per_subject["mean_observed"],
        per_subject["mean_null"],
        alternative=alternative,
    )
    t = stats.ttest_rel(
        per_subject["mean_observed"],
        per_subject["mean_null"],
        alternative=alternative,
    )
    paired_wilcoxon = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
    paired_t = {
        "statistic": float(t.statistic),
        "p_value": float(t.pvalue),
        "df": int(n_subjects - 1),
    }

    # 2) Mixed model: per-(subject, structure) mean excess, subject random intercept.
    per_unit = (
        excess_df.groupby(["subject", "structure"], observed=True)["excess_globality"]
        .mean()
        .reset_index(name="mean_excess")
    )
    mixedlm: dict = {"n_units": len(per_unit), "n_groups": per_unit["subject"].nunique()}
    try:
        import statsmodels.formula.api as smf

        fit = smf.mixedlm(
            "mean_excess ~ 1", per_unit, groups=per_unit["subject"]
        ).fit(reml=True)
        coef = float(fit.params["Intercept"])
        se = float(fit.bse["Intercept"])
        z = coef / se if se > 0 else np.nan
        # statsmodels reports a two-sided p; one-side it for the directional claim.
        p_two = float(fit.pvalues["Intercept"])
        if alternative == "greater":
            p_one = p_two / 2 if coef > 0 else 1 - p_two / 2
        elif alternative == "less":
            p_one = p_two / 2 if coef < 0 else 1 - p_two / 2
        else:
            p_one = p_two
        mixedlm.update(
            intercept=coef, se=se, z=z, p_value=p_one, p_two_sided=p_two,
            converged=bool(fit.converged),
        )
    except Exception as exc:  # noqa: BLE001 -- surface, don't crash the report
        mixedlm.update(error=f"{type(exc).__name__}: {exc}")

    return {
        "per_subject": per_subject,
        "paired_wilcoxon": paired_wilcoxon,
        "paired_t": paired_t,
        "mixedlm": mixedlm,
        "n_subjects": n_subjects,
        "alternative": alternative,
    }


# -------------------- Pipeline step --------------------


def do_subject(
    subject: str,
    *,
    off_source: str = DEFAULT_OFF_SOURCE,
    conditions: list[str] | None = None,
    n_shuffles: int = 200,
    peth_window: float = 0.5,
    peth_bin_size: float = 0.01,
    overwrite: bool = False,
    no_jitter: bool = False,
    no_legend: bool = False,
) -> None:
    """Analyze cross-structure OFF period relationships for a single subject.

    Loads OFFs from the chosen ``off_source``, computes overlap counts, pairwise
    overlaps, PETHs, jitter null distributions, and condition comparisons. Saves
    all results as parquet files (under a source-specific directory) and
    generates diagnostic figures.

    Args:
        subject: Subject name.
        off_source: OFF source key, one of :data:`OFF_SOURCES`.
        conditions: Conditions to analyze. Defaults to ``const.CORE_CONDITIONS``.
        n_shuffles: Number of circular jitter shuffles for null distribution.
        peth_window: PETH half-window in seconds.
        peth_bin_size: PETH bin width in seconds.
        overwrite: If True, recompute even if outputs exist.
        no_jitter: If True, skip jitter null computation and plotting. Forced True
            for ``morphological-full48h`` (the circular jitter is invalid for the
            absolute, whole-recording timestamps of the full-48h OFFs).
        no_legend: If True, suppress legends on all plots.
    """
    _check_off_source(off_source)
    if conditions is None:
        conditions = list(const.CORE_CONDITIONS)

    if off_source == "morphological-full48h" and not no_jitter:
        utils.log_step(
            "Skipping jitter null (invalid for absolute-time full-48h OFFs)",
            subject=subject,
        )
        no_jitter = True

    if not overwrite and _all_outputs_exist(subject, off_source):
        utils.log_step("SKIPPING (outputs exist)", subject=subject)
        return

    utils.log_step("Loading OFFs", subject=subject, off_source=off_source)
    offs = load_cross_structure_offs(off_source, subject=subject, conditions=conditions)

    structure_labels = sorted(offs["structure"].dropna().unique())
    n_structures = len(structure_labels)

    if n_structures < 2:
        utils.log_step(
            "SKIPPING (fewer than 2 structures)",
            subject=subject,
            n_structures=n_structures,
        )
        return

    utils.log_step(
        "Analyzing cross-structure OFFs",
        subject=subject,
        structures=structure_labels,
        n_structures=n_structures,
    )

    # Load condition durations.
    ref_probe = offs["probe"].iloc[0]
    condition_durations = _compute_condition_durations(subject, ref_probe, conditions)

    # Build offs_dict.
    offs_dict = _build_offs_dict(offs, structure_labels, conditions)

    # Overlap counts.
    utils.log_step("Computing overlap counts", subject=subject)
    overlap_df, raw_counts = _compute_all_overlap_counts(
        offs_dict, structure_labels, conditions, subject
    )

    # Chance baselines.
    utils.log_step("Computing chance baselines", subject=subject)
    baselines_df = _compute_chance_baselines(
        offs_dict, structure_labels, conditions, condition_durations, raw_counts
    )

    # Pairwise overlaps.
    utils.log_step("Computing pairwise overlaps", subject=subject)
    pairwise_df = _compute_all_pairwise_overlaps(
        offs_dict, structure_labels, conditions
    )

    # PETHs.
    utils.log_step("Computing PETHs", subject=subject)
    peths_df = _compute_all_peths(
        offs_dict, structure_labels, conditions, peth_window, peth_bin_size
    )

    # Jitter null.
    if no_jitter:
        jitter_df = pd.DataFrame()
    else:
        utils.log_step(
            "Computing jitter null distribution",
            subject=subject,
            n_shuffles=n_shuffles,
        )
        jitter_df = _compute_jitter_null(
            offs_dict,
            structure_labels,
            conditions,
            condition_durations,
            raw_counts,
            n_shuffles,
        )

    # Condition comparison.
    utils.log_step("Computing condition comparison", subject=subject)
    comparison_df = _compute_condition_comparison(
        structure_labels, conditions, raw_counts
    )

    # Save outputs.
    output_dir = _get_output_dir(subject, off_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    utils.log_step("Saving results", subject=subject, output_dir=str(output_dir))
    overlap_df.to_parquet(output_dir / "overlap_counts.parquet")
    baselines_df.to_parquet(output_dir / "chance_baselines.parquet")
    pairwise_df.to_parquet(output_dir / "pairwise_overlaps.parquet")
    peths_df.to_parquet(output_dir / "peths.parquet")
    if not no_jitter:
        jitter_df.to_parquet(output_dir / "jitter_null.parquet")
    comparison_df.to_parquet(output_dir / "condition_comparison.parquet")

    # Generate figures.
    utils.log_step("Generating figures", subject=subject)
    from cnpix_local_sleep.morphological.pipeline import plot_cross_structure_offs

    plot_cross_structure_offs.do_subject(
        subject=subject,
        offs_dict=offs_dict,
        structure_labels=structure_labels,
        conditions=conditions,
        condition_durations=condition_durations,
        raw_counts=raw_counts,
        pairwise_df=pairwise_df,
        peths_df=peths_df,
        jitter_df=jitter_df,
        comparison_df=comparison_df,
        baselines_df=baselines_df,
        peth_window=peth_window,
        peth_bin_size=peth_bin_size,
        no_jitter=no_jitter,
        no_legend=no_legend,
        off_source=off_source,
    )

    utils.log_step("DONE", subject=subject)


def do_experiment(
    *,
    off_source: str = DEFAULT_OFF_SOURCE,
    conditions: list[str] | None = None,
    n_shuffles: int = 200,
    overwrite: bool = False,
    no_jitter: bool = False,
    no_legend: bool = False,
) -> None:
    """Run cross-structure OFF analysis for all qualifying subjects.

    Qualifying subjects are those with OFFs in more than one cortical region,
    as determined by ``get_multi_cortical_subjects(off_source)``.

    Args:
        off_source: OFF source key, one of :data:`OFF_SOURCES`.
        conditions: Conditions to analyze. Defaults to ``const.CORE_CONDITIONS``.
        n_shuffles: Number of circular jitter shuffles for null distribution.
        overwrite: If True, recompute even if outputs exist.
        no_jitter: If True, skip jitter null computation and plotting.
        no_legend: If True, suppress legends on all plots.
    """
    _check_off_source(off_source)
    subjects = get_multi_cortical_subjects(off_source)
    utils.log_step(
        "Cross-structure OFF analysis",
        off_source=off_source,
        n_subjects=len(subjects),
        subjects=subjects,
    )
    for subject in subjects:
        do_subject(
            subject,
            off_source=off_source,
            conditions=conditions,
            n_shuffles=n_shuffles,
            overwrite=overwrite,
            no_jitter=no_jitter,
            no_legend=no_legend,
        )