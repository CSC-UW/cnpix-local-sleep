"""Export cross-structure OFF "locality" summaries for the r-offp locality analyses.

An additive, fully separable companion to :mod:`cross_structure_offs` and to
:func:`aggregate_experiment_offs.do_experiment_full`. It reproduces the
group-level data behind ``notebooks/figures/group_cross_structure_offs.ipynb``
(the right subplot of ``cross_structure_3_condition_comparison.svg`` and
``cross_structure_4b_local_vs_overlapping_all_conditions.svg``), plus a
whole-recording NREM/Wake split, and writes three flat parquets into the r-offp
``inst/extdata`` directory so the "locality" questions (Local vs Overlapping OFFs)
can be tested with the house mixed-model machinery.

"Local" / "Overlapping" mirrors the laminar ``supra`` / ``infra`` within-cell
factor: an OFF is Local when no other cortical structure has a temporally
overlapping OFF (``n_overlapping_structures == 0``) and Overlapping otherwise
(:func:`cross_structure_offs.compute_overlap_counts`).

Three parquets (all LLAS-level, matching the notebook figures):

1. ``summarized_locality_overlap_offs.parquet``: one row per
   ``(subject, structure, condition)``; ``mean_overlap_degree`` (+ the other
   per-condition overlap fractions). Drives request 1 (condition contrasts on
   "Mean # overlapping structures").
2. ``summarized_locality_per_condition_llas_offs.parquet``: one row per
   ``(subject, probe, structure, condition, overlap_status)``; per-group medians
   of ``median_duration`` / ``span`` / ``area``. Drives request 2
   (within-condition Local vs Overlapping).
3. ``summarized_locality_full48h_llas_offs.parquet``: one row per
   ``(subject, probe, structure, state, overlap_status)`` with
   ``state in {NREM, Wake}``, overlap computed over the whole 48 h recording
   (not condition-subset). Drives request 3.

Writes only into r-offp, never NFS. Requires NFS mounted: it reads the
whole-recording detection through the :mod:`cross_structure_offs` loaders. Delete
this module and the ``export-locality-offs`` CLI command to remove the Python
side entirely; nothing else imports it.
"""

import pathlib

import pandas as pd

from cnpix_local_sleep import atlas, const
from cnpix_local_sleep.morphological.pipeline import cross_structure_offs as cso

# Output response-variable column -> per-OFF source column. Matches the panels of
# plot 4b (median over OFFs of the per-OFF column). ``median_duration`` is the
# per-OFF per-channel median OFF duration (the notebook's ``median_duration``
# panel), NOT the per-OFF ``duration``.
_MEASURE_SOURCES = {
    "median_duration": "median_duration",
    "median_span": "span",
    "median_area": "area",
}

# Whole-recording states retained for request 3.
_STATES_KEEP = ("NREM", "Wake")

_OVERLAP_STATUS = {True: "Local", False: "Overlapping"}


# -------------------- Shared helpers --------------------


def _aggregate_measures(offs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Per-group median of the three locality measures (+ ``count``, ``clade``)."""
    agg = {out: (src, "median") for out, src in _MEASURE_SOURCES.items()}
    agg["count"] = ("start_time", "count")
    summarized = offs.groupby(group_cols, observed=True).agg(**agg).reset_index()
    summarized["clade"] = "Cx"
    return summarized


def _load_full_hypnogram(subject: str, probe: str):
    """Whole-recording consolidated hypnogram (states) for state tagging.

    Thin wrapper over :func:`cnpix_local_sleep.hyp.load_whole_recording_hypnogram` (kept for
    call-site stability).
    """
    from cnpix_local_sleep import hyp

    return hyp.load_whole_recording_hypnogram(subject, probe)


# Request 1: mean overlap degree per (subject, structure, condition)


def summarize_overlap_degree(off_source: str = cso.DEFAULT_OFF_SOURCE) -> pd.DataFrame:
    """Concatenate per-subject ``condition_comparison.parquet`` (the notebook's
    ``group_comparison``); one row per ``(subject, structure, condition)``."""
    cso.do_experiment(off_source=off_source, no_jitter=True)  # idempotent
    subjects = cso.get_multi_cortical_subjects(off_source)
    frames = []
    for subject in subjects:
        path = cso._get_output_path(
            subject, "condition_comparison.parquet", off_source
        )
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["subject"] = subject
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["clade"] = "Cx"
    return out[
        [
            "subject",
            "structure",
            "condition",
            "n_offs",
            "frac_local",
            "frac_any_overlap",
            "mean_overlap_degree",
            "clade",
        ]
    ]


# Request 2: Local vs Overlapping measures within each statistical condition


def _annotate_per_condition(off_source: str) -> pd.DataFrame:
    """Reproduce the notebook's ``annotated_offs``: per-OFF properties merged with
    the per-OFF overlap annotation, tagged ``overlap_status``."""
    conditions = list(const.CORE_CONDITIONS)
    props = cso.load_cross_structure_offs(off_source, conditions=conditions)

    subjects = cso.get_multi_cortical_subjects(off_source)
    overlaps = []
    for subject in subjects:
        path = cso._get_output_path(subject, "overlap_counts.parquet", off_source)
        if not path.exists():
            continue
        overlaps.append(pd.read_parquet(path))
    group_overlaps = pd.concat(overlaps, ignore_index=True)

    keys = ["subject", "structure", "condition", "start_time", "end_time"]
    left = props.astype({"subject": str, "structure": str, "condition": str})
    right = group_overlaps.astype(
        {"subject": str, "structure": str, "condition": str}
    )
    annotated = left.merge(
        right[keys + ["n_overlapping_structures", "is_local"]],
        on=keys,
        how="inner",
    )
    annotated["overlap_status"] = annotated["is_local"].map(_OVERLAP_STATUS)
    return annotated


def summarize_per_condition_locality(
    off_source: str = cso.DEFAULT_OFF_SOURCE,
) -> pd.DataFrame:
    """Per-group medians per ``(subject, probe, structure, condition,
    overlap_status)`` for the six statistical conditions."""
    annotated = _annotate_per_condition(off_source)
    return _aggregate_measures(
        annotated,
        group_cols=["subject", "probe", "structure", "condition", "overlap_status"],
    )


# Request 3: Local vs Overlapping within NREM / Wake on whole-recording OFFs


def _tag_states(offs: pd.DataFrame) -> pd.Series:
    """Label each OFF's whole-recording sleep state from the consolidated
    hypnogram of its own ``(subject, probe)``.

    OFFs within a ``(subject, probe)`` group are concatenated across structures,
    so ``start_time`` is not globally sorted; ``FloatHypnogram.get_states``
    requires increasing times, so we sort per group and scatter the result back
    to the original rows.
    """
    states = pd.Series("", index=offs.index, dtype=object)
    for (subject, probe), grp in offs.groupby(["subject", "probe"], observed=True):
        hg = _load_full_hypnogram(subject, probe)
        order = grp["start_time"].to_numpy().argsort()
        idx_sorted = grp.index.to_numpy()[order]
        starts_sorted = grp["start_time"].to_numpy()[order]
        states.loc[idx_sorted] = hg.get_states(starts_sorted, default_value="")
    return states


def summarize_whole_recording_locality(
    off_source: str = cso.DEFAULT_OFF_SOURCE,
) -> pd.DataFrame:
    """Per-group medians per ``(subject, probe, structure, state, overlap_status)``.

    Overlap is computed over the whole 48 h recording (the notebook's
    ``whole_annotated_offs``); each OFF is then tagged NREM/Wake and only those
    two states are kept. A ``condition`` column equal to ``state`` is emitted so
    the R runner's ``conditions=`` filter works unchanged.
    """
    if off_source != "morphological-full48h":
        raise ValueError(
            "summarize_whole_recording_locality is only defined for "
            f"off_source='morphological-full48h', got {off_source!r}"
        )
    subjects = cso.get_multi_cortical_subjects(off_source)
    parts = []
    for subject in subjects:
        s_offs = cso.load_whole_recording_offs(off_source, subject=subject)
        s_structs = sorted(s_offs["structure"].dropna().unique())
        if len(s_structs) < 2:
            continue
        s_cond = s_offs["condition"].iloc[0]  # "Full48h"
        s_dict = cso._build_offs_dict(s_offs, s_structs, [s_cond])
        s_counts = cso.compute_overlap_counts(s_dict, s_structs, s_cond)
        for struct in s_structs:
            g = s_dict[(struct, s_cond)].copy()
            if g.empty:
                continue
            g["is_local"] = s_counts[struct] == 0
            parts.append(g)
    whole = pd.concat(parts, ignore_index=True)
    whole["overlap_status"] = whole["is_local"].map(_OVERLAP_STATUS)
    whole["state"] = _tag_states(whole)
    whole = whole[whole["state"].isin(_STATES_KEEP)].copy()
    whole["condition"] = whole["state"]
    return _aggregate_measures(
        whole,
        group_cols=[
            "subject",
            "probe",
            "structure",
            "state",
            "condition",
            "overlap_status",
        ],
    )


# -------------------- Driver --------------------


def export_locality_offs(
    output_dir: pathlib.Path | str,
    *,
    off_source: str = cso.DEFAULT_OFF_SOURCE,
) -> None:
    """Write the three ``summarized_locality_*`` parquets into ``output_dir``."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interim OFF-analysis structure consolidation (e.g. mPPC -> PPC), applied to
    # each summary before writing. Renaming a structure never changes the
    # cross-structure overlap counts (no probe carries both mPPC and PPC, so the
    # SET of structures on a probe is unchanged).
    print("Summarizing overlap degree (request 1)...")
    overlap = atlas.consolidate_off_structure_columns(summarize_overlap_degree(off_source))
    overlap.to_parquet(output_dir / "summarized_locality_overlap_offs.parquet")
    print(f"  {len(overlap)} (subject, structure, condition) rows")

    print("Summarizing per-condition Local vs Overlapping (request 2)...")
    per_cond = atlas.consolidate_off_structure_columns(
        summarize_per_condition_locality(off_source)
    )
    per_cond.to_parquet(
        output_dir / "summarized_locality_per_condition_llas_offs.parquet"
    )
    print(f"  {len(per_cond)} (subject, structure, condition, status) rows")

    print("Summarizing whole-recording NREM/Wake Local vs Overlapping (request 3)...")
    whole = atlas.consolidate_off_structure_columns(
        summarize_whole_recording_locality(off_source)
    )
    whole.to_parquet(output_dir / "summarized_locality_full48h_llas_offs.parquet")
    print(f"  {len(whole)} (subject, structure, state, status) rows")
