"""Subject-probe-structure configuration for OFF period detection.

Provides functions for loading and querying the cross-method
``subject_probe_structure_config.csv``, which carries per-structure
identifiers and anatomical classification only.

Each detection method (``morphological``, ``unit_based``) owns its
own inclusion CSV, as does the non-method ``annotation-grid`` cohort. The mapping from method
name to inclusion CSV lives in :data:`METHOD_INCLUSION_REGISTRY`.
Call any helper that filters on inclusion with the ``method=`` argument.

A (subject, probe, structure) tuple absent from a method's inclusion
CSV is treated as ``include=False``; every method's CSV must list the
full set of candidate tuples.

This module is consumed by every detection method, so it lives at the
``cnpix_local_sleep`` top level rather than under any method-specific subpackage.
The CSV data file is packaged at ``cnpix_local_sleep.data``;
a later refactor phase may relocate it.
"""

from collections import OrderedDict
from importlib import resources

import pandas as pd


METHOD_INCLUSION_REGISTRY: dict[str, tuple[str, str]] = {
    "morphological": ("cnpix_local_sleep.morphological.mua.data", "quantile_thresholds.csv"),
    # Not a detection method: the (subject, probe, structure) rows annotated on
    # the napari image stacks. Inherited from the retired tom-bugnon variant,
    # whose ``include`` column defines the 26-pair stack cohort. Kept because
    # the stacks and every manual OFF label are pinned to that grid; see
    # ``cnpix_local_sleep.files.get_preprocessed_ap_path``.
    "annotation-grid": ("cnpix_local_sleep.stacks.data", "annotation_grid_cohort.csv"),
    "unit_based": ("cnpix_local_sleep.unit_based.data", "structure_config.csv"),
}


def register_method_inclusion(method: str, package: str, filename: str) -> None:
    """Register an inclusion CSV for a detection method defined outside this package.

    Detection methods that are not part of the manuscript live in the development
    repository, so their inclusion tables cannot be listed above without inverting
    the dev -> manuscript dependency. Those packages call this at import time
    instead; ``offproj.harding`` registers ``"harding"`` this way.
    """
    METHOD_INCLUSION_REGISTRY[method] = (package, filename)

_KEY_COLS = ["subject", "probe", "structure_acronym"]

# Subjective signal-quality tiers, best -> worst. Populated per
# (subject, probe, structure) in the morphological ``quantile_thresholds.csv``
# ``quality_tier`` column from the user's review of the full 48h traces.
QUALITY_TIER_ORDER: tuple[str, ...] = (
    "excellent",
    "very_good",
    "good",
    "mid_to_low",
    "maybe_exclude",
    "probably_exclude",
    "definitely_exclude",
)


def load_config() -> pd.DataFrame:
    """Load the subject-probe-structure configuration CSV.

    Returns a DataFrame with columns:

    - subject, probe, structure_acronym (identifiers)
    - is_cortex, is_thalamus, is_striatum, is_laminar
    - flip_supra_infra (supra/infra layer order is vertically flipped
      relative to probe geometry; see :func:`get_flipped_laminar_combos`)
    - notes

    Inclusion flags previously lived here too but were split into
    per-method CSVs; load them via :func:`load_method_inclusion`.
    """
    with resources.path(
        "cnpix_local_sleep.data", "subject_probe_structure_config.csv"
    ) as f:
        df = pd.read_csv(f)
    return df


def _load_method_csv(method: str) -> pd.DataFrame:
    """Load a method's full inclusion CSV (all columns) as a DataFrame."""
    if method not in METHOD_INCLUSION_REGISTRY:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Valid methods: {list(METHOD_INCLUSION_REGISTRY)}"
        )
    package, filename = METHOD_INCLUSION_REGISTRY[method]
    with resources.path(package, filename) as f:
        return pd.read_csv(f)


def load_method_inclusion(method: str) -> pd.DataFrame:
    """Load the per-method inclusion table for ``method``.

    Returns a DataFrame with columns ``subject``, ``probe``,
    ``structure_acronym``, ``include``. The source CSV may carry
    additional columns (e.g. quantile thresholds for morphological variants);
    those are dropped here.
    """
    return _load_method_csv(method)[_KEY_COLS + ["include"]]


def load_quality_tiers(method: str = "morphological") -> pd.DataFrame:
    """Load per-(subject, probe, structure) signal-quality tiers.

    Returns a DataFrame with columns ``subject``, ``probe``,
    ``structure_acronym``, ``quality_tier``. Only methods whose inclusion
    CSV carries a ``quality_tier`` column are supported (currently
    ``morphological``); others raise. Blank cells appear as NaN.
    """
    df = _load_method_csv(method)
    if "quality_tier" not in df.columns:
        raise ValueError(
            f"Method {method!r} has no quality_tier column in its "
            "inclusion CSV; quality tiers are only defined for the morphological method."
        )
    return df[_KEY_COLS + ["quality_tier"]]


def _spsl_sort_key(x: tuple[str, str, str]) -> tuple[int, str, str]:
    subject, probe, structure = x
    subject_num = int("".join(c for c in subject if c.isdigit()))
    return (subject_num, probe, structure)


def get_analysis_spsl(
    method: str = "morphological",
    *,
    include_maybe_exclude: bool = True,
) -> list[tuple[str, str, str]]:
    """(subject, probe, structure) tuples good enough for advanced reporting.

    "Advanced" = the condition re-aggregation comparison and the intrusion
    sweep; these are reported only for combos of acceptable signal quality.

    The default cutoff admits tiers through ``maybe_exclude`` (i.e. excludes
    ``probably_exclude`` and ``definitely_exclude``). Passing
    ``include_maybe_exclude=False`` tightens the cutoff to ``mid_to_low``,
    dropping the maybe-exclude combos, the one-flag regeneration path for
    "remove the borderline combos". Rows with a blank/unknown tier are
    excluded (conservative).
    """
    cutoff = "maybe_exclude" if include_maybe_exclude else "mid_to_low"
    max_idx = QUALITY_TIER_ORDER.index(cutoff)
    allowed = set(QUALITY_TIER_ORDER[: max_idx + 1])
    tiers = load_quality_tiers(method)
    df = tiers[tiers["quality_tier"].isin(allowed)]
    lst = list(
        zip(df["subject"], df["probe"], df["structure_acronym"])
    )
    return sorted(lst, key=_spsl_sort_key)


def get_plottable_spsl(method: str = "morphological") -> list[tuple[str, str, str]]:
    """(subject, probe, structure) tuples that have (or can have) full-48h OFFs.

    A combo is plottable if both quantile thresholds are populated, meaning
    full-recording detection can run for it. 48h timecourse plots are allowed
    for any such combo regardless of quality tier ("fine to run for
    curiosity"), so this is broader than :func:`get_analysis_spsl`.
    """
    df = _load_method_csv(method)
    has_thresh = (
        df["nrem_quantile_threshold"].notna()
        & df["wake_quantile_threshold"].notna()
    )
    df = df[has_thresh]
    lst = list(
        zip(df["subject"], df["probe"], df["structure_acronym"])
    )
    return sorted(lst, key=_spsl_sort_key)


def _sort_by_first_number(
    lst: list[tuple[str, ...] | str],
) -> list[tuple[str, ...] | str]:
    """Sort list of strings or string tuples by first number in each string."""
    return sorted(
        lst, key=lambda x: int("".join(c for c in x[0] if c.isdigit()))
    )


def get_excluded_structures(method: str) -> list[tuple[str, str, str]]:
    """Return (subject, probe, structure) tuples excluded under ``method``.

    A tuple is excluded if its row in the method's inclusion CSV has
    ``include=False`` *or* if the tuple is absent from the method's CSV
    altogether (missing-row rule).
    """
    sps = load_config()
    inc = load_method_inclusion(method)
    merged = sps.merge(inc, on=_KEY_COLS, how="left")
    excluded = merged[~merged["include"].fillna(False).astype(bool)]
    lst = list(
        zip(excluded["subject"], excluded["probe"], excluded["structure_acronym"])
    )
    return _sort_by_first_number(lst)


def get_laminar_structures() -> list[tuple[str, str, str]]:
    """Return list of (subject, probe, structure) tuples that are laminar."""
    st = load_config()
    df = st[st["is_laminar"]]
    lst = list(zip(df["subject"], df["probe"], df["structure_acronym"]))
    return _sort_by_first_number(lst)


def get_flipped_laminar_combos() -> set[tuple[str, str, str]]:
    """Return (subject, probe, structure) combos with flipped supra/infra order.

    For these combos, brain curvature flips the structure vertically relative
    to the probe geometry, so the supragranular layer sits at *lower* y/depth
    (and infragranular at *higher* y), the opposite of the convention baked
    into :func:`cnpix_local_sleep.channel_anatomy.get_layer_borders`. Consumers that
    derive supra/infra concentration from the raw ``supra_area``/``infra_area``
    columns (which are stored in geometric order) must swap the two for these
    combos. The canonical consumer is
    :func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations`.
    """
    st = load_config()
    df = st[st["flip_supra_infra"]]
    return set(zip(df["subject"], df["probe"], df["structure_acronym"]))


def get_subject_probe_structure_list(
    *,
    method: str | None = None,
    exclude_cortex: bool = False,
    exclude_thalamus: bool = False,
    exclude_striatum: bool = False,
    exclude_other: bool = False,
    exclude_nonlaminar: bool = False,
    respect_exclusions: bool = True,
) -> list[tuple[str, str, str]]:
    """Return unique (subject, probe, structure) tuples from the config CSV.

    Sorted by (subject_number, probe).

    Args:
        method: Detection method whose inclusion CSV governs exclusion.
            Required when ``respect_exclusions=True``. Must be a key of
            :data:`METHOD_INCLUSION_REGISTRY`.
        exclude_cortex: If True, exclude cortical structures.
        exclude_thalamus: If True, exclude thalamic structures.
        exclude_striatum: If True, exclude striatal structures.
        exclude_other: If True, exclude structures not in cortex, thalamus,
            or striatum.
        exclude_nonlaminar: If True, exclude non-laminar structures.
        respect_exclusions: If True, exclude structures whose ``include``
            flag in the method's inclusion CSV is False or missing.
    """
    st = load_config()

    if exclude_cortex:
        st = st[~st["is_cortex"]]
    if exclude_thalamus:
        st = st[~st["is_thalamus"]]
    if exclude_striatum:
        st = st[~st["is_striatum"]]
    if exclude_other:
        is_other = ~(st["is_cortex"] | st["is_thalamus"] | st["is_striatum"])
        st = st[~is_other]
    if exclude_nonlaminar:
        st = st[st["is_laminar"]]
    if respect_exclusions:
        if method is None:
            raise ValueError(
                "method= is required when respect_exclusions=True. "
                f"Valid methods: {list(METHOD_INCLUSION_REGISTRY)}"
            )
        inc = load_method_inclusion(method)
        st = st.merge(inc, on=_KEY_COLS, how="left")
        st = st[st["include"].fillna(False).astype(bool)]

    tuples = list(zip(st["subject"], st["probe"], st["structure_acronym"]))

    def sort_key(x: tuple[str, str, str]) -> tuple[int, str]:
        subject, probe, _ = x
        subject_num = int("".join(c for c in subject if c.isdigit()))
        return (subject_num, probe)

    tuples.sort(key=sort_key)
    return tuples


def get_subject_probe_list(
    expand_probes: bool = True,
    **kwargs,
) -> list[tuple[str, str]] | list[tuple[str, tuple[str, ...]]]:
    """Return unique (subject, probe) tuples.

    If expand_probes is True, return list of tuples of (subject, probe).
    If expand_probes is False, return list of tuples of
    (subject, tuple of probes).

    Additional keyword arguments are passed to
    get_subject_probe_structure_list(); ``method=`` is required unless
    ``respect_exclusions=False`` is passed.
    """
    sps_tuples = get_subject_probe_structure_list(**kwargs)

    # Get unique (subject, probe) pairs, preserving order
    seen = set()
    sp_list = []
    for subject, probe, _ in sps_tuples:
        if (subject, probe) not in seen:
            seen.add((subject, probe))
            sp_list.append((subject, probe))

    if expand_probes:
        return sp_list
    else:
        # Group probes by subject
        subject_probes: OrderedDict[str, list[str]] = OrderedDict()
        for subject, probe in sp_list:
            if subject not in subject_probes:
                subject_probes[subject] = []
            subject_probes[subject].append(probe)
        return [
            (subject, tuple(probes))
            for subject, probes in subject_probes.items()
        ]


def get_finalized_corticothalamic_subject_probe_structure_list(
    method: str,
) -> list[tuple[str, str, str]]:
    spsl_cx = get_subject_probe_structure_list(
        method=method,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )

    spsl_thal = get_subject_probe_structure_list(
        method=method,
        exclude_cortex=True,
        exclude_striatum=True,
        exclude_other=True,
    )

    return spsl_cx + spsl_thal


