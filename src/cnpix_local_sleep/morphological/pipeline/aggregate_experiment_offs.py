import functools
import pathlib

import numpy as np
import pandas as pd
from ecephys import utils
from scipy import stats

from cnpix_local_sleep import atlas, const, hyp
from cnpix_local_sleep.const import CONDITIONS
from cnpix_local_sleep.morphological.mua import files as files
from cnpix_local_sleep import off_tables
from cnpix_local_sleep import sps_conf

_POSTPROCESSED_COLUMNS = [
    "clade",
    "AP.Coord",
    "Cx.AP.group",
    "max_span",
    "span_rel2max",
    "area_rel2span",
]


def _collect_all_offs() -> pd.DataFrame:
    """Collect all OFF period dataframes from existing files.

    Loads (subject, probe, structure) combos from
    ``subject_probe_structure_config.csv``, finds all relevant OFFs on disk,
    and concatenates into a single dataframe.

    No filters based on e.g. span or duration are applied here.

    Returns
    -------
    pd.DataFrame
        DataFrame of all OFF periods across all subjects, probes, structures,
        threshold groups, and conditions.
    """

    offs = []

    spsl_cx = sps_conf.get_subject_probe_structure_list(
        method=files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    for subject, probe, structure in spsl_cx:
        for condition in CONDITIONS:
            pathspec = {
                "subject": subject,
                "probe": probe,
                "structure": structure,
                "threshold_group": None,
                "condition": condition,
            }
            fpath = files.get_offs_path(**pathspec)
            if fpath.exists():
                _offs = (
                    pd.read_parquet(fpath).assign(**pathspec).dropna(axis=1, how="all")
                )
                offs.append(_offs)

    return _finalize_collected_offs(offs)


def _finalize_collected_offs(
    frames: list[pd.DataFrame], condition_subset: bool = True
) -> pd.DataFrame:
    """Concatenate per-(spc) OFF frames and apply shared cleanup + categoricals.

    Shared by :func:`_collect_all_offs` (per-condition detection files) and
    :func:`_collect_all_offs_from_full` (48h files subset by condition) so both
    collection paths produce an identically-typed frame for aggregation.

    When ``condition_subset=False`` the frames carry no ``condition`` column
    (whole-recording detection that was never tagged/subset), so the ordered
    ``condition`` categorical step is skipped.
    """
    if not frames:
        return pd.DataFrame()

    offs = pd.concat(frames, ignore_index=True)

    # Drop columns not needed for aggregation
    # - label: Only useful for visualization with lbl_ixs (not saved to disk)
    drop_cols = ["label"]
    offs = offs.drop(columns=[c for c in drop_cols if c in offs.columns])

    # Interim OFF-analysis structure consolidation (e.g. mPPC -> PPC), applied
    # after detection + postprocess (raw-acronym-keyed) and before the structure
    # categorical below, so aggregation/export see the consolidated label only.
    offs = atlas.consolidate_off_structure_columns(offs)

    # Convert to categoricals with specified order
    offs["subject"] = pd.Categorical(
        offs["subject"],
        categories=utils.misc.sort_strings_by_integer(offs["subject"].unique()),
        ordered=True,
    )
    offs["probe"] = pd.Categorical(
        offs["probe"], categories=["imec0", "imec1"], ordered=True
    )
    if condition_subset and "condition" in offs.columns:
        offs["condition"] = pd.Categorical(
            offs["condition"],
            categories=[
                "Early.BSL.NREM",
                "Early.REC.NREM.Match",
                "Early.NOD.Wake",
                "Late.NOD.Wake",
                "Early.REC.NREM",
                "Late.REC.NREM",
            ],
            ordered=True,
        )

    # Convert to categoricals without specified order
    cat_cols = [
        "structure",
        "clade",
    ]
    for col in cat_cols:
        if col in offs.columns:
            offs[col] = offs[col].astype("category")

    # Convert to ordered categorical (anterior -> posterior)
    offs["Cx.AP.group"] = pd.Categorical(
        offs["Cx.AP.group"],
        categories=["ant", "cent-ant", "cent-post", "post"],
        ordered=True,
    )

    return offs


def _collect_all_offs_from_full(condition_subset: bool = True) -> pd.DataFrame:
    """Collect OFFs from the full-48h detection, optionally subset to conditions.

    Mirrors :func:`_collect_all_offs` but sources from the condition-agnostic
    full-recording ``offs.parquet`` (``mua.files.get_full_offs_path``) instead
    of the per-condition detection files. Postprocessing columns (clade, A/P
    group, normalized features) are added in memory so the downstream
    filter/aggregate steps are identical to the canonical path.

    When ``condition_subset=True`` (default) each OFF is tagged with the core
    condition whose statistical hypnogram covers its ``start_time`` (the windows
    are disjoint, so at most one matches), and OFFs outside every core condition
    are dropped. When ``condition_subset=False`` no condition tagging or drop is
    performed, so the result is the whole-recording OFF set (no ``condition``
    column); use this for continuous-timecourse / whole-recording analyses.

    Cortex-only restriction is applied by callers (e.g.
    :func:`_collect_cortical_48h`), matching the canonical LLAS/CLAS/BLAS scope.
    """
    from cnpix_local_sleep.morphological.pipeline import postprocess_offs as pp

    spsl_cx = sps_conf.get_subject_probe_structure_list(
        method=files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )

    frames = []
    for subject, probe, structure in spsl_cx:
        fpath = files.get_full_offs_path(subject, probe, structure)
        if not fpath.exists():
            continue
        offs = pd.read_parquet(fpath)
        if offs.empty:
            continue

        if condition_subset:
            hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
            starts = offs["start_time"].to_numpy()
            condition = pd.Series(pd.NA, index=offs.index, dtype="object")
            for cond in const.CORE_CONDITIONS:
                if cond not in hgs:
                    continue
                mask = hgs[cond].covers_time(starts) & condition.isna().to_numpy()
                condition.loc[mask] = cond
            offs["condition"] = condition.to_numpy()
            offs = offs.loc[offs["condition"].notna()].copy()
            if offs.empty:
                continue

        offs["subject"] = subject
        offs["probe"] = probe
        offs["structure"] = structure
        pp.postprocess_offs_frame(offs, structure)
        frames.append(offs.dropna(axis=1, how="all"))

    return _finalize_collected_offs(frames, condition_subset=condition_subset)


def _apply_additional_exclusions(offs: pd.DataFrame) -> pd.DataFrame:
    """Apply exclusions not specified in subject_probe_structure_config.csv.

    Exclusions specified in `subject_probe_structure_config.csv` are based on signal/data quality,
    and should already have been applied in `get_subject_probe_structure_list()`.

    Exclusions performed here are for missing data.

    Giuseppe's first NOD hour is genuinely missing, and is not the stretch
    recovered on 2026-08-12 when three mis-named visbrain files were made
    discoverable again (that recovery lengthened his *late* NOD wake window).
    ``Early.NOD`` is listed alongside ``Early.NOD.Wake`` so the period-level
    condition is covered too, should it ever be tagged: neither it nor
    ``Early.EXT`` appears in :data:`cnpix_local_sleep.const.CONDITIONS` today, so those
    two entries are inert guards rather than active exclusions.
    """
    drop = (offs["subject"] == "CNPIX7-Giuseppe") & (
        offs["condition"].isin(["Early.NOD", "Early.NOD.Wake", "Early.EXT"])
    )
    drop |= (offs["subject"] == "CNPIX15-Claude") & (
        offs["condition"].isin(["Early.NOD.Wake"])
    )
    return offs.loc[~drop].reset_index(drop=True)


def _apply_filters(offs: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filter criteria to OFF periods.

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame of OFF periods.
    filters : dict
        Dictionary mapping column names to (min, max) tuples.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows that pass all filter criteria.
    """
    mask = pd.Series(True, index=offs.index)
    for col, (lo, hi) in filters.items():
        mask &= offs[col].between(lo, hi)
    return offs.loc[mask].reset_index(drop=True)


def _filter_cortical_by_category(
    cortical: pd.DataFrame, filter_name: str
) -> pd.DataFrame:
    """Filter the untagged cortical OFF frame by a LAS category.

    Supports the ``llas_exclusive`` = ``llas & ~clas`` adjacent-partition
    complement (OFFs admitted by the LLAS filter but rejected by CLAS) in addition
    to the plain named filters. The exclusive mask reuses
    :func:`off_tables.off_filter_mask`, the same column-threshold SPOT behind
    ``EXCLUSIVE_FILTERS`` in ``morphological/mua/pipeline/full48h.py`` and
    :func:`export_full48h_exclusive_offs`, so a llas_exclusive summary is the
    exact per-condition analogue of the committed
    ``summarized_full48h_llas_exclusive_offs.parquet``.
    """
    if filter_name == "llas_exclusive":
        mask = off_tables.off_filter_mask(cortical, "llas") & ~off_tables.off_filter_mask(
            cortical, "clas"
        )
        return cortical.loc[mask].reset_index(drop=True)
    return _apply_filters(cortical, off_tables.NAMED_FILTERS[filter_name])


def _get_condition_durations(offs: pd.DataFrame) -> pd.DataFrame:
    """Get condition durations for all (subject, probe, condition) combos present in
    OFFs.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex: (subject, probe, condition) and column "duration".
    """
    # Get unique combinations of (subject, probe, condition) from the OFFs
    cols = ["subject", "probe", "condition"]
    spc = offs[cols].drop_duplicates().set_index(cols).sort_index()

    # Get unique combinations of (subject, probe) from the multi-index
    subject_probe_pairs = spc.index.droplevel(["condition"]).unique()

    # Load hypnograms for each unique (subject, probe) pair
    for subj, prb in subject_probe_pairs:
        hgs = hyp.load_statistical_condition_hypnograms(subj, prb)
        conditions = spc.loc[subj, prb].index.get_level_values("condition").unique()
        for cond in conditions:
            sel = (subj, prb, cond)
            spc.loc[sel, "duration"] = hgs[cond]["duration"].sum()

    return spc


def _get_summarized_metrics(
    offs: pd.DataFrame,
    durs: pd.DataFrame,
    grouping_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute summarized OFF period metrics for each group.

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame of OFF periods.
    durs : pd.DataFrame
        DataFrame of condition durations with MultiIndex:
        (subject, probe, condition) and column "duration".
    grouping_cols : list[str] | None
        Columns to group by. If None, defaults to
        ["subject", "probe", "structure", "condition"].

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex defined by ``grouping_cols`` and
        summarized OFF period metrics as columns.
    """
    if grouping_cols is None:
        grouping_cols = [
            "subject",
            "probe",
            "structure",
            "condition",
        ]
    grouped = offs.groupby(grouping_cols, observed=True)

    # Compute metrics that can be computed directly from OFF periods
    agg_dict = {
        "median_duration": ("duration", "median"),
        "mean_boxcox_duration": ("boxcox_duration", "mean"),
        "median_median_duration": ("median_duration", "median"),
        "mean_median_duration": ("median_duration", "mean"),
        "mean_boxcox_median_duration": ("boxcox_median_duration", "mean"),
        "median_span": ("span", "median"),
        "mean_span": ("span", "mean"),
        "mean_boxcox_span": ("boxcox_span", "mean"),
        "median_span_rel2max": ("span_rel2max", "median"),
        "mean_boxcox_span_rel2max": ("boxcox_span_rel2max", "mean"),
        "median_area": ("area", "median"),
        "mean_area": ("area", "mean"),
        "mean_boxcox_area": ("boxcox_area", "mean"),
        "median_area_rel2span": ("area_rel2span", "median"),
        "mean_boxcox_area_rel2span": ("boxcox_area_rel2span", "mean"),
        "total_area": ("area", "sum"),
        "total_area_rel2span": ("area_rel2span", "sum"),
        "count": ("start_time", "count"),
        "max_span": ("max_span", "max"),
        "median_median_trace": ("median_trace", "median"),
        "mean_median_trace": ("median_trace", "mean"),
        "median_min_trace": ("min_trace", "median"),
        "median_mad_trace": ("mad_trace", "median"),
        "mean_mad_trace": ("mad_trace", "mean"),
        "median_onset_slope": ("onset_slope", "median"),
        "mad_onset_slope": ("onset_slope", stats.median_abs_deviation),
        "median_abs_onset_slope": ("onset_slope", lambda x: np.median(np.abs(x))),
        "median_onset_r2": ("onset_r2", "median"),
        "median_onset_jitter": ("onset_jitter", "median"),
        "median_onset_mad": ("onset_mad", "median"),
        "mean_onset_mad": ("onset_mad", "mean"),
        "median_offset_slope": ("offset_slope", "median"),
        "mad_offset_slope": ("offset_slope", stats.median_abs_deviation),
        "median_abs_offset_slope": ("offset_slope", lambda x: np.median(np.abs(x))),
        "median_offset_r2": ("offset_r2", "median"),
        "median_offset_jitter": ("offset_jitter", "median"),
        "median_offset_mad": ("offset_mad", "median"),
        "mean_offset_mad": ("offset_mad", "mean"),
        "median_onset_offset_wedge": ("onset_offset_wedge", "median"),
        "median_abs_onset_offset_wedge": (
            "onset_offset_wedge",
            lambda x: np.median(np.abs(x)),
        ),
    }

    # Include grouped Box-Cox means when grouped_boxcox_* columns are present
    grouped_boxcox_cols = [c for c in offs.columns if c.startswith("grouped_boxcox_")]
    for col in grouped_boxcox_cols:
        agg_dict[f"mean_{col}"] = (col, "mean")

    summarized = grouped.agg(**agg_dict)

    # Compute metrics that depend on other summarized metrics
    summarized["rate"] = summarized["count"] / durs["duration"]
    summarized["total_area_norm"] = summarized["total_area_rel2span"] / durs["duration"]

    return summarized


def _apply_boxcox(
    offs: pd.DataFrame,
    rows_to_transform: pd.Series,
    cols_to_transform: list[str] = [
        "duration",
        "median_duration",
        "span",
        "span_rel2max",
        "area",
        "area_rel2span",
    ],
    prefix: str = "boxcox",
) -> pd.DataFrame:
    """Apply transformations (e.g. log, boxcox) to OFF periods dataframe.

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame of OFF periods.
    rows_to_transform : pd.Series
        Boolean series indicating which rows to apply transformations to.
    cols_to_transform : list[str]
        List of column names to apply transformations to.
    prefix : str
        Prefix for the transformed column names. Default is "boxcox", which
        produces columns like "boxcox_duration". Use "grouped_boxcox" to
        produce columns like "grouped_boxcox_duration".

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed feature columns.
    dict
        Dictionary of Box-Cox lambda values for each transformed column.
    """
    lambds = {}
    for col in cols_to_transform:
        x = offs.loc[rows_to_transform, col].to_numpy()
        if np.all(x[0] == x):
            print(f"Column {col} has zero variance; skipping Box-Cox transformation.")
            lambds[col] = np.nan
            offs.loc[rows_to_transform, f"{prefix}_{col}"] = np.nan
        else:
            y, lambds[col] = stats.boxcox(x)
            offs.loc[rows_to_transform, f"{prefix}_{col}"] = y
    return offs, lambds


_QUARTILE_LABELS = ["oQ1", "oQ2", "oQ3", "oQ4"]


def _assign_min_trace_quartiles(
    offs: pd.DataFrame,
    group_cols: list[str] = ["subject", "probe", "structure"],
) -> pd.DataFrame:
    """Assign a ``min_trace_quartile`` label to each OFF period.

    Quartile bin edges are computed separately for each group defined by
    ``group_cols``, so that the quartile boundaries reflect the distribution
    within each (subject, probe, structure) combination.

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame of OFF periods. Must contain a ``min_trace`` column.
    group_cols : list[str]
        Columns to group by when computing quartile boundaries.

    Returns
    -------
    pd.DataFrame
        Copy of ``offs`` with an added ``min_trace_quartile`` categorical
        column (values: oQ1, oQ2, oQ3, oQ4).
    """
    offs = offs.copy()
    offs["min_trace_quartile"] = pd.Categorical(
        [None] * len(offs), categories=_QUARTILE_LABELS, ordered=True
    )
    for _, group_df in offs.groupby(group_cols, observed=True):
        idx = group_df.index
        offs.loc[idx, "min_trace_quartile"] = pd.qcut(
            offs.loc[idx, "min_trace"],
            q=4,
            labels=_QUARTILE_LABELS[
                ::-1
            ],  # Reverse so that Q1 = highest min_trace (least negative)
        ).cat.reorder_categories(_QUARTILE_LABELS, ordered=True)
    return offs


_BOXCOX_GROUP_COLS = ["subject", "probe", "structure"]


def _apply_grouped_boxcox(
    offs: pd.DataFrame,
    cols_to_transform: list[str] = [
        "duration",
        "median_duration",
        "span",
        "span_rel2max",
        "area",
        "area_rel2span",
    ],
) -> tuple[pd.DataFrame, dict[tuple, dict[str, float]]]:
    """Apply Box-Cox transformations separately per (subject, probe, structure) group.

    Transformed columns are named with the ``grouped_boxcox_`` prefix (e.g.
    ``grouped_boxcox_duration``).

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame of OFF periods.
    cols_to_transform : list[str]
        List of column names to apply transformations to.

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed feature columns.
    dict[tuple, dict[str, float]]
        Nested dictionary of Box-Cox lambda values, keyed by
        ``(subject, probe, structure)`` tuples.
    """
    grouped_lambdas = {}
    for group_key, group_df in offs.groupby(_BOXCOX_GROUP_COLS, observed=True):
        mask = offs.index.isin(group_df.index)
        offs, lambds = _apply_boxcox(
            offs, mask, cols_to_transform, prefix="grouped_boxcox"
        )
        grouped_lambdas[group_key] = lambds
    return offs, grouped_lambdas


def _boxcox_category(offs: pd.DataFrame, grouped_boxcox: bool = False) -> pd.DataFrame:
    """Apply global (and, if requested, per-group) Box-Cox to a filtered category.

    The shared Box-Cox core, used by both the on-disk writer
    (:func:`_save_filtered_category`) and the in-memory summarizer
    (:func:`summarize_subset_of_48h_offs`). Returns a copy of *offs* with
    ``boxcox_*`` (and, when ``grouped_boxcox``, ``grouped_boxcox_*``) columns and
    the fitted lambdas recorded in ``.attrs``.
    """
    offs = offs.copy()
    xfrm_mask = pd.Series(True, index=offs.index)
    offs, ungrouped_lambds = _apply_boxcox(offs, xfrm_mask)
    offs.attrs["boxcox_lambdas"] = ungrouped_lambds

    if grouped_boxcox:
        offs, grouped_lambds = _apply_grouped_boxcox(offs)
        offs.attrs["grouped_boxcox_lambdas"] = {
            str(k): v for k, v in grouped_lambds.items()
        }
    return offs


def _summarize_with_struct_info(
    offs: pd.DataFrame, durs: pd.DataFrame, grouping_cols: list[str] | None = None
) -> pd.DataFrame:
    """Summarized metrics for *offs*, merged with per-structure metadata.

    *offs* must already carry the Box-Cox columns. The shared summary core, used
    by both the on-disk writer and :func:`summarize_subset_of_48h_offs`.
    """
    extra = {} if grouping_cols is None else {"grouping_cols": grouping_cols}
    summarized = _get_summarized_metrics(offs, durs, **extra).reset_index()
    struct_info = offs[
        ["structure", "clade", "AP.Coord", "Cx.AP.group"]
    ].drop_duplicates()
    return summarized.merge(struct_info, on="structure", how="left")


def _resolve_out_path(
    filename: str, output_dir: pathlib.Path | None
) -> pathlib.Path:
    """Resolve an output path for an aggregation artifact.

    When ``output_dir`` is ``None`` (the default for :func:`do_experiment`), the
    canonical NFS path is used via ``mua.files.get_path`` (``method=morphological``).
    When ``output_dir`` is given (e.g. :func:`do_experiment_full` writing into
    ``r-offp/inst/extdata``), the file is written there as a flat ``output_dir /
    filename``, never on NFS.
    """
    if output_dir is None:
        return files.get_path(filename)
    return pathlib.Path(output_dir) / filename


def _save_filtered_category(
    name: str,
    offs: pd.DataFrame,
    durs: pd.DataFrame,
    grouped_boxcox: bool = False,
    output_dir: pathlib.Path | None = None,
) -> None:
    """Apply Box-Cox transforms, save OFFs and summarized metrics for one category.

    Ungrouped (global) Box-Cox transforms are always applied, producing
    ``boxcox_*`` columns. When ``grouped_boxcox=True``, per-group transforms
    are also applied, producing additional ``grouped_boxcox_*`` columns.

    Parameters
    ----------
    name : str
        Category name (e.g. "llas", "clas", "blas"). Used to construct filenames.
    offs : pd.DataFrame
        DataFrame of already-filtered OFF periods for this category.
    durs : pd.DataFrame
        DataFrame of condition durations with MultiIndex:
        (subject, probe, condition) and column "duration".
    grouped_boxcox : bool
        If True, also fit separate Box-Cox lambdas per (subject, probe,
        structure) group, in addition to the global lambdas.
    output_dir : pathlib.Path | None
        If ``None`` (default), write to the canonical NFS path. Otherwise write
        flat files into ``output_dir`` (see :func:`_resolve_out_path`).
    """
    print(f"Applying Box-Cox transformations for {name} OFFs...")
    offs = _boxcox_category(offs, grouped_boxcox=grouped_boxcox)

    print(f"Saving {name}_offs.parquet...")
    offs_path = _resolve_out_path(f"{name}_offs.parquet", output_dir)
    offs_path.parent.mkdir(parents=True, exist_ok=True)
    offs.to_parquet(offs_path)

    print(f"Computing summarized metrics for {name} OFFs...")
    summarized = _summarize_with_struct_info(offs, durs)

    summarized_path = _resolve_out_path(f"summarized_{name}_offs.parquet", output_dir)
    summarized_path.parent.mkdir(parents=True, exist_ok=True)
    summarized.to_parquet(summarized_path)

    # Also save summarized metrics grouped by min_trace quartile
    print(f"Computing summarized metrics for {name} OFFs by min_trace quartile...")
    offs_q = _assign_min_trace_quartiles(offs)
    quartile_grouping = [
        "subject",
        "probe",
        "structure",
        "condition",
        "min_trace_quartile",
    ]
    summarized_by_quartile = _summarize_with_struct_info(
        offs_q, durs, grouping_cols=quartile_grouping
    )
    quartile_path = _resolve_out_path(
        f"summarized_{name}_offs_by_min_trace_quartile.parquet", output_dir
    )
    quartile_path.parent.mkdir(parents=True, exist_ok=True)
    summarized_by_quartile.to_parquet(quartile_path)


def do_experiment(grouped_boxcox: bool = False):
    """Aggregate cortical spatial OFF periods for the project/experiment.

    Produces three filtered categories of OFFs (LLAS, CLAS, BLAS), each saved
    as both individual OFF periods and summarized metrics.

    Parameters
    ----------
    grouped_boxcox : bool
        If True, also fit separate Box-Cox lambdas per (subject, probe,
        structure) group, producing ``grouped_boxcox_*`` columns alongside
        the global ``boxcox_*`` columns. If False (default), only global
        lambdas are fitted.

    Pipeline:

    1. Collect raw OFFs from disk (expects postprocessed columns from
       ``postprocess-offs``).
    2. Apply additional exclusions based on missing data.
    3. Retain only cortical (Cx) OFFs (using ``clade`` from postprocessing).
    4. Apply LLAS filters (most liberal).
    5. For each category (LLAS, CLAS, BLAS):
        - Apply category-specific filters (CLAS from LLAS, BLAS from CLAS).
        - Apply global Box-Cox transformations (and grouped, if requested).
        - Save individual OFFs and summarized metrics.

    Outputs:
        - "llas_offs.parquet", "summarized_llas_offs.parquet"
        - "clas_offs.parquet", "summarized_clas_offs.parquet"
        - "blas_offs.parquet", "summarized_blas_offs.parquet"
        - "summarized_{llas,clas,blas}_offs_by_min_trace_quartile.parquet"
        - "condition_durations.parquet"
    """
    print("Collecting all OFFs...")
    offs = _collect_all_offs()

    missing = [c for c in _POSTPROCESSED_COLUMNS if c not in offs.columns]
    if missing:
        raise ValueError(
            f"Missing postprocessed columns: {missing}. "
            "Run `postprocess-offs` before `aggregate-offs`."
        )

    print("Applying additional exclusions...")
    offs = _apply_additional_exclusions(offs)

    print("Retaining cortical OFFs...")
    offs = offs.loc[offs["clade"] == "Cx"].reset_index(drop=True)

    print("Applying LLAS filters...")
    llas_offs = _apply_filters(offs, off_tables.llas_filters)

    print("Getting condition durations...")
    durs = _get_condition_durations(llas_offs)
    condition_durations_path = files.get_path("condition_durations.parquet")
    condition_durations_path.parent.mkdir(parents=True, exist_ok=True)
    durs.to_parquet(condition_durations_path)

    _save_filtered_category("llas", llas_offs, durs, grouped_boxcox=grouped_boxcox)

    print("Applying CLAS filters...")
    clas_offs = _apply_filters(llas_offs, off_tables.clas_filters)
    _save_filtered_category("clas", clas_offs, durs, grouped_boxcox=grouped_boxcox)

    print("Applying BLAS filters...")
    blas_offs = _apply_filters(clas_offs, off_tables.blas_filters)
    _save_filtered_category("blas", blas_offs, durs, grouped_boxcox=grouped_boxcox)


def do_experiment_full(
    output_dir: pathlib.Path | str,
    *,
    grouped_boxcox: bool = False,
    name_prefix: str = "full48h_",
) -> None:
    """Aggregate full-48h cortical OFFs and write parity parquets to ``output_dir``.

    The full-48h counterpart of :func:`do_experiment`. Instead of the
    per-condition detection files, it sources OFFs from the whole-recording
    state-aware detection subset to the six statistical conditions
    (:func:`_collect_cortical_48h`), and writes the LLAS/CLAS/BLAS artifacts as
    flat files under ``output_dir`` with a ``name_prefix`` (default
    ``"full48h_"``), e.g. ``r-offp/inst/extdata`` so the R package can consume
    them as its ``full48h`` OFF source.

    Writes (per category, with ``name_prefix``):
        - "{prefix}{cat}_offs.parquet", "summarized_{prefix}{cat}_offs.parquet"
        - "summarized_{prefix}{cat}_offs_by_min_trace_quartile.parquet"
        - "{prefix}condition_durations.parquet"

    Never writes to NFS; this is the deliberate replacement for the retired NFS
    ``full48h_*`` artifacts (which had no consumer and were a drift hazard). The
    summarized outputs are guaranteed identical to the in-memory loaders
    (:func:`summarize_subset_of_48h_offs`,
    :func:`summarize_subset_of_48h_offs_by_min_trace_quartile`) used by the
    exploratory homeostasis plots, since both share the same Box-Cox / summary /
    quartile helpers.

    Parameters
    ----------
    output_dir : pathlib.Path | str
        Destination directory for the flat parquet files (created if needed).
    grouped_boxcox : bool
        If True, also fit per-(subject, probe, structure) Box-Cox lambdas.
    name_prefix : str
        Filename prefix for every artifact. Default ``"full48h_"``.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting full-48h cortical OFFs (condition-subset)...")
    cortical = _collect_cortical_48h()

    print("Applying LLAS filters...")
    llas_offs = _apply_filters(cortical, off_tables.llas_filters)

    print("Getting condition durations...")
    durs = _get_condition_durations(llas_offs)
    durs.to_parquet(output_dir / f"{name_prefix}condition_durations.parquet")

    _save_filtered_category(
        f"{name_prefix}llas",
        llas_offs,
        durs,
        grouped_boxcox=grouped_boxcox,
        output_dir=output_dir,
    )

    print("Applying CLAS filters...")
    clas_offs = _apply_filters(llas_offs, off_tables.clas_filters)
    _save_filtered_category(
        f"{name_prefix}clas",
        clas_offs,
        durs,
        grouped_boxcox=grouped_boxcox,
        output_dir=output_dir,
    )

    print("Applying BLAS filters...")
    blas_offs = _apply_filters(clas_offs, off_tables.blas_filters)
    _save_filtered_category(
        f"{name_prefix}blas",
        blas_offs,
        durs,
        grouped_boxcox=grouped_boxcox,
        output_dir=output_dir,
    )


def export_full48h_exclusive_offs(
    output_dir: pathlib.Path | str,
    *,
    name_prefix: str = "full48h_",
) -> None:
    """Write the summarized full-48h LLAS-exclusive OFF parquet into ``output_dir``.

    An additive, offline companion to :func:`do_experiment_full`. It does
    NOT re-run detection or touch NFS: it reads the already-exported
    ``{prefix}llas_offs.parquet`` and
    ``{prefix}condition_durations.parquet`` in ``output_dir`` (e.g.
    ``r-offp/inst/extdata``), keeps the OFFs admitted by the LLAS filter but
    rejected by the CLAS filter (the "adjacent partition" complement
    ``llas & ~clas``), and writes
    ``summarized_{prefix}llas_exclusive_offs.parquet`` using the same canonical
    summarizer (:func:`_summarize_with_struct_info`) as every other aggregation
    artifact. Existing artifacts are left untouched.

    The exclusive subset is a pure, deterministic column-threshold derivation of
    the committed LLAS event table (via :func:`off_tables.off_filter_mask`, the
    single point of truth for the CLAS thresholds), so no re-detection is needed.
    The LLAS condition durations are reused unchanged: the denominator for
    ``rate``/``total_area_norm`` is the condition's wall-clock duration, identical
    across LLAS/CLAS/BLAS.

    The read LLAS frame is summarized directly (its global ``boxcox_*`` columns,
    fitted on the full LLAS set, are carried through rather than re-fit on the
    subset); the ``mean_boxcox_*`` response variables are unused by the
    cx_homeostasis config, so this is immaterial.

    Parameters
    ----------
    output_dir : pathlib.Path | str
        Directory holding the ``{prefix}llas_offs.parquet`` and
        ``{prefix}condition_durations.parquet`` inputs; the new parquet is
        written here too.
    name_prefix : str
        Filename prefix for the inputs and output. Default ``"full48h_"``.
    """
    output_dir = pathlib.Path(output_dir)

    durs = pd.read_parquet(output_dir / f"{name_prefix}condition_durations.parquet")
    if not isinstance(durs.index, pd.MultiIndex):
        durs = durs.set_index(["subject", "probe", "condition"])
    durs = durs.sort_index()

    offs = pd.read_parquet(output_dir / f"{name_prefix}llas_offs.parquet")
    exclusive = offs.loc[~off_tables.off_filter_mask(offs, "clas")].reset_index(drop=True)
    print(
        f"Computing summarized metrics for {name_prefix}llas_exclusive OFFs "
        f"({len(exclusive)} of {len(offs)} LLAS OFFs)..."
    )
    summarized = _summarize_with_struct_info(exclusive, durs)
    out_path = output_dir / f"summarized_{name_prefix}llas_exclusive_offs.parquet"
    summarized.to_parquet(out_path)


@functools.cache
def _collect_cortical_48h() -> pd.DataFrame:
    """Collect + exclude + cortex-restrict the full-48h condition-subset OFFs.

    The shared, process-memoized base for the in-memory loaders below. Mirrors
    the first three steps of :func:`do_experiment` for the full-48h source
    (:func:`_collect_all_offs_from_full` -> :func:`_apply_additional_exclusions`
    -> cortex filter). Derived fresh from the raw whole-recording detection, so
    it can never drift from a stale on-disk artifact. Requires NFS.
    """
    offs = _collect_all_offs_from_full()
    offs = _apply_additional_exclusions(offs)
    return offs.loc[offs["clade"] == "Cx"].reset_index(drop=True)


@functools.cache
def _collect_cortical_48h_whole() -> pd.DataFrame:
    """Collect cortex-restricted full-48h OFFs WITHOUT condition subsetting.

    The whole-recording counterpart of :func:`_collect_cortical_48h`: the OFFs
    are kept regardless of which (if any) statistical-condition window covers
    them, so the result has no ``condition`` column. ``_apply_additional_exclusions``
    is intentionally skipped (it is condition-scoped). Process-memoized so
    repeated callers (e.g. notebooks) re-read NFS at most once. Requires NFS.
    """
    offs = _collect_all_offs_from_full(condition_subset=False)
    return offs.loc[offs["clade"] == "Cx"].reset_index(drop=True)


def load_subset_of_48h_offs(filter_name: str = "blas") -> pd.DataFrame:
    """Event-level full-48h OFFs subset to the six statistical conditions.

    Derived in memory from the raw whole-recording detection
    (``mua.files.get_full_offs_path``), the in-memory replacement for reading
    ``full48h_{filter}_offs.parquet`` from disk. Each OFF is tagged with the core
    condition covering its ``start_time`` and dropped if it falls in none; the
    result is cortex-only and filtered to the ``filter_name`` LAS category
    (``"llas"`` / ``"clas"`` / ``"blas"``).

    This is a CONDITION-SUBSET view: it is empty outside the six condition
    windows. For whole-recording OFF coverage (e.g. continuous timecourses or
    OFF excision), read the raw per-(probe, structure) ``get_full_offs_path``
    directly instead.

    Memoized via :func:`_collect_cortical_48h`; treat the result as read-only.
    """
    cortical = _collect_cortical_48h()
    return _apply_filters(
        cortical, off_tables.NAMED_FILTERS[filter_name]
    ).reset_index(drop=True)


def summarize_subset_of_48h_offs(
    filter_name: str = "blas", *, grouped_boxcox: bool = False
) -> pd.DataFrame:
    """Summarized per-(subject, probe, structure, condition) metrics for the
    full-48h OFFs, subset to the six statistical conditions.

    The in-memory replacement for reading ``summarized_full48h_{filter}_offs.parquet``.
    Reuses the same Box-Cox and summary helpers as the on-disk writer
    (:func:`_save_filtered_category`), so it matches that file column-for-column.
    See :func:`load_subset_of_48h_offs` for the condition-subset caveat.
    """
    cortical = _collect_cortical_48h()
    durs = _get_condition_durations(_apply_filters(cortical, off_tables.llas_filters))
    offs = _apply_filters(
        cortical, off_tables.NAMED_FILTERS[filter_name]
    ).reset_index(drop=True)
    offs = _boxcox_category(offs, grouped_boxcox=grouped_boxcox)
    return _summarize_with_struct_info(offs, durs)


def summarize_subset_of_48h_offs_by_min_trace_quartile(
    filter_name: str = "blas", *, grouped_boxcox: bool = False
) -> pd.DataFrame:
    """Summarized full-48h metrics grouped by ``min_trace`` quartile.

    The in-memory replacement for reading
    ``summarized_{filter}_offs_by_min_trace_quartile.parquet``. Mirrors the
    by-quartile block of :func:`_save_filtered_category`, reusing the same
    helpers, so it matches that file column-for-column. Subset to the six
    statistical conditions; see :func:`load_subset_of_48h_offs` for the caveat.
    """
    cortical = _collect_cortical_48h()
    durs = _get_condition_durations(_apply_filters(cortical, off_tables.llas_filters))
    offs = _apply_filters(
        cortical, off_tables.NAMED_FILTERS[filter_name]
    ).reset_index(drop=True)
    offs = _boxcox_category(offs, grouped_boxcox=grouped_boxcox)
    offs_q = _assign_min_trace_quartiles(offs)
    quartile_grouping = [
        "subject",
        "probe",
        "structure",
        "condition",
        "min_trace_quartile",
    ]
    return _summarize_with_struct_info(
        offs_q, durs, grouping_cols=quartile_grouping
    )


@functools.cache
def _collect_cortical_48h_untagged() -> pd.DataFrame:
    """Cortex-only full-48h OFFs, NOT tagged with any condition.

    The condition-agnostic counterpart of :func:`_collect_cortical_48h`. Loads the
    whole-recording ``offs.parquet`` for each cortical (subject, probe, structure),
    adds the in-memory postprocessing columns, and keeps cortical OFFs, but
    unlike :func:`_collect_all_offs_from_full`, it does NOT assign each OFF to a
    core condition or drop OFFs that fall outside the six core windows. This lets
    callers tag OFFs with an arbitrary, possibly OVERLAPPING set of condition
    windows (e.g. ``NOD`` / ``NOD.Wake`` / ``Late.NOD``) downstream.

    Missing-data exclusions are deliberately NOT applied here; they key on the
    ``condition`` column, so apply :func:`_apply_additional_exclusions` after
    tagging. Memoized; treat the result as read-only. Requires NFS.
    """
    from cnpix_local_sleep.morphological.pipeline import postprocess_offs as pp

    spsl_cx = sps_conf.get_subject_probe_structure_list(
        method=files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )

    frames = []
    for subject, probe, structure in spsl_cx:
        fpath = files.get_full_offs_path(subject, probe, structure)
        if not fpath.exists():
            continue
        offs = pd.read_parquet(fpath)
        if offs.empty:
            continue
        offs["subject"] = subject
        offs["probe"] = probe
        offs["structure"] = structure
        pp.postprocess_offs_frame(offs, structure)
        frames.append(offs.dropna(axis=1, how="all"))

    if not frames:
        return pd.DataFrame()
    offs = pd.concat(frames, ignore_index=True)
    # Interim OFF-analysis structure consolidation (this collector bypasses
    # _finalize_collected_offs, so apply it here before returning).
    offs = atlas.consolidate_off_structure_columns(offs)
    return offs.loc[offs["clade"] == "Cx"].reset_index(drop=True)


def summarize_48h_offs_for_conditions(
    conditions: list[str],
    filter_name: str = "llas",
    *,
    grouped_boxcox: bool = False,
) -> pd.DataFrame:
    """Summarized per-(subject, probe, structure, condition) metrics for an
    ARBITRARY list of condition windows, from the full-48h morphological source.

    Generalizes :func:`summarize_subset_of_48h_offs` (which is restricted to the
    six disjoint core conditions) to any set of statistical-condition windows,
    including OVERLAPPING ones such as ``NOD`` / ``NOD.Wake`` / ``Late.NOD`` and
    their ``Early.*`` / ``Late.*`` sub-windows. Each OFF is emitted once per
    requested condition whose hypnogram covers its ``start_time``, so a single OFF
    can contribute to several overlapping conditions.

    Reuses the same filter / Box-Cox / duration / summary helpers as the canonical
    pipeline, so for the shared core conditions the numbers match
    :func:`summarize_subset_of_48h_offs` column-for-column. Box-Cox is fitted once
    on the unique filtered OFF set (before the per-condition duplication) so the
    transform distributions are not distorted by OFFs that belong to multiple
    windows.

    Parameters
    ----------
    conditions : list[str]
        Statistical-condition names (keys of
        ``hyp.load_statistical_condition_hypnograms``). Names absent from a
        subject's hypnogram dict are skipped for that subject.
    filter_name : str
        LAS category: ``"llas"`` / ``"clas"`` / ``"blas"``, or the exclusive
        partition ``"llas_exclusive"`` (``llas & ~clas``).
    grouped_boxcox : bool
        If True, also fit per-(subject, probe, structure) Box-Cox lambdas.

    Returns
    -------
    pd.DataFrame
        Summarized metrics with a plain-string ``condition`` column (not the
        six-category core ordering), one row per
        (subject, probe, structure, condition). Empty if no OFF falls in any
        requested window.
    """
    cortical = _collect_cortical_48h_untagged()
    filtered = _filter_cortical_by_category(cortical, filter_name)
    filtered = _boxcox_category(filtered, grouped_boxcox=grouped_boxcox)

    # Tag OFFs by each requested (possibly overlapping) condition window. An OFF
    # covered by N requested windows is emitted N times, once per window. Tagging
    # is per (subject, probe, structure): the hypnograms depend only on
    # (subject, probe), but ``covers_time`` requires monotonically increasing
    # start times, which only holds within a single structure's OFF train.
    tagged_frames = []
    sp_pairs = filtered[["subject", "probe"]].drop_duplicates()
    for subj, prb in sp_pairs.itertuples(index=False):
        hgs = hyp.load_statistical_condition_hypnograms(subj, prb)
        sp_offs = filtered[
            (filtered["subject"] == subj) & (filtered["probe"] == prb)
        ]
        for _, struct_offs in sp_offs.groupby("structure", observed=True):
            struct_offs = struct_offs.sort_values("start_time")
            starts = struct_offs["start_time"].to_numpy()
            for cond in conditions:
                if cond not in hgs:
                    continue
                mask = hgs[cond].covers_time(starts)
                if not mask.any():
                    continue
                cond_offs = struct_offs.loc[mask].copy()
                cond_offs["condition"] = cond
                tagged_frames.append(cond_offs)

    if not tagged_frames:
        return pd.DataFrame()
    long = pd.concat(tagged_frames, ignore_index=True)
    long = _apply_additional_exclusions(long)

    durs = _get_condition_durations(long)
    return _summarize_with_struct_info(long, durs)