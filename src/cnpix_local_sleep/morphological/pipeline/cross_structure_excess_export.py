"""Export cross-structure OFF "excess globality" summaries for r-offp.

An additive, fully separable companion to :mod:`cross_structure_offs`,
mirroring :mod:`cross_structure_locality_export`. It ports the per-OFF
"excess globality" assertion from
``cross_structure_offs.test_excess_above_chance`` (the subject-level paired test
+ intercept-only mixed model behind the *Plot 5 statistics* cell of
``notebooks/figures/group_cross_structure_offs.ipynb``) into a single flat parquet
the r-offp ``excess_globality`` pipeline consumes.

For each multi-cortical subject we score every whole-recording NREM OFF against
the windowed local-shift null (:func:`cross_structure_offs.do_subject_excess_globality`,
``null_scope="whole_recording"``), which yields a per-OFF ``observed_degree`` and
a duration-matched ``null_mean``. We then aggregate to one mean degree per
``(subject, structure)`` (the same unit of analysis the Python mixed model used,
which avoids per-OFF pseudoreplication), and emit a long frame with a
two-level ``quantity`` factor:

- ``quantity == "observed"`` -> mean observed overlap degree of that cell.
- ``quantity == "null"``     -> mean windowed-null overlap degree of that cell.

In r-offp the LRT main effect of ``quantity`` (``value ~ quantity + (1|subject) +
(1|subject:structure)`` vs the intercept-only null) is the "observed >
chance" test, structurally identical to the locality ``overlap_status`` paired
model. ``null`` is laid out as the factor reference so an ``"observed - null"``
contrast reads as the excess.

One parquet:

``summarized_excess_globality_offs.parquet``: one row per
``(subject, structure, quantity)`` with ``value`` (mean degree), ``count``
(scored OFFs in the cell), ``clade``, ``condition == "NREM"`` (so the R runner's
``conditions=`` filter works unchanged), and provenance columns ``null_scope`` /
``window`` / ``n_shuffles``.

Writes only into r-offp, never NFS. Requires NFS mounted: it reads the
whole-recording detection through :mod:`cross_structure_offs`. Delete this module
and the ``export-excess-globality-offs`` CLI command to remove the Python side
entirely; nothing else imports it.
"""

import pathlib

import pandas as pd

from cnpix_local_sleep import atlas
from cnpix_local_sleep.morphological.pipeline import cross_structure_offs as cso

# Long-format quantity levels (``null`` is the r-offp factor reference).
_QUANTITY_OBSERVED = "observed"
_QUANTITY_NULL = "null"


def _aggregate_long(per_off: pd.DataFrame) -> pd.DataFrame:
    """Per-OFF excess frame -> long ``(subject, structure, quantity)`` summary.

    Pure helper (no I/O) so the observed/null aggregation is unit-testable. Takes
    the concatenated per-OFF output of
    :func:`cross_structure_offs.do_subject_excess_globality` (columns
    ``subject``, ``structure``, ``observed_degree``, ``null_mean``) and returns a
    long frame with two rows per ``(subject, structure)`` cell.
    """
    grouped = (
        per_off.groupby(["subject", "structure"], observed=True)
        .agg(
            observed=("observed_degree", "mean"),
            null=("null_mean", "mean"),
            count=("observed_degree", "size"),
        )
        .reset_index()
    )
    long = grouped.melt(
        id_vars=["subject", "structure", "count"],
        value_vars=[_QUANTITY_OBSERVED, _QUANTITY_NULL],
        var_name="quantity",
        value_name="value",
    )
    long["clade"] = "Cx"
    long["condition"] = "NREM"
    return long


def _build_per_off(
    off_source: str = cso.DEFAULT_OFF_SOURCE,
    *,
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Concatenated per-OFF frame (``observed_degree``, ``null_mean``, and OFF size
    columns) across every multi-cortical subject, scored against the windowed
    local-shift null. Slow (re-runs the null); requires NFS."""
    subjects = cso.get_multi_cortical_subjects(off_source)
    per_off_frames = []
    for subject in subjects:
        df = cso.do_subject_excess_globality(
            subject,
            off_source=off_source,
            null_scope="whole_recording",
            window=window,
            n_shuffles=n_shuffles,
            seed=seed,
        )
        if not df.empty:
            per_off_frames.append(df)
    if not per_off_frames:
        return pd.DataFrame()
    return pd.concat(per_off_frames, ignore_index=True)


def summarize_excess_globality(
    off_source: str = cso.DEFAULT_OFF_SOURCE,
    *,
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Long ``(subject, structure, quantity)`` excess-globality summary.

    Scores every multi-cortical subject's whole-recording NREM OFFs against the
    windowed local-shift null and aggregates to one mean degree per cell. Requires
    ``off_source="morphological-full48h"`` (the regime
    :func:`cross_structure_offs.do_subject_excess_globality` is defined for).
    """
    per_off = _build_per_off(off_source, window=window, n_shuffles=n_shuffles, seed=seed)
    if per_off.empty:
        return pd.DataFrame()
    long = _aggregate_long(per_off)
    long["null_scope"] = "whole_recording"
    long["window"] = window
    long["n_shuffles"] = n_shuffles
    return long


def summarize_size_globality_correlations(
    off_source: str = cso.DEFAULT_OFF_SOURCE,
    *,
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
    min_offs: int = 10,
    per_off: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Manuscript Fig. 4d-f: per-(subject, structure) Spearman of each OFF size
    property vs the RAW overlap degree and vs the EXCESS globality
    (``observed_degree - null_mean``), pooled across combinations by the same
    random-effects (DerSimonian-Laird) meta-analysis used for the per-event
    correlations. A raw rho that collapses toward 0 for the excess target means the
    size-globality link is largely mechanical; a surviving positive excess rho is
    genuine coordination. Ports ``group_cross_structure_offs.ipynb`` cell 5a.
    """
    from cnpix_local_sleep.morphological import correlation_stats

    if per_off is None:
        per_off = _build_per_off(off_source, window=window, n_shuffles=n_shuffles, seed=seed)
    if per_off.empty:
        return pd.DataFrame()
    per_off = per_off.copy()
    per_off["excess_globality"] = per_off["observed_degree"] - per_off["null_mean"]
    props = [c for c in ["duration", "span", "area"] if c in per_off.columns]
    targets = [("observed_degree", "raw degree"), ("excess_globality", "excess")]
    rows = []
    for prop in props:
        for target_col, target_label in targets:
            gc = correlation_stats.compute_group_correlations(
                per_off, prop, target_col,
                group_cols=["subject", "structure"], method="spearman",
            )
            if gc is None or len(gc) == 0:
                continue
            gc = gc[gc["n"] >= min_offs]
            if len(gc) == 0:
                continue
            meta = correlation_stats.meta_analyze_correlations(gc)
            rows.append({
                "property": prop, "target": target_label,
                "pooled_rho": meta["overall_rho"], "ci_lo": meta["ci_lo"],
                "ci_hi": meta["ci_hi"], "p_value": meta["p_value"],
                "i_squared": meta["i_squared"], "q_pvalue": meta["q_pvalue"],
                "k": meta["k"],
            })
    return pd.DataFrame(rows)


def export_size_globality_correlations(
    output_dir: pathlib.Path | str,
    *,
    off_source: str = cso.DEFAULT_OFF_SOURCE,
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
) -> pathlib.Path:
    """Write ``manuscript_size_globality_correlations.csv`` (Fig. 4d-f) into
    ``output_dir``. Slow (re-runs the windowed null); requires NFS."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Summarizing size vs globality Spearman correlations (raw + excess)...")
    df = summarize_size_globality_correlations(
        off_source, window=window, n_shuffles=n_shuffles, seed=seed
    )
    out = output_dir / "manuscript_size_globality_correlations.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {len(df)} rows -> {out}")
    return out


def export_excess_globality_offs(
    output_dir: pathlib.Path | str,
    *,
    off_source: str = cso.DEFAULT_OFF_SOURCE,
    window: float = 60.0,
    n_shuffles: int = 200,
    seed: int = 42,
) -> None:
    """Write ``summarized_excess_globality_offs.parquet`` into ``output_dir``."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Summarizing cross-structure excess globality (observed vs windowed null)...")
    summary = summarize_excess_globality(
        off_source, window=window, n_shuffles=n_shuffles, seed=seed
    )
    # Interim OFF-analysis structure consolidation (e.g. mPPC -> PPC).
    summary = atlas.consolidate_off_structure_columns(summary)
    summary.to_parquet(output_dir / "summarized_excess_globality_offs.parquet")
    n_cells = len(summary) // 2 if len(summary) else 0
    print(f"  {n_cells} (subject, structure) cells x 2 quantities = {len(summary)} rows")
