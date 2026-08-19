"""Click-based CLI for shared unit-free pipeline commands.

Morphological-specific detection commands (detect-offs, detect-offs-full) are in
``cnpix_local_sleep.morphological.cli`` (entry point: ``morphological-offs``). MUA trace
preprocessing is in ``cnpix.mua`` (entry point: ``cnpix-mua``).
"""

from pathlib import Path

import click

from cnpix_local_sleep import const
from cnpix_local_sleep import files
from cnpix_local_sleep import release_data
from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.morphological.pipeline.logging_utils import PipelineLogger


def resolve_structures(
    subject: str,
    probe: str,
    structure: str | None,
    descendants_of: tuple[str, ...],
) -> list[str]:
    """Resolve structure(s) to process.

    If structure is provided, returns [structure].
    Otherwise, queries database for available structures.
    """
    if structure is not None:
        return [structure]

    import wisc_ecephys_tools as wet

    subject_prb_struct_list = wet.rats.sortings.get_subject_probe_structure_list(
        const.EXPERIMENT,
        "full",
        select_descendants_of=list(descendants_of),
        exclude_descendants_of=["HY"],
    )
    excluded = set(sps_conf.get_excluded_structures(method="morphological"))
    filtered = [dat for dat in subject_prb_struct_list if dat not in excluded]
    available_structures = [
        descr[2] for descr in filtered if descr[0] == subject and descr[1] == probe
    ]

    if not available_structures:
        raise click.UsageError(
            f"No structures available for subject '{subject}' and probe '{probe}'. "
            "Provide a structure explicitly as an argument."
        )

    return available_structures


def run_on_structures(subject, probe, structures, func, **kwargs):
    """Run a function on each structure."""
    for target_structure in structures:
        func(subject, probe, target_structure, **kwargs)


@click.group()
@click.option("--no-log", is_flag=True, help="Disable log file output.")
@click.option(
    "--log-dir",
    type=click.Path(),
    default=".",
    help="Directory for log files (default: current directory).",
)
@click.pass_context
def main(ctx, no_log: bool, log_dir: str):
    """Shared unit-free OFF period pipeline commands.

    For morphological-specific detection commands, use ``morphological-offs``.
    """
    ctx.ensure_object(dict)
    ctx.obj["logger"] = PipelineLogger(
        log_dir=Path(log_dir) if not no_log else None,
        enabled=not no_log,
    )


# -------------------- Postprocessing --------------------


@main.command("postprocess-offs")
@click.argument("subject", required=False)
@click.argument("probe", required=False)
@click.argument("structure", required=False)
@click.option(
    "--descendants-of",
    multiple=True,
    default=["Cx"],
    help="Ancestor structures to include when structure is omitted.",
)
def postprocess_offs(
    subject: str | None,
    probe: str | None,
    structure: str | None,
    descendants_of: tuple[str, ...],
):
    """Postprocess OFF parquet files with derived columns.

    Adds clade, A/P group and normalized features to each offs.parquet file
    in-place.

    Can be run at experiment level (no arguments), or for a specific
    subject/probe/structure.
    """
    from cnpix_local_sleep.morphological.pipeline import postprocess_offs as pp_module

    if subject is None:
        click.echo("Postprocessing OFFs (experiment-wide)")
        pp_module.do_experiment()
    else:
        if probe is None:
            raise click.UsageError("probe is required when subject is given")
        structures = resolve_structures(subject, probe, structure, descendants_of)
        click.echo(f"Postprocessing OFFs for {subject}, {probe}")
        run_on_structures(subject, probe, structures, pp_module.do_structure)


# -------------------- Aggregation --------------------


@main.command("aggregate-offs")
@click.option(
    "--grouped-boxcox",
    is_flag=True,
    help="Apply grouped Box-Cox transformations (by subject, probe, structure).",
)
def aggregate_offs(grouped_boxcox: bool):
    """Aggregate experiment-level OFF metrics."""
    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs

    click.echo("Aggregating experiment-level OFF metrics")
    aggregate_experiment_offs.do_experiment(grouped_boxcox=grouped_boxcox)




@main.command("export-full48h-offs")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option(
    "--grouped-boxcox",
    is_flag=True,
    help="Apply grouped Box-Cox transformations (by subject, probe, structure).",
)
def export_full48h_offs(output_dir: Path | None, grouped_boxcox: bool):
    """Export full-48h morphological OFF parquets for the r-offp ``full48h`` source.

    Re-aggregates the whole-recording state-aware detection (subset to the six
    statistical conditions) into LLAS/CLAS/BLAS ``full48h_*`` parquets and writes
    them as flat files into OUTPUT-DIR (default: ``r-offp/inst/extdata``). Never writes to NFS; this is the deliberate,
    consumer-backed replacement for the retired NFS ``full48h_*`` artifacts.
    """
    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting full-48h OFFs to {output_dir}")
    aggregate_experiment_offs.do_experiment_full(
        output_dir, grouped_boxcox=grouped_boxcox
    )


@main.command("export-bandpower-condition-means")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
def export_bandpower_condition_means(output_dir: Path | None):
    """Export per-condition instantaneous bipolar band-power means for r-offp.

    Summarizes, per cortical ``(subject, probe, structure, condition)``, the mean
    z-scored log10 instantaneous bipolar band power (the same power trace that
    annotates OFF periods), writing ``summarized_full48h_bandpower_offs.parquet``
    into OUTPUT-DIR (default: ``r-offp/inst/extdata``). Consumed by
    the r-offp ``bandpower`` homeostasis pipeline. Requires the per-structure
    band-power zarrs on NFS; never writes to NFS.
    """
    from cnpix_local_sleep.morphological.pipeline import export_bandpower_condition_means as exp

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting per-condition bandpower means to {output_dir}")
    out_path = exp.export_bandpower_condition_means(output_dir, verbose=True)
    click.echo(f"Wrote {out_path}")


@main.command("export-full48h-exclusive-offs")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
def export_full48h_exclusive_offs(output_dir: Path | None):
    """Export the summarized full-48h LLAS-exclusive OFF parquet for r-offp.

    Additive, offline companion to ``export-full48h-offs``: reads the already
    exported ``full48h_llas_offs.parquet`` + ``full48h_condition_durations.parquet``
    in OUTPUT-DIR (default: ``r-offp/inst/extdata``), keeps the OFFs
    admitted by the LLAS filter but rejected by the CLAS filter (the
    ``llas & ~clas`` adjacent-partition complement), and writes
    ``summarized_full48h_llas_exclusive_offs.parquet``. Requires no NFS and leaves
    all existing artifacts untouched. Run ``export-full48h-offs`` first to produce
    the inputs.
    """
    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting full-48h LLAS-exclusive OFFs to {output_dir}")
    aggregate_experiment_offs.export_full48h_exclusive_offs(output_dir)


@main.command("export-adjusted-edge-statistics")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option(
    "--n-boot",
    type=int,
    default=0,
    help="Bootstrap replicates (time-block clustered) for the adjusted means.",
)
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    type=click.Choice(["llas", "clas", "llas_exclusive"]),
    help="OFF classes to export (repeatable; default: all).",
)
@click.option(
    "--size-term",
    type=click.Choice(["shared-curve", "per-combo-factor"]),
    default="shared-curve",
    show_default=True,
    help="How the size dependence is represented (per-combo-factor is the "
    "sensitivity specification, not the reported one).",
)
@click.option(
    "--free-lambda-min-events",
    type=int,
    default=60,
    show_default=True,
    help="Events a (combo, state) block needs before it fits its own curve "
    "amplitude; smaller blocks are given the pooled one. Shared-curve only.",
)
def export_adjusted_edge_statistics(
    output_dir: Path | None,
    n_boot: int,
    datasets: tuple[str, ...],
    size_term: str,
    free_lambda_min_events: int,
):
    """Export size-adjusted onset/offset MAD cell means for r-offp.

    Additive, offline companion to ``export-full48h-offs``: reads the already
    exported ``full48h_llas_offs.parquet`` in OUTPUT-DIR (default: sibling
    ``r-offp/inst/extdata``) and writes
    ``summarized_full48h_<dataset>_edge_adjusted.parquet`` per OFF class.

    Onset/offset MAD depend on how many channels an event spans for reasons that
    are mechanical rather than neural, so a condition effect on the raw statistic
    partly reflects a condition effect on OFF size. Each cell mean is therefore
    re-estimated by marginal standardization (the g-formula): a per-combo model of
    the statistic on condition and event size is used to predict every condition
    over one common population of event sizes. Size enters through a single curve
    estimated once across all cells and applied with a free per-combo amplitude.
    Event duration is deliberately *not* adjusted for; it is a mediator, not a
    measurement confound.

    Requires no NFS and leaves all existing artifacts untouched. Run
    ``export-full48h-offs`` first to produce the input.
    """
    from cnpix_local_sleep.morphological import edge_synchrony_validation

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting size-adjusted edge statistics to {output_dir}")
    written = edge_synchrony_validation.export_adjusted_edge_statistics(
        output_dir,
        datasets=datasets or tuple(edge_synchrony_validation.ADJUSTED_DATASETS),
        n_boot=n_boot,
        size_term=size_term.replace("-", "_"),
        free_lambda_min_events=free_lambda_min_events,
    )
    for dataset, path in written.items():
        click.echo(f"  {dataset}: {path.name}")


@main.command("export-nod-rebound-correlation")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option(
    "--grouped-boxcox",
    is_flag=True,
    help="Apply grouped Box-Cox transformations (by subject, probe, structure).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    type=click.Choice(["llas", "clas", "blas", "llas_exclusive"]),
    help="Categories to export (repeatable; default: all). "
    "e.g. --category llas_exclusive writes only that parquet.",
)
def export_nod_rebound_correlation(
    output_dir: Path | None, grouped_boxcox: bool, categories: tuple[str, ...]
):
    """Export NOD-vs-NREM.Rebound correlation-input parquets for r-offp.

    Additive, fully separable companion to ``export-full48h-offs``. Summarizes
    the full-48h morphological OFFs over the two whole-period predictors ``NOD``
    (Wake+NREM) and ``NOD.Wake`` (wake-only) and the two ``Early.REC.NREM`` /
    ``Early.REC.NREM.Match`` conditions (whose difference is ``NREM.Rebound``)
    for each category, writing
    ``nod_rebound_correlation_{llas,clas,blas,llas_exclusive}_offs.parquet`` as
    flat files into OUTPUT-DIR (default: ``r-offp/inst/extdata``).
    ``llas_exclusive`` is the ``llas & ~clas`` adjacent-partition complement.
    Never writes to NFS. Requires NFS mounted (reads the raw whole-recording
    detection).
    """
    from cnpix_local_sleep.morphological.pipeline import export_correlation_inputs

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting NOD-rebound correlation inputs to {output_dir}")
    export_correlation_inputs.export_nod_rebound_correlation(
        output_dir, categories=categories or None, grouped_boxcox=grouped_boxcox
    )


@main.command("export-locality-offs")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
def export_locality_offs(output_dir: Path | None):
    """Export cross-structure "locality" (Local vs Overlapping) summaries for r-offp.

    Additive, fully separable companion. Reproduces the group-level data behind
    ``notebooks/figures/group_cross_structure_offs.ipynb`` (plots 3 and 4b) plus a
    whole-recording NREM/Wake split, writing three flat parquets
    (``summarized_locality_overlap_offs.parquet``,
    ``summarized_locality_per_condition_llas_offs.parquet``,
    ``summarized_locality_full48h_llas_offs.parquet``) into OUTPUT-DIR (default:
    ``r-offp/inst/extdata``). Never writes to NFS. Requires NFS
    mounted (reads the whole-recording detection).
    """
    from cnpix_local_sleep.morphological.pipeline import cross_structure_locality_export

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting locality (Local vs Overlapping) OFFs to {output_dir}")
    cross_structure_locality_export.export_locality_offs(output_dir)


@main.command("export-excess-globality-offs")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option(
    "--window",
    type=float,
    default=60.0,
    show_default=True,
    help="Local-shift null window in seconds (recommended 30-180).",
)
@click.option(
    "--n-shuffles",
    type=int,
    default=200,
    show_default=True,
    help="Number of windowed-shift shuffles per OFF.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="RNG seed.")
def export_excess_globality_offs(
    output_dir: Path | None, window: float, n_shuffles: int, seed: int
):
    """Export cross-structure "excess globality" (observed vs windowed null) for r-offp.

    Additive, fully separable companion. Ports the per-OFF excess-globality
    assertion (``cross_structure_offs.test_excess_above_chance``; the *Plot 5
    statistics* cell of ``notebooks/figures/group_cross_structure_offs.ipynb``)
    into a single flat parquet (``summarized_excess_globality_offs.parquet``)
    written into OUTPUT-DIR (default: ``r-offp/inst/extdata``). One
    row per ``(subject, structure, quantity)`` with ``quantity in
    {observed, null}``. Never writes to NFS. Requires NFS mounted (reads the
    whole-recording detection and scores each NREM OFF against the windowed null,
    so this is slow).
    """
    from cnpix_local_sleep.morphological.pipeline import cross_structure_excess_export

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting excess-globality (observed vs windowed null) OFFs to {output_dir}")
    cross_structure_excess_export.export_excess_globality_offs(
        output_dir, window=window, n_shuffles=n_shuffles, seed=seed
    )


@main.command("export-size-globality-correlations")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option("--window", type=float, default=60.0, show_default=True,
              help="Local-shift null window in seconds.")
@click.option("--n-shuffles", type=int, default=200, show_default=True,
              help="Number of windowed-shift shuffles per OFF.")
@click.option("--seed", type=int, default=42, show_default=True, help="RNG seed.")
def export_size_globality_correlations(output_dir, window, n_shuffles, seed):
    """Export the OFF-size vs globality Spearman correlations (manuscript Fig. 4d-f;
    Table S3b): per-(subject, structure) Spearman of OFF duration/span/area vs the
    raw overlap degree and vs the excess globality (observed - windowed-null),
    pooled by random-effects meta-analysis. Writes
    ``manuscript_size_globality_correlations.csv`` into OUTPUT-DIR (default: the
    ``r-offp/inst/extdata``). Requires NFS (re-runs the windowed null; slow).
    """
    from cnpix_local_sleep.morphological.pipeline import cross_structure_excess_export

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting size-vs-globality correlations to {output_dir}")
    cross_structure_excess_export.export_size_globality_correlations(
        output_dir, window=window, n_shuffles=n_shuffles, seed=seed
    )


@main.command("export-depth-profile-summary")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option(
    "--n-perm",
    type=int,
    default=200,
    show_default=True,
    help="Permutations for the occupancy null test.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="RNG seed.")
def export_depth_profile_summary(output_dir: Path | None, n_perm: int, seed: int):
    """Export the laminar-trimodality mechanical-null per-structure summary for r-offp.

    Additive, fully separable companion. Distils
    ``notebooks/figures/laminar_trimodality_null.ipynb`` to one tidy row per
    ``(subject, probe, structure)`` laminar combo: the Wasserstein skill-score
    attributions of ``supra_concentration`` and COM to the mechanical null (both
    the in-detection no-clip ``feasible`` null and the whole-structure ``uniform``
    null) plus the time/count occupancy effect sizes vs the depth
    null, writing ``summarized_depth_profile.parquet`` into OUTPUT-DIR (default:
    ``r-offp/inst/extdata``). Feeds the r-offp laminar-null group
    analysis (``scripts/depth_profile_summary.R``). Never writes to NFS. Requires
    NFS mounted (reads the full-48h ``morphological`` detection; slow).
    """
    from cnpix_local_sleep.morphological.pipeline import depth_profile_export

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting laminar-null per-structure summary to {output_dir}")
    depth_profile_export.export_depth_profile_summary(
        output_dir, seed=seed, n_perm_occ=n_perm
    )


@main.command("export-manuscript-correlations")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (default: r-offp/inst/extdata).",
)
@click.option("--skip-per-event", is_flag=True, help="Skip the per-event correlation export.")
@click.option("--skip-epoched", is_flag=True, help="Skip the epoched-partial export.")
def export_manuscript_correlations(output_dir, skip_per_event, skip_epoched):
    """Export the two correlation analyses (per-event + epoched partial) for the
    manuscript statistical tables (r-offp Table S2a/S2b).

    Additive, fully separable companion. Recomputes the ``has_value.ipynb`` pooled
    Spearman correlations from the cached OFF+bandpower parquet and tidies the
    ``incremental_added_value{,_wake}.ipynb`` 2-tier "area" pooled parquets, writing
    ``manuscript_per_event_correlations.csv`` and
    ``manuscript_epoched_partial_correlations.csv`` into OUTPUT-DIR (default: the
    ``r-offp/inst/extdata``). Reads the committed local caches only; does
    NOT require NFS.
    """
    from cnpix_local_sleep.morphological.pipeline import export_manuscript_correlations as exp

    if output_dir is None:
        output_dir = files.get_r_offp_extdata_dir()
    click.echo(f"Exporting manuscript correlation summaries to {output_dir}")
    exp.export_manuscript_correlations(
        output_dir,
        do_per_event=not skip_per_event,
        do_epoched=not skip_epoched,
    )


# -------------------- Data publication --------------------


@main.command("publish-release-data")
@click.option(
    "--extdata-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory to publish (default: r-offp/inst/extdata).",
)
@click.option(
    "--repo",
    default=release_data.REPO,
    show_default=True,
    help="Repository whose Release hosts the data.",
)
@click.option(
    "--tag",
    default=release_data.TAG,
    show_default=True,
    help="Release tag. Rolling; pin it to a fixed version at acceptance.",
)
@click.option(
    "--event-level-only",
    is_flag=True,
    help="Upload only the three tables that cannot be committed.",
)
@click.option("--dry-run", is_flag=True, help="List what would be uploaded and stop.")
def publish_release_data(
    extdata_dir: Path | None,
    repo: str,
    tag: str,
    event_level_only: bool,
    dry_run: bool,
):
    """Attach the exported OFF tables to a GitHub Release.

    Reads whatever the exporters already wrote; computes nothing. The three
    event-level ``full48h_*_offs.parquet`` tables (347 MB) have to be hosted here
    because one exceeds GitHub's 100 MB per-file limit. The 24 summarized tables are
    committed and need no download, but are uploaded too by default so that one asset
    set is a complete copy of the data.

    Creates the release if it does not exist and replaces assets in place
    (``--clobber``), which is what invalidates a stale download cache.
    """
    if extdata_dir is None:
        extdata_dir = files.get_r_offp_extdata_dir()
    if dry_run:
        names = (
            list(release_data.EVENT_LEVEL_ASSETS)
            if event_level_only
            else sorted(p.name for p in extdata_dir.iterdir() if p.is_file())
        )
        click.echo(f"Would upload {len(names)} assets to {repo} @ {tag}:")
        for name in names:
            size = (extdata_dir / name).stat().st_size / 1e6
            click.echo(f"  {name}  ({size:.2f} MB)")
        return
    click.echo(f"Publishing {extdata_dir} to {repo} @ {tag}")
    uploaded = release_data.publish(
        extdata_dir, repo=repo, tag=tag, event_level_only=event_level_only
    )
    click.echo(f"Uploaded {len(uploaded)} assets.")


# -------------------- Bandpowers --------------------


@main.command("aggregate-bandpowers")
@click.option("--bipolar", is_flag=True, help="Use bipolar rereferencing.")
@click.option(
    "--kind",
    type=click.Choice(["stft", "inst"]),
    required=True,
    help="Bandpower computation method.",
)
def aggregate_bandpowers(bipolar: bool, kind: str):
    """Aggregate bandpower data across subjects/probes/structures."""
    from cnpix_local_sleep.morphological.pipeline import aggregate_bandpowers as agg_bp

    bipolar_str = " (bipolar)" if bipolar else ""
    click.echo(f"Aggregating {kind} bandpowers{bipolar_str}")
    agg_bp.do_project(bipolar=bipolar, kind=kind)


@main.command("extract-inst-bandpowers")
@click.argument("subject")
@click.argument("probe")
@click.argument("band_name")
@click.argument("structure", required=False)
@click.option("--plot-ipower", is_flag=True, help="Plot instantaneous power.")
@click.option(
    "--descendants-of",
    multiple=True,
    default=["Cx"],
    help="Ancestor structures to include when structure is omitted.",
)
def extract_inst_bandpowers(
    subject: str,
    probe: str,
    band_name: str,
    structure: str | None,
    plot_ipower: bool,
    descendants_of: tuple[str, ...],
):
    """Extract instantaneous bandpower from LFP data."""
    from cnpix_local_sleep.morphological.pipeline import (
        extract_inst_bandpowers as extract_inst_bp,
    )

    structures = resolve_structures(subject, probe, structure, descendants_of)
    click.echo(f"Extracting instantaneous {band_name} bandpower")
    run_on_structures(
        subject,
        probe,
        structures,
        extract_inst_bp.do_structure,
        band_name=band_name,
        plot_ipower=plot_ipower,
    )


# -------------------- Experiment-level plots --------------------


@main.command("plot-offs-vs-time")
def plot_offs_vs_time():
    """Plot OFF period rates over time with power overlays."""
    from cnpix_local_sleep.morphological import mua
    from cnpix_local_sleep.morphological.pipeline import plot_offs_vs_time as povt

    click.echo("Plotting OFFs vs time (morphological)")
    povt.do_project(mua.SOURCE_CONFIG)


# -------------------- Structure-level plots --------------------


# -------------------- Cross-structure analysis --------------------


@main.command("cross-structure-offs")
@click.argument("subject", required=False)
@click.option(
    "--off-source",
    type=click.Choice(["morphological-full48h", "morphological"]),
    default="morphological-full48h",
    show_default=True,
    help="OFF source to analyze.",
)
@click.option(
    "--n-shuffles", default=200, type=int, help="Number of jitter shuffles."
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@click.option(
    "--no-jitter",
    is_flag=True,
    help="Skip jitter null computation and plotting.",
)
@click.option(
    "--no-legend", is_flag=True, help="Suppress legends on all plots."
)
def cross_structure_offs(
    subject: str | None,
    off_source: str,
    n_shuffles: int,
    overwrite: bool,
    no_jitter: bool,
    no_legend: bool,
):
    """Analyze cross-structure OFF period relationships.

    If SUBJECT is provided, runs for that subject only. Otherwise, runs
    for all subjects with OFFs in more than one cortical region.
    """
    from cnpix_local_sleep.morphological.pipeline import cross_structure_offs as module

    if subject is None:
        click.echo(
            "Running cross-structure OFF analysis for all qualifying subjects "
            f"(off_source={off_source})"
        )
        module.do_experiment(
            off_source=off_source,
            n_shuffles=n_shuffles,
            overwrite=overwrite,
            no_jitter=no_jitter,
            no_legend=no_legend,
        )
    else:
        click.echo(
            f"Running cross-structure OFF analysis for {subject} "
            f"(off_source={off_source})"
        )
        module.do_subject(
            subject,
            off_source=off_source,
            n_shuffles=n_shuffles,
            overwrite=overwrite,
            no_jitter=no_jitter,
            no_legend=no_legend,
        )


if __name__ == "__main__":
    main()