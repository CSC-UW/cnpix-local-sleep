"""Click-based CLI for unit-based (pooled spike-train) OFF detection."""

import click

from cnpix_local_sleep.unit_based import const as ub_const


@click.group()
def main():
    """Unit-based (pooled spike-train) OFF period detection pipeline."""
    pass


@main.command("detect")
@click.argument("subject")
@click.argument("probe")
@click.argument("structure")
@click.option(
    "--algo",
    type=click.Choice(ub_const.ALGOS),
    default=ub_const.DEFAULT_ALGO,
    show_default=True,
    help="Detection algorithm (pooled spike train).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing OFFs.")
def detect(subject, probe, structure, algo, overwrite):
    """Detect pooled OFFs for one (subject, probe, structure)."""
    from cnpix_local_sleep.unit_based.pipeline import detect_full

    detect_full.do_structure(
        subject, probe, structure, algo=algo, overwrite=overwrite
    )


@main.command("detect-experiment")
@click.option(
    "--algo",
    type=click.Choice(ub_const.ALGOS),
    default=ub_const.DEFAULT_ALGO,
    show_default=True,
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing OFFs.")
def detect_experiment(algo, overwrite):
    """Detect pooled OFFs for every included (subject, probe, structure)."""
    from cnpix_local_sleep.unit_based.pipeline import detect_full

    detect_full.do_experiment(algo=algo, overwrite=overwrite)


@main.command("detect-banded")
@click.argument("subject")
@click.argument("probe")
@click.argument("structure")
@click.option(
    "--algo",
    type=click.Choice(ub_const.ALGOS),
    default=ub_const.DEFAULT_ALGO,
    show_default=True,
    help="Detection algorithm (run within each depth band).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing banded OFFs.")
def detect_banded(subject, probe, structure, algo, overwrite):
    """Detect spatially-resolved (banded) OFFs for one (subject, probe, structure)."""
    from cnpix_local_sleep.unit_based import banded

    banded.do_structure_banded(
        subject, probe, structure, algo=algo, overwrite=overwrite
    )


@main.command("detect-banded-experiment")
@click.option(
    "--algo",
    type=click.Choice(ub_const.ALGOS),
    default=ub_const.DEFAULT_ALGO,
    show_default=True,
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Structures to detect concurrently (each a spawn process). Auto-clamped by "
        "RAM (~22 GB/structure) and cores, leaving headroom on the shared host. "
        "1 = sequential; try 16-48 on a big host."
    ),
)
@click.option(
    "--threads-per-job",
    type=int,
    default=1,
    show_default=True,
    help="BLAS/OpenMP threads per worker (1 is best; sticky compute is single-threaded).",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing banded OFFs.")
def detect_banded_experiment(algo, jobs, threads_per_job, overwrite):
    """Detect banded OFFs for every included (subject, probe, structure)."""
    from cnpix_local_sleep.unit_based import banded

    banded.do_experiment_banded(
        algo=algo, overwrite=overwrite, n_jobs=jobs, threads_per_job=threads_per_job
    )


if __name__ == "__main__":
    main()
