"""Click-based CLI for the morphological OFF detection pipeline."""

from pathlib import Path

import click
import yaml

from cnpix_local_sleep.morphological.analysis_cli import resolve_structures, run_on_structures
from cnpix_local_sleep.morphological.pipeline.logging_utils import PipelineLogger


def load_opts(options_path: str | None) -> dict | None:
    """Load options from YAML file."""
    if options_path is None:
        return None
    with open(options_path) as f:
        return yaml.safe_load(f)


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
    """Morphological (quantile-threshold) OFF period detection pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["logger"] = PipelineLogger(
        log_dir=Path(log_dir) if not no_log else None,
        enabled=not no_log,
    )


# -------------------- Preprocessing --------------------


# -------------------- Detection --------------------


@main.command("detect-offs")
@click.argument("subject")
@click.argument("probe")
@click.argument("structure", required=False)
@click.option(
    "--options-path",
    type=click.Path(exists=True),
    required=True,
    help="YAML config file (required).",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite existing outputs."
)
@click.option(
    "--descendants-of",
    multiple=True,
    default=["Cx"],
    help="Ancestor structures to include when structure is omitted.",
)
def detect_offs(
    subject: str,
    probe: str,
    structure: str | None,
    options_path: str,
    overwrite: bool,
    descendants_of: tuple[str, ...],
):
    """Detect morphological OFF periods and write results to disk.

    This command performs threshold computation and OFF detection in a
    single step, loading data once and performing both operations before
    writing results. Results are written under ``method=morphological``.
    """
    from cnpix_local_sleep.morphological import detect as detect_module
    from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG
    from cnpix_local_sleep.morphological.types import validate_detection_opts

    opts = load_opts(options_path)
    validate_detection_opts(opts)

    structures = resolve_structures(
        subject, probe, structure, descendants_of
    )
    click.echo("Detecting OFF events (variant=morphological)")
    run_on_structures(
        subject,
        probe,
        structures,
        detect_module.do_structure,
        opts=opts,
        overwrite=overwrite,
        source_config=SOURCE_CONFIG,
    )


@main.command("detect-offs-full")
@click.argument("subject")
@click.argument("probe")
@click.argument("structure", required=False)
@click.option(
    "--options-path",
    type=click.Path(exists=True),
    required=True,
    help="YAML config file (required).",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite existing outputs."
)
@click.option(
    "--descendants-of",
    multiple=True,
    default=["Cx"],
    help="Ancestor structures to include when structure is omitted.",
)
def detect_offs_full(
    subject: str,
    probe: str,
    structure: str | None,
    options_path: str,
    overwrite: bool,
    descendants_of: tuple[str, ...],
):
    """Detect OFF periods across the full recording.

    Computes absolute NREM and Wake thresholds from all data in the
    respective states, then runs detection on the entire trace with
    state-dependent thresholds. Results are saved without a condition
    axis.
    """
    from cnpix_local_sleep.morphological import detect_full as detect_full_module
    from cnpix_local_sleep.morphological.types import validate_detection_opts

    opts = load_opts(options_path)
    validate_detection_opts(opts)

    structures = resolve_structures(
        subject, probe, structure, descendants_of
    )
    click.echo("Detecting OFF events (full recording)")
    run_on_structures(
        subject,
        probe,
        structures,
        detect_full_module.do_structure,
        opts=opts,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()