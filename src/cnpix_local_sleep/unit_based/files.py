"""File path functions for unit-based OFF detection (``method=unit_based``).

All paths use :func:`cnpix_local_sleep.files.get_path` with ``method="unit_based"``,
producing paths like::

    {project}/{experiment}/{subject}/method=unit_based/probe={probe}/
        structure={structure}/detection_mode=pooled-{algo}/...

The detection algorithm (``threshold``/``hmmem``/``sticky``) is encoded in the
``detection_mode`` path segment as ``pooled-{algo}`` so that runs of different
algorithms are kept separate on disk. Detection is structure-level *pooled*
(spike trains of all units in a structure are merged), so there is no spatial
(channel x time) component, unlike ``morphological``.

Modeled on :mod:`cnpix_local_sleep.morphological.mua.files`.
"""

from __future__ import annotations

import pathlib

from cnpix_local_sleep import const, files

METHOD = "unit_based"
_METHOD = METHOD


def get_path(
    filename: str,
    pathspec: dict | None = None,
    *,
    subject: str | None = None,
    experiment: str | None = const.EXPERIMENT,
    project: str = "offproj",
    enforce_in_schema: bool = True,
    enforce_schema_order: bool = True,
    **kwargs,
) -> pathlib.Path:
    """Construct file paths with automatic ``method=unit_based`` injection.

    Wrapper around :func:`cnpix_local_sleep.files.get_path` that organizes all unit-based
    files under a ``method=unit_based`` directory. ``method`` must not be passed
    by the caller; the wrapper injects it.
    """
    merged_pathspec = {"method": _METHOD}

    if pathspec is not None:
        for key, value in pathspec.items():
            if key == "method":
                raise ValueError(
                    f"method is injected by unit_based.files.get_path(); "
                    f"do not pass method={value!r}"
                )
            merged_pathspec[key] = value

    for key, value in kwargs.items():
        if key == "method":
            raise ValueError(
                f"method is injected by unit_based.files.get_path(); "
                f"do not pass method={value!r}"
            )
        merged_pathspec[key] = value

    return files.get_path(
        filename,
        pathspec=merged_pathspec,
        subject=subject,
        experiment=experiment,
        project=project,
        enforce_in_schema=enforce_in_schema,
        enforce_schema_order=enforce_schema_order,
    )


def detection_mode(algo: str) -> str:
    """Return the ``detection_mode`` path segment for a pooled algorithm."""
    return f"pooled-{algo}"


def banded_detection_mode(algo: str) -> str:
    """Return the ``detection_mode`` segment for banded (spatially-resolved) detection.

    Banded detection runs the algorithm within depth *bands* and merges OFFs across
    them, so its OFFs carry a real depth x time footprint -- distinct from the pooled
    mode (``pooled-{algo}``) and from ``morphological`` spatial OFFs. See
    :mod:`cnpix_local_sleep.unit_based.banded` (provisional, under validation).
    """
    return f"banded-{algo}"


def get_full_banded_offs_path(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pathlib.Path:
    """Path for the condition-agnostic full-recording banded OFF dataframe."""
    return get_path(
        "offs.parquet",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode=banded_detection_mode(algo),
    )


def get_full_banded_detection_info_path(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pathlib.Path:
    """Path for per-pass banded detection metadata (bands, fitted params, FR)."""
    return get_path(
        "detection_info.pickle",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode=banded_detection_mode(algo),
    )


# Detection outputs
#
# Detection runs condition-agnostically over each macro-state (NREM, NOD-Wake)
# and writes a single full-recording OFF dataframe. Per-condition OFFs are
# obtained downstream by tagging each OFF with the statistical condition
# hypnogram that covers its start_time, mirroring the morphological full-48h
# aggregation path.


def get_full_offs_path(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pathlib.Path:
    """Path for the condition-agnostic full-recording OFF dataframe."""
    return get_path(
        "offs.parquet",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode=detection_mode(algo),
    )


def get_full_detection_info_path(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pathlib.Path:
    """Path for per-pass detection metadata (fitted params, FR, algo used)."""
    return get_path(
        "detection_info.pickle",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode=detection_mode(algo),
    )


# Interactive tuner caches (algorithm-agnostic; regenerable)


def get_interactive_raster_cache_path(
    subject: str,
    probe: str,
    structure: str,
) -> pathlib.Path:
    """Path for the interactive OFF-tuner raster cache (pooled spike trains).

    Algorithm-agnostic (no ``detection_mode`` segment): the cached per-unit spike
    trains and pooled 10 ms counts are reused across all detection algorithms.
    A regenerable cache, not a durable data product.
    """
    return get_path(
        "interactive_unit_raster.npz",
        subject=subject,
        probe=probe,
        structure=structure,
    )


def get_tuned_params_path(
    subject: str,
    probe: str,
    structure: str,
    algo: str,
) -> pathlib.Path:
    """Path for a tuned-parameters JSON sidecar saved from the interactive tuner."""
    return get_path(
        "tuned_params.json",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode=detection_mode(algo),
    )


# -------------------- Experiment-level outputs --------------------


def get_experiment_dir(algo: str, *, banded: bool = False) -> pathlib.Path:
    """Directory for experiment-level (cross-subject) unit-based outputs.

    Sits above the per-subject tree, at the ``detection_mode`` level. Validation
    readouts (e.g. ``banded_validation/``) are written into subdirectories here.
    """
    mode = banded_detection_mode(algo) if banded else detection_mode(algo)
    return get_path("", detection_mode=mode, subject=None)
