"""File path functions for morphological OFF detection (``method=morphological`` on disk).

All paths use ``cnpix_local_sleep.files.get_path()`` with
``method="morphological"``, producing paths like::

    {project}/{experiment}/{subject}/method=morphological/probe={probe}/...

Modeled on harding/files.py.
"""

from __future__ import annotations

import pathlib

from cnpix_local_sleep import const, files


METHOD = "morphological"
_METHOD = METHOD  # back-compat alias for existing internal references


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
    """Construct file paths with automatic method=morphological injection.

    Wrapper around ``cnpix_local_sleep.files.get_path()`` that organizes all
    outputs under a ``method=morphological`` directory. ``method``
    must not be passed by the caller; the wrapper injects it.
    """
    merged_pathspec = {"method": _METHOD}

    if pathspec is not None:
        for key, value in pathspec.items():
            if key == "method":
                raise ValueError(
                    f"method is injected by morphological.mua.files.get_path(); "
                    f"do not pass method={value!r}"
                )
            merged_pathspec[key] = value

    for key, value in kwargs.items():
        if key == "method":
            raise ValueError(
                f"method is injected by morphological.mua.files.get_path(); "
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


# Border files (shared project, cross-method anatomy)


def get_restricted_structure_borders_path(
    subject: str, probe: str
) -> pathlib.Path:
    """Get path for OFF-detection structure borders.

    Structure borders are cross-method anatomy, not per-method
    detection outputs. All methods resolve to the same file. Delegates
    to :func:`cnpix_local_sleep.files.get_off_detection_structure_borders_path`.
    """
    return files.get_off_detection_structure_borders_path(subject, probe)


# -------------------- Border files (cnpix_local_sleep project) --------------------


# -------------------- Channel masks --------------------


# Detection outputs (per-condition spatial detection)


def get_channel_thresholds_path(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    threshold_group: str | None,
) -> pathlib.Path:
    """Get path for channel thresholds file."""
    return get_path(
        "channel_thresholds.zarr",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode="spatial",
        threshold_group=threshold_group,
        condition=condition,
    )


def get_off_label_indices_path(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    threshold_group: str | None,
) -> pathlib.Path:
    """Get path for OFF label indices file."""
    return get_path(
        "off_label_indices.parquet",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode="spatial",
        threshold_group=threshold_group,
        condition=condition,
    )


def get_offs_path(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    threshold_group: str | None,
) -> pathlib.Path:
    """Get path for OFF periods dataframe."""
    return get_path(
        "offs.parquet",
        subject=subject,
        probe=probe,
        structure=structure,
        detection_mode="spatial",
        threshold_group=threshold_group,
        condition=condition,
    )


# -------------------- Full-recording detection outputs --------------------


def get_full_offs_path(
    subject: str,
    probe: str,
    structure: str,
) -> pathlib.Path:
    """Get path for full-recording OFF periods dataframe."""
    return get_path(
        "offs.parquet",
        subject=subject,
        probe=probe,
        structure=structure,
    )


def get_full_off_label_indices_path(
    subject: str,
    probe: str,
    structure: str,
) -> pathlib.Path:
    """Get path for full-recording OFF label indices file."""
    return get_path(
        "off_label_indices.parquet",
        subject=subject,
        probe=probe,
        structure=structure,
    )


def get_full_channel_thresholds_path(
    subject: str,
    probe: str,
    structure: str,
    threshold_type: str,
) -> pathlib.Path:
    """Get path for full-recording channel thresholds file.

    Args:
        subject: Subject identifier.
        probe: Probe identifier.
        structure: Brain structure name.
        threshold_type: Which threshold: ``"nrem"``, ``"wake"``,
            or a core condition name (for reference thresholds).
    """
    return get_path(
        "channel_thresholds.zarr",
        subject=subject,
        probe=probe,
        structure=structure,
        condition=threshold_type,
    )


def get_interactive_thresholds_path(
    subject: str,
    probe: str,
    structure: str,
) -> pathlib.Path:
    """Get path for interactive morphological threshold cache."""
    return get_path(
        "interactive_thresholds.npz",
        subject=subject,
        probe=probe,
        structure=structure,
    )
