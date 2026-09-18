"""File paths for OFF period image stacks.

The two reader getters refuse a stack directory carrying a ``STALE.json``
marker (see :func:`cnpix.evaluation.paths.check_not_stale`) unless
``allow_stale=True``; the savedir getter does not, so writers can create new
directories freely.
"""

import pathlib

from cnpix.evaluation import paths as label_paths

from cnpix_local_sleep import files

# Image-stack paths in the samoffs project (no method or detection_mode component)


def get_sam3_savedir_path(
    subject: str,
    probe: str,
    condition: str,
    structure_acronym: str | None = None,
) -> pathlib.Path:
    """Get the save directory path for SAM3 image stacks."""
    return files.get_path(
        "",
        subject=subject,
        project=label_paths.MODEL_LABELS_PROJECT,
        probe=probe,
        structure=structure_acronym,
        condition=condition,
    )


def get_sam3_off_stacks_ome_zarr_path(
    subject: str,
    probe: str,
    condition: str,
    structure: str | None = None,
    *,
    allow_stale: bool = False,
) -> pathlib.Path:
    """Get path for SAM3 OME-Zarr off stacks store."""
    path = files.get_path(
        "off_stacks.ome.zarr",
        subject=subject,
        project=label_paths.MODEL_LABELS_PROJECT,
        probe=probe,
        structure=structure,
        condition=condition,
    )
    label_paths.check_not_stale(path.parent, allow_stale=allow_stale)
    return path


def get_sam3_off_stacks_timestamps_path(
    subject: str,
    probe: str,
    condition: str,
    structure: str | None = None,
    *,
    allow_stale: bool = False,
) -> pathlib.Path:
    """Get path for SAM3 timestamps zarr array."""
    path = files.get_path(
        "timestamps.zarr",
        subject=subject,
        project=label_paths.MODEL_LABELS_PROJECT,
        probe=probe,
        structure=structure,
        condition=condition,
    )
    label_paths.check_not_stale(path.parent, allow_stale=allow_stale)
    return path