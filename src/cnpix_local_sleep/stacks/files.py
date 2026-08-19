"""File paths for OFF period image stacks."""

import pathlib

from cnpix_local_sleep import files


# SAM3 paths (method=sam3, no detection_mode component)


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
        method="sam3",
        probe=probe,
        structure=structure_acronym,
        condition=condition,
    )


def get_sam3_off_stacks_ome_zarr_path(
    subject: str,
    probe: str,
    condition: str,
    structure: str | None = None,
) -> pathlib.Path:
    """Get path for SAM3 OME-Zarr off stacks store."""
    return files.get_path(
        "off_stacks.ome.zarr",
        subject=subject,
        method="sam3",
        probe=probe,
        structure=structure,
        condition=condition,
    )


def get_sam3_off_stacks_timestamps_path(
    subject: str,
    probe: str,
    condition: str,
    structure: str | None = None,
) -> pathlib.Path:
    """Get path for SAM3 timestamps zarr array."""
    return files.get_path(
        "timestamps.zarr",
        subject=subject,
        method="sam3",
        probe=probe,
        structure=structure,
        condition=condition,
    )