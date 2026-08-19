from __future__ import annotations

import types
from typing import Literal


from cnpix_local_sleep.morphological.mua import files


def detection_outputs_exist(
    subject: str,
    probe: str,
    structure: str,
    condition: str,
    threshold_group: str | None = None,
    files_module: types.ModuleType | None = None,
) -> bool:
    """Check if OFF detection outputs exist for a given condition.

    Args:
        subject: Subject identifier
        probe: Probe identifier
        structure: Brain structure name
        condition: Experimental condition
        threshold_group: Threshold group name, or None if not using
            threshold groups
        files_module: Module providing ``get_offs_path()`` and
            ``get_off_label_indices_path()``. Defaults to
            ``cnpix_local_sleep.morphological.mua.files``.

    Returns:
        True if both label indices and offs files exist, False otherwise
    """
    fm = files if files_module is None else files_module
    f1 = fm.get_off_label_indices_path(
        subject=subject,
        probe=probe,
        structure=structure,
        condition=condition,
        threshold_group=threshold_group,
    )
    f2 = fm.get_offs_path(
        subject=subject,
        probe=probe,
        structure=structure,
        condition=condition,
        threshold_group=threshold_group,
    )
    return f1.exists() and f2.exists()


def log_step(step: str, **kwargs) -> None:
    """Helper to standardize logging output for processing steps.

    Args:
        step: The step name (e.g., "RUNNING", "PASSING", "DONE")
        **kwargs: Key-value pairs to include in the log message
    """
    details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"{step}: {details}")


def get_threshold_groups_to_run(
    condition: str,
    threshold_method: Literal["from_value"],
) -> list[str]:
    """Determine which contrasts to run for a condition given threshold method.

    With ``"from_value"`` -- the only surviving method -- the same threshold
    applies to every condition in a contrast, so each condition is detected
    once and no per-contrast reruns are needed.

    Args:
        condition: The experimental condition
        threshold_method: Method for computing thresholds

    Returns:
        Empty list.
    """
    if threshold_method != "from_value":
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    return []