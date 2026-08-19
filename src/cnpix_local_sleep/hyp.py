from typing import Final

import wisc_ecephys_tools as wet
from ecephys import hypnogram
from wisc_ecephys_tools.rats import cnd_hgs

from cnpix_local_sleep import const

ALL_STATES: Final[set[str]] = {
    "Wake",
    "NREM",
    "IS",
    "REM",
    "MA",
    "Artifact",
    "NoData",
    "Other",
}


def load_statistical_condition_hypnograms(
    subject: str,
    probe: str | None,
) -> dict[str, hypnogram.FloatHypnogram]:
    """CNPIX6-Eugene's imec0 has hippocampal LFP issues that ruin it's circadian match,
    but apparently Tom deemed that it can be used for a cortical circadian match. This
    loads a project-specific set of hypnograms for CNPIX6-Eugene,
    (see scripts/get_statistical_condition_hypnograms.py),
    whereas all other subjects use the shared project-agnostic hypnograms."""
    if subject == "CNPIX6-Eugene":
        project = wet.get_sglx_project("offproj")
    else:
        project = wet.get_sglx_project("shared")
    hgs = cnd_hgs.load_statistical_condition_hypnograms(
        subject, const.EXPERIMENT, probe, project
    )

    # Assert that no unexpected states are present
    for condition, hg in hgs.items():
        states_in_hg = set(hg["state"].unique())
        unexpected_states = states_in_hg - ALL_STATES
        if unexpected_states:
            raise ValueError(
                f"Unexpected states found in condition '{condition}': {unexpected_states}. "
                f"Expected states are: {ALL_STATES}"
            )
    return hgs


def load_whole_recording_hypnogram(
    subject: str,
    probe: str | None,
    *,
    simplify: bool = True,
    fallback: bool = True,
) -> hypnogram.FloatHypnogram:
    """Whole-recording consolidated hypnogram (states) for a ``(subject, probe)``.

    Mirrors the project selection in :func:`load_statistical_condition_hypnograms`
    (CNPIX6-Eugene uses the project-specific hypnograms; everyone else the shared
    ones), but returns the *whole-recording* consolidated hypnogram rather than
    the six statistical-condition windows.
    """
    from wisc_ecephys_tools.rats import exp_hgs

    project = wet.get_sglx_project(
        "offproj" if subject == "CNPIX6-Eugene" else "shared"
    )
    return exp_hgs.load_consolidated_hypnogram(
        project, const.EXPERIMENT, subject, probe, simplify=simplify, fallback=fallback
    )
