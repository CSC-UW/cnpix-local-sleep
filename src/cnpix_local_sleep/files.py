"""Generic path-building infrastructure and file definitions.

Provides ``get_path()`` for constructing hierarchical file paths where analysis
parameters are encoded as directory components (e.g., ``probe=imec0/structure=PPC/``),
path-building functions for bandpower and PSD zarr stores, and ``Files`` for
aggregated bandpower filename patterns.

See also ``cnpix_local_sleep.morphological.mua.files``, which wraps ``get_path()`` with
``method='morphological'`` injection.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import wisc_ecephys_tools as wet

from cnpix_local_sleep import const
from cnpix_local_sleep.const import EXPERIMENT

DEFAULT_PATH_SCHEMA = (
    "project",
    "experiment",
    "subject",
    "package",
    "method",
    "model",
    "probe",
    "structure",
    "layer",
    "detection_mode",
    "threshold_group",
    "condition",
    "filters",
)  # TODO: "clade" should go immediately before "structure", and "contrast" should go immediately before "condition".
# "model" identifies a trained-model run (e.g. SAM3 instance-segmentation
# models). It sits just after "method" so that all outputs of a given model
# nest under method=<m>/model=<id>/. Encoded as a single value such as
# "trained-on-Early.REC.NREM.2026-05-09" (no "=" or "_-_"; get_path forbids them).


class Files(StrEnum):
    @staticmethod
    def BANDPOWER_MEANS(bipolar: bool, kind: Literal["stft", "inst"]) -> str:
        stem = "bandpower_condition_means"
        if bipolar:
            stem += ".bipolar"
        return f"{stem}.{kind}.pqt"

    @staticmethod
    def BANDPOWER_CONTRASTS(bipolar: bool, kind: Literal["stft", "inst"]) -> str:
        stem = "bandpower_condition_contrasts"
        if bipolar:
            stem += ".bipolar"
        return f"{stem}.{kind}.pqt"


def get_path(
    filename: str,
    pathspec: dict | None = None,
    *,
    project: str = "offproj",
    experiment: str | None = EXPERIMENT,
    subject: str | None = None,
    enforce_in_schema: bool = True,
    enforce_schema_order: bool = True,
    **kwargs,
) -> Path:
    """Construct file paths with hierarchical parameter organization.

    This function creates paths where analysis parameters are organized into
    directories rather than encoded in filenames. This provides self-documenting
    file organization and prevents filename explosion.

    Args:
        filename: Name of the file (no path components). Can be empty string to
            get directory path only.
        pathspec: Dictionary of parameter key-value pairs to encode in the path.
            Can be nested for complex parameter groupings. If None, only kwargs
            are used.
        project: Project name for WNE infrastructure (default: "offproj").
        experiment: Experiment name for WNE infrastructure (default: EXPERIMENT).
            Can be None to omit experiment from path.
        subject: Subject name for WNE infrastructure. Can be None to omit subject
            from path.
        enforce_in_schema: If True (default), raise error for keys not in schema.
        enforce_schema_order: If True (default), reorder pathspec by schema and
            warn if order changed.
        **kwargs: Additional parameter key-value pairs. Merged with pathspec.

    Returns:
        Path object with structure:
        {project_dir}/{experiment}/{subject}/{key1=val1}/{key2=val2}/.../filename

    Reserved Keywords:
        The following cannot be used as keys in pathspec or kwargs:
        - "filename", "pathspec", "project", "experiment", "subject"
        - "enforce_in_schema", "enforce_schema_order"

    Path Encoding Rules:
        - Keys and values cannot contain "=" or "_-_" (reserved for encoding)
        - "." is allowed in keys and values (R-friendly column names)
        - None values are omitted from path entirely
        - Nested dicts encoded as: key=(k1=v1_-_k2=v2)
        - Path components ordered by DEFAULT_PATH_SCHEMA

    Examples:
        >>> # Basic usage
        >>> get_path(
        ...     "off_df.parquet",
        ...     subject="CNPIX15-Claude",
        ...     probe="imec0",
        ...     structure="PPC",
        ...     condition="Early.NOD.Wake"
        ... )
        Path('.../novel_objects_deprivation/CNPIX15-Claude/probe=imec0/structure=PPC/condition=Early.NOD.Wake/off_df.parquet')

        >>> # With nested parameters
        >>> get_path(
        ...     "thresholds.zarr",
        ...     subject="CNPIX15-Claude",
        ...     probe="imec0",
        ...     structure="PPC",
        ...     threshold_group={"contrast": "NOD.Incline", "param": "slope"}
        ... )
        Path('.../CNPIX15-Claude/.../threshold_group=(contrast=NOD.Incline_-_param=slope)/thresholds.zarr')

        >>> # Omitting None values
        >>> get_path("data.zarr", subject="CNPIX15-Claude", layer=None, structure="PPC")
        Path('.../CNPIX15-Claude/structure=PPC/data.zarr')  # layer omitted

        >>> # Using pathspec dict
        >>> get_path(
        ...     "summary.csv",
        ...     pathspec={"probe": "imec0", "structure": "PPC"},
        ...     subject="CNPIX15-Claude"
        ... )
        Path('.../CNPIX15-Claude/probe=imec0/structure=PPC/summary.csv')
    """
    # Reserved keywords that cannot be in pathspec or kwargs
    reserved = {
        "filename",
        "pathspec",
        "project",
        "experiment",
        "subject",
        "enforce_in_schema",
        "enforce_schema_order",
    }

    # Merge pathspec and kwargs
    working_dict = {}
    if pathspec is not None:
        working_dict.update(pathspec)
    working_dict.update(kwargs)

    # Check for reserved keywords in working_dict
    conflicts = reserved & set(working_dict.keys())
    if conflicts:
        raise ValueError(
            f"Reserved keywords found in pathspec/kwargs: {conflicts}. "
            f"Reserved keywords are: {reserved}"
        )

    # Extract WNE schema components if present in working_dict
    wne_keys = {"project", "experiment", "subject"}
    wne_from_dict = {k: working_dict.pop(k) for k in wne_keys if k in working_dict}

    # Cross-reference extracted WNE components against function arguments
    wne_from_args = {"project": project, "experiment": experiment, "subject": subject}
    for key in wne_keys:
        dict_val = wne_from_dict.get(key)
        arg_val = wne_from_args[key]
        if dict_val is not None and arg_val is not None and dict_val != arg_val:
            raise ValueError(
                f"Conflict for '{key}': pathspec/kwargs has '{dict_val}' "
                f"but function argument is '{arg_val}'"
            )
        # Use dict value if provided, otherwise use arg value
        if dict_val is not None:
            wne_from_args[key] = dict_val

    # Validate all keys and values: reject if contains "=" or "_-_"
    def validate_key_value(key, value, path=""):
        """Recursively validate keys and values."""
        current_path = f"{path}.{key}" if path else key

        if not isinstance(key, str):
            raise TypeError(
                f"Key must be string, got {type(key).__name__} at '{current_path}'"
            )

        if "=" in key:
            raise ValueError(
                f"Key contains reserved substring '=': '{key}' at '{current_path}'"
            )
        if "_-_" in key:
            raise ValueError(
                f"Key contains reserved substring '_-_': '{key}' at '{current_path}'"
            )

        if value is None:
            return  # None values are allowed and will be omitted

        if isinstance(value, dict):
            # Recursively validate nested dict
            for nested_key, nested_val in value.items():
                validate_key_value(nested_key, nested_val, current_path)
        else:
            # Validate value
            value_str = str(value)
            if "=" in value_str:
                raise ValueError(
                    f"Value contains reserved substring '=': '{value_str}' "
                    f"for key '{current_path}'"
                )
            if "_-_" in value_str:
                raise ValueError(
                    f"Value contains reserved substring '_-_': '{value_str}' "
                    f"for key '{current_path}'"
                )

    for key, value in working_dict.items():
        validate_key_value(key, value)

    # Check if keys are in schema
    if enforce_in_schema:
        unknown_keys = set(working_dict.keys()) - set(DEFAULT_PATH_SCHEMA)
        if unknown_keys:
            raise ValueError(
                f"Keys not in schema: {unknown_keys}. "
                f"Schema is: {DEFAULT_PATH_SCHEMA}. "
                f"Set enforce_in_schema=False to allow."
            )

    # Reorder by schema if requested
    if enforce_schema_order:
        original_keys = list(working_dict.keys())
        # Order by schema, keeping only keys that exist in working_dict
        # Keys in schema come first in schema order
        ordered_keys = [k for k in DEFAULT_PATH_SCHEMA if k in working_dict]
        # Keys not in schema come after, in original order
        keys_not_in_schema = [k for k in original_keys if k not in DEFAULT_PATH_SCHEMA]
        ordered_keys.extend(keys_not_in_schema)

        if original_keys != ordered_keys:
            import warnings

            warnings.warn(
                f"Pathspec keys reordered to match schema. "
                f"Original: {original_keys}, Reordered: {ordered_keys}",
                UserWarning,
                stacklevel=2,
            )
        # Reorder the dict
        working_dict = {k: working_dict[k] for k in ordered_keys}

    # Convert pathspec to path components
    def encode_value(value):
        """Encode a value for path component."""
        if isinstance(value, dict):
            # Nested dict: convert to (k1=v1_-_k2=v2)
            pairs = [f"{k}={encode_value(v)}" for k, v in value.items()]
            return f"({('_-_'.join(pairs))})"
        else:
            return str(value)

    path_components = []
    for key, value in working_dict.items():
        if value is not None:  # Omit None values
            encoded = encode_value(value)
            path_components.append(f"{key}={encoded}")

    # Construct the final path
    # Start with project directory
    base_path = wet.get_sglx_project(wne_from_args["project"]).dir

    # Add experiment if provided
    if wne_from_args["experiment"] is not None:
        base_path = base_path / wne_from_args["experiment"]

    # Add subject if provided
    if wne_from_args["subject"] is not None:
        base_path = base_path / wne_from_args["subject"]

    # Add parameter directories
    for component in path_components:
        base_path = base_path / component

    # Add filename
    if filename:
        base_path = base_path / filename

    return base_path


def get_structure_bandpower_path(
    subject: str,
    probe: str,
    structure: str,
    band_name: str,
    bipolar: bool,
    kind: Literal["stft", "inst"],
) -> Path:
    """Get path for a structure-level bandpower zarr store.

    Args:
        subject: Subject name.
        probe: Probe name (e.g. "imec0").
        structure: Brain structure acronym (e.g. "VPM").
        band_name: Frequency band name (e.g. "delta", "eta").
        bipolar: Whether bipolar referencing was used.
        kind: "stft" for STFT-based bandpower, "inst" for instantaneous.

    Returns:
        Path like ``{subject}/probe=imec0/structure=VPM/eta.bipolar.inst.zarr``.
    """
    stem = band_name
    if bipolar:
        stem += ".bipolar"
    return get_path(f"{stem}.{kind}.zarr", subject=subject, probe=probe, structure=structure)


def get_structure_psds_path(
    subject: str,
    probe: str,
    structure: str,
    bipolar: bool,
) -> Path:
    """Get path for a structure-level PSD zarr store.

    Args:
        subject: Subject name.
        probe: Probe name (e.g. "imec0").
        structure: Brain structure acronym (e.g. "VPM").
        bipolar: Whether bipolar referencing was used.

    Returns:
        Path like ``{subject}/probe=imec0/structure=VPM/bipolar.psds.zarr``.
    """
    stem = "bipolar.psds" if bipolar else "psds"
    return get_path(f"{stem}.zarr", subject=subject, probe=probe, structure=structure)


# OFF detection anatomy files (cross-method)
#
# Structure borders and manual layer borders describe anatomical regions of
# interest for OFF detection, NOT per-method detection outputs. Every
# detection method (morphological, harding, unit_based) reads the same borders.
# They live flat in the shared-project subject directory, alongside the
# existing WNE ``<probe>.structures.htsv`` convention, deliberately
# outside any ``method=`` subdirectory.
#
# Prior home (pre-refactor) was
# ``<shared>/<experiment>/<subject>/package=offproj/method=unit_free/probe=<probe>/structure_borders.htsv``
# which conflated anatomy with the legacy tom-era on-disk ``method=unit_free``
# segment. The one-shot migration script that renamed it has since been deleted.
# for the one-shot on-disk move.


def get_off_detection_structure_borders_path(
    subject: str, probe: str
) -> Path:
    """Get path to an OFF-detection structure-borders htsv file.

    Structure borders restrict which channels of a probe cover each
    brain structure *for OFF detection purposes*. Distinct from the
    broader WNE ``<probe>.structures.htsv`` file in that the ``lo``/
    ``hi`` columns can be more restrictive and some structures may
    be omitted entirely.

    Returns a path of the form
    ``<shared>/<experiment>/<subject>/<probe>.off_detection_structure_borders.htsv``.
    """
    subject_dir = wet.get_sglx_project("shared").get_experiment_subject_directory(
        const.EXPERIMENT, subject
    )
    return subject_dir / f"{probe}.off_detection_structure_borders.htsv"


def get_off_detection_layer_borders_path(subject: str, probe: str) -> Path:
    """Get path to an OFF-detection manual layer-borders htsv file.

    Optional. When present, its supra/infra boundaries override the
    automatic top-45%/bottom-45% split of the detection structure
    borders.

    Returns a path of the form
    ``<shared>/<experiment>/<subject>/<probe>.off_detection_layer_borders.htsv``.
    """
    subject_dir = wet.get_sglx_project("shared").get_experiment_subject_directory(
        const.EXPERIMENT, subject
    )
    return subject_dir / f"{probe}.off_detection_layer_borders.htsv"


def get_plot_dir() -> Path:
    path = (
        wet.get_sglx_project("offproj").get_experiment_directory(const.EXPERIMENT)
        / "gf_plots"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_subject_plot_dir(subject: str) -> Path:
    path = (
        wet.get_sglx_project("offproj").get_experiment_subject_directory(
            const.EXPERIMENT, subject
        )
        / "gf_plots"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_r_offp_extdata_dir() -> Path:
    """The R package's ``inst/extdata``, which lives inside this repo.

    ``r-offp/`` sits at the repo root, so this is a fixed hop up from this module
    rather than a guess about how the checkout is laid out. Keeping the R package
    and the exporters in one repo is what makes this a relative path at all --
    it was two independent ``parents[4]`` walks to a *sibling* clone before.
    """
    return Path(__file__).resolve().parents[2] / "r-offp" / "inst" / "extdata"


# The napari annotation stacks, the manual OFF labels, and Table 1's scoring are
# all pinned to the y-coordinates of this legacy Tom-Bugnon ``processed_ap.zarr``.
# The ``method=tom-bugnon`` segment is therefore a *grid* identity, not a live
# dependency on the tom detection pipeline, which is why this path outlives it.
# It goes away only when the stacks are re-rendered from
# ``mua_traces.zarr`` and the labels are re-pinned (see the audit's section 6).
_ANNOTATION_GRID_METHOD = "tom-bugnon"


def get_preprocessed_ap_path(
    subject: str,
    probe: str,
    condition: str | None = None,
    style: Literal["v1", "v2", "v3"] = "v1",
) -> Path:
    """Get path for preprocessed AP-band data (the manual-annotation grid).

    Files containg both light periods and 10s chunks are ~30GB per probe per 48h,
    with ~128MB chunks.

    Values are float32 with arbitrary units.
    Dimensions: (time, channel)
    Coordinates:
      - time: float64 - Timestamps at ~300 Hz sampling rate.
      - channel: string - Channel names of the form 'AP0', 'AP2', etc.
      - y: (channel, float64) - Channel depths in microns.
      - struct: (channel, string) - Brain structure acronyms for each channel.

    Args:
        subject: Subject name
        probe: Probe name (e.g., 'imec0', 'imec1')
        style: Style version of the file path ("v1", "v2", or "v3")

    Returns:
        Path to preprocessed AP-band data zarr store.

    Examples:
        >>> get_preprocessed_ap_path("CNPIX15-Claude", "imec0")
        Path('.../CNPIX15-Claude/method=tom-bugnon/probe=imec0/processed_ap.zarr')
    """
    fname = {
        "v1": "processed_ap.zarr",
        "v2": "preprocessed_ap.zarr",
        "v3": "preprocessed_ap_v3.zarr",
    }[style]
    return get_path(
        fname,
        subject=subject,
        method=_ANNOTATION_GRID_METHOD,
        probe=probe,
        condition=condition,
    )
