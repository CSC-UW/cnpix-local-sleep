"""Probe-channel anatomy: which structure and layer each channel sits in.

Everything here is backed by the recording's own htsv tables (``<probe>.structures.htsv``,
the detection borders) rather than by an atlas query; :mod:`cnpix_local_sleep.atlas` is the
BrainGlobe side, deliberately isolated because it is slow to import.
"""

from __future__ import annotations


import ecephys.utils
import numpy as np
import pandas as pd
import wisc_ecephys_tools as wet
import xarray as xr
from ecephys import wne

from cnpix_local_sleep import atlas, const, files as _op_files


def load_structures(subject: str, probe: str) -> pd.DataFrame:
    """Load per-subject structure boundaries from network storage.

    This reads the ``<probe>.structures.htsv`` file from the shared WNE project,
    which contains the structure boundaries for each probe in the experiment.
    Distinct from ``cnpix_local_sleep.sps_conf.load_config()``, which reads
    bundled CSV metadata.
    """
    ext = wne.constants.FileExtensions.STRUCTURES
    return ecephys.utils.read_htsv(
        wet.get_sglx_project("shared").get_experiment_subject_file(
            const.EXPERIMENT, subject, f"{probe}{ext}"
        )
    )

def assign_structures(da: xr.DataArray, structs: pd.DataFrame) -> xr.DataArray:
    depths = da.y
    structures = np.array(["???"] * len(depths), dtype=object)
    for structure in structs.itertuples():
        lo = getattr(structure, "lo")
        hi = getattr(structure, "hi")
        mask = (depths >= lo) & (depths <= hi)
        structures[np.where(mask)] = getattr(structure, "acronym")
    return da.assign_coords(struct=("channel", structures))

def compute_channel_mask(
    y_coords: xr.DataArray,
    subject: str,
    probe: str,
    structure: str,
    layer: str | None = None,
) -> xr.DataArray:
    """Compute boolean channel mask from border definitions.

    This computes the mask on-the-fly from the structure/layer border files,
    rather than loading a pre-computed mask from disk.

    Parameters
    ----------
    y_coords
        DataArray containing the y-coordinates (depths in µm) of each channel.
        Must have a 'channel' coordinate.
    subject
        Subject identifier.
    probe
        Probe identifier.
    structure
        Brain structure acronym (e.g., 'V1', 'PPC').
    layer
        Optional layer name ('supra' or 'infra'). If None, uses structure-level
        borders from structure_borders.htsv. If provided, computes layer-level
        borders on-the-fly via get_layer_borders().

    Returns
    -------
    xr.DataArray
        Boolean DataArray with same 'channel' coordinate as y_coords.
        True for channels within the specified borders, False otherwise.
    """
    if layer is None:
        borders = load_detection_borders(subject, probe).set_index("acronym")
        lo = borders.loc[structure, "lo"]
        hi = borders.loc[structure, "hi"]
    else:
        borders = get_layer_borders(subject, probe, structure)
        layer_row = borders[borders.layer == layer]
        if layer_row.empty:
            raise ValueError(f"Layer '{layer}' not found for structure '{structure}'")
        lo = layer_row.iloc[0]["lo"]
        hi = layer_row.iloc[0]["hi"]

    keep = np.logical_and(y_coords.data >= lo, y_coords.data <= hi)
    return xr.DataArray(
        data=keep,
        dims=("channel",),
        coords={"channel": y_coords.channel},
        name="Detection channels",
    )

def get_layer_borders(
    subject: str,
    probe: str,
    structure: str,
) -> pd.DataFrame:
    """Get layer borders for a cortical structure, computing on-the-fly if needed.

    Layer borders are defined either:
        - From manual values, if `layer_borders.htsv` file is found
        - Or computed as the top and bottom 45% of channels used in structure-wide
          detection

    Parameters
    ----------
    subject
        Subject identifier.
    probe
        Probe identifier.
    structure
        BrainGlobe atlas structure acronym (e.g., 'V1', 'PPC').
        Must be a cortical structure.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: acronym, layer, hi, lo
        - acronym: Original structure acronym (NOT modified with layer suffix)
        - layer: 'supra' or 'infra'
        - hi: Upper depth boundary in micrometers
        - lo: Lower depth boundary in micrometers

    Raises
    ------
    FileNotFoundError
        If detection borders file does not exist.
    ValueError
        If structure is not found in detection borders.
    AssertionError
        If structure is not cortical.
    """
    assert atlas.get_clade(structure) == "Cx", (
        "Cannot compute layer borders for non-cortical structures."
    )

    # Check for manual layer borders first. Cross-method anatomy; resolves
    # to <shared>/<experiment>/<subject>/<probe>.off_detection_layer_borders.htsv.
    manual_layer_borders_file = _op_files.get_off_detection_layer_borders_path(
        subject, probe
    )
    if manual_layer_borders_file.exists():
        layers = ecephys.utils.read_htsv(manual_layer_borders_file)

        # Filter to requested structure - handle both old and new schema formats
        if "structure" in layers.columns:
            layers = layers[layers.structure == structure].copy()
        else:
            # Filter by acronym - handles both "V1" and "V1-supra" formats
            mask = (layers.acronym == structure) | layers.acronym.str.startswith(
                f"{structure}-"
            )
            layers = layers[mask].copy()

        # Normalize to new schema
        layers["acronym"] = structure
        if "structure" in layers.columns:
            layers = layers.drop(columns=["structure"])

        return layers[["acronym", "layer", "hi", "lo"]]

    # Compute from detection borders (cross-method anatomy).
    detection_borders_file = _op_files.get_off_detection_structure_borders_path(
        subject, probe
    )
    if not detection_borders_file.exists():
        raise FileNotFoundError(
            f"Could not find detection borders file: {detection_borders_file}"
        )

    detection_borders = ecephys.utils.read_htsv(detection_borders_file).set_index(
        "acronym"
    )
    if structure not in detection_borders.index:
        raise ValueError(
            f"Could not find structure acronym {structure} in detection borders "
            f"file: {detection_borders_file}."
        )

    # NOTE: "supra" is hard-wired to the top band (higher y/depth) and "infra"
    # to the bottom band. This geometric convention is WRONG for combos where
    # brain curvature flips the structure vertically vs the probe
    # (``sps_conf.get_flipped_laminar_combos``, e.g. CNPIX12-Santiago/imec0/VO).
    # That per-combo orientation is currently corrected downstream at
    # consumption, in
    # ``cnpix_local_sleep.morphological.pipeline.postprocess_offs.laminar_concentrations``.
    # TODO(source-fix): relabel the bands here for flipped combos (swap the
    # "supra"/"infra" assignment) so callers like ``detect.add_laminar_areas``
    # persist anatomically honest areas, then re-run detection + re-export. If
    # you do, you MUST remove the consumption-time swap in
    # ``laminar_concentrations`` simultaneously to avoid a silent double-flip.
    supra_hi = detection_borders.loc[structure, "hi"]
    infra_lo = detection_borders.loc[structure, "lo"]
    span = supra_hi - infra_lo

    span_ratio = 0.45
    supra_lo = supra_hi - span_ratio * span
    infra_hi = infra_lo + span_ratio * span

    layers = pd.DataFrame(
        [
            {
                "acronym": structure,
                "layer": "supra",
                "hi": supra_hi,
                "lo": supra_lo,
            },
            {
                "acronym": structure,
                "layer": "infra",
                "hi": infra_hi,
                "lo": infra_lo,
            },
        ]
    )

    return layers

def load_detection_borders(subject: str, probe: str) -> pd.DataFrame:
    """Load the region borders used for unit-free OFF detection.

    Borders of the region of interest within each structure used for OFF detection,
    in microns. The contents of this file resemble `<probe>.structures.htsv`,
    but the `lo` and `hi` columns specify the channels used for detection within
    each structure, and therefore are more restricted.
    There may be gaps between structures, and entire structures may be absent.

    Resolves to
    ``<shared>/<experiment>/<subject>/<probe>.off_detection_structure_borders.htsv``
    via :func:`cnpix_local_sleep.files.get_off_detection_structure_borders_path`.
    """
    return ecephys.utils.read_htsv(
        _op_files.get_off_detection_structure_borders_path(
            subject=subject, probe=probe
        )
    )
