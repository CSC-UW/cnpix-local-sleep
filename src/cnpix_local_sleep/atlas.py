"""Brain atlas utilities for cnpix_local_sleep.

This module is separated from core.py to isolate the brainglobe_atlasapi import,
which is slow to load. Import this module only when atlas functionality is needed.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas


def get_atlas() -> BrainGlobeAtlas:
    """Atlas version 1.2"""
    return BrainGlobeAtlas("whs_sd_rat_39um", check_latest=False)


# Bin edges for anterior-posterior axis grouping.
# Specific to the whs_sd_rat_39um atlas (version 1.2).
AP_COORD_BINS = [0, 12900, 18370, 20100, float("inf")]
AP_COORD_BIN_LABELS = ["ant", "cent-ant", "cent-post", "post"]

#: Supported estimators for a structure's A/P centre. ``"mesh"`` is the historical
#: default and must stay the default: ``AP.Coord`` computed with it is already
#: materialized in every exported parquet, and silently changing the estimator would
#: shift every downstream consumer (the hazard flagged for the mPPC->PPC rename in
#: gfys_workspace/docs/reports/2026-07-02_mppc_to_ppc_rename_consequences.md).
AP_CENTROID_METHODS = ("mesh", "volume")

_AP_COORD_CACHE: dict[tuple[str, str], float] = {}
_AP_PROFILE_CACHE: dict[str, np.ndarray] = {}


def get_anterior_posterior_axis_profile(structure: str) -> np.ndarray:
    """Voxel count per anterior-posterior slice for a structure.

    Returns a 1-D array over the atlas' first (A/P) axis, giving the number of
    annotation voxels belonging to *structure* (including its descendants) in
    each A/P slice. This is the structure's mass distribution along A/P, and
    backs :func:`get_anterior_posterior_axis_coord` (``method="volume"``).

    ``BrainGlobeAtlas.get_structure_mask`` returns a mask whose non-zero entries
    carry the *structure id*, not ``1``; it must be binarized before counting,
    or structures with descendants (e.g. ``V2`` -> ``V2M``/``V2L``) would be
    weighted by their childrens' ids.
    """
    if structure in _AP_PROFILE_CACHE:
        return _AP_PROFILE_CACHE[structure]
    mask = get_atlas().get_structure_mask(structure) > 0
    profile = mask.sum(axis=(1, 2)).astype(np.float64)
    _AP_PROFILE_CACHE[structure] = profile
    return profile


def get_anterior_posterior_axis_coord(
    structure: str, method: str = "mesh"
) -> float:
    """Get the anterior-posterior axis coordinate for a brain structure, in um.

    Larger values are more posterior (the whs_sd_rat_39um atlas is ``asr``
    oriented, so axis 0 runs anterior -> posterior). The coordinate is in atlas
    voxel space, *not* relative to bregma.

    Parameters
    ----------
    structure
        Structure acronym.
    method
        ``"mesh"`` (default) takes the mean of the structure's mesh vertices
        along the A/P axis. It is biased by tessellation density, which tracks
        local surface curvature rather than volume.
        ``"volume"`` takes the mass centroid of the annotation volume, the
        voxel-count-weighted mean A/P position, including descendant
        structures. This is the unbiased centroid of the structure's volume.

        The two agree closely on the cortical structures used here (Pearson
        r ~ 0.997 across the 12 OFF-analysis structures), so ``"mesh"`` remains
        the default for backwards compatibility with the exported ``AP.Coord``.
    """
    if method not in AP_CENTROID_METHODS:
        raise ValueError(
            f"Unknown A/P centroid method {method!r}. "
            f"Supported: {', '.join(AP_CENTROID_METHODS)}"
        )
    key = (structure, method)
    if key in _AP_COORD_CACHE:
        return _AP_COORD_CACHE[key]
    if method == "mesh":
        mesh = get_atlas().mesh_from_structure(structure)
        coord = float(np.mean(mesh.points, axis=0)[0])
    else:
        profile = get_anterior_posterior_axis_profile(structure)
        resolution = get_atlas().resolution[0]
        positions = np.arange(len(profile)) * resolution
        coord = float((positions * profile).sum() / profile.sum())
    _AP_COORD_CACHE[key] = coord
    return coord


def sort_structures_by_anterior_posterior(
    structures: Sequence[str],
) -> list[str]:
    """Sort brain structures by their anterior-posterior axis coordinate.

    Parameters
    ----------
    structures
        Structure acronyms to sort.

    Returns
    -------
    list[str]
        Structures sorted by A/P coordinate (ascending).
    """
    return sorted(structures, key=get_anterior_posterior_axis_coord)


_CLADE_CACHE: dict[str, str] = {}


def get_clade(structure: str) -> str:
    if structure in _CLADE_CACHE:
        return _CLADE_CACHE[structure]

    ancestors = get_atlas().get_structure_ancestors(structure)
    if "Thal-D" in ancestors or structure == "Thal-D":
        clade = "Thal-D"
    elif "Cx" in ancestors or structure == "Cx":
        if "HF" in ancestors or structure == "HF":
            clade = "HF"
        else:
            clade = "Cx"
    elif "Str" in ancestors or structure == "Str":
        clade = "Str"
    elif structure == "CLA":
        clade = "Cx"
    else:
        clade = "Other"

    _CLADE_CACHE[structure] = clade
    return clade


def cx_ap_group(ap_coord: float, clade: str) -> str | float:
    """Bin a cortical structure's A/P coordinate into an ``AP_COORD_BIN_LABELS`` group.

    Returns ``pd.NA`` for non-cortical clades (the A/P grouping is only defined
    for cortex). Shared by :func:`postprocess_offs.postprocess_offs_frame` and
    :func:`consolidate_off_structure_columns` so the binning lives in one place.
    """
    if clade != "Cx":
        return pd.NA
    ap_group = pd.cut([ap_coord], AP_COORD_BINS, labels=AP_COORD_BIN_LABELS)[0]
    return str(ap_group)


# -------------------- Interim OFF-analysis structure consolidation --------------------
#: SPOT alias map for the interim OFF-analysis structure consolidation. ``mPPC``
#: ("Parietal association cortex, medial area") is treated as ``PPC`` ("Posterior
#: parietal cortex") for the OFF-analysis projects, because the finer distinction
#: reflects a precision we do not uniformly have. This is
#: an ANALYSIS-LAYER relabel only: the source ``*.structures.htsv`` anatomy,
#: detection thresholds/borders, and the ``sps_conf`` registry are unchanged and
#: still keyed by the raw acronym. Apply this at export boundaries (after
#: detection + postprocess, before summarize/group-by).
OFF_ANALYSIS_STRUCTURE_ALIASES: dict[str, str] = {"mPPC": "PPC"}


def consolidate_structure_label(structure: str) -> str:
    """Map a raw structure acronym to its consolidated OFF-analysis label."""
    return OFF_ANALYSIS_STRUCTURE_ALIASES.get(structure, structure)


def consolidate_off_structure_columns(offs: pd.DataFrame) -> pd.DataFrame:
    """Relabel aliased structures and recompute atlas-derived per-structure columns.

    Applies :data:`OFF_ANALYSIS_STRUCTURE_ALIASES` to the ``structure`` column and,
    for any of ``clade`` / ``AP.Coord`` / ``Cx.AP.group`` that are present, recomputes
    them from the consolidated label. Recomputation is required (not cosmetic): the
    aggregation summary merges per-structure metadata via
    ``struct_info.drop_duplicates()`` + ``merge(on="structure")``; if a relabeled
    ``PPC`` row kept the aliased structure's ``AP.Coord`` alongside a native ``PPC``
    row's, the merge would fan out. After recompute all rows sharing a consolidated
    label carry identical atlas values, so exactly one ``struct_info`` row remains.

    Idempotent, and a no-op (returns *offs* unchanged) when no aliased structure is
    present or the frame is empty. Per-OFF columns such as ``laminar_class`` (already
    computed from the correct raw identity upstream) are preserved untouched.
    """
    if "structure" not in offs.columns or offs.empty:
        return offs

    aliased = offs["structure"].isin(OFF_ANALYSIS_STRUCTURE_ALIASES)
    if not aliased.any():
        return offs

    offs = offs.copy()
    offs["structure"] = offs["structure"].map(consolidate_structure_label)
    if isinstance(offs["structure"].dtype, pd.CategoricalDtype):
        offs["structure"] = offs["structure"].cat.remove_unused_categories()

    # Recompute atlas-derived per-structure columns for every consolidated label so
    # each label carries a single consistent (clade, AP.Coord, Cx.AP.group) triple.
    consolidated_labels = set(OFF_ANALYSIS_STRUCTURE_ALIASES.values())
    for label in consolidated_labels:
        rows = offs["structure"] == label
        if not rows.any():
            continue
        clade = get_clade(label)
        ap_coord = get_anterior_posterior_axis_coord(label)
        if "clade" in offs.columns:
            offs.loc[rows, "clade"] = clade
        if "AP.Coord" in offs.columns:
            offs.loc[rows, "AP.Coord"] = ap_coord
        if "Cx.AP.group" in offs.columns:
            offs.loc[rows, "Cx.AP.group"] = cx_ap_group(ap_coord, clade)

    return offs
