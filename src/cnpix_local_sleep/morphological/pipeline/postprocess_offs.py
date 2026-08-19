"""Postprocess OFF period parquet files with derived columns.

This pipeline step runs after detection and before aggregation. It adds
per-file metadata columns (clade, A/P group, normalized features) so that
aggregation can simply collect the files without recomputing these values.

Only operates on cortical structures.
"""

from __future__ import annotations

import types

import pandas as pd

from cnpix_local_sleep import atlas, const
from cnpix_local_sleep.morphological.mua import files
from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.morphological.pipeline import utils


def postprocess_offs_file(
    offs_path: str,
    structure: str,
    *,
    subject: str | None = None,
    probe: str | None = None,
) -> None:
    """Add derived columns to an OFF parquet file in-place.

    Columns added:
        - ``clade``: Anatomical clade (e.g. "Cx").
        - ``AP.Coord``: Anterior-posterior axis coordinate.
        - ``Cx.AP.group``: Anterior-posterior bin label (cortical only).
        - ``span_rel2max``: span / max_span.
        - ``area_rel2span``: area / max_span.

    Args:
        offs_path: Path to offs.parquet file.
        structure: Brain structure acronym.
        subject: Subject identifier. Needed for the per-combo supra/infra
            orientation correction (see :func:`laminar_concentrations`) when
            *offs* lacks subject/probe/structure columns.
        probe: Probe identifier. See *subject*.
    """
    offs = pd.read_parquet(offs_path)
    postprocess_offs_frame(offs, structure, subject=subject, probe=probe)
    offs.to_parquet(offs_path, index=False)


def postprocess_offs_frame(
    offs: pd.DataFrame,
    structure: str,
    *,
    subject: str | None = None,
    probe: str | None = None,
) -> pd.DataFrame:
    """Add the derived postprocessing columns to *offs* in place.

    This is the in-memory core of :func:`postprocess_offs_file`; callers that
    already hold an OFF DataFrame (e.g. full-48h re-aggregation) use this
    directly instead of round-tripping through a parquet file. Returns the
    same DataFrame for convenience. See :func:`postprocess_offs_file` for the
    list of columns added.

    *subject*/*probe* (together with *structure*) identify the combo for the
    per-combo supra/infra orientation correction used by
    :func:`laminar_concentrations`. Pass them when *offs* is a single-combo
    frame without subject/probe/structure columns (e.g. the full-48h
    ``offs.parquet``); multi-combo frames that carry those columns are resolved
    per row instead.
    """
    if offs.empty:
        offs["clade"] = pd.Series(dtype="object")
        offs["AP.Coord"] = pd.Series(dtype="float64")
        offs["Cx.AP.group"] = pd.Series(dtype="object")
        offs["span_rel2max"] = pd.Series(dtype="float64")
        offs["area_rel2span"] = pd.Series(dtype="float64")
        offs["onset_offset_wedge"] = pd.Series(dtype="float64")
        return offs

    # Clade
    clade = atlas.get_clade(structure)
    offs["clade"] = clade

    # Anterior-posterior coordinate and group
    ap_coord = atlas.get_anterior_posterior_axis_coord(structure)
    offs["AP.Coord"] = ap_coord

    offs["Cx.AP.group"] = atlas.cx_ap_group(ap_coord, clade)

    # Normalized features
    offs["span_rel2max"] = offs["span"] / offs["max_span"]
    offs["area_rel2span"] = offs["area"] / offs["max_span"]
    offs["onset_offset_wedge"] = offs["onset_slope"] - offs["offset_slope"]

    return offs


def _laminar_flip_mask(
    offs: pd.DataFrame,
    *,
    subject: str | None = None,
    probe: str | None = None,
    structure: str | None = None,
) -> pd.Series:
    """Boolean Series flagging rows whose supra/infra order is flipped.

    Resolves each row's (subject, probe, structure) identity against
    :func:`sps_conf.get_flipped_laminar_combos`. Identity comes either from the
    scalar *subject*/*probe*/*structure* args (single-combo frames) or, when
    those are not all given, from per-row ``subject``/``probe``/``structure``
    columns (multi-combo frames). If identity cannot be resolved either way,
    returns all-``False`` (no flip), safe because the flipped set is a small,
    explicit allowlist and unresolvable frames carry no laminar combos.
    """
    flips = sps_conf.get_flipped_laminar_combos()
    if not flips:
        return pd.Series(False, index=offs.index)

    if subject is not None and probe is not None and structure is not None:
        is_flipped = (subject, probe, structure) in flips
        return pd.Series(is_flipped, index=offs.index)

    if {"subject", "probe", "structure"}.issubset(offs.columns):
        keys = zip(offs["subject"], offs["probe"], offs["structure"])
        return pd.Series([k in flips for k in keys], index=offs.index)

    return pd.Series(False, index=offs.index)


def laminar_concentrations(
    offs: pd.DataFrame,
    *,
    subject: str | None = None,
    probe: str | None = None,
    structure: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return orientation-corrected ``(supra_concentration, infra_concentration)``.

    ::

        supra_concentration = supra_area / (supra_area + infra_area)
        infra_concentration = infra_area / (supra_area + infra_area)

    The raw ``supra_area``/``infra_area`` columns are stored in *geometric*
    order ("supra" = top 45% band, higher y/depth). For combos flagged
    ``flip_supra_infra`` in the sps config (brain curvature flips the structure
    vertically vs probe geometry; see
    :func:`sps_conf.get_flipped_laminar_combos`), the two concentrations are
    swapped so that "supra" denotes the true supragranular layer.

    This is the single point of truth for the supra/infra fraction, shared by
    the depth-profile null (:mod:`cnpix_local_sleep.morphological.laminar_null`) and the
    trimodality notebook. It does not mutate *offs* (postprocessing stays
    idempotent).

    Combo identity is resolved as in :func:`_laminar_flip_mask`.
    """
    total_area = offs["supra_area"] + offs["infra_area"]
    supra_conc = offs["supra_area"] / total_area
    infra_conc = offs["infra_area"] / total_area

    flipped = _laminar_flip_mask(
        offs, subject=subject, probe=probe, structure=structure
    )
    corrected_supra = supra_conc.where(~flipped, infra_conc)
    corrected_infra = infra_conc.where(~flipped, supra_conc)
    return corrected_supra, corrected_infra


def do_structure(
    subject: str,
    probe: str,
    structure: str,
    files_module: types.ModuleType | None = None,
) -> None:
    """Postprocess OFFs for all conditions in a structure.

    Args:
        subject: Subject identifier.
        probe: Probe identifier.
        structure: Brain structure name.
        files_module: Module providing ``get_offs_path()``. Defaults to
            ``cnpix_local_sleep.morphological.mua.files``.
    """
    fm = files if files_module is None else files_module
    for condition in const.CORE_CONDITIONS:
        offs_path = fm.get_offs_path(
            subject=subject,
            probe=probe,
            structure=structure,
            condition=condition,
            threshold_group=None,
        )
        if offs_path.exists():
            utils.log_step(
                "Postprocessing OFFs",
                condition=condition,
            )
            postprocess_offs_file(
                str(offs_path), structure, subject=subject, probe=probe
            )


def do_experiment(files_module: types.ModuleType | None = None) -> None:
    """Postprocess OFFs for all cortical structures in the experiment.

    Args:
        files_module: Module providing ``get_offs_path()``. Defaults to
            ``cnpix_local_sleep.morphological.mua.files``.
    """
    fm = files if files_module is None else files_module
    spsl_cx = sps_conf.get_subject_probe_structure_list(
        method=fm.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )
    for subject, probe, structure in spsl_cx:
        print(f"Postprocessing OFFs: {subject}, {probe}, {structure}")
        do_structure(subject, probe, structure, files_module=fm)