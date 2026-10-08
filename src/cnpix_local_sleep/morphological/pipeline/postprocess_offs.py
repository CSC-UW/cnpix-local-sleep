"""Derived OFF columns (clade, A/P group, normalized features, laminar concentrations).

Applied in memory by the full-48h aggregation before filtering.
"""

from __future__ import annotations

import pandas as pd

from cnpix_local_sleep import atlas
from cnpix_local_sleep import sps_conf


def postprocess_offs_frame(
    offs: pd.DataFrame,
    structure: str,
    *,
    subject: str | None = None,
    probe: str | None = None,
) -> pd.DataFrame:
    """Add the derived postprocessing columns to *offs* in place.

    Returns the same DataFrame for convenience. Columns added:
        - ``clade``: Anatomical clade (e.g. "Cx").
        - ``AP.Coord``: Anterior-posterior axis coordinate.
        - ``Cx.AP.group``: Anterior-posterior bin label (cortical only).
        - ``span_rel2max``: span / max_span.
        - ``area_rel2span``: area / max_span.
        - ``onset_offset_wedge``: onset_slope - offset_slope.

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
