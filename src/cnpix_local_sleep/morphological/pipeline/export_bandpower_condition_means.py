"""Export per-condition instantaneous bipolar band-power means for r-offp.

Additive, fully separable companion (mirrors the other r-offp exporters such as
``export_correlation_inputs.py`` / ``cross_structure_excess_export.py``). It
summarizes, per ``(subject, probe, structure, condition)``, the mean of the
z-scored log10 instantaneous bipolar band power: the *same* power trace that
annotates OFF periods in
:mod:`cnpix_local_sleep.morphological.pipeline.add_bandpower_to_offs` (``zscore(log10(power))``,
z-scored over the whole clean ``Full.Conservative`` recording, per structure).

The output parquet mirrors the schema of the summarized full-48h OFF parquets
(keys ``subject, probe, structure, condition`` plus struct metadata
``clade, AP.Coord, Cx.AP.group``), so the r-offp ``bandpower`` homeostasis
pipeline can fit it through the identical condition-homeostasis model and draw
the same six-condition violins with significance bars.

This computes over the cortical subject-probe-structure list (the "full
cortical" list backing the full-48h morphological OFFs). Requires the per-structure
instantaneous band-power zarrs on NFS; never writes to NFS.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from cnpix_local_sleep import atlas, const, sps_conf
from cnpix_local_sleep.morphological import agg

#: Bands to summarize. ``delta`` is the headline SWA band; ``eta`` is a companion.
DEFAULT_BAND_NAMES: tuple[str, ...] = ("delta", "eta")

#: Output filename. Uses the ``full48h_`` infix so the r-offp loader / plotting /
#: summary tooling (``summary_filename``/``load_offs_summary``) resolves it as the
#: ``full48h`` source of the ``bandpower`` dataset. The band power is a property of
#: the whole 48h recording, so ``full48h`` is the honest provenance label.
OUTPUT_FILENAME: str = "summarized_full48h_bandpower_offs.parquet"


def get_cortical_spsl() -> list[tuple[str, str, str]]:
    """The full cortical (subject, probe, structure) list for morphological detection."""
    return sps_conf.get_subject_probe_structure_list(
        method="morphological",
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )


def _add_struct_info(means: pd.DataFrame) -> pd.DataFrame:
    """Attach ``clade``/``AP.Coord``/``Cx.AP.group`` per structure.

    Reproduces the canonical anatomical labeling used by the OFF summaries
    (:func:`cnpix_local_sleep.morphological.pipeline.postprocess_offs.postprocess_offs_frame`) so
    the bandpower parquet is schema-compatible with the summarized OFF parquets.
    """
    structures = means["structure"].unique()
    records = []
    for structure in structures:
        clade = atlas.get_clade(structure)
        ap_coord = atlas.get_anterior_posterior_axis_coord(structure)
        records.append(
            {
                "structure": structure,
                "clade": clade,
                "AP.Coord": ap_coord,
                "Cx.AP.group": atlas.cx_ap_group(ap_coord, clade),
            }
        )
    struct_info = pd.DataFrame.from_records(records)
    return means.merge(struct_info, on="structure", how="left")


def summarize_bandpower_condition_means(
    band_names: tuple[str, ...] = DEFAULT_BAND_NAMES,
    bipolar: bool = True,
    kind: Literal["stft", "inst"] = "inst",
    spsl: list[tuple[str, str, str]] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Per-condition band-power means, one row per (sps, condition).

    Loads the per-timepoint ``zlog_<band>``/``log_<band>`` traces (via
    :func:`cnpix_local_sleep.morphological.agg.aggregate_bandpowers`, same transform/baseline as
    the OFF annotation), tags each timepoint by statistical condition, and
    averages within each ``(subject, probe, structure, condition)`` cell.

    Returns columns ``subject, probe, structure, condition``, the ``mean_*``
    power columns (e.g. ``mean_zlog_delta``, ``mean_log_delta``, ``mean_delta``
    and the ``eta`` variants), and struct metadata ``clade, AP.Coord,
    Cx.AP.group``.
    """
    if spsl is None:
        spsl = get_cortical_spsl()

    # Per-timepoint zlog/log power, condition-tagged (wide boolean columns).
    pwr = agg.aggregate_bandpowers(
        bipolar, kind, band_names=list(band_names), verbose=verbose, spsl=spsl
    )
    # One row per (timepoint, condition), then mean within each cell.
    # ``aggregate_bandpowers`` tags each timepoint with membership in *every*
    # statistical-condition hypnogram, so restrict to the band-power value
    # columns before averaging; otherwise meaningless mean_<other_condition>
    # overlap-fraction columns leak into the output.
    key_cols = ["subject", "probe", "structure", "condition"]
    value_cols = []
    for band in band_names:
        value_cols += [band, f"log_{band}", f"zlog_{band}"]
    c_pwr = agg.aggregated_events_wide_to_long(pwr)
    value_cols = [c for c in value_cols if c in c_pwr.columns]
    means = (
        c_pwr[key_cols + value_cols]
        .groupby(key_cols, observed=True)
        .mean(numeric_only=True)
        .add_prefix("mean_")
        .reset_index()
    )

    # Order conditions like the OFF summaries (anterior->posterior irrelevant here;
    # this is the canonical six-condition order used throughout this package and r-offp).
    means["condition"] = pd.Categorical(
        means["condition"], categories=list(const.CORE_CONDITIONS), ordered=True
    )
    means = means.sort_values(
        ["subject", "probe", "structure", "condition"]
    ).reset_index(drop=True)

    means = _add_struct_info(means)
    # Interim OFF-analysis structure consolidation (e.g. mPPC -> PPC): relabel and
    # recompute the just-attached struct metadata from the consolidated label.
    return atlas.consolidate_off_structure_columns(means)


def export_bandpower_condition_means(
    output_dir: Path,
    band_names: tuple[str, ...] = DEFAULT_BAND_NAMES,
    bipolar: bool = True,
    kind: Literal["stft", "inst"] = "inst",
    spsl: list[tuple[str, str, str]] | None = None,
    verbose: bool = False,
) -> Path:
    """Compute and write ``summarized_full48h_bandpower_offs.parquet``.

    Writes flat into ``output_dir`` (``r-offp/inst/extdata``); never
    writes to NFS. Returns the written path.
    """
    means = summarize_bandpower_condition_means(
        band_names=band_names,
        bipolar=bipolar,
        kind=kind,
        spsl=spsl,
        verbose=verbose,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_FILENAME
    # Categorical condition -> str for a stable, tool-agnostic parquet.
    means = means.copy()
    means["condition"] = means["condition"].astype(str)
    means.to_parquet(out_path)
    return out_path
