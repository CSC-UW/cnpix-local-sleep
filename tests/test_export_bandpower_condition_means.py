"""Tests for the per-condition band-power means exporter.

The struct-metadata attachment and the cortical SPS selection are pure (no NFS);
the full compute/export smoke test needs mounted production data, so it is marked
``requires_nfs`` and gated behind ``RUN_NFS_TESTS=1`` like the other
production-data smoke tests in this package.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from cnpix_local_sleep import atlas
from cnpix_local_sleep.morphological.pipeline import export_bandpower_condition_means as bp


def _skip_without_nfs() -> None:
    if os.environ.get("RUN_NFS_TESTS") != "1":
        pytest.skip("Set RUN_NFS_TESTS=1 to run production-data smoke test")


# -------------------- Pure: cortical SPS selection --------------------


def test_get_cortical_spsl_is_cortical_only():
    spsl = bp.get_cortical_spsl()
    assert len(spsl) > 0
    # Every returned structure must be cortical clade.
    assert all(atlas.get_clade(structure) == "Cx" for _, _, structure in spsl)


# Pure: struct-metadata attachment matches the canonical anatomical labeling


def test_add_struct_info_attaches_clade_and_ap():
    means = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "probe": ["imec0", "imec0", "imec0"],
            "structure": ["PPC", "PPC", "M2"],
            "condition": ["Early.REC.NREM", "Late.REC.NREM", "Early.REC.NREM"],
            "mean_zlog_delta": [1.7, 1.1, 1.5],
        }
    )
    out = bp._add_struct_info(means)

    # Columns added, rows preserved.
    assert set(["clade", "AP.Coord", "Cx.AP.group"]).issubset(out.columns)
    assert len(out) == len(means)

    # Values match the canonical atlas labeling used by the OFF summaries.
    for structure in means["structure"].unique():
        row = out.loc[out["structure"] == structure].iloc[0]
        assert row["clade"] == atlas.get_clade(structure)
        assert row["AP.Coord"] == atlas.get_anterior_posterior_axis_coord(structure)
        assert row["clade"] == "Cx"
        assert isinstance(row["Cx.AP.group"], str)


# -------------------- Production-data smoke test (needs NFS) --------------------


@pytest.mark.requires_nfs
def test_summarize_bandpower_condition_means_smoke():
    _skip_without_nfs()
    spsl = bp.get_cortical_spsl()[:1]
    means = bp.summarize_bandpower_condition_means(spsl=spsl)

    expected_cols = {
        "subject",
        "probe",
        "structure",
        "condition",
        "mean_delta",
        "mean_log_delta",
        "mean_zlog_delta",
        "mean_eta",
        "mean_log_eta",
        "mean_zlog_eta",
        "clade",
        "AP.Coord",
        "Cx.AP.group",
    }
    assert set(means.columns) == expected_cols
    # One row per (sps, condition); the six core conditions.
    assert means["clade"].eq("Cx").all()
    assert means["mean_zlog_delta"].notna().all()
