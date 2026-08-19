"""Tests for the locality (Local vs Overlapping) export.

Pure aggregation checks need no data. The full ``export_locality_offs`` smoke
test needs mounted production data, so it is marked ``requires_nfs`` and gated
behind ``RUN_NFS_TESTS=1`` like the other production-data smoke tests.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from cnpix_local_sleep.morphological.pipeline import cross_structure_locality_export as loc


def _skip_without_nfs() -> None:
    if os.environ.get("RUN_NFS_TESTS") != "1":
        pytest.skip("Set RUN_NFS_TESTS=1 to run production-data smoke test")


# -------------------- Measure vocabulary --------------------


def test_measure_sources_and_states():
    assert loc._MEASURE_SOURCES == {
        "median_duration": "median_duration",
        "median_span": "span",
        "median_area": "area",
    }
    assert loc._STATES_KEEP == ("NREM", "Wake")
    assert loc._OVERLAP_STATUS == {True: "Local", False: "Overlapping"}


# Pure aggregation: one row per group; medians over the per-OFF source columns


def _off(subject, structure, condition, status, *, median_duration, span, area):
    return {
        "subject": subject,
        "probe": "imec0",
        "structure": structure,
        "condition": condition,
        "overlap_status": status,
        "median_duration": median_duration,
        "span": span,
        "area": area,
        "start_time": 0.0,
    }


def test_aggregate_measures_grain_and_medians():
    offs = pd.DataFrame(
        [
            _off("S", "M2", "C", "Local", median_duration=0.1, span=100.0, area=10.0),
            _off("S", "M2", "C", "Local", median_duration=0.3, span=300.0, area=30.0),
            _off("S", "M2", "C", "Overlapping", median_duration=0.5, span=500.0, area=50.0),
        ]
    )
    group_cols = ["subject", "probe", "structure", "condition", "overlap_status"]
    out = loc._aggregate_measures(offs, group_cols)

    # One row per (subject, probe, structure, condition, overlap_status).
    assert len(out) == 2
    assert not out.duplicated(subset=group_cols).any()
    assert set(out.columns) == set(
        group_cols + ["median_duration", "median_span", "median_area", "count", "clade"]
    )

    local = out[out["overlap_status"] == "Local"].iloc[0]
    assert local["median_duration"] == pytest.approx(0.2)  # median(0.1, 0.3)
    assert local["median_span"] == pytest.approx(200.0)
    assert local["median_area"] == pytest.approx(20.0)
    assert local["count"] == 2
    assert (out["clade"] == "Cx").all()


# -------------------- NFS-gated end-to-end smoke test --------------------


def test_summarize_whole_recording_requires_full48h():
    with pytest.raises(ValueError, match="morphological-full48h"):
        loc.summarize_whole_recording_locality("morphological")


@pytest.mark.requires_nfs
def test_export_locality_offs_writes_three_parquets(tmp_path):
    _skip_without_nfs()
    loc.export_locality_offs(tmp_path)

    overlap = pd.read_parquet(tmp_path / "summarized_locality_overlap_offs.parquet")
    assert {"subject", "structure", "condition", "mean_overlap_degree", "clade"} <= set(
        overlap.columns
    )
    assert not overlap.duplicated(subset=["subject", "structure", "condition"]).any()

    per_cond = pd.read_parquet(
        tmp_path / "summarized_locality_per_condition_llas_offs.parquet"
    )
    grain = ["subject", "probe", "structure", "condition", "overlap_status"]
    assert {"median_duration", "median_span", "median_area"} <= set(per_cond.columns)
    assert not per_cond.duplicated(subset=grain).any()
    assert set(per_cond["overlap_status"]) <= {"Local", "Overlapping"}

    whole = pd.read_parquet(
        tmp_path / "summarized_locality_full48h_llas_offs.parquet"
    )
    whole_grain = ["subject", "probe", "structure", "state", "overlap_status"]
    assert {"median_duration", "median_span", "median_area"} <= set(whole.columns)
    assert not whole.duplicated(subset=whole_grain).any()
    assert set(whole["state"]) <= {"NREM", "Wake"}
    assert (whole["condition"] == whole["state"]).all()
