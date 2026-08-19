"""Tests for unit-based OFF detection plumbing (NFS-free).

Covers the schema mapping (on_off_df -> Off) and path construction. Detection
itself requires production data and is exercised in validation runs, not here.
"""

import pandas as pd

from cnpix_local_sleep.unit_based import files, loading
from cnpix_local_sleep.unit_based.pipeline import detect_full


# -------------------- detect_full: FR gate (hmmem-only) --------------------


def test_gate_algo_only_gates_hmmem():
    # sticky and threshold run at any FR; only hmmem is gated, falling back to
    # sticky (not threshold) below MIN_POOLED_FR.
    assert detect_full._gate_algo("sticky", 5.0) == "sticky"
    assert detect_full._gate_algo("threshold", 5.0) == "threshold"
    assert detect_full._gate_algo("hmmem", 5.0) == "sticky"
    assert detect_full._gate_algo("hmmem", 500.0) == "hmmem"


# -------------------- files.py --------------------


def test_full_offs_path_injects_method_and_detection_mode():
    p = str(files.get_full_offs_path("CNPIX12-Santiago", "imec0", "M2", "sticky"))
    assert "method=unit_based" in p
    assert "detection_mode=pooled-sticky" in p
    assert p.endswith("offs.parquet")


def test_detection_mode_helper():
    assert files.detection_mode("hmmem") == "pooled-hmmem"


# -------------------- loading.on_off_df_to_off_frame --------------------


def test_on_off_df_to_off_frame_schema_and_fills():
    off_rows = pd.DataFrame(
        {"start_time": [1.0, 5.0], "end_time": [1.2, 5.3], "duration": [0.2, 0.3]}
    )
    fr = loading.on_off_df_to_off_frame(
        off_rows, depth_lo=100.0, depth_hi=900.0, binsize=0.010
    )
    # All Off schema columns present, plus max_span.
    assert set(loading.OFF_COLUMNS).issubset(fr.columns)
    assert "max_span" in fr.columns
    # span == max_span == structure extent; span_rel2max would be 1.0.
    assert (fr["span"] == 800.0).all()
    assert (fr["max_span"] == 800.0).all()
    # area is a positive bin-count proxy.
    assert (fr["area"] >= 1).all()
    assert fr["area"].tolist() == [20, 30]
    # Morphology is NaN (no spatial propagation for pooled OFFs).
    assert fr["onset_slope"].isna().all()
    assert fr["offset_slope"].isna().all()


def test_on_off_df_to_off_frame_empty():
    empty = loading.on_off_df_to_off_frame(
        pd.DataFrame({"start_time": [], "end_time": [], "duration": []}),
        depth_lo=0.0,
        depth_hi=10.0,
    )
    assert len(empty) == 0
    assert "start_time" in empty.columns
    assert "max_span" in empty.columns
