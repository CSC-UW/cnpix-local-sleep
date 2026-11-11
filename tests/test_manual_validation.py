"""NFS-free unit tests for the manual-validation plumbing.

Covers the SPOT LAS filter (:func:`cnpix_local_sleep.off_tables.filter_offs`), the
label-index normalizer, and the full-recording MUA timebase accessor that the
true-mask path depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cnpix_local_sleep.morphological import manual_validation as mv
from cnpix_local_sleep import off_tables, trace_io


# off_tables.filter_offs (SPOT LAS filter)
def _offs_frame():
    # rows chosen to straddle the llas/clas/blas thresholds
    return pd.DataFrame(
        {
            "span": [50.0, 120.0, 250.0, 300.0],       # llas>=100, clas>=200
            "duration": [0.10, 0.10, 0.10, 0.10],       # all pass duration floors
            "median_duration": [0.10, 0.10, 0.10, 0.10],
            "max_span": [400.0, 400.0, 400.0, 400.0],   # span_rel2max = span/max_span
        }
    )


def test_filter_offs_llas_clas_thresholds():
    offs = _offs_frame()
    assert list(off_tables.filter_offs(offs, "llas")["span"]) == [120.0, 250.0, 300.0]
    assert list(off_tables.filter_offs(offs, "clas")["span"]) == [250.0, 300.0]


def test_filter_offs_blas_thresholds_on_derived_span_rel2max():
    offs = _offs_frame()  # no span_rel2max column present
    out = off_tables.filter_offs(offs, "blas")
    # blas = clas (span>=200) AND span_rel2max in [0.75, 1.0]; 250/400=0.625 fails, 300/400=0.75 passes
    assert list(out["span"]) == [300.0]
    # The derived span_rel2max lives in off_filter_mask and is deliberately NOT
    # written back: filter_offs must not add columns to the caller's schema.
    assert "span_rel2max" not in out.columns


def test_filter_offs_none_is_passthrough():
    offs = _offs_frame()
    out = off_tables.filter_offs(offs, None)
    assert len(out) == len(offs)
    assert list(out.index) == list(range(len(offs)))  # index reset


def test_filter_offs_matches_prior_inline_behavior():
    """filter_offs reproduces the inline column-threshold mask."""
    offs = _offs_frame().assign(span_rel2max=lambda d: d["span"] / d["max_span"])
    for name in ("llas", "clas", "blas"):
        filters = off_tables.NAMED_FILTERS[name]
        mask = pd.Series(True, index=offs.index)
        for col, (lo, hi) in filters.items():
            mask &= offs[col].between(lo, hi)
        expected = offs.loc[mask].reset_index(drop=True)
        pd.testing.assert_frame_equal(off_tables.filter_offs(offs, name), expected)


# manual_validation._normalize_label_indices (0-d scalar / NaN robustness)
def test_normalize_label_indices_drops_nonfinite_and_pairs():
    offs = pd.DataFrame(
        {
            "label": [1, 2, 3],
            # row 1: clean; row 2: a NaN pixel to drop; row 3: a 0-d scalar
            "time_ixs": [
                np.array([10, 11, 12]),
                np.array([5.0, np.nan, 7.0]),
                np.array(3),
            ],
            "chan_ixs": [
                np.array([0, 1, 2]),
                np.array([0.0, 1.0, 2.0]),
                np.array(4),
            ],
        }
    )
    out = mv._normalize_label_indices(offs)
    # row 1 unchanged (int), row 2 drops the NaN pixel, row 3 becomes a length-1 int array
    assert out["time_ixs"].iloc[0].dtype.kind == "i"
    np.testing.assert_array_equal(out["time_ixs"].iloc[0], [10, 11, 12])
    np.testing.assert_array_equal(out["time_ixs"].iloc[1], [5, 7])
    np.testing.assert_array_equal(out["chan_ixs"].iloc[1], [0, 2])
    np.testing.assert_array_equal(out["time_ixs"].iloc[2], [3])
    np.testing.assert_array_equal(out["chan_ixs"].iloc[2], [4])


# manual_validation._get_mua_full_times (full-recording, unmasked)
def test_get_mua_full_times_is_unmasked(monkeypatch):
    times = np.arange(100.0)

    class _FakeDA:
        time = type("T", (), {"values": times})()

    monkeypatch.setattr(
        "cnpix.mua.files.get_mua_traces_path", lambda s, p: "/fake/path"
    )
    monkeypatch.setattr(
        trace_io, "open_si_zarr_recording_as_xarray", lambda path: _FakeDA()
    )
    mv._get_mua_full_times.cache_clear()
    out = mv._get_mua_full_times("SUBJ", "imec0")
    np.testing.assert_array_equal(out, times)  # no condition mask applied
    mv._get_mua_full_times.cache_clear()



# manual_validation._load_translated_mua_offs (grid overlap + gap-aware pixel mapping)
def test_load_translated_mua_offs_keeps_grid_overlap_and_drops_gap_pixels(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    # Stack grid: two bouts, 0-4 s and 10-14 s, 1 s apart; MUA every 0.5 s.
    ts_flat = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    mua_times = np.arange(0.0, 30.0, 0.5)
    ix = lambda *t: (np.asarray(t) / 0.5).astype(int)
    offs = pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5],
            # 1 inside; 2 starts in the gap; 3 wholly in the gap; 4 runs past the
            # bout end; 5 after the grid
            "start_time": [1.0, 7.0, 5.0, 4.0, 20.0],
            "end_time": [3.0, 11.0, 8.0, 6.0, 25.0],
            "span": [300.0] * 5,
            "duration": [0.1] * 5,
            "median_duration": [0.1] * 5,
            "max_span": [400.0] * 5,
        }
    )
    lbls = pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5],
            "time_ixs": [ix(1, 2, 3), ix(7, 10, 11), ix(5, 8), ix(4, 5, 6), ix(20, 25)],
            "chan_ixs": [[0, 1, 2], [7, 8, 9], [0, 0], [3, 4, 5], [0, 0]],
        }
    )
    offs.to_parquet(tmp_path / "offs.parquet")
    lbls.to_parquet(tmp_path / "lbls.parquet")
    fm = SimpleNamespace(
        get_full_offs_path=lambda s, p, st: tmp_path / "offs.parquet",
        get_full_off_label_indices_path=lambda s, p, st: tmp_path / "lbls.parquet",
    )
    monkeypatch.setattr(mv, "_get_mua_full_times", lambda s, p: mua_times)

    out = mv._load_translated_mua_offs(
        "SUBJ", "imec0", "M2", "llas",
        source_config=SimpleNamespace(files_module=fm),
        stack_times_flat=ts_flat,
    ).set_index("label")

    assert list(out.index) == [1, 2, 4]
    np.testing.assert_array_equal(out.loc[1, "time_ixs"], [1, 2, 3])
    np.testing.assert_array_equal(out.loc[2, "time_ixs"], [5, 6])  # 7 s pixel dropped
    np.testing.assert_array_equal(out.loc[2, "chan_ixs"], [8, 9])
    np.testing.assert_array_equal(out.loc[4, "time_ixs"], [4])  # tail not moved to 10 s
    np.testing.assert_array_equal(out.loc[4, "chan_ixs"], [3])
