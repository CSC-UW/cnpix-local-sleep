"""Tests for banded (spatially-resolved) unit-based OFF detection helpers.

NFS-free: exercises the pure helpers (Off-schema mapping with real spatial fields,
the adaptive per-band parameter builder) on synthetic data. The full
``detect_structure_banded`` path requires mounted production data and is covered
elsewhere / manually.
"""

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
from on_off_detection import SpatialOffModel

from cnpix_local_sleep.unit_based import banded as bmod
from cnpix_local_sleep.unit_based import banded_eval, loading
from cnpix_local_sleep.unit_based.banded import (
    _assemble_spatial_params,
    banded_on_off_df_to_off_frame,
    build_per_band_params,
)

STICKY_CAP0 = {
    "binsize": 0.010,
    "min_dwell": 0.050,
    "off_rate_max": 0.0,
    "n_iter_EM": 100,
    "tol": 1e-4,
    "min_off_duration": None,
}


def _fake_merged_off_df():
    """Mimic SpatialOffModel.run() output: intersection (core) + union (full) windows."""
    return pd.DataFrame(
        {
            "state": ["off", "off", "off"],
            "intersection_start_time": [10.0, 20.0, 30.0],
            "intersection_end_time": [10.2, 20.3, 30.1],
            "intersection_duration": [0.2, 0.3, 0.1],
            "union_start_time": [9.95, 19.9, 29.8],
            "union_end_time": [10.25, 20.4, 30.2],
            "union_duration": [0.30, 0.50, 0.40],
            "lo": [0.0, 200.0, 500.0],
            "hi": [1000.0, 400.0, 700.0],
            "span": [1000.0, 200.0, 200.0],
            "N_merged": [11, 3, 1],
            "merged_band_offs_indices": [[0, 1, 2], [3, 4], [5]],
        }
    )


def test_banded_off_frame_has_off_schema_columns():
    frame = banded_on_off_df_to_off_frame(_fake_merged_off_df())
    assert set(frame.columns) == set(loading.empty_off_frame().columns)


def test_banded_off_frame_real_spatial_fields():
    df = _fake_merged_off_df()
    frame = banded_on_off_df_to_off_frame(df)
    # lo/hi/span vary per OFF (the point of banded detection), not structure-constant.
    assert frame["span"].nunique() > 1
    np.testing.assert_allclose(sorted(frame["span"]), [200.0, 200.0, 1000.0])
    # center_of_mass_depth is the per-OFF midpoint.
    row = frame[frame["lo"] == 200.0].iloc[0]
    assert row["center_of_mass_depth"] == pytest.approx(300.0)
    # max_span is the per-structure max -> span_rel2max varies downstream.
    assert (frame["max_span"] == 1000.0).all()
    # morphology fields stay NaN (no per-channel traces from pooled-per-band).
    assert frame["onset_slope"].isna().all()
    assert frame["median_trace"].isna().all()
    # area proxy >= 1.
    assert (frame["area"] >= 1).all()


def test_banded_off_frame_uses_union_window():
    df = _fake_merged_off_df()
    frame = banded_on_off_df_to_off_frame(df)
    # The reported start/end/duration are the UNION (full extent), not the core.
    np.testing.assert_allclose(sorted(frame["start_time"]), [9.95, 19.9, 29.8])
    np.testing.assert_allclose(sorted(frame["end_time"]), [10.25, 20.4, 30.2])
    np.testing.assert_allclose(sorted(frame["duration"]), [0.30, 0.40, 0.50])


def test_banded_off_frame_empty():
    frame = banded_on_off_df_to_off_frame(pd.DataFrame())
    assert len(frame) == 0
    assert set(frame.columns) == set(loading.empty_off_frame().columns)


# -------------------- adaptive per-band params --------------------
def _small_model():
    rng = np.random.default_rng(0)
    n_units = 10
    depths = np.linspace(0.0, 1000.0, n_units)
    trains = [np.sort(rng.uniform(0, 60.0, size=2000)) for _ in range(n_units)]
    bouts = pd.DataFrame(
        {"start_time": [0.0], "end_time": [60.0], "duration": [60.0], "state": ["NREM"]}
    )
    return SpatialOffModel(
        trains, depths, bouts,
        cluster_ids=list(range(n_units)),
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [500.0, None]},
        verbose=False,
    )


def test_build_per_band_params_per_unit_scales_with_units():
    model = _small_model()
    per_band = build_per_band_params(model, STICKY_CAP0, scheme="per_unit")
    assert len(per_band) == len(model.bands_df)
    for p, (_, row) in zip(per_band, model.bands_df.iterrows()):
        n = len(row["band_cluster_indices"])
        assert p["off_rate_max"] == pytest.approx(0.0893 * n)
    # The whole-structure band (most units) gets the largest cap.
    caps = [p["off_rate_max"] for p in per_band]
    assert max(caps) > min(caps)


def test_build_per_band_params_cap0_is_silent():
    model = _small_model()
    per_band = build_per_band_params(model, STICKY_CAP0, scheme="cap0")
    assert all(p["off_rate_max"] == 0.0 for p in per_band)


def test_build_per_band_params_bad_scheme_raises():
    model = _small_model()
    with pytest.raises(ValueError):
        build_per_band_params(model, STICKY_CAP0, scheme="bogus")


# -------------------- union-of-band-boxes footprint --------------------
def test_off_frame_area_from_union_area():
    df = _fake_merged_off_df()
    df["union_area"] = [1000.0, 500.0, 200.0]  # s*um
    frame = banded_on_off_df_to_off_frame(df, binsize=0.01)
    # area = round(union_area / binsize), in start_time order (10, 20, 30).
    np.testing.assert_array_equal(
        frame["area"].to_numpy(),
        np.maximum(1, np.round(np.array([1000.0, 500.0, 200.0]) / 0.01)).astype(int),
    )


# duration-cleaning param plumbing (_assemble_spatial_params)
def _base_sp_kwargs(**over):
    kw = dict(
        band_definition="greedy_fr",
        band_sizes=[250.0, None],
        tile_start="superficial",
        min_band_off_duration=None,
        min_merged_off_duration=None,
        spatial_params=None,
    )
    kw.update(over)
    return kw


def test_assemble_spatial_params_defaults_omit_duration_keys():
    sp = _assemble_spatial_params(**_base_sp_kwargs())
    # When nothing is set, the engine defaults apply (keys not injected here).
    assert "min_band_off_duration" not in sp
    assert "min_merged_off_duration" not in sp
    assert sp["band_definition"] == "greedy_fr"


def test_assemble_spatial_params_pre_merge_only():
    sp = _assemble_spatial_params(**_base_sp_kwargs(min_band_off_duration=0.05))
    assert sp["min_band_off_duration"] == 0.05
    assert "min_merged_off_duration" not in sp


def test_assemble_spatial_params_post_merge_floor():
    sp = _assemble_spatial_params(**_base_sp_kwargs(min_merged_off_duration=0.10))
    assert sp["min_merged_off_duration"] == 0.10


def test_assemble_spatial_params_escape_hatch_wins():
    sp = _assemble_spatial_params(
        **_base_sp_kwargs(
            min_band_off_duration=0.05,
            spatial_params={"min_band_off_duration": 0.07, "band_overlap": 0.25},
        )
    )
    assert sp["min_band_off_duration"] == 0.07  # explicit spatial_params overrides
    assert sp["band_overlap"] == 0.25


def test_assemble_spatial_params_keys_are_engine_recognized():
    # Every key we emit must be a real SPATIAL_PARAMS key, else the engine setter raises.
    from on_off_detection.spatial_off import SPATIAL_PARAMS

    sp = _assemble_spatial_params(
        **_base_sp_kwargs(min_band_off_duration=0.05, min_merged_off_duration=0.10)
    )
    assert set(sp).issubset(set(SPATIAL_PARAMS))


# -------------------- diagnostic plotting helpers (NFS-free) --------------------
def test_structure_rows_is_deep_at_top():
    from cnpix_local_sleep.unit_based import banded_plots

    y_coords = np.arange(10) * 100.0  # ascending channel depths
    # structure = channels in [300, 700] um.
    row_mask = np.zeros(10, dtype=bool)
    chan = np.where((y_coords >= 300) & (y_coords <= 700))[0]
    row_mask[(10 - 1) - chan] = True
    r0, r1, depths = banded_plots._structure_rows(y_coords, row_mask)
    assert r1 > r0
    # depths descending: the first drawn row (r0) is the deepest -> rendered at top.
    assert depths[0] > depths[-1]
    assert depths[0] == pytest.approx(700.0)
    assert depths[-1] == pytest.approx(300.0)


def test_pick_windows_ranks_and_caps():
    from cnpix_local_sleep.unit_based import banded_plots

    n_chunks, n_rows, spc = 3, 8, 30
    presence = np.zeros((n_chunks, n_rows, spc), dtype=bool)
    # chunk 1 has the most activity, chunk 2 some, chunk 0 none.
    presence[1, 2:5, 0:20] = True
    presence[2, 2:5, 0:5] = True
    windows = banded_plots._pick_windows(
        presence, chunks=[0, 1, 2], r0=0, r1=n_rows, win=10, max_windows=2
    )
    assert len(windows) == 2
    # highest-activity window first; chunk 0 (empty) never selected.
    assert windows[0][0] == 1
    assert all(c != 0 for c, _, _ in windows)


def test_plot_window_writes_png(tmp_path):
    from cnpix_local_sleep.unit_based import banded_plots

    n_chunks, n_rows, spc = 1, 6, 50
    rng = np.random.default_rng(0)
    arrays = {
        k: (rng.random((n_chunks, n_rows, spc)) > 0.7).astype(np.int32)
        for k in ("manual", "banded", "mua")
    }
    arrays["trace"] = (rng.random((n_chunks, n_rows, spc)) * 255).astype(np.uint8)
    panels = [
        {"label": "manual", "key": "manual", "cmap": "Greens", "kind": "mask"},
        {"label": "banded", "key": "banded", "cmap": "Reds", "kind": "mask"},
        {"label": "morphological", "key": "mua", "cmap": "Blues", "kind": "mask"},
        {"label": "MUA image", "key": "trace", "cmap": "gray", "kind": "image"},
    ]
    depths = np.array([600.0, 500.0, 400.0, 300.0, 200.0, 100.0])  # descending (deep top)
    out = tmp_path / "w0.png"
    path = banded_plots.plot_window(
        arrays, panels, c=0, s0=0, s1=50, r0=0, r1=6, depths=depths, dt=0.1,
        title="test", t0_abs=12.3, save_path=str(out),
    )
    assert pathlib.Path(path).exists() and pathlib.Path(path).stat().st_size > 0


# production rollout drivers + persistence wiring (NFS-free, mocked)
def test_do_structure_banded_pins_rollout_config(monkeypatch):
    from cnpix_local_sleep.unit_based import banded

    captured = {}

    def fake_detect(subject, probe, structure, **kwargs):
        captured.update(kwargs)
        captured["spc"] = (subject, probe, structure)
        return (loading.empty_off_frame(), {})

    monkeypatch.setattr(banded, "detect_structure_banded", fake_detect)
    banded.do_structure_banded("CNPIXx", "imec0", "M2")

    # Persists, full 48h (no bouts_by_pass), and pins the validated M2 config.
    assert captured["persist"] is True
    assert captured.get("bouts_by_pass") is None
    for k, v in banded.ROLLOUT_CONFIG.items():
        assert captured[k] == v


def test_do_structure_banded_config_override(monkeypatch):
    from cnpix_local_sleep.unit_based import banded

    captured = {}
    monkeypatch.setattr(
        banded, "detect_structure_banded",
        lambda s, p, st, **kw: captured.update(kw) or (loading.empty_off_frame(), {}),
    )
    banded.do_structure_banded("S", "imec0", "M2", band_definition="fixed_tiled")
    assert captured["band_definition"] == "fixed_tiled"  # override wins
    assert captured["algo"] == "sticky"  # untouched rollout key remains


def test_do_experiment_banded_continues_on_error(monkeypatch):
    from cnpix_local_sleep import sps_conf
    from cnpix_local_sleep.unit_based import banded

    spsl = [("S1", "imec0", "A"), ("S1", "imec0", "BOOM"), ("S2", "imec1", "C")]
    monkeypatch.setattr(
        sps_conf, "get_subject_probe_structure_list", lambda **kw: spsl
    )
    done = []

    def fake_do(subject, probe, structure, **kw):
        if structure == "BOOM":
            raise RuntimeError("boom")
        done.append((subject, probe, structure))

    monkeypatch.setattr(banded, "do_structure_banded", fake_do)
    banded.do_experiment_banded()  # must not raise
    assert done == [("S1", "imec0", "A"), ("S2", "imec1", "C")]


def test_banded_experiment_dir_is_distinct_from_pooled():
    from cnpix_local_sleep.unit_based import files

    pooled = str(files.get_experiment_dir("sticky"))
    banded = str(files.get_experiment_dir("sticky", banded=True))
    assert "pooled-sticky" in pooled and "banded-sticky" in banded


# parallel sweep helpers (NFS-free; the pool path itself is integration-tested by
# the real run -- spawn re-imports defeat in-process monkeypatching)
def test_plan_jobs_clamps_by_resources():
    from cnpix_local_sleep.unit_based import banded

    # RAM-bound: 100 GB avail, 22 GB/job, 15% headroom -> floor(85/22) = 3
    jobs, note = banded._plan_jobs(50, 16, 1, mem_avail_gb=100, n_cpus=224)
    assert jobs == 3 and "mem_cap=3" in note
    # core-bound: 8 cpus, 4 threads/job, 15% headroom -> floor(6.8)//4 = 1
    jobs2, _ = banded._plan_jobs(50, 16, 4, mem_avail_gb=10000, n_cpus=8)
    assert jobs2 == 1
    # pending-bound: only 2 structures left
    jobs3, _ = banded._plan_jobs(2, 16, 1, mem_avail_gb=10000, n_cpus=224)
    assert jobs3 == 2
    # never below 1, even on a tiny host
    assert banded._plan_jobs(0, 16, 1, mem_avail_gb=1, n_cpus=1)[0] == 1
    # not inflated above the request
    assert banded._plan_jobs(100, 4, 1, mem_avail_gb=10000, n_cpus=224)[0] == 4


def test_worker_init_sets_thread_caps():
    from cnpix_local_sleep.unit_based import banded

    vars_ = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in vars_}
    try:
        banded._worker_init(3)
        assert all(os.environ[k] == "3" for k in vars_)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_detect_structure_banded_worker_ok_and_failed(monkeypatch):
    from cnpix_local_sleep.unit_based import banded

    calls = []
    monkeypatch.setattr(
        banded, "do_structure_banded",
        lambda s, p, st, **kw: calls.append((s, p, st, kw)),
    )
    spc = ("S", "imec0", "M2")
    assert banded._detect_structure_banded_worker(spc, False, {"algo": "sticky"}) == (
        spc, "ok", None
    )
    assert calls[0][3]["verbose"] is False  # workers stay quiet

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(banded, "do_structure_banded", boom)
    spc2, status, err = banded._detect_structure_banded_worker(spc, True, {})
    assert spc2 == spc and status == "failed" and "RuntimeError" in err


def test_pending_banded_structures_skips_existing(monkeypatch):
    from cnpix_local_sleep.unit_based import banded

    spsl = [("S", "imec0", "A"), ("S", "imec0", "B"), ("S", "imec1", "C")]
    done = {("S", "imec0", "A")}

    class _FakePath:
        def __init__(self, exists):
            self._exists = exists

        def exists(self):
            return self._exists

    monkeypatch.setattr(
        banded.files, "get_full_banded_offs_path",
        lambda s, p, st, a: _FakePath((s, p, st) in done),
    )
    assert banded._pending_banded_structures(spsl, "sticky", overwrite=False) == [
        ("S", "imec0", "B"), ("S", "imec1", "C")
    ]
    # overwrite -> redo everything
    assert banded._pending_banded_structures(spsl, "sticky", overwrite=True) == spsl


def test_rasterize_banded_union_is_the_l_not_the_bbox():
    from cnpix_local_sleep.evaluation import rasterize
    from cnpix_local_sleep.unit_based import banded_eval

    y = np.array([0, 100, 200, 300, 400, 500, 600], dtype=float)
    ts = np.arange(10) * 0.1
    shape = (1, 7, 10)
    # Two band-OFFs forming an L: A shallow+early, B deep+late, sharing a corner.
    all_bands = pd.DataFrame(
        {
            "state": ["off", "off"],
            "start_time": [0.0, 0.2],
            "end_time": [0.3, 0.5],
            "lo": [0.0, 300.0],
            "hi": [300.0, 600.0],
        }
    )  # index 0, 1
    off_df = pd.DataFrame({"merged_band_offs_indices": [[0, 1]]})

    union = banded_eval.rasterize_banded_union(off_df, all_bands, ts, y, shape)
    # Same merged event, but as a single bounding box [0,0.5] x [0,600]:
    bbox = rasterize.rasterize_offs(
        pd.DataFrame({"start_time": [0.0], "end_time": [0.5], "lo": [0.0], "hi": [600.0]}),
        ts, y, shape,
    )
    # The union excludes the empty L-corners that the bounding box fills.
    assert (union > 0).sum() < (bbox > 0).sum()
    # late + shallow corner: empty in the union, filled in the bbox.
    assert union[0, 5, 5] == 0 and bbox[0, 5, 5] > 0
    # the overlap region is covered in the union.
    assert union[0, 3, 2] > 0
    # one event => single non-zero label.
    assert set(np.unique(union)) <= {0, 1}


def test_row_mask_explicit_override_wins():
    """An explicit row_mask overrides depth_lo/hi and restrict_to_structure."""
    y = np.array([0, 100, 200, 300], dtype=float)
    manual = np.zeros((1, 4, 5))
    custom = np.array([True, False, True, False])
    out = banded_eval.resolve_row_mask(
        y, manual, depth_lo=0.0, depth_hi=300.0, restrict_to_structure=True,
        row_mask=custom,
    )
    np.testing.assert_array_equal(out, custom)


def test_row_mask_depth_range_when_no_override():
    y = np.array([0, 100, 200, 300], dtype=float)
    manual = np.zeros((1, 4, 5))
    out = banded_eval.resolve_row_mask(y, manual, 100.0, 200.0, True)
    # structure_row_mask flips channel indices; just assert it equals that helper.
    np.testing.assert_array_equal(out, banded_eval.structure_row_mask(y, 100.0, 200.0))


def test_row_mask_all_rows_when_unrestricted():
    y = np.array([0, 100, 200, 300], dtype=float)
    manual = np.zeros((1, 4, 5))
    out = banded_eval.resolve_row_mask(y, manual, None, None, False)
    assert out.all() and out.shape == (4,)


def test_row_mask_requires_depths_when_restricted():
    y = np.array([0, 100, 200, 300], dtype=float)
    manual = np.zeros((1, 4, 5))
    with pytest.raises(ValueError, match="depth_lo/depth_hi"):
        banded_eval.resolve_row_mask(y, manual, None, None, True)


def _fake_sglx_subject(name):
    return type("S", (), {"name": name})()


def test_load_structure_inputs_none_when_no_units(monkeypatch):
    """No units -> None (so detect_structure_banded can skip the structure)."""
    monkeypatch.setattr(bmod.wet, "get_sglx_subject", _fake_sglx_subject)
    monkeypatch.setattr(
        bmod.units, "load_structure_sorting",
        lambda *a, **k: type("Sorting", (), {"properties": pd.DataFrame()})(),
    )
    assert bmod.load_structure_inputs("SUBJ", "imec0", "M2") is None


def test_detect_banded_preloaded_skips_loader(monkeypatch):
    """With preloaded= given, detect_structure_banded must NOT call load_structure_inputs.

    Uses an empty pass (no bouts) so detection returns early without building a model
    (no real spike trains needed) -- the point is only that the loader is bypassed.
    """
    monkeypatch.setattr(bmod.wet, "get_sglx_subject", _fake_sglx_subject)

    def _boom(*a, **k):  # would fire if the loader were called
        raise AssertionError("load_structure_inputs called despite preloaded=")

    monkeypatch.setattr(bmod, "load_structure_inputs", _boom)
    empty_bouts = pd.DataFrame(
        columns=["start_time", "end_time", "duration", "state"]
    )
    off, infos = bmod.detect_structure_banded(
        "SUBJ", "imec0", "M2",
        preloaded={"trains": [], "depths": np.array([]), "cluster_ids": [], "hgs": {}},
        bouts_by_pass={"NREM": empty_bouts},
        verbose=False,
    )
    assert not len(off) and infos == {}
