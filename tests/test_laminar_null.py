"""Tests for the laminar-dominance mechanical null (:mod:`cnpix_local_sleep.morphological.laminar_null`).

The pure-array tests need no production data: they monkeypatch the two external
lookups inside :func:`cnpix_local_sleep.morphological.detect.add_laminar_areas` (``atlas.get_clade``
and ``readers.get_layer_borders``) so the real band-counting code runs against a
synthetic 20-channel probe. The zero-shift identity test against a real
``offs.parquet`` is gated behind ``RUN_NFS_TESTS=1`` like the other production
smoke tests in this package.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import cnpix_local_sleep.atlas
import cnpix_local_sleep.channel_anatomy
from cnpix_local_sleep.morphological import laminar_null as ln

# Synthetic probe: 20 channels, 20 um pitch, depths 0..380 um.
Y_COORDS = np.arange(20, dtype=float) * 20.0
STRUCTURE = "TESTSTRUCT"

# Explicit, deterministic layer borders (independent of the 45/45/10 logic):
#   infra = bottom half  [0, 180]   -> channel indices 0..9
#   supra = top half     [200, 380] -> channel indices 10..19
_BORDERS = pd.DataFrame(
    [
        {"acronym": STRUCTURE, "layer": "supra", "hi": 380.0, "lo": 200.0},
        {"acronym": STRUCTURE, "layer": "infra", "hi": 180.0, "lo": 0.0},
    ]
)


@pytest.fixture
def patched_bands(monkeypatch):
    """Make the real band code treat ``STRUCTURE`` as cortical with fixed borders."""
    monkeypatch.setattr(cnpix_local_sleep.atlas, "get_clade", lambda _s: "Cx")
    monkeypatch.setattr(
        cnpix_local_sleep.channel_anatomy,
        "get_layer_borders",
        lambda _subj, _probe, _struct: _BORDERS,
    )


@pytest.fixture
def footprints() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Three OFFs: supra-only, infra-only, and a straddling one."""
    return {
        1: (  # supra only: depths 300, 300, 320 -> 3 supra pixels
            np.array([0, 1, 0], dtype=np.int64),
            np.array([15, 15, 16], dtype=np.int64),
        ),
        2: (  # infra only: depths 40, 60 -> 2 infra pixels
            np.array([0, 1], dtype=np.int64),
            np.array([2, 3], dtype=np.int64),
        ),
        3: (  # straddle: depth 100 (infra) + depth 300 (supra)
            np.array([0, 0], dtype=np.int64),
            np.array([5, 15], dtype=np.int64),
        ),
    }


def test_laminar_areas_reuse_real_code(patched_bands, footprints):
    out = ln.laminar_areas_for_footprints(
        footprints, Y_COORDS, "subj", "probe", STRUCTURE
    ).set_index("label")
    assert out.loc[1, "supra_area"] == 3
    assert out.loc[1, "infra_area"] == 0
    assert out.loc[2, "supra_area"] == 0
    assert out.loc[2, "infra_area"] == 2
    assert out.loc[3, "supra_area"] == 1
    assert out.loc[3, "infra_area"] == 1


def test_center_of_mass_is_pixel_weighted(footprints):
    com = ln.center_of_mass_depths(footprints, Y_COORDS)
    # OFF 1 is pixel-weighted: mean(300, 300, 320), not mean(300, 320).
    assert com[1] == pytest.approx((300 + 300 + 320) / 3)
    assert com[2] == pytest.approx((40 + 60) / 2)
    assert com[3] == pytest.approx((100 + 300) / 2)


def test_measure_footprints_concentrations(patched_bands, footprints):
    out = ln.measure_footprints(
        footprints, Y_COORDS, "subj", "probe", STRUCTURE
    ).set_index("label")
    assert out.loc[1, "supra_concentration"] == pytest.approx(1.0)
    assert out.loc[2, "supra_concentration"] == pytest.approx(0.0)
    assert out.loc[3, "supra_concentration"] == pytest.approx(0.5)
    assert out.loc[3, "center_of_mass_depth"] == pytest.approx(200.0)


def test_shift_no_clip_preserves_pixel_count(footprints):
    rng = np.random.default_rng(0)
    shifted = ln._shift_footprints(footprints, Y_COORDS.size, rng, edge_clip=False)
    for label, (_t, chan) in shifted.items():
        assert chan.size == footprints[label][0].size  # same number of pixels
        assert chan.min() >= 0 and chan.max() < Y_COORDS.size


def test_shift_clip_never_grows_and_stays_in_range(footprints):
    rng = np.random.default_rng(1)
    for _ in range(50):
        shifted = ln._shift_footprints(
            footprints, Y_COORDS.size, rng, edge_clip=True
        )
        for label, (time, chan) in shifted.items():
            assert chan.size == time.size  # paired drop of time + channel
            assert chan.size <= footprints[label][1].size
            assert chan.size >= 1  # target center in range => never empty
            if chan.size:
                assert chan.min() >= 0 and chan.max() < Y_COORDS.size


def test_feasible_shift_preserves_all_pixels_in_bounds(footprints):
    rng = np.random.default_rng(0)
    for _ in range(50):
        shifted = ln._shift_footprints(
            footprints, Y_COORDS.size, rng, edge_clip=False, placement="feasible"
        )
        for label, (time, chan) in shifted.items():
            # No pixel is ever dropped (size preserved exactly)...
            assert chan.size == footprints[label][1].size
            assert time.size == footprints[label][0].size
            # ...and every channel stays in range (no clipping needed).
            assert chan.min() >= 0 and chan.max() < Y_COORDS.size
            # The footprint is rigidly translated: pairwise channel gaps preserved.
            np.testing.assert_array_equal(
                np.diff(chan), np.diff(footprints[label][1])
            )


def test_feasible_full_span_off_has_no_positional_freedom():
    # A footprint already spanning the whole probe has a single feasible
    # placement (delta == 0): it cannot be moved without losing a pixel.
    full = {0: (np.zeros(Y_COORDS.size, dtype=np.int64),
                np.arange(Y_COORDS.size, dtype=np.int64))}
    rng = np.random.default_rng(0)
    for _ in range(20):
        shifted = ln._shift_footprints(
            full, Y_COORDS.size, rng, edge_clip=False, placement="feasible"
        )
        np.testing.assert_array_equal(shifted[0][1], full[0][1])


def test_feasible_delta_bounds_match_in_bounds_range():
    # OFF spanning channels [5, 15] on a 20-channel probe: delta in [-5, 4].
    lo, hi = ln._feasible_delta_bounds(np.array([5]), np.array([15]), 20)
    assert lo[0] == -5 and hi[0] == 4
    # Full-span [0, 19] collapses to {0}.
    lo, hi = ln._feasible_delta_bounds(np.array([0]), np.array([19]), 20)
    assert lo[0] == 0 and hi[0] == 0


def test_draw_deltas_rejects_unknown_placement():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ln._draw_deltas("bogus", 20, np.array([5]), np.array([5]), rng)


def test_placement_delta_bounds_structure_window():
    # OFF spanning channels [5, 15] in a structure window [-10, 29] (the structure
    # extends 10 channels below and above the 20-channel detection probe):
    #   delta in [ceil(-10 - 5), floor(29 - 15)] = [-15, 14].
    lo, hi = ln._placement_delta_bounds(np.array([5]), np.array([15]), -10.0, 29.0)
    assert lo[0] == -15 and hi[0] == 14
    # In-detection window [0, 19] reproduces the feasible range [-5, 4].
    lo, hi = ln._placement_delta_bounds(np.array([5]), np.array([15]), 0.0, 19.0)
    assert lo[0] == -5 and hi[0] == 4


def test_uniform_struct_bounds_overhang_clips_and_can_vanish():
    # A structure window wider than the detection probe lets the uniform null place
    # an OFF partly (or wholly) outside the detection window; edge_clip then drops
    # the overhang, so the observed pixel count can SHRINK or hit zero.
    rng = np.random.default_rng(0)
    fp = {0: (np.zeros(3, dtype=np.int64), np.array([8, 9, 10], dtype=np.int64))}
    n = Y_COORDS.size  # 20
    sizes, sizes_clipped = set(), False
    for _ in range(400):
        shifted = ln._shift_footprints(
            fp, n, rng, edge_clip=True, placement="uniform",
            struct_bounds=(-15.0, float(n - 1 + 15)),
        )
        chan = shifted[0][1]
        if chan.size:
            assert chan.min() >= 0 and chan.max() < n  # observed part in-window
        sizes.add(int(chan.size))
        sizes_clipped |= chan.size < 3
    assert sizes_clipped and 0 in sizes  # both partial and fully-clipped occur
    # With no structure window (None) the uniform null stays in the detection
    # window, so nothing is ever clipped.
    rng2 = np.random.default_rng(0)
    for _ in range(50):
        shifted = ln._shift_footprints(fp, n, rng2, edge_clip=True, placement="uniform")
        assert shifted[0][1].size == 3


def test_null_measures_per_off_feasible_label_aligned(patched_bands, footprints, monkeypatch):
    import cnpix_local_sleep.sps_conf
    monkeypatch.setattr(cnpix_local_sleep.sps_conf, "get_flipped_laminar_combos", lambda: set())
    out = ln.null_measures_per_off(
        footprints, Y_COORDS, "s", "p", STRUCTURE, np.random.default_rng(0),
        placement="feasible",
    )
    assert list(out["label"]) == list(footprints)
    # Concentration is a fraction in [0, 1]; COM stays on the probe.
    assert out["supra_concentration"].between(0, 1).all()
    assert out["center_of_mass_depth"].between(Y_COORDS.min(), Y_COORDS.max()).all()


def test_structure_index_bounds_maps_anatomy_to_index(monkeypatch):
    # Detection probe: 20 channels, 20 um pitch, depths 0..380. The anatomical
    # structure spans 200 um (10 channels) below and above the detection window.
    structs = pd.DataFrame(
        [{"acronym": STRUCTURE, "lo": -200.0, "hi": 580.0}]
    )
    monkeypatch.setattr(
        cnpix_local_sleep.channel_anatomy, "load_structures",
        lambda _subj, _probe: structs,
    )
    lo, hi = ln.structure_index_bounds(Y_COORDS, "s", "p", STRUCTURE)
    # (lo - y0)/step = (-200)/20 = -10 ; (580)/20 = 29.
    assert lo == pytest.approx(-10.0)
    assert hi == pytest.approx(29.0)
    # A structure entirely inside the detection window is unioned up to [0, n-1].
    structs_narrow = pd.DataFrame([{"acronym": STRUCTURE, "lo": 100.0, "hi": 200.0}])
    monkeypatch.setattr(
        cnpix_local_sleep.channel_anatomy, "load_structures",
        lambda _subj, _probe: structs_narrow,
    )
    lo, hi = ln.structure_index_bounds(Y_COORDS, "s", "p", STRUCTURE)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(19.0)


def test_collapse_footprints_fields(footprints):
    c = ln.collapse_footprints(footprints)
    assert c.labels == list(footprints)
    # mins/maxs are the per-OFF channel extents...
    for i, (_t, chan) in enumerate(footprints.values()):
        assert c.mins[i] == chan.min()
        assert c.maxs[i] == chan.max()
    # ...and the rest agrees with the internal 5-tuple collapse.
    ch, w, sizes, means, seg = ln._collapse_footprints(footprints)
    np.testing.assert_array_equal(c.ch, ch)
    np.testing.assert_array_equal(c.w, w)
    np.testing.assert_array_equal(c.sizes, sizes)
    np.testing.assert_allclose(c.means, means)
    np.testing.assert_array_equal(c.seg, seg)


def test_collapsed_reuse_matches_internal_occupancy(footprints):
    # Passing a pre-built collapse must give bit-identical results to letting the
    # function collapse internally (collapse consumes no RNG, so the draw stream
    # is unchanged).
    c = ln.collapse_footprints(footprints)
    r_int = ln.occupancy_null_test(
        footprints, Y_COORDS, np.random.default_rng(3), n_perm=20
    )
    r_pre = ln.occupancy_null_test(
        footprints, Y_COORDS, np.random.default_rng(3), n_perm=20, collapsed=c
    )
    np.testing.assert_array_equal(r_int["obs_p"], r_pre["obs_p"])
    np.testing.assert_array_equal(r_int["null_mean"], r_pre["null_mean"])
    assert r_int["w1_um"] == r_pre["w1_um"]


def test_collapsed_reuse_matches_internal_per_off(patched_bands, footprints, monkeypatch):
    import cnpix_local_sleep.sps_conf
    monkeypatch.setattr(cnpix_local_sleep.sps_conf, "get_flipped_laminar_combos", lambda: set())
    c = ln.collapse_footprints(footprints)
    a = ln.null_measures_per_off(
        footprints, Y_COORDS, "s", "p", STRUCTURE, np.random.default_rng(1),
        placement="feasible",
    )
    b = ln.null_measures_per_off(
        footprints, Y_COORDS, "s", "p", STRUCTURE, np.random.default_rng(1),
        placement="feasible", collapsed=c,
    )
    pd.testing.assert_frame_equal(a, b)








def test_mechanical_attribution_identical_and_flat_null():
    rng = np.random.default_rng(3)
    emp = np.concatenate([np.zeros(2000), np.ones(2000)])  # spiky, far from flat
    # Null == empirical => the null reproduces it fully.
    res_same = ln.mechanical_attribution(emp, emp.copy(), support=(0.0, 1.0), rng=rng)
    assert res_same["attribution"] == pytest.approx(1.0, abs=1e-9)
    assert res_same["w_null"] == pytest.approx(0.0, abs=1e-9)
    assert res_same["resolvable"]
    # Null == uniform (the structureless baseline) => attribution ~ 0.
    flat_null = rng.uniform(0, 1, size=200_000)
    res_flat = ln.mechanical_attribution(emp, flat_null, support=(0.0, 1.0), rng=rng)
    assert abs(res_flat["attribution"]) < 0.05


def test_occupancy_null_test_detects_concentration():
    rng = np.random.default_rng(4)
    n_chans = 40
    y = np.arange(n_chans, dtype=float) * 20.0
    # Footprints all jammed onto the same few channels => occupancy is a spike
    # that uniform-depth placement cannot reproduce.
    lbl = {
        i: (np.zeros(3, dtype=np.int64), np.array([18, 19, 20], dtype=np.int64))
        for i in range(300)
    }
    res = ln.occupancy_null_test(lbl, y, rng, n_perm=100)
    assert res["n_off"] == 300
    assert res["w1_um"] > 0
    assert res["tv"] > 0.5            # large effect: obs far from null envelope
    assert res["p_global"] < 0.05     # and significant
    assert res["sig"].any()           # localized excess flagged
    # A genuinely uniform-depth process should sit near its own null envelope.
    rng2 = np.random.default_rng(5)
    centers = rng2.integers(2, n_chans - 2, size=4000)
    lbl_unif = {
        i: (np.zeros(3, dtype=np.int64),
            np.array([c - 1, c, c + 1], dtype=np.int64))
        for i, c in enumerate(centers)
    }
    res_u = ln.occupancy_null_test(lbl_unif, y, rng2, n_perm=100)
    assert res_u["tv"] < res["tv"]    # much smaller departure than the spike case


def test_occupancy_null_renormalized_to_unit_mass_under_deficit():
    # Whole-structure (uniform) placement over a structure 5x wider than the
    # detection window clips most placements out of the window (a large, f-dependent
    # partial-visibility deficit). The null must be renormalized to UNIT mass so the
    # test compares shape only, never sitting mechanically below the data.
    rng = np.random.default_rng(0)
    n = 20
    y = np.arange(n, dtype=float) * 20.0
    centers = rng.integers(0, n - 2, size=2000)
    lbl = {
        i: (np.zeros(3, dtype=np.int64), np.array([c, c + 1, c + 2], dtype=np.int64))
        for i, c in enumerate(centers)
    }
    # Structure window centered on the detection window but 5x wider.
    res = ln.occupancy_null_test(
        lbl, y, rng, n_perm=60, placement="uniform",
        struct_bounds=(-40.0, float(n - 1 + 40)),
    )
    assert res["null_mean"].sum() == pytest.approx(1.0, abs=1e-9)
    assert res["obs_p"].sum() == pytest.approx(1.0, abs=1e-9)
    # Uniform-random footprints vs a (symmetric) whole-structure null => both ~flat
    # within the window, so TV is a small shape distance, NOT the ~0.5*(1-f)~0.4
    # mechanical floor it would carry if the deficit were left in.
    assert res["tv"] < 0.2


def test_shifted_measures_zero_shift_matches_production(patched_bands, footprints):
    # At zero shift the vectorized band/COM arithmetic must reproduce the real
    # add_laminar_areas + center_of_mass_depths (the null-band ground truth).
    ch, w, sizes, means, seg = ln._collapse_footprints(footprints)
    supra_mask = (Y_COORDS >= 200.0) & (Y_COORDS <= 380.0)
    infra_mask = (Y_COORDS >= 0.0) & (Y_COORDS <= 180.0)
    delta = np.zeros(sizes.size, dtype=np.int32)
    conc, com = ln._shifted_measures(
        ch, w, sizes, seg, delta, Y_COORDS, supra_mask, infra_mask,
        flipped=False, edge_clip=True,
    )
    ref = ln.measure_footprints(footprints, Y_COORDS, "s", "p", STRUCTURE)
    ref = ref.set_index("label").loc[list(footprints)]
    np.testing.assert_allclose(conc, ref["supra_concentration"].to_numpy())
    np.testing.assert_allclose(com, ref["center_of_mass_depth"].to_numpy())


def test_null_measure_bands_shapes_and_density(patched_bands, footprints, monkeypatch):
    import cnpix_local_sleep.sps_conf
    monkeypatch.setattr(cnpix_local_sleep.sps_conf, "get_flipped_laminar_combos", lambda: set())
    # Uniform placement now needs the anatomical structure window; pin it to the
    # detection extent (no overhang/clipping) so this test stays about shapes.
    monkeypatch.setattr(
        cnpix_local_sleep.channel_anatomy, "load_structures",
        lambda _s, _p: pd.DataFrame(
            [{"acronym": STRUCTURE, "lo": float(Y_COORDS.min()),
              "hi": float(Y_COORDS.max())}]
        ),
    )
    conc_bins = np.linspace(0, 1, 11)
    depth_bins = np.linspace(0, 380, 9)
    out = ln.null_measure_bands(
        footprints, Y_COORDS, "s", "p", STRUCTURE, np.random.default_rng(0),
        n_reps=8, conc_bins=conc_bins, depth_bins=depth_bins,
    )
    assert out["conc_mean"].shape == (10,)
    assert out["com_mean"].shape == (8,)
    assert (out["conc_sd"] >= 0).all() and (out["com_sd"] >= 0).all()
    # densities integrate to ~1 over their bins
    assert out["conc_mean"].sum() * np.diff(conc_bins)[0] == pytest.approx(1.0, abs=1e-6)


def test_occupancy_count_weighting_differs_from_time():
    rng = np.random.default_rng(6)
    n_chans = 40
    y = np.arange(n_chans, dtype=float) * 20.0
    # One long, tall OFF (dominates time-occupancy) plus many short narrow OFFs
    # spread across depth (dominate count-occupancy). The two readouts must differ.
    lbl = {0: (np.tile(np.arange(50), 30), np.repeat(np.arange(5, 35), 50))}
    for i, c in enumerate(rng.integers(2, n_chans - 2, size=400), start=1):
        lbl[i] = (np.zeros(2, dtype=np.int64), np.array([c, c + 1], dtype=np.int64))
    res_t = ln.occupancy_null_test(lbl, y, np.random.default_rng(7),
                                   n_perm=50, weighting="time")
    res_c = ln.occupancy_null_test(lbl, y, np.random.default_rng(7),
                                   n_perm=50, weighting="count")
    assert res_t["weighting"] == "time" and res_c["weighting"] == "count"
    # Different readouts => different observed marginals.
    assert not np.allclose(res_t["obs_p"], res_c["obs_p"])
    with pytest.raises(ValueError):
        ln.occupancy_null_test(lbl, y, rng, n_perm=5, weighting="bogus")


def test_occupancy_asymmetry_sign_and_bounds(patched_bands):
    # Borders fixture: supra (superficial) = top half (channels 10-19, y>190).
    n = Y_COORDS.size
    flat = np.full(n, 1.0 / n)
    obs_sup = np.zeros(n)
    obs_sup[15:] = 1.0
    obs_sup /= obs_sup.sum()
    r = ln.occupancy_asymmetry(obs_sup, flat, Y_COORDS, "s", "p", STRUCTURE)
    assert r["asym"] > 0 and 0.0 < r["asym_norm"] <= 1.0   # excess toward surface
    obs_deep = np.zeros(n)
    obs_deep[:5] = 1.0
    obs_deep /= obs_deep.sum()
    r2 = ln.occupancy_asymmetry(obs_deep, flat, Y_COORDS, "s", "p", STRUCTURE)
    assert r2["asym"] < 0 and -1.0 <= r2["asym_norm"] < 0.0
    # Identical distributions => no asymmetry (and tv==0 => asym_norm is nan).
    r3 = ln.occupancy_asymmetry(flat, flat, Y_COORDS, "s", "p", STRUCTURE)
    assert r3["asym"] == pytest.approx(0.0)
    assert np.isnan(r3["asym_norm"])


def test_occupancy_asymmetry_flip_reverses_sign(patched_bands, monkeypatch):
    import cnpix_local_sleep.sps_conf
    n = Y_COORDS.size
    flat = np.full(n, 1.0 / n)
    obs_sup = np.zeros(n)
    obs_sup[15:] = 1.0
    obs_sup /= obs_sup.sum()
    # Not flipped: the high-y end is superficial => surface-heavy obs is positive.
    monkeypatch.setattr(cnpix_local_sleep.sps_conf, "get_flipped_laminar_combos", lambda: set())
    r = ln.occupancy_asymmetry(obs_sup, flat, Y_COORDS, "s", "p", STRUCTURE)
    # Flipped: the high-y end is now the DEEP end => same obs flips sign, same |asym|.
    monkeypatch.setattr(
        cnpix_local_sleep.sps_conf, "get_flipped_laminar_combos",
        lambda: {("s", "p", STRUCTURE)},
    )
    rf = ln.occupancy_asymmetry(obs_sup, flat, Y_COORDS, "s", "p", STRUCTURE)
    assert r["asym"] > 0 and rf["asym"] < 0
    assert rf["asym"] == pytest.approx(-r["asym"])


# Production-data identity check (zero shift must reproduce offs.parquet)
@pytest.mark.requires_nfs
def test_zero_shift_reproduces_parquet():
    if os.environ.get("RUN_NFS_TESTS") != "1":
        pytest.skip("Set RUN_NFS_TESTS=1 to run production-data identity test")

    from cnpix_local_sleep import sps_conf
    from cnpix_local_sleep.morphological import mua

    spsl = sps_conf.get_subject_probe_structure_list(
        method=mua.files.METHOD,
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
        exclude_nonlaminar=True,
    )
    combo = next(
        (
            (s, p, st)
            for (s, p, st) in spsl
            if mua.files.get_full_offs_path(s, p, st).exists()
            and mua.files.get_full_off_label_indices_path(s, p, st).exists()
        ),
        None,
    )
    if combo is None:
        pytest.skip("No full-48h morphological results available on this mount")
    subject, probe, structure = combo

    offs, lbl_ixs, y_coords = ln.load_structure_data(subject, probe, structure)
    measured = ln.measure_footprints(
        lbl_ixs, y_coords, subject, probe, structure
    ).set_index("label")

    # Compare on labels present in both the parquet and the footprint file.
    ref = offs.set_index("label")
    common = ref.index.intersection(measured.index)
    assert len(common) > 0
    ref = ref.loc[common]
    got = measured.loc[common]

    np.testing.assert_array_equal(
        got["supra_area"].to_numpy(), ref["supra_area"].to_numpy()
    )
    np.testing.assert_array_equal(
        got["infra_area"].to_numpy(), ref["infra_area"].to_numpy()
    )
    np.testing.assert_allclose(
        got["center_of_mass_depth"].to_numpy(),
        ref["center_of_mass_depth"].to_numpy(),
        rtol=0,
        atol=1e-6,
    )
