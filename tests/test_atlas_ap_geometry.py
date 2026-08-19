"""A/P centroid estimators (:mod:`cnpix_local_sleep.atlas`).

``AP.Coord`` as materialized in every exported parquet is the *mesh* centroid.
The mesh default must not move: changing it would silently shift every existing
``AP.Coord`` consumer. ``method="volume"`` is the mass centroid of the annotation
volume, kept as a cross-check on the mesh estimate.

Tests that only exercise caching or validation seed the caches, so they do not
need the BrainGlobe atlas download; the ones that read the atlas are marked
``requires_atlas`` and skip when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from cnpix_local_sleep import atlas


@pytest.fixture
def _clear_caches():
    ap_coord = dict(atlas._AP_COORD_CACHE)
    ap_profile = dict(atlas._AP_PROFILE_CACHE)
    atlas._AP_COORD_CACHE.clear()
    atlas._AP_PROFILE_CACHE.clear()
    yield
    atlas._AP_COORD_CACHE.clear()
    atlas._AP_COORD_CACHE.update(ap_coord)
    atlas._AP_PROFILE_CACHE.clear()
    atlas._AP_PROFILE_CACHE.update(ap_profile)


def _atlas_available() -> bool:
    try:
        atlas.get_atlas()
    except Exception:
        return False
    return True


requires_atlas = pytest.mark.skipif(
    not _atlas_available(), reason="whs_sd_rat_39um atlas not downloaded"
)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown A/P centroid method"):
        atlas.get_anterior_posterior_axis_coord("M2", method="bogus")


def test_mesh_is_the_default_method(_clear_caches):
    """The default must stay "mesh" so exported AP.Coord values do not move."""
    atlas._AP_COORD_CACHE[("FAKE", "mesh")] = 1234.0
    atlas._AP_COORD_CACHE[("FAKE", "volume")] = 9999.0
    assert atlas.get_anterior_posterior_axis_coord("FAKE") == 1234.0


def test_cache_is_keyed_by_method(_clear_caches):
    atlas._AP_COORD_CACHE[("FAKE", "mesh")] = 1234.0
    atlas._AP_COORD_CACHE[("FAKE", "volume")] = 9999.0
    assert atlas.get_anterior_posterior_axis_coord("FAKE", method="mesh") == 1234.0
    assert atlas.get_anterior_posterior_axis_coord("FAKE", method="volume") == 9999.0


def test_volume_centroid_is_mass_weighted(_clear_caches, monkeypatch):
    """A profile with all its mass in one slice centroids onto that slice."""

    class _FakeAtlas:
        resolution = (39.0, 39.0, 39.0)

    monkeypatch.setattr(atlas, "get_atlas", lambda: _FakeAtlas())
    profile = np.zeros(100)
    profile[10] = 5.0
    atlas._AP_PROFILE_CACHE["FAKE"] = profile
    coord = atlas.get_anterior_posterior_axis_coord("FAKE", method="volume")
    assert coord == pytest.approx(10 * 39.0)


@requires_atlas
def test_mesh_and_volume_centroids_agree_on_cortical_structures():
    """The two estimators must not disagree materially, or the choice matters.

    They correlate at r ~ 0.997 across the OFF-analysis structures; the largest
    single disagreement is M2 (~800 um), the most A/P-elongated of them.
    """
    structures = ["MO", "VO", "PrL", "IL", "M2", "M1", "CLA", "Cg1", "PPC", "V2"]
    mesh = np.array(
        [atlas.get_anterior_posterior_axis_coord(s) for s in structures]
    )
    volume = np.array(
        [
            atlas.get_anterior_posterior_axis_coord(s, method="volume")
            for s in structures
        ]
    )
    assert np.corrcoef(mesh, volume)[0, 1] > 0.99
    assert np.abs(mesh - volume).max() < 1000.0


@requires_atlas
def test_descendant_bearing_structure_is_binarized():
    """V2's annotation is labelled V2M/V2L, not V2.

    ``get_structure_mask`` returns an *id-valued* mask, so failing to binarize
    would weight each voxel by its structure id and put V2's centroid nowhere
    near the structure.
    """
    profile = atlas.get_anterior_posterior_axis_profile("V2")
    assert profile.sum() < atlas.get_atlas().annotation.size
    occupied = profile.nonzero()[0]
    resolution = atlas.get_atlas().resolution[0]
    centroid = atlas.get_anterior_posterior_axis_coord("V2", method="volume")
    assert occupied[0] * resolution <= centroid <= occupied[-1] * resolution


@requires_atlas
def test_ap_coord_increases_posteriorly():
    """Sign convention: larger AP.Coord is more posterior.

    Anything that orders structures along A/P depends on this.
    """
    frontal = atlas.get_anterior_posterior_axis_coord("PrL")
    occipital = atlas.get_anterior_posterior_axis_coord("V2")
    assert frontal < occipital
    for method in atlas.AP_CENTROID_METHODS:
        assert atlas.get_anterior_posterior_axis_coord(
            "PrL", method=method
        ) < atlas.get_anterior_posterior_axis_coord("V2", method=method)
