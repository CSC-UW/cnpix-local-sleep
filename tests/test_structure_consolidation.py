"""Interim OFF-analysis structure consolidation (e.g. ``mPPC`` -> ``PPC``).

The consolidation is applied at OFF-analysis export boundaries only: it relabels
aliased structures and recomputes the atlas-derived per-structure columns
(``clade``/``AP.Coord``/``Cx.AP.group``) from the consolidated label, so a
relabeled row carries the *same* metadata as a native row of the target label and
the downstream ``struct_info`` merge does not fan out.

The atlas caches are pre-populated so these tests do not require the BrainGlobe
atlas download.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cnpix_local_sleep import atlas


@pytest.fixture(autouse=True)
def _seed_atlas_caches():
    """Populate the atlas caches with the PPC-family values used below.

    ``mPPC`` (Parietal association cortex, medial area) sits ~560 um anterior to
    ``PPC`` (Posterior parietal cortex); both fall in the ``cent-post`` A/P bin.
    """
    atlas._CLADE_CACHE.update({"mPPC": "Cx", "PPC": "Cx", "V2": "Cx"})
    # Keyed by (structure, method) -- see atlas.AP_CENTROID_METHODS.
    atlas._AP_COORD_CACHE.update(
        {
            ("mPPC", "mesh"): 18386.0,
            ("PPC", "mesh"): 18946.0,
            ("V2", "mesh"): 19500.0,
        }
    )
    yield


def _mixed_frame() -> pd.DataFrame:
    """A frame with a native-PPC row, an mPPC row (distinct AP.Coord), and V2."""
    return pd.DataFrame(
        {
            "subject": ["CNPIX2-Segundo", "CNPIX10-Charles", "CNPIX10-Charles"],
            "structure": ["PPC", "mPPC", "V2"],
            "clade": ["Cx", "Cx", "Cx"],
            "AP.Coord": [18946.0, 18386.0, 19500.0],
            "Cx.AP.group": ["cent-post", "cent-post", "cent-post"],
            "laminar_class": ["supra", "infra", "mixed"],
        }
    )


def test_relabels_alias_and_leaves_others():
    out = atlas.consolidate_off_structure_columns(_mixed_frame())
    assert "mPPC" not in set(out["structure"])
    assert set(out["structure"]) == {"PPC", "V2"}


def test_no_fanout_single_metadata_per_label():
    """After consolidation every label carries one (clade, AP.Coord, Cx.AP.group)."""
    out = atlas.consolidate_off_structure_columns(_mixed_frame())
    struct_info = out[["structure", "clade", "AP.Coord", "Cx.AP.group"]].drop_duplicates()
    # One row per structure -> merge(on="structure") cannot duplicate rows.
    assert struct_info["structure"].is_unique
    ppc = struct_info.loc[struct_info["structure"] == "PPC"].iloc[0]
    # The relabeled row now carries PPC's atlas coordinate, not mPPC's.
    assert ppc["AP.Coord"] == pytest.approx(18946.0)
    assert ppc["clade"] == "Cx"
    assert ppc["Cx.AP.group"] == "cent-post"


def test_preserves_per_off_columns():
    """Per-OFF results computed from the raw identity upstream are untouched."""
    frame = _mixed_frame()
    out = atlas.consolidate_off_structure_columns(frame)
    assert list(out["laminar_class"]) == list(frame["laminar_class"])


def test_idempotent():
    once = atlas.consolidate_off_structure_columns(_mixed_frame())
    twice = atlas.consolidate_off_structure_columns(once)
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


def test_noop_when_no_alias_present():
    frame = _mixed_frame()
    frame = frame.loc[frame["structure"] != "mPPC"].reset_index(drop=True)
    out = atlas.consolidate_off_structure_columns(frame)
    pd.testing.assert_frame_equal(out, frame)


def test_noop_on_empty_frame():
    empty = pd.DataFrame({"structure": pd.Series(dtype="object")})
    out = atlas.consolidate_off_structure_columns(empty)
    assert out.empty


def test_handles_categorical_structure_column():
    """A categorical ``structure`` (as produced during aggregation) is relabeled
    and drops the now-unused ``mPPC`` category level."""
    frame = _mixed_frame()
    frame["structure"] = frame["structure"].astype("category")
    out = atlas.consolidate_off_structure_columns(frame)
    assert "mPPC" not in set(out["structure"])
    if isinstance(out["structure"].dtype, pd.CategoricalDtype):
        assert "mPPC" not in list(out["structure"].cat.categories)


def test_recomputes_only_present_metadata_columns():
    """A frame lacking AP.Coord/Cx.AP.group is still relabeled without error."""
    frame = _mixed_frame()[["subject", "structure", "clade"]]
    out = atlas.consolidate_off_structure_columns(frame)
    assert set(out["structure"]) == {"PPC", "V2"}
    assert "AP.Coord" not in out.columns


def test_scalar_label_helper():
    assert atlas.consolidate_structure_label("mPPC") == "PPC"
    assert atlas.consolidate_structure_label("PPC") == "PPC"
    assert atlas.consolidate_structure_label("V2") == "V2"
