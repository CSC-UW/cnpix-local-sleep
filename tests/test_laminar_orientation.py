"""Per-combo supra/infra orientation correction in laminar classification.

A few (subject, probe, structure) combos have their supragranular/infragranular
order flipped vertically relative to the probe geometry (brain curvature; see
``sps_conf.get_flipped_laminar_combos``). ``supra_area``/``infra_area`` are stored
in geometric order, so the orientation correction must be applied at consumption:
``laminar_concentrations`` swaps the two for flipped combos, and everything that
derives supra/infra from the areas (the depth-profile null, the trimodality
notebook) routes through it.
"""

from __future__ import annotations

import pandas as pd

from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.morphological.pipeline import postprocess_offs as ppo

# A genuinely flipped combo and a non-flipped laminar combo, taken from config.
FLIPPED = ("CNPIX12-Santiago", "imec0", "VO")
NORMAL = ("CNPIX2-Segundo", "imec0", "PPC")


def _areas_frame(rows):
    """Build a minimal OFF frame with supra_area/infra_area (+ identity) columns."""
    return pd.DataFrame(rows)


def test_fixtures_match_config():
    flips = sps_conf.get_flipped_laminar_combos()
    assert FLIPPED in flips
    assert NORMAL not in flips


# -------------------- laminar_concentrations --------------------


def test_concentrations_scalar_path_unflipped_combo_unchanged():
    offs = _areas_frame([{"supra_area": 9, "infra_area": 1}])
    supra, infra = ppo.laminar_concentrations(
        offs, subject=NORMAL[0], probe=NORMAL[1], structure=NORMAL[2]
    )
    assert supra.iloc[0] == 0.9
    assert infra.iloc[0] == 0.1


def test_concentrations_scalar_path_flipped_combo_is_swapped():
    offs = _areas_frame([{"supra_area": 9, "infra_area": 1}])
    supra, infra = ppo.laminar_concentrations(
        offs, subject=FLIPPED[0], probe=FLIPPED[1], structure=FLIPPED[2]
    )
    # Geometric "supra" (top band) is really infragranular here, so the
    # supragranular concentration is the small one.
    assert supra.iloc[0] == 0.1
    assert infra.iloc[0] == 0.9


def test_concentrations_column_path_swaps_only_flipped_rows():
    offs = _areas_frame(
        [
            {**dict(zip(("subject", "probe", "structure"), FLIPPED)),
             "supra_area": 9, "infra_area": 1},
            {**dict(zip(("subject", "probe", "structure"), NORMAL)),
             "supra_area": 9, "infra_area": 1},
        ]
    )
    supra, infra = ppo.laminar_concentrations(offs)
    # Flipped row swapped; normal row unchanged.
    assert list(supra) == [0.1, 0.9]
    assert list(infra) == [0.9, 0.1]


def test_concentrations_do_not_mutate_area_columns():
    offs = _areas_frame([{"supra_area": 9, "infra_area": 1}])
    before = offs.copy()
    ppo.laminar_concentrations(
        offs, subject=FLIPPED[0], probe=FLIPPED[1], structure=FLIPPED[2]
    )
    pd.testing.assert_frame_equal(offs, before)
