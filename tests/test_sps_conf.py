"""Tests for the per-method inclusion dispatch in ``cnpix_local_sleep.sps_conf``."""

from __future__ import annotations

import pandas as pd
import pytest

from cnpix_local_sleep import sps_conf


def test_load_method_inclusion_returns_expected_columns():
    df = sps_conf.load_method_inclusion("morphological")
    assert list(df.columns) == ["subject", "probe", "structure_acronym", "include"]
    assert df["include"].dtype == bool


def test_every_registered_method_has_inclusion_csv():
    for method in sps_conf.METHOD_INCLUSION_REGISTRY:
        df = sps_conf.load_method_inclusion(method)
        assert not df.empty, f"{method} inclusion CSV is empty"


def test_load_method_inclusion_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        sps_conf.load_method_inclusion("not-a-method")


def test_get_subject_probe_structure_list_requires_method_when_respecting_exclusions():
    with pytest.raises(ValueError, match="method= is required"):
        sps_conf.get_subject_probe_structure_list()


def test_respect_exclusions_false_does_not_require_method():
    # No method= passed, but respect_exclusions=False; should not raise.
    tuples = sps_conf.get_subject_probe_structure_list(respect_exclusions=False)
    assert len(tuples) > 0


def test_missing_row_is_treated_as_excluded(tmp_path, monkeypatch):
    """A (subject, probe, structure) absent from a method's CSV is excluded."""
    sps = sps_conf.load_config()

    # Drop one row from the morphological inclusion table; reload should treat
    # that tuple as excluded.
    full_inc = sps_conf.load_method_inclusion("morphological")
    truncated = full_inc.iloc[1:].copy()
    dropped = full_inc.iloc[0]
    truncated_csv = tmp_path / "quantile_thresholds.csv"
    truncated.to_csv(truncated_csv, index=False)

    def fake_load(method: str) -> pd.DataFrame:
        if method == "morphological":
            return truncated[["subject", "probe", "structure_acronym", "include"]]
        return sps_conf.load_method_inclusion.__wrapped__(method)  # type: ignore[attr-defined]

    monkeypatch.setattr(sps_conf, "load_method_inclusion", fake_load)

    spsl = sps_conf.get_subject_probe_structure_list(method="morphological")
    dropped_tuple = (
        dropped["subject"],
        dropped["probe"],
        dropped["structure_acronym"],
    )
    assert dropped_tuple not in spsl
    # Sanity: the row exists in sps_conf itself.
    sps_tuples = set(
        zip(sps["subject"], sps["probe"], sps["structure_acronym"])
    )
    assert dropped_tuple in sps_tuples


def test_methods_can_diverge():
    """Different methods can yield different inclusion sets."""
    mua = set(sps_conf.get_subject_probe_structure_list(method="morphological"))
    unit_based = set(sps_conf.get_subject_probe_structure_list(method="unit_based"))
    # Both pre-populated from the same sps_conf at migration time, but
    # the dispatch should still be calling the right CSVs (i.e. they're
    # at minimum readable and produce non-empty lists).
    assert len(mua) > 0
    assert len(unit_based) > 0


def test_sps_conf_no_longer_carries_include_column():
    """The `include` column moved out; sps_conf is now anatomy-only."""
    sps = sps_conf.load_config()
    assert "include" not in sps.columns


def test_load_quality_tiers_columns_and_values():
    tiers = sps_conf.load_quality_tiers("morphological")
    assert list(tiers.columns) == [
        "subject",
        "probe",
        "structure_acronym",
        "quality_tier",
    ]
    # Every non-blank tier is one of the known tiers.
    nonblank = tiers["quality_tier"].dropna()
    assert set(nonblank) <= set(sps_conf.QUALITY_TIER_ORDER)


def test_load_quality_tiers_rejects_method_without_column():
    # unit_based's CSV has no quality_tier column.
    with pytest.raises(ValueError, match="no quality_tier column"):
        sps_conf.load_quality_tiers("unit_based")


def test_get_analysis_spsl_cutoffs_are_nested_and_exclude_low_tiers():
    with_maybe = sps_conf.get_analysis_spsl(include_maybe_exclude=True)
    without_maybe = sps_conf.get_analysis_spsl(include_maybe_exclude=False)

    # Dropping maybe-exclude is a strict subset (the one-flag regeneration).
    assert set(without_maybe) < set(with_maybe)

    tiers = sps_conf.load_quality_tiers("morphological").set_index(
        ["subject", "probe", "structure_acronym"]
    )["quality_tier"]
    # The default set never contains probably/definitely-exclude combos.
    for sps in with_maybe:
        assert tiers.loc[sps] not in {"probably_exclude", "definitely_exclude"}
    # The tightened set additionally drops maybe-exclude.
    for sps in without_maybe:
        assert tiers.loc[sps] != "maybe_exclude"
    # Exactly the maybe-exclude combos are dropped between the two.
    dropped = set(with_maybe) - set(without_maybe)
    assert all(tiers.loc[sps] == "maybe_exclude" for sps in dropped)


def test_low_tier_excluded_from_analysis_but_plottable():
    # A probably-exclude combo with a populated threshold is plottable "for
    # curiosity" but must never enter advanced reporting.
    combo = ("CNPIX18-Pier", "imec1", "V1")
    tier = (
        sps_conf.load_quality_tiers("morphological")
        .set_index(["subject", "probe", "structure_acronym"])["quality_tier"]
        .loc[combo]
    )
    assert tier == "probably_exclude"
    assert combo not in sps_conf.get_analysis_spsl(include_maybe_exclude=True)
    assert combo in sps_conf.get_plottable_spsl()


def test_plottable_is_superset_of_analysis():
    plottable = set(sps_conf.get_plottable_spsl())
    analysis = set(sps_conf.get_analysis_spsl(include_maybe_exclude=True))
    assert analysis <= plottable


def test_load_config_includes_flip_supra_infra_bool_column():
    cfg = sps_conf.load_config()
    assert "flip_supra_infra" in cfg.columns
    assert cfg["flip_supra_infra"].dtype == bool


def test_get_flipped_laminar_combos_is_exactly_santiago_vo():
    # Brain curvature flips VO vertically vs the imec0 probe for this subject;
    # it is currently the only known flipped combo.
    assert sps_conf.get_flipped_laminar_combos() == {
        ("CNPIX12-Santiago", "imec0", "VO")
    }


def test_flipped_combos_are_a_subset_of_laminar_structures():
    # A flip only matters for laminar structures (supra/infra split).
    assert sps_conf.get_flipped_laminar_combos() <= set(
        sps_conf.get_laminar_structures()
    )
