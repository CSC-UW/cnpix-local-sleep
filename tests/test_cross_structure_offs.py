"""Tests for the OFF-source-aware cross-structure OFF pipeline.

The pure path/categorization checks need no data. The loader smoke tests need
mounted production data, so they are marked ``requires_nfs`` and gated behind
``RUN_NFS_TESTS=1`` like the other production-data smoke tests in this package.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from cnpix_local_sleep import const
from cnpix_local_sleep.morphological.pipeline import cross_structure_offs as csx


def _skip_without_nfs() -> None:
    if os.environ.get("RUN_NFS_TESTS") != "1":
        pytest.skip("Set RUN_NFS_TESTS=1 to run production-data smoke test")


# -------------------- OFF source vocabulary --------------------


def test_off_sources_and_default():
    assert csx.OFF_SOURCES == ("morphological-full48h", "morphological")
    assert csx.DEFAULT_OFF_SOURCE == "morphological-full48h"


def test_check_off_source_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown off_source"):
        csx.get_multi_cortical_subjects("not-a-source")


# Output-path scheme: the three sources must not collide


def test_output_dirs_are_distinct_and_encode_source():
    subject = "CNPIX12-Santiago"
    full = str(csx._get_output_dir(subject, "morphological-full48h"))
    mua = str(csx._get_output_dir(subject, "morphological"))

    # Whole-recording and per-condition OFFs share a ``method=`` segment, so
    # only ``off_source=`` keeps them from overwriting each other.
    assert full != mua

    assert "method=morphological" in full
    assert "analysis=cross_structure_offs" in full
    assert "off_source=full48h" in full

    assert "method=morphological" in mua
    assert "off_source=per_condition" in mua


def test_output_path_appends_filename():
    p = csx._get_output_path(
        "CNPIX12-Santiago", "overlap_counts.parquet", "morphological-full48h"
    )
    assert p.name == "overlap_counts.parquet"
    assert "off_source=full48h" in str(p)


# LAS categorization (BLAS > CLAS > LLAS by set membership)


def _row(start, end, struct="M2"):
    return {
        "subject": "S",
        "probe": "imec0",
        "structure": struct,
        "start_time": start,
        "end_time": end,
    }


def test_assign_las_category_most_restrictive():
    llas = pd.DataFrame([_row(0.0, 1.0), _row(2.0, 3.0), _row(4.0, 5.0)])
    clas = pd.DataFrame([_row(0.0, 1.0), _row(2.0, 3.0)])  # subset of llas
    blas = pd.DataFrame([_row(0.0, 1.0)])  # subset of clas

    cat = csx._assign_las_category(llas, clas, blas)
    assert list(cat) == ["BLAS", "CLAS", "LLAS"]
    assert list(cat.categories) == ["LLAS", "CLAS", "BLAS"]
    assert cat.ordered


# -------------------- Whole-recording loader is full-48h only --------------------


@pytest.mark.parametrize("off_source", ["morphological", "tom-bugnon"])
def test_load_whole_recording_offs_requires_full48h(off_source):
    with pytest.raises(ValueError, match="morphological-full48h"):
        csx.load_whole_recording_offs(off_source)


# -------------------- NFS-gated loader smoke tests --------------------


@pytest.mark.requires_nfs
def test_load_cross_structure_offs_full48h_multistructure():
    _skip_without_nfs()
    subject = "CNPIX12-Santiago"
    offs = csx.load_cross_structure_offs("morphological-full48h", subject=subject)

    assert not offs.empty
    assert (offs["subject"] == subject).all()
    assert offs["structure"].nunique() >= 2
    assert "category" in offs.columns
    assert set(offs["category"].dropna()).issubset({"LLAS", "CLAS", "BLAS"})
    assert set(offs["condition"]).issubset(set(const.CORE_CONDITIONS))


@pytest.mark.requires_nfs
def test_whole_recording_is_superset_of_condition_subset():
    _skip_without_nfs()
    subject = "CNPIX12-Santiago"
    subset = csx.load_cross_structure_offs("morphological-full48h", subject=subject)
    whole = csx.load_whole_recording_offs("morphological-full48h", subject=subject)

    # The whole recording covers every condition window plus the gaps between
    # them, so it can never have fewer OFFs than the condition-subset view.
    assert len(whole) >= len(subset)
    assert whole["structure"].nunique() >= 2
    assert (whole["condition"] == "Full48h").all()


# Windowed local-shift null and per-OFF excess globality (pure helpers)


def _rand_offs(rng, n, T=1000.0):
    starts = np.sort(rng.uniform(0, T, n))
    durs = rng.uniform(0.05, 2.0, n)
    return pd.DataFrame({"start_time": starts, "end_time": starts + durs})


def test_compute_overlap_degree_matches_compute_overlap_counts():
    # The vectorized degree used by the null must equal the canonical counts.
    rng = np.random.default_rng(0)
    structs = ["A", "B", "C"]
    cond = "X"
    offs_dict = {(s, cond): _rand_offs(rng, 1500) for s in structs}
    old = csx.compute_overlap_counts(offs_dict, structs, cond)
    new = csx.compute_overlap_degree(offs_dict, structs, cond)
    for s in structs:
        assert np.array_equal(old[s], new[s])


def test_overlaps_any_definition():
    # ref [10, 20]; partners chosen to probe the boundary conditions.
    ref_s = np.array([10.0])
    ref_e = np.array([20.0])
    # touching-at-end (no overlap), nested, touching-at-start (no overlap).
    o_s = np.array([5.0, 12.0, 20.0])
    o_e = np.array([10.0, 15.0, 25.0])
    out = csx._overlaps_any(ref_s, ref_e, o_s, o_e)
    assert out.tolist() == [True]  # the nested [12,15] overlaps
    # only the touching intervals -> no strict overlap.
    out2 = csx._overlaps_any(ref_s, ref_e, np.array([5.0, 20.0]), np.array([10.0, 25.0]))
    assert out2.tolist() == [False]


def test_assign_bouts_full_containment():
    bouts = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    starts = np.array([5.0, 11.0, 25.0, 29.9, 45.0, 49.9, 100.0])
    ends = np.array([6.0, 12.0, 26.0, 30.5, 46.0, 50.0, 101.0])
    out = csx._assign_bouts(starts, ends, bouts)
    # in-bout, gap, in-bout, crosses-end, in-bout, ends-at-edge, past-end.
    assert out.tolist() == [0, -1, 1, -1, 2, 2, -1]


def test_windowed_jitter_preserves_duration_and_stays_in_bout():
    rng = np.random.default_rng(1)
    bouts = np.array([[0.0, 100.0], [200.0, 300.0]])
    starts = np.array([10.0, 50.0, 210.0, 250.0])
    durs = np.array([1.0, 2.0, 0.5, 3.0])
    offs = pd.DataFrame({"start_time": starts, "end_time": starts + durs})
    bout_idx = csx._assign_bouts(starts, starts + durs, bouts)
    jit = csx.windowed_jitter_off_times(offs, bout_idx, bouts, window=60.0, rng=rng)
    # Duration preserved (multiset), and every OFF stays inside some bout.
    assert np.allclose(
        np.sort((jit["end_time"] - jit["start_time"]).to_numpy()), np.sort(durs)
    )
    js = jit["start_time"].to_numpy()
    je = jit["end_time"].to_numpy()
    inside = (
        (js >= bouts[:, 0][:, None]) & (je <= bouts[:, 1][:, None])
    ).any(axis=0)
    assert inside.all()
    # Start order is sorted.
    assert jit["start_time"].is_monotonic_increasing


def test_windowed_jitter_drops_out_of_domain():
    rng = np.random.default_rng(2)
    bouts = np.array([[0.0, 100.0]])
    offs = pd.DataFrame({"start_time": [10.0, 500.0], "end_time": [11.0, 501.0]})
    bout_idx = csx._assign_bouts(
        offs["start_time"].to_numpy(), offs["end_time"].to_numpy(), bouts
    )
    jit = csx.windowed_jitter_off_times(offs, bout_idx, bouts, window=30.0, rng=rng)
    assert len(jit) == 1  # the out-of-bout OFF is dropped


@pytest.mark.parametrize("off_source", ["morphological", "tom-bugnon"])
def test_excess_globality_requires_full48h(off_source):
    with pytest.raises(ValueError, match="morphological-full48h"):
        csx.do_subject_excess_globality("CNPIX12-Santiago", off_source=off_source)


def _excess_df_with_offset(rng, offset, n_subjects=9, n_structs=2, n_offs=200):
    """Synthetic excess_df where every subject's observed exceeds null by ~offset."""
    rows = []
    for si in range(n_subjects):
        base = rng.uniform(0.3, 0.7)  # subject-level null level
        for st in range(n_structs):
            null = rng.uniform(base, base + 0.1, n_offs)
            obs = null + offset + rng.normal(0, 0.05, n_offs)
            rows.append(
                pd.DataFrame(
                    {
                        "subject": f"S{si}",
                        "structure": f"st{st}",
                        "observed_degree": obs,
                        "null_mean": null,
                        "excess_globality": obs - null,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_excess_above_chance_detects_positive_offset():
    rng = np.random.default_rng(0)
    df = _excess_df_with_offset(rng, offset=0.2)
    out = csx.test_excess_above_chance(df)
    assert out["n_subjects"] == 9
    # Every subject is above null -> one-sided paired tests should be significant.
    assert out["paired_wilcoxon"]["p_value"] < 0.01
    assert out["paired_t"]["p_value"] < 0.01
    assert (out["per_subject"]["mean_excess"] > 0).all()
    # Mixed-model intercept positive and significant.
    assert out["mixedlm"].get("intercept", 0) > 0
    assert out["mixedlm"].get("p_value", 1.0) < 0.05


def test_excess_above_chance_null_offset_not_significant():
    rng = np.random.default_rng(1)
    df = _excess_df_with_offset(rng, offset=0.0)
    out = csx.test_excess_above_chance(df)
    # No real effect -> one-sided "greater" paired test should not be significant.
    assert out["paired_wilcoxon"]["p_value"] > 0.05


def test_excess_above_chance_requires_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        csx.test_excess_above_chance(pd.DataFrame({"subject": ["a"]}))


@pytest.mark.requires_nfs
def test_do_subject_excess_globality_smoke():
    _skip_without_nfs()
    res = csx.do_subject_excess_globality(
        "CNPIX12-Santiago",
        null_scope="whole_recording",
        window=60.0,
        n_shuffles=20,
    )
    assert not res.empty
    # excess = observed - null_mean, by construction.
    assert np.allclose(
        res["excess_globality"], res["observed_degree"] - res["null_mean"]
    )
    # Every scored OFF is in NREM domain; degree is bounded by #partners.
    assert (res["observed_degree"] >= 0).all()
    assert res["p_greater"].between(0.0, 1.0).all()
    assert {"span", "area", "duration"}.issubset(res.columns)
