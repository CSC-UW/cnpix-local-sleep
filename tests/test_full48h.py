"""Tests for full-48h OFF analysis (timecourses, re-aggregation, intrusion sweep).

The substantive checks need mounted production data, so they are marked
``requires_nfs`` and gated behind ``RUN_NFS_TESTS=1`` like the other
production-data smoke tests in this package.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cnpix_local_sleep.morphological.mua.pipeline import full48h


def _skip_without_nfs() -> None:
    if os.environ.get("RUN_NFS_TESTS") != "1":
        pytest.skip("Set RUN_NFS_TESTS=1 to run production-data smoke test")


def test_intrusion_states_exclude_wake_and_artifact():
    # "All non-Wake except artifact": the gate must never admit Wake/Artifact/
    # NoData, and must admit the sleep-like + arousal states.
    assert "Wake" not in full48h.INTRUSION_STATES
    assert "Artifact" not in full48h.INTRUSION_STATES
    assert "NoData" not in full48h.INTRUSION_STATES
    assert set(full48h.INTRUSION_STATES) == {"NREM", "IS", "REM", "MA", "Other"}


def test_additive_metrics_registry_normalize_flags():
    # count/total_area are raw sums (no fixed-duration denominator); rate and
    # total_area_norm divide by the canonical window duration.
    assert full48h.ADDITIVE_METRICS["count"] == (full48h._COUNT, False)
    assert full48h.ADDITIVE_METRICS["rate"] == (full48h._COUNT, True)
    assert full48h.ADDITIVE_METRICS["total_area"] == ("area", False)
    assert full48h.ADDITIVE_METRICS["total_area_norm"] == ("area_rel2span", True)


def test_intrusion_sweep_rejects_non_additive_metric():
    # Validation happens before any data load, so this needs no NFS.
    with pytest.raises(ValueError, match="additive"):
        full48h.intrusion_sweep(
            "CNPIX8-Allan", "imec0", "M2", metric="median_span"
        )


def test_compute_intrusion_sweeps_rejects_non_additive_metric():
    with pytest.raises(ValueError, match="additive"):
        full48h.compute_intrusion_sweeps(metrics=("median_span",))


@pytest.mark.requires_nfs
def test_intrusion_sweep_t0_recovers_canonical_incline():
    _skip_without_nfs()
    sweep, intrusions = full48h.intrusion_sweep("CNPIX8-Allan", "imec0", "M2")

    base = sweep.loc[sweep["threshold_s"] == 0].iloc[0]
    # T=0 admits no intrusions, so nothing is added to either bucket.
    assert base["n_intrusions_le_thr"] == 0
    assert base["nod_intrusion_value"] == pytest.approx(0.0)

    # Admitting intrusions only ever ADDS OFF area, so the Early/Late burdens
    # are monotonically non-decreasing in the threshold.
    assert sweep["early_value"].is_monotonic_increasing
    assert sweep["late_value"].is_monotonic_increasing
    # The cumulative NOD intrusion contribution is non-decreasing too.
    assert sweep["nod_intrusion_value"].is_monotonic_increasing

    # Every intrusion bout is one of the configured intrusion states.
    assert set(intrusions["state"]).issubset(set(full48h.INTRUSION_STATES))


@pytest.mark.requires_nfs
def test_intrusion_sweep_count_metric_and_clas_filter():
    _skip_without_nfs()
    # count is a non-normalized additive metric (raw OFF counts), and clas is a
    # stricter filter than the default llas.
    sweep, intrusions = full48h.intrusion_sweep(
        "CNPIX8-Allan", "imec0", "M2", metric="count", filter_name="clas"
    )
    assert (sweep["metric"] == "count").all()
    assert (sweep["filter"] == "clas").all()
    # OFF counts are integer-valued sums and monotonically non-decreasing in T.
    assert sweep["early_value"].is_monotonic_increasing
    assert sweep["late_value"].is_monotonic_increasing
    # Each admitted bout contributes an integer number of OFFs.
    assert np.allclose(intrusions["value"], np.round(intrusions["value"]))


@pytest.mark.requires_nfs
def test_collect_all_offs_from_full_is_cortical_and_postprocessed():
    _skip_without_nfs()
    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs as agg

    offs = agg._collect_all_offs_from_full()
    assert not offs.empty
    # Cortex-only, and every OFF tagged with a core condition.
    assert set(offs["clade"].unique()) == {"Cx"}
    assert set(offs["condition"].dropna().unique()).issubset(
        set(np.asarray(agg.const.CORE_CONDITIONS))
    )
    # Postprocessing columns required by the aggregation are present.
    for col in agg._POSTPROCESSED_COLUMNS:
        assert col in offs.columns


# do_experiment_full writer (r-offp `full48h` export)


def test_do_experiment_full_orchestration(tmp_path, monkeypatch):
    """Routing + prefix + LLAS->CLAS->BLAS filter chain, no NFS.

    The real ``_collect_cortical_48h`` and ``_get_condition_durations`` need
    production data, so they are stubbed; the LAS filter chain runs for real and
    ``_save_filtered_category`` is stubbed to record how it was invoked.
    """
    import pandas as pd

    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs as agg

    # Only the columns the LAS filters touch (span, median_duration, duration,
    # span_rel2max). Counts by category: llas=5, clas=4, blas=3.
    cortical = pd.DataFrame(
        {
            "subject": ["s"] * 6,
            "probe": ["imec0"] * 6,
            "structure": ["M2"] * 6,
            "condition": ["Early.BSL.NREM"] * 6,
            "span": [50, 150, 250, 250, 250, 300],
            "median_duration": [0.01, 0.03, 0.06, 0.06, 0.06, 0.07],
            "duration": [0.01, 0.05, 0.10, 0.10, 0.10, 0.12],
            "span_rel2max": [0.5, 0.5, 0.5, 0.80, 0.90, 1.00],
        }
    )
    monkeypatch.setattr(agg, "_collect_cortical_48h", lambda: cortical.copy())

    fake_durs = pd.DataFrame(
        {"duration": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("s", "imec0", "Early.BSL.NREM")],
            names=["subject", "probe", "condition"],
        ),
    )
    monkeypatch.setattr(agg, "_get_condition_durations", lambda offs: fake_durs)

    calls = []

    def fake_save(name, offs, durs, grouped_boxcox=False, output_dir=None):
        calls.append((name, len(offs), output_dir))
        (output_dir / f"{name}_offs.parquet").touch()
        (output_dir / f"summarized_{name}_offs.parquet").touch()

    monkeypatch.setattr(agg, "_save_filtered_category", fake_save)

    agg.do_experiment_full(tmp_path)

    # Prefixed category names, in the LLAS -> CLAS -> BLAS order.
    assert [c[0] for c in calls] == [
        "full48h_llas",
        "full48h_clas",
        "full48h_blas",
    ]
    # Every category routed to output_dir (never NFS).
    assert all(c[2] == tmp_path for c in calls)
    # The filter chain strictly narrows.
    counts = {c[0]: c[1] for c in calls}
    assert counts["full48h_llas"] == 5
    assert counts["full48h_clas"] == 4
    assert counts["full48h_blas"] == 3
    # condition_durations + summarized artifacts written with the prefix.
    assert (tmp_path / "full48h_condition_durations.parquet").exists()
    for cat in ("llas", "clas", "blas"):
        assert (tmp_path / f"summarized_full48h_{cat}_offs.parquet").exists()


@pytest.mark.requires_nfs
def test_do_experiment_full_matches_in_memory_loaders(tmp_path):
    """Drift-lock: the written summarized parquets equal the in-memory loaders
    the homeostasis_plots notebook uses, so r-offp and the notebook cannot diverge.
    """
    _skip_without_nfs()
    import pandas as pd

    from cnpix_local_sleep.morphological.pipeline import aggregate_experiment_offs as agg

    agg.do_experiment_full(tmp_path)

    for ds in ("llas", "clas", "blas"):
        written = pd.read_parquet(
            tmp_path / f"summarized_full48h_{ds}_offs.parquet"
        )
        pd.testing.assert_frame_equal(
            written.reset_index(drop=True),
            agg.summarize_subset_of_48h_offs(ds).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )

        written_q = pd.read_parquet(
            tmp_path
            / f"summarized_full48h_{ds}_offs_by_min_trace_quartile.parquet"
        )
        pd.testing.assert_frame_equal(
            written_q.reset_index(drop=True),
            agg.summarize_subset_of_48h_offs_by_min_trace_quartile(
                ds
            ).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )

        # Full parity: event-level OFFs are written too.
        assert (tmp_path / f"full48h_{ds}_offs.parquet").exists()

    assert (tmp_path / "full48h_condition_durations.parquet").exists()
    # No laminar-class summaries (laminar is not built in the in-memory 48h path).
    assert not list(tmp_path.glob("*by_laminar_class*"))
