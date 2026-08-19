"""Tests for the additive excess-globality exporter.

The aggregation helper is pure (no NFS); the exporter smoke test needs mounted
production data, so it is marked ``requires_nfs`` and gated behind
``RUN_NFS_TESTS=1`` like the other production-data smoke tests in this package.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from cnpix_local_sleep.morphological.pipeline import cross_structure_excess_export as cex


def _skip_without_nfs() -> None:
    if os.environ.get("RUN_NFS_TESTS") != "1":
        pytest.skip("Set RUN_NFS_TESTS=1 to run production-data smoke test")


# Pure aggregation: per-OFF excess frame -> long (subject, structure, quantity)


def _per_off(subject, structure, observed, null):
    return pd.DataFrame(
        {
            "subject": subject,
            "structure": structure,
            "observed_degree": np.asarray(observed, dtype=float),
            "null_mean": np.asarray(null, dtype=float),
        }
    )


def test_aggregate_long_shape_and_levels():
    per_off = pd.concat(
        [
            _per_off("S1", "M2", [1.0, 2.0, 3.0], [0.5, 1.0, 1.5]),
            _per_off("S1", "PPC", [0.0, 2.0], [0.0, 1.0]),
            _per_off("S2", "M2", [4.0], [2.0]),
        ],
        ignore_index=True,
    )
    long = cex._aggregate_long(per_off)

    # Exactly two rows per (subject, structure) cell.
    assert len(long) == 3 * 2
    assert set(long["quantity"]) == {"observed", "null"}
    counts = long.groupby(["subject", "structure"]).size()
    assert (counts == 2).all()
    # Provenance/identity columns the R loader relies on.
    assert (long["clade"] == "Cx").all()
    assert (long["condition"] == "NREM").all()


def test_aggregate_long_values_are_cell_means():
    per_off = _per_off("S1", "M2", [1.0, 2.0, 3.0], [0.5, 1.0, 1.5])
    long = cex._aggregate_long(per_off)

    obs = long.loc[long["quantity"] == "observed", "value"].iloc[0]
    null = long.loc[long["quantity"] == "null", "value"].iloc[0]
    count = long["count"].iloc[0]

    assert obs == pytest.approx(2.0)  # mean([1,2,3])
    assert null == pytest.approx(1.0)  # mean([0.5,1,1.5])
    # Excess = observed - null is recoverable from the two rows.
    assert obs - null == pytest.approx(1.0)
    assert count == 3


def test_aggregate_long_count_is_scored_offs_per_cell():
    per_off = pd.concat(
        [
            _per_off("S1", "M2", [1.0, 2.0], [0.0, 0.0]),
            _per_off("S1", "PPC", [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]),
        ],
        ignore_index=True,
    )
    long = cex._aggregate_long(per_off)
    counts = (
        long.drop_duplicates(["subject", "structure"])
        .set_index("structure")["count"]
        .to_dict()
    )
    assert counts == {"M2": 2, "PPC": 4}


# -------------------- NFS-gated exporter smoke test --------------------


@pytest.mark.requires_nfs
def test_export_excess_globality_offs_writes_parquet(tmp_path):
    _skip_without_nfs()
    cex.export_excess_globality_offs(tmp_path, window=60.0, n_shuffles=20)
    out = tmp_path / "summarized_excess_globality_offs.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert not df.empty
    assert set(df["quantity"]) == {"observed", "null"}
    assert {"subject", "structure", "value", "count", "clade", "condition"}.issubset(
        df.columns
    )
    # Two rows per (subject, structure) cell.
    counts = df.groupby(["subject", "structure"]).size()
    assert (counts == 2).all()
