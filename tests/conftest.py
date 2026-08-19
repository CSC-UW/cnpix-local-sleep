"""Shared pytest fixtures and collection gates for cnpix_local_sleep tests."""

from __future__ import annotations

import pytest

import ecephys.testing


@pytest.fixture(scope="module")
def np1_recording():
    """A small synthetic Neuropixels 1.0 recording (float32, 384ch, 2s)."""
    rec, _ = ecephys.testing.generate_neuropixels_recording(
        duration_s=2.0,
        dtype="float32",
    )
    return rec


@pytest.fixture(scope="module")
def np1_recording_int16():
    """A small synthetic Neuropixels 1.0 recording (int16, 384ch, 2s)."""
    rec, _ = ecephys.testing.generate_neuropixels_recording(
        duration_s=2.0,
        dtype="int16",
    )
    return rec


@pytest.fixture
def surrogate_zarr(tmp_path):
    """A small surrogate si_recording.zarr on disk (int16)."""
    zarr_path = tmp_path / "surrogate.si_recording.zarr"
    return ecephys.testing.save_surrogate_zarr(
        path=zarr_path,
        duration_s=2.0,
        dtype="int16",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``requires_release_data`` tests when the event-level tables are not local.

    The three ``full48h_*_offs.parquet`` tables are Release assets, not committed.
    :func:`cnpix_local_sleep.release_data.get_event_table_path` would download them,
    but a test run is not the place to pull 347 MB over the network -- so these tests
    skip unless the tables are already in ``r-offp/inst/extdata`` (any machine that
    has run the exporters) or in the download cache. No env var: unlike NFS, having
    the data is a question with a local answer.
    """
    from cnpix_local_sleep import release_data

    if all(release_data.is_available(d) for d in ("llas", "clas", "blas")):
        return
    skip = pytest.mark.skip(
        reason="event-level OFF tables are not local; get them with "
        "`python -c \'from cnpix_local_sleep import release_data as r; "
        "[r.get_event_table_path(d) for d in (\"llas\",\"clas\",\"blas\")]\'` "
        "(347 MB), or by running `off-analysis export-full48h-offs`"
    )
    for item in items:
        if "requires_release_data" in item.keywords:
            item.add_marker(skip)
