"""Tests for the morphological per-bin threshold computation path."""

from __future__ import annotations

import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

from cnpix_local_sleep.morphological import detect as morphological_detect
from cnpix_local_sleep.morphological.mua.readers import bin_boundaries_from_chunks
from cnpix_local_sleep.morphological import morphology


def _make_da(
    values: np.ndarray, *, chunk_size: int | None = None
) -> xr.DataArray:
    n_time, n_channels = values.shape
    if chunk_size is None:
        chunk_size = n_time
    data = dask_array.from_array(values, chunks=(chunk_size, n_channels))
    return xr.DataArray(
        data,
        dims=("time", "channel"),
        coords={
            "channel": np.arange(n_channels),
            "y": ("channel", np.arange(n_channels) * 20.0),
        },
        attrs={"fs": 500.0},
    )


def test_bin_boundaries_from_chunks_uniform():
    da = _make_da(np.zeros((600, 4)), chunk_size=200)
    boundaries = bin_boundaries_from_chunks(da)
    np.testing.assert_array_equal(boundaries, np.array([0, 200, 400, 600]))


def test_bin_boundaries_from_chunks_nonuniform_tail():
    arr = np.zeros((550, 2))
    da = _make_da(arr, chunk_size=200)
    boundaries = bin_boundaries_from_chunks(da)
    np.testing.assert_array_equal(boundaries, np.array([0, 200, 400, 550]))


def test_bin_boundaries_from_chunks_after_compute_raises():
    da = _make_da(np.zeros((100, 2)), chunk_size=50).compute()
    with pytest.raises(TypeError, match="dask-backed"):
        bin_boundaries_from_chunks(da)


def test_per_bin_quantile_matches_local_distribution():
    rng = np.random.default_rng(42)
    n_bins = 4
    bin_size = 1000
    n_channels = 3
    bins = []
    for i in range(n_bins):
        # Each bin has a different mean to ensure per-bin quantile != whole.
        bins.append(rng.normal(loc=float(i), size=(bin_size, n_channels)))
    values = np.concatenate(bins, axis=0)
    da = _make_da(values, chunk_size=bin_size)
    boundaries = bin_boundaries_from_chunks(da)
    da_computed = da.compute()

    q = 0.2
    thresholds = morphological_detect.compute_per_bin_thresholds(
        da_computed,
        q,
        boundaries,
        threshold_method="from_value",
        ndimage_filter_type=None,
        ndimage_filter_kwargs=None,
    )

    assert thresholds.dims == ("bin", "channel")
    assert thresholds.shape == (n_bins, n_channels)

    for bi in range(n_bins):
        lo, hi = int(boundaries[bi]), int(boundaries[bi + 1])
        expected = np.quantile(values[lo:hi], q, axis=0)
        np.testing.assert_allclose(thresholds.values[bi], expected, atol=1e-9)


def test_per_bin_threshold_attrs_round_trip():
    values = np.arange(800 * 2, dtype=np.float64).reshape(800, 2)
    da = _make_da(values, chunk_size=400).compute()
    boundaries = np.array([0, 400, 800])

    thresholds = morphological_detect.compute_per_bin_thresholds(
        da,
        0.5,
        boundaries,
        threshold_method="from_value",
        ndimage_filter_type="median",
        ndimage_filter_kwargs={"size": [17, 1]},
    )

    assert thresholds.attrs["bin_boundaries"] == [0, 400, 800]
    assert thresholds.attrs["quantile"] == 0.5
    assert thresholds.attrs["threshold_method"] == "from_value"
    assert thresholds.attrs["ndimage_filter_type"] == "median"
    np.testing.assert_array_equal(
        thresholds.coords["bin_start_sample"].values, np.array([0, 400])
    )


def test_per_bin_threshold_with_derivation_mask_excludes_others():
    rng = np.random.default_rng(0)
    n_time = 800
    n_channels = 2
    values = rng.normal(loc=0.0, size=(n_time, n_channels))
    # Half the samples are flagged "ineligible" with very low values
    # that would dominate a quantile if they weren't masked out.
    derivation_mask = np.ones(n_time, dtype=bool)
    derivation_mask[::2] = False
    values[~derivation_mask] = -1000.0

    da = _make_da(values, chunk_size=200).compute()
    boundaries = np.array([0, 200, 400, 600, 800])

    thresholds = morphological_detect.compute_per_bin_thresholds(
        da,
        0.2,
        boundaries,
        derivation_mask=derivation_mask,
        threshold_method="from_value",
        ndimage_filter_type=None,
        ndimage_filter_kwargs=None,
    )

    # If the mask were ignored, every threshold would sit near -1000.
    # With the mask, thresholds reflect the eligible ~Gaussian samples.
    assert (thresholds.values > -10).all()


def test_empty_bins_filled_from_nearest_neighbor():
    rng = np.random.default_rng(1)
    n_time = 600
    values = rng.normal(size=(n_time, 1))
    derivation_mask = np.ones(n_time, dtype=bool)
    # Bin 1 has zero eligible samples.
    derivation_mask[200:400] = False

    da = _make_da(values, chunk_size=200).compute()
    boundaries = np.array([0, 200, 400, 600])

    thresholds = morphological_detect.compute_per_bin_thresholds(
        da,
        0.5,
        boundaries,
        derivation_mask=derivation_mask,
        threshold_method="from_value",
        ndimage_filter_type=None,
        ndimage_filter_kwargs=None,
    )

    # Bin 1 should equal one of its non-empty neighbors (bin 0 or bin 2).
    bin_0 = thresholds.values[0]
    bin_1 = thresholds.values[1]
    bin_2 = thresholds.values[2]
    assert np.isclose(bin_1, bin_0).all() or np.isclose(bin_1, bin_2).all()
    assert not np.isnan(thresholds.values).any()


def test_below_threshold_mask_per_bin_matches_explicit_loop():
    rng = np.random.default_rng(7)
    n_time = 500
    n_channels = 4
    n_bins = 5
    bin_size = n_time // n_bins
    values = rng.normal(size=(n_time, n_channels))
    boundaries = np.array(
        [i * bin_size for i in range(n_bins + 1)], dtype=np.int64
    )
    thresholds_arr = rng.normal(size=(n_bins, n_channels))
    thresholds_da = xr.DataArray(
        thresholds_arr,
        dims=("bin", "channel"),
        coords={
            "channel": np.arange(n_channels),
            "y": ("channel", np.arange(n_channels) * 20.0),
            "bin_start_sample": ("bin", boundaries[:-1]),
        },
        attrs={"bin_boundaries": boundaries.tolist()},
    )

    expected = np.empty(values.shape, dtype=bool)
    for bi in range(n_bins):
        lo, hi = int(boundaries[bi]), int(boundaries[bi + 1])
        expected[lo:hi] = values[lo:hi] < thresholds_arr[bi]

    actual = morphology._below_threshold_mask(values, thresholds_da)

    np.testing.assert_array_equal(actual, expected)


def test_below_threshold_mask_scalar_path_unchanged():
    """Tom-Bugnon (scalar per channel) keeps working through the same helper."""
    rng = np.random.default_rng(9)
    values = rng.normal(size=(20, 3))
    thresholds = xr.DataArray(
        np.array([0.0, 0.5, -0.5]),
        dims=("channel",),
        coords={"channel": np.arange(3)},
    )

    actual = morphology._below_threshold_mask(values, thresholds)
    expected = values < thresholds.values

    np.testing.assert_array_equal(actual, expected)


def test_compute_per_bin_thresholds_rejects_misaligned_boundaries():
    da = _make_da(np.zeros((100, 2)), chunk_size=50).compute()
    with pytest.raises(ValueError, match="bin_boundaries"):
        morphological_detect.compute_per_bin_thresholds(
            da,
            0.5,
            np.array([0, 50, 99]),  # wrong end
            threshold_method="from_value",
            ndimage_filter_type=None,
            ndimage_filter_kwargs=None,
        )


def test_below_threshold_mask_rejects_misaligned_boundaries():
    values = np.zeros((100, 2))
    thresholds = xr.DataArray(
        np.zeros((2, 2)),
        dims=("bin", "channel"),
        coords={"channel": np.arange(2)},
        attrs={"bin_boundaries": [0, 50, 99]},  # wrong end
    )
    with pytest.raises(ValueError, match="bin_boundaries"):
        morphology._below_threshold_mask(values, thresholds)
