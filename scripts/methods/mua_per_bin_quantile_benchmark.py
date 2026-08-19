# %% [markdown]
# # Per-bin quantile benchmark
#
# Goal: time the per-bin Q20 computation for all (bin × channel) pairs on CNPIX2-Segundo imec0 PPC
# under three quantile primitives, for direct comparison:
#
# 1. numpy: `np.quantile(bin_vals, 0.20, axis=0)`: one vectorized call per bin, over (n_state, K).
# 2. dask default: `dask_array.percentile(ch_vals, [20.0], internal_method="default")`: per
#    channel per bin; "default" is the sort-based path.
# 3. dask tdigest: same as above but `internal_method="tdigest"`: approximate.
#
# Each method computes the same output: a `(n_bins, n_channels)` array of Q20 values. Results are
# cross-verified for numerical agreement (numpy ~ dask-default exact; tdigest within a small tolerance).

# %%
from __future__ import annotations

import time
import warnings
from pathlib import Path

import dask.array as dask_array
import dask_image.ndfilters as dndf
import numpy as np
import xarray as xr

from cnpix_local_sleep import hyp
from cnpix.mua import files as mua_files
from cnpix_local_sleep import channel_anatomy
from cnpix_local_sleep import trace_io

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
SPS = ("CNPIX2-Segundo", "imec0", "PPC")
STATE = "nrem"  # NREM-only for the benchmark (wake would just double wall time)
NREM_STATES = ("NREM", "IS")
GLOBAL_QUANTILE = 0.20
MIN_BIN_STATE_FRACTION = 0.05
GAUSSIAN_FREQ_MAX = 20.0


def load_raw_traces(subject: str, probe: str, structure: str) -> xr.DataArray:
    path = mua_files.get_mua_traces_path(subject, probe)
    da = trace_io.open_si_zarr_recording_as_xarray(path)
    da = trace_io.scale_to_uV(da)
    structs = channel_anatomy.load_structures(subject, probe)
    da = channel_anatomy.assign_structures(da, structs)
    da = da.sel(channel=(da["struct"] == structure))
    keep = channel_anatomy.compute_channel_mask(da.y, subject, probe, structure)
    return da.sel(channel=keep)


def build_nrem_mask(da: xr.DataArray, subject: str, probe: str) -> np.ndarray:
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    times = da.time.values
    mask = np.zeros(times.shape, dtype=bool)
    for hg in hgs.values():
        hg_state = hg.keep_states(list(NREM_STATES))
        if len(hg_state) > 0:
            mask |= hg_state.covers_time(times)
    return mask


def bin_boundaries(da: xr.DataArray) -> np.ndarray:
    sizes = np.asarray(da.chunks[0], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(sizes)])


# %% [markdown]
# ## Load + Gaussian smooth (prepare trace once; shared across all benchmark runs)

# %%
subject, probe, structure = SPS
print(f"combo: {subject} {probe} {structure}")

da = load_raw_traces(subject, probe, structure)
fs = float(da.attrs["fs"])
boundaries = bin_boundaries(da)
mask = build_nrem_mask(da, subject, probe)
print(
    f"trace {da.shape}, fs={fs}, {len(boundaries) - 1} bins, NREM samples={mask.sum():,}"
)

# Keep TWO copies of the smoothed trace: numpy (for np.quantile) and dask (for dask_array.percentile).
# Use matching chunk sizes so each has the same parallel layout opportunities.
sigma_samples = fs / (2 * np.pi * GAUSSIAN_FREQ_MAX)
t0 = time.perf_counter()
arr_np = (
    dndf.gaussian_filter(da.data, sigma=[sigma_samples, 0]).compute().astype(np.float64)
)
print(f"load + gaussian smooth (numpy): {time.perf_counter() - t0:.1f}s")

# For the dask path, re-wrap the numpy array as a dask array with the same time chunking
# as the zarr. This is equivalent to the in-pipeline dask trace but ensures both methods
# are benchmarked from the SAME data (no re-load).
chunk_sizes = tuple(int(c) for c in da.chunks[0])
arr_dask = dask_array.from_array(arr_np, chunks=(chunk_sizes, -1))
print(f"dask wrapper: shape={arr_dask.shape}, chunks={arr_dask.chunks[0][:2]}... + {arr_dask.chunks[1]}")

# %% [markdown]
# ## Method 1: numpy `np.quantile`
#
# Per bin, vectorize across channels in one call: `np.quantile(vals_2d, 0.20, axis=0)`.

# %%
def per_bin_q20_numpy(
    arr: np.ndarray, mask: np.ndarray, boundaries: np.ndarray, q: float
) -> np.ndarray:
    n_bins = len(boundaries) - 1
    n_chans = arr.shape[1]
    out = np.full((n_bins, n_chans), np.nan, dtype=np.float64)
    for bi in range(n_bins):
        lo, hi = boundaries[bi], boundaries[bi + 1]
        bin_mask = mask[lo:hi]
        n_state = int(bin_mask.sum())
        if n_state < max(1, int(MIN_BIN_STATE_FRACTION * (hi - lo))):
            continue
        vals = arr[lo:hi][bin_mask, :]
        out[bi, :] = np.quantile(vals, q, axis=0)
    return out


t0 = time.perf_counter()
q20_numpy = per_bin_q20_numpy(arr_np, mask, boundaries, GLOBAL_QUANTILE)
t_numpy = time.perf_counter() - t0
kept_numpy = int(np.sum(~np.isnan(q20_numpy).all(axis=1)))
print(f"numpy: {t_numpy:.2f}s  ({kept_numpy}/{len(boundaries) - 1} bins kept)")

# %% [markdown]
# ## Method 2: `dask_array.percentile(..., internal_method="default")`
#
# dask's default is a sort-based exact quantile. 1D API; loop over channels per bin.

# %%
def per_bin_q20_dask(
    arr: dask_array.Array,
    mask: np.ndarray,
    boundaries: np.ndarray,
    q: float,
    internal_method: str,
) -> np.ndarray:
    """Build all (bin, channel) percentile tasks lazily; single dask.compute at the end.

    `dask_array.percentile` is 1D-only so we still loop per channel within a bin,
    but all n_bins × n_channels tasks are accumulated into one task graph and
    materialized in a single scheduler invocation. This lets dask schedule the
    entire workload in parallel, amortizing per-call setup overhead.
    """
    n_bins = len(boundaries) - 1
    n_chans = arr.shape[1]
    out = np.full((n_bins, n_chans), np.nan, dtype=np.float64)
    percentiles = np.array([q * 100.0])

    tasks = []
    idx: list[tuple[int, int]] = []
    for bi in range(n_bins):
        lo, hi = boundaries[bi], boundaries[bi + 1]
        bin_mask = mask[lo:hi]
        n_state = int(bin_mask.sum())
        if n_state < max(1, int(MIN_BIN_STATE_FRACTION * (hi - lo))):
            continue
        bin_dask = arr[lo:hi]
        for ci in range(n_chans):
            ch = bin_dask[:, ci][bin_mask]
            tasks.append(
                dask_array.percentile(ch, percentiles, internal_method=internal_method)
            )
            idx.append((bi, ci))

    # Single compute call: all ~n_bins*n_chans percentile tasks dispatched together.
    results = dask_array.compute(*tasks)
    for (bi, ci), r in zip(idx, results):
        out[bi, ci] = float(np.asarray(r).ravel()[0])
    return out


t0 = time.perf_counter()
q20_dask_default = per_bin_q20_dask(
    arr_dask, mask, boundaries, GLOBAL_QUANTILE, "default"
)
t_dask_default = time.perf_counter() - t0
print(f"dask default: {t_dask_default:.2f}s")

# %%
t0 = time.perf_counter()
q20_dask_tdigest = per_bin_q20_dask(
    arr_dask, mask, boundaries, GLOBAL_QUANTILE, "tdigest"
)
t_dask_tdigest = time.perf_counter() - t0
print(f"dask tdigest: {t_dask_tdigest:.2f}s")

# %% [markdown]
# ## Correctness verification

# %%
def abs_rel_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return float("nan"), float("nan")
    abs_err = np.abs(a[both] - b[both])
    denom = np.maximum(np.abs(a[both]), np.finfo(float).eps)
    return float(abs_err.max()), float((abs_err / denom).max())


abs_nd, rel_nd = abs_rel_err(q20_numpy, q20_dask_default)
abs_nt, rel_nt = abs_rel_err(q20_numpy, q20_dask_tdigest)
print(f"numpy vs dask-default:   max|abs err|={abs_nd:.4f} uV, max rel err={rel_nd:.2e}")
print(f"numpy vs dask-tdigest:   max|abs err|={abs_nt:.4f} uV, max rel err={rel_nt:.2e}")

# %% [markdown]
# ## Summary

# %%
print(f"\n=== per-bin Q20 ({kept_numpy} bins × {arr_np.shape[1]} channels, NREM-only) ===")
print(f"  numpy          : {t_numpy:7.2f} s  (baseline)")
print(f"  dask default   : {t_dask_default:7.2f} s  ({t_dask_default / t_numpy:.2f}x numpy)")
print(f"  dask tdigest   : {t_dask_tdigest:7.2f} s  ({t_dask_tdigest / t_numpy:.2f}x numpy)")

# Save.
out = Path(
    "/Volumes/npx_nfs/nobak/offproj/novel_objects_deprivation/_exploration/mua_drift_detrending_check/per_bin_quantile_benchmark.json"
)
out.parent.mkdir(parents=True, exist_ok=True)
import json

with open(out, "w") as f:
    json.dump(
        {
            "subject": subject,
            "probe": probe,
            "structure": structure,
            "state": STATE,
            "n_bins_kept": kept_numpy,
            "n_channels": arr_np.shape[1],
            "quantile": GLOBAL_QUANTILE,
            "seconds": {
                "numpy": t_numpy,
                "dask_default": t_dask_default,
                "dask_tdigest": t_dask_tdigest,
            },
            "abs_err_vs_numpy_uV": {
                "dask_default": abs_nd,
                "dask_tdigest": abs_nt,
            },
        },
        f,
        indent=2,
    )
print(f"\nsaved -> {out}")
