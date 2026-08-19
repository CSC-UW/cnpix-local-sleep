# %% [markdown]
# # MUA drift measurement
#
# Quantify per-channel MUA trace-value distribution drift over each recording, on raw MUA
# (no Gaussian smoothing, no 17×1 median filter). For each channel we slice the recording into
# the zarr's native ~5.5-min dask chunks, restrict to state-masked samples, and compute per-bin
# distribution summaries.
#
# Cohort (cortical only): CNPIX2-Segundo, CNPIX12-Santiago, CNPIX15-Claude.
#
# ## Per-bin summaries computed
#
# - `q20`, `q50`, `iqr`: three points of the per-bin distribution.
# - `pctile_at_T`: the percentile within this bin's distribution at which the whole-recording Q20
#   falls. If the bin's distribution were identical to the whole-recording, this would be 0.20 in
#   every bin. Drift causes it to swing.
#
# ## Drift scalars computed
#
# - `drift_scalar_ptp = (max - min) / IQR_whole_recording` of per-bin Q20.
# - `drift_scalar_robust = (Q90 - Q10) / IQR_whole_recording` of per-bin Q20 (resistant to single-bin outliers).
# - `pctile_drift_ptp = max - min` of per-bin `pctile_at_T`.
# - `pctile_drift_robust = Q90 - Q10` of per-bin `pctile_at_T`. Direct, interpretable units:
#   "the threshold drifts from Q{50-d/2} to Q{50+d/2} across 80% of bins" (where d is in
#   percentile points). Tells you, in plain language, what fraction of percentile shift the
#   detection threshold experiences over time.

# %%
from __future__ import annotations

import time
import warnings
from pathlib import Path

import dask_image.ndfilters as dndf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from cnpix_local_sleep import hyp, sps_conf
from cnpix.mua import files as mua_files
from cnpix_local_sleep import channel_anatomy
from cnpix_local_sleep import trace_io

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Configuration
SUBJECTS = ["CNPIX2-Segundo", "CNPIX12-Santiago", "CNPIX15-Claude"]

NREM_STATES = ("NREM", "IS")  # REM excluded per earlier directive
WAKE_STATES = ("Wake", "MA", "Other")
STATE_GROUPS = {"nrem": NREM_STATES, "wake": WAKE_STATES}

GLOBAL_QUANTILE = (
    0.20  # threshold-equivalent quantile (matches OFF detection convention)
)
MIN_BIN_STATE_FRACTION = 0.05  # drop bins with <5% state samples

# Gaussian smoothing applied to the trace before computing any quantiles.
# Matches the production pre-detection chain's Gaussian step (20 Hz cutoff).
# The raw zarr stores heavily-quantized int16*gain values, which creates tied
# samples at the nominal quantile (F(whole_q20) > 0.20). Gaussian smoothing
# produces float-valued samples that are essentially tie-free, so F(whole_q20)
# converges back to 0.20. We deliberately skip the 17x1 median filter that
# production also applies; it's a denoiser, not a de-tier, and it's not
# needed for the drift measurement.
GAUSSIAN_FREQ_MAX = 20.0  # Hz

OUT_DIR = Path(
    "/Volumes/npx_nfs/nobak/offproj/novel_objects_deprivation/_exploration/mua_drift"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

BINS_PATH = OUT_DIR / "per_bin_summaries.parquet"
DRIFT_PATH = OUT_DIR / "drift_scalars.parquet"

print(f"Outputs -> {OUT_DIR}")

# %% [markdown]
# ## Raw trace loader
#
# Goes straight through `open_si_zarr_recording_as_xarray`, `scale_to_uV`, and `assign_structures`:
# no Gaussian smoothing and no 17×1 median filter. Applies the detection channel mask so we measure
# the same channels the OFF detection pipeline uses.


# %%
def load_raw_traces(subject: str, probe: str, structure: str) -> xr.DataArray:
    path = mua_files.get_mua_traces_path(subject, probe)
    if not path.exists():
        raise FileNotFoundError(f"MUA traces not found: {path}")
    da = trace_io.open_si_zarr_recording_as_xarray(path)
    da = trace_io.scale_to_uV(da)
    structs = channel_anatomy.load_structures(subject, probe)
    da = channel_anatomy.assign_structures(da, structs)
    da = da.sel(channel=(da["struct"] == structure))
    keep = channel_anatomy.compute_channel_mask(da.y, subject, probe, structure)
    da = da.sel(channel=keep)
    return da


def build_state_masks(
    da: xr.DataArray, subject: str, probe: str
) -> dict[str, np.ndarray]:
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    times = da.time.values
    masks = {name: np.zeros(times.shape, dtype=bool) for name in STATE_GROUPS}
    for hg in hgs.values():
        for name, states in STATE_GROUPS.items():
            hg_state = hg.keep_states(list(states))
            if len(hg_state) > 0:
                masks[name] |= hg_state.covers_time(times)
    return masks


def bin_boundaries(da: xr.DataArray) -> np.ndarray:
    """Return the zarr dask-chunk boundaries along time as a (n_bins+1,) index array."""
    sizes = np.asarray(da.chunks[0], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(sizes)])


# %% [markdown]
# ## Per-bin summaries for one combo
#
# Loads the full probe at once, computes whole-recording stats per (channel, state) once,
# then loops over bins × states to record per-bin q20/q50/iqr and the percentile of the
# whole-recording Q20 within each bin's distribution.


# %%
def compute_per_bin_summaries(subject: str, probe: str, structure: str) -> pd.DataFrame:
    da = load_raw_traces(subject, probe, structure)
    times = da.time.values
    boundaries = bin_boundaries(da)
    state_masks = build_state_masks(da, subject, probe)

    # Fuse load + Gaussian smooth via dask_image so the raw trace is never
    # separately materialized (halves peak memory to ~trace size).
    # sigma = fs / (2*pi*freq_max) matches production (morphological/mua/readers.py).
    # Halo overhead per chunk boundary is ~4*sigma*2 ~ 32 samples, negligible
    # vs 165 000-sample chunks (<0.02%).
    fs = float(da.attrs["fs"])
    sigma_samples = fs / (2 * np.pi * GAUSSIAN_FREQ_MAX)
    t0 = time.perf_counter()
    arr = (
        dndf.gaussian_filter(da.data, sigma=[sigma_samples, 0])
        .compute()
        .astype(np.float64)
    )
    t_load = time.perf_counter() - t0
    print(
        f"  load + gaussian smooth (sigma={sigma_samples:.2f} samples) -> "
        f"{arr.shape} ({arr.nbytes / 1e9:.1f} GB) in {t_load:.1f}s"
    )

    chan_labels = [str(c) for c in da.channel.values]
    y_vals = (
        [float(y) for y in da.y.values] if "y" in da.coords else [np.nan] * arr.shape[1]
    )

    # Whole-recording stats per (channel, state). Used both for normalization
    # (IQR) and as the global threshold whose per-bin percentile we track (Q20).
    # Also record `whole_pctile_at_T` = P(sample <= whole_q20) over the pool.
    # On heavily-quantized data (MUA is int16*gain, ~0.78 uV steps), tied values
    # at the nominal quantile push F(whole_q20) far above 0.20; this is the
    # *actual* threshold-percentile that quantile-based OFF detection uses.
    t0 = time.perf_counter()
    whole_q20: dict[str, np.ndarray] = {}
    whole_iqr: dict[str, np.ndarray] = {}
    whole_pctile_at_T: dict[str, np.ndarray] = {}
    for state_name, mask in state_masks.items():
        if not mask.any():
            continue
        state_arr = arr[mask, :]
        q = np.quantile(state_arr, [GLOBAL_QUANTILE, 0.25, 0.75], axis=0)
        whole_q20[state_name] = q[0]
        whole_iqr[state_name] = q[2] - q[1]
        whole_pctile_at_T[state_name] = (state_arr <= q[0][None, :]).mean(axis=0)
    t_whole = time.perf_counter() - t0
    print(f"  whole-recording q20+iqr per state: {t_whole:.1f}s")

    bin_centers = 0.5 * (times[boundaries[:-1]] + times[boundaries[1:] - 1])
    n_bins = len(bin_centers)
    n_chans = arr.shape[1]

    t0 = time.perf_counter()
    rows: list[dict] = []
    for state_name, mask in state_masks.items():
        if state_name not in whole_q20:
            continue
        T_global = whole_q20[state_name]  # (K,) global threshold per channel
        iqr_global = whole_iqr[state_name]  # (K,)
        pool_pctile_at_T = whole_pctile_at_T[state_name]  # (K,)
        for bi in range(n_bins):
            lo, hi = boundaries[bi], boundaries[bi + 1]
            bin_mask = mask[lo:hi]
            n_state = int(bin_mask.sum())
            if n_state < max(1, int(MIN_BIN_STATE_FRACTION * (hi - lo))):
                continue
            vals = arr[lo:hi][bin_mask, :]  # (n_state, K)
            q = np.quantile(vals, [GLOBAL_QUANTILE, 0.50, 0.25, 0.75], axis=0)  # (4, K)
            q20, q50, q25, q75 = q[0], q[1], q[2], q[3]
            iqr = q75 - q25
            # pctile_at_T: empirical CDF of bin samples evaluated at T_global per channel.
            # = fraction of bin samples <= T_global[ci]
            pctile_at_T = (vals <= T_global[None, :]).mean(axis=0)  # (K,)
            for ci in range(n_chans):
                rows.append(
                    {
                        "subject": subject,
                        "probe": probe,
                        "structure": structure,
                        "channel": chan_labels[ci],
                        "y": y_vals[ci],
                        "state": state_name,
                        "bin_idx": bi,
                        "bin_center_s": float(bin_centers[bi]),
                        "n_state_samples": n_state,
                        "q20": float(q20[ci]),
                        "q50": float(q50[ci]),
                        "iqr": float(iqr[ci]),
                        "pctile_at_T": float(pctile_at_T[ci]),
                        "whole_q20": float(T_global[ci]),
                        "whole_iqr": float(iqr_global[ci]),
                        # Actual pool-level F(T); differs from GLOBAL_QUANTILE
                        # because np.quantile + tied (quantized) values don't
                        # satisfy F(T) = q exactly.
                        "whole_pctile_at_T": float(pool_pctile_at_T[ci]),
                    }
                )
    t_bins = time.perf_counter() - t0
    print(
        f"  per-bin summaries: {len(rows)} rows over {n_bins} bins x {n_chans} chan x {len(state_masks)} states in {t_bins:.1f}s"
    )

    del arr
    return pd.DataFrame(rows)


def _append_parquet(path: Path, new_df: pd.DataFrame) -> None:
    if new_df.empty:
        return
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)


REQUIRED_BIN_COLUMNS = {
    "subject",
    "probe",
    "structure",
    "channel",
    "y",
    "state",
    "bin_idx",
    "bin_center_s",
    "n_state_samples",
    "q20",
    "q50",
    "iqr",
    "pctile_at_T",
    "whole_q20",
    "whole_iqr",
    "whole_pctile_at_T",
}


def _existing_combos(path: Path) -> set[tuple[str, str, str]]:
    """Return the (subject, probe, structure) tuples already in path.

    If the parquet's schema is stale (missing columns added by a later
    notebook version), delete it and return an empty set, forcing a fresh run.
    """
    if not path.exists():
        return set()
    sample = pd.read_parquet(path).head(1)
    missing = REQUIRED_BIN_COLUMNS - set(sample.columns)
    if missing:
        print(
            f"  stale schema in {path.name} (missing {sorted(missing)}); "
            f"deleting and re-running from scratch"
        )
        path.unlink()
        if DRIFT_PATH.exists():
            DRIFT_PATH.unlink()
        return set()
    df = pd.read_parquet(path, columns=["subject", "probe", "structure"])
    return set(
        map(
            tuple,
            df[["subject", "probe", "structure"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )


# %% [markdown]
# ## Cohort enumeration

# %%
spsl_cx = sps_conf.get_subject_probe_structure_list(
    exclude_thalamus=True,
    exclude_striatum=True,
    exclude_other=True,
)
spsl = [t for t in spsl_cx if t[0] in SUBJECTS]
print(f"{len(spsl)} cortical combos:")
for t in spsl:
    print(" ", t)

# %% [markdown]
# ## Main loop

# %%
OVERWRITE = False
done = set() if OVERWRITE else _existing_combos(BINS_PATH)
for subject, probe, structure in spsl:
    if (subject, probe, structure) in done:
        print(f"skip {subject} {probe} {structure} (already done)")
        continue
    print(f"processing {subject} {probe} {structure}")
    t0 = time.perf_counter()
    df = compute_per_bin_summaries(subject, probe, structure)
    print(f"  combo done in {(time.perf_counter() - t0) / 60:.1f} min")
    _append_parquet(BINS_PATH, df)

print(f"per-bin parquet -> {BINS_PATH}")

# %% [markdown]
# ## Drift scalars per channel
#
# Reads the per-bin parquet (which now carries `whole_q20`, `whole_iqr`, and `pctile_at_T`)
# and computes per-channel × per-state drift summaries.

# %%
bins_df = pd.read_parquet(BINS_PATH)
print(f"per-bin rows: {len(bins_df):,}")


def compute_drift_scalars(bins_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict] = []
    group_cols = ["subject", "probe", "structure", "channel", "y", "state"]
    for keys, g in bins_df.groupby(group_cols, sort=False):
        subject, probe, structure, channel, y, state = keys
        if len(g) < 2:
            continue
        q20_vals = g["q20"].to_numpy()
        pctile_vals = g["pctile_at_T"].to_numpy()
        n_vals = g["n_state_samples"].to_numpy().astype(np.float64)
        whole_iqr = float(g["whole_iqr"].iloc[0])
        whole_q20 = float(g["whole_q20"].iloc[0])
        whole_pctile = float(g["whole_pctile_at_T"].iloc[0])

        # Sanity: sample-count-weighted mean of pctile_at_T over kept bins
        # should match `whole_pctile_at_T` (pool-level F(T)) up to the samples
        # excluded by MIN_BIN_STATE_FRACTION. It does NOT generally equal
        # GLOBAL_QUANTILE because np.quantile on heavily-quantized (tied) data
        # produces a threshold where F(T) > q.
        n_sum = n_vals.sum()
        pctile_weighted_mean = (
            float((pctile_vals * n_vals).sum() / n_sum) if n_sum > 0 else np.nan
        )

        q20_ptp = float(np.ptp(q20_vals))
        q20_p10, q20_p90 = np.quantile(q20_vals, [0.10, 0.90])
        q20_iqd = float(q20_p90 - q20_p10)  # inter-decile

        pctile_ptp = float(np.ptp(pctile_vals))
        pctile_p10, pctile_p90 = np.quantile(pctile_vals, [0.10, 0.90])
        pctile_iqd = float(pctile_p90 - pctile_p10)
        pctile_min = float(pctile_vals.min())
        pctile_max = float(pctile_vals.max())

        out_rows.append(
            {
                "subject": subject,
                "probe": probe,
                "structure": structure,
                "channel": channel,
                "y": y,
                "state": state,
                "n_bins": int(len(g)),
                "whole_q20": whole_q20,
                "whole_iqr": whole_iqr,
                "whole_pctile_at_T": whole_pctile,
                # Q20 drift in absolute uV
                "q20_range_ptp": q20_ptp,
                "q20_range_p10p90": q20_iqd,
                # Q20 drift normalized by within-channel IQR
                "drift_scalar_ptp": q20_ptp / whole_iqr if whole_iqr > 0 else np.nan,
                "drift_scalar_robust": q20_iqd / whole_iqr if whole_iqr > 0 else np.nan,
                # Percentile-equivalent drift: how much does the global threshold's percentile
                # within bin-distributions move across bins?
                "pctile_at_T_min": pctile_min,
                "pctile_at_T_max": pctile_max,
                "pctile_at_T_p10": float(pctile_p10),
                "pctile_at_T_p90": float(pctile_p90),
                "pctile_drift_ptp": pctile_ptp,
                "pctile_drift_robust": pctile_iqd,
                # Sanity: sample-count-weighted mean of pctile_at_T, must equal
                # GLOBAL_QUANTILE (0.20) up to floating point.  Deviation => bug.
                "pctile_weighted_mean": pctile_weighted_mean,
            }
        )
    return pd.DataFrame(out_rows)


drift_df = compute_drift_scalars(bins_df)
drift_df.to_parquet(DRIFT_PATH, index=False)
print(f"drift scalars -> {DRIFT_PATH} ({len(drift_df)} rows)")

# Sanity check: weighted mean of pctile_at_T over kept bins should match
# `whole_pctile_at_T` (pool-level F(T)) up to the fraction of samples in bins
# dropped by MIN_BIN_STATE_FRACTION. It does NOT equal GLOBAL_QUANTILE because
# np.quantile on heavily-quantized (tied) MUA values lands inside a dense mode
# where F(T) > q. Also report how far F(T) is from the nominal q, a separate
# measurement of the tie effect.
_dev = (drift_df["pctile_weighted_mean"] - drift_df["whole_pctile_at_T"]).abs()
_tie = (drift_df["whole_pctile_at_T"] - GLOBAL_QUANTILE).abs()
print(
    "\nSanity: weighted mean of pctile_at_T (kept bins) vs pool-level F(T) per channel"
)
print(
    f"  max |dev|: {_dev.max():.6f} | median |dev|: {_dev.median():.6f} "
    f"| rows > 1e-3: {int((_dev > 1e-3).sum())} / {len(drift_df)}"
)
print(
    f"\nTie-breaking effect: F(whole_q20) vs nominal q={GLOBAL_QUANTILE:.3f} "
    "(large = heavy quantization / tied values at threshold)"
)
print(f"  max |dev|: {_tie.max():.6f} | median |dev|: {_tie.median():.6f}")
if (_dev > 1e-3).any():
    print(
        "\n  WARNING: sanity-check deviation > 1e-3; likely dropped-bin fraction effect:"
    )
    print(
        drift_df.loc[
            _dev.nlargest(5).index,
            [
                "subject",
                "probe",
                "structure",
                "channel",
                "state",
                "pctile_weighted_mean",
                "whole_pctile_at_T",
                "n_bins",
            ],
        ].to_string()
    )

# %% [markdown]
# ## Plots
#
# Per (subject, probe, structure):
#
# - Triple time series (q50 / q20 / iqr), one row per state. Shows central tendency, low-tail,
#   and spread drift independently.
# - Q20 normalized by `whole_iqr`, with horizontal reference at zero shift.
# - Percentile-at-T over time: most interpretable view of detection-threshold drift.
# - Drift scalar vs. depth (y): diagnostic for spatial patterns (superficial vs. deep).


# %%
def plot_triple_timeseries(
    bins_df: pd.DataFrame, subject: str, probe: str, structure: str
) -> Path:
    sub = bins_df[
        (bins_df["subject"] == subject)
        & (bins_df["probe"] == probe)
        & (bins_df["structure"] == structure)
    ]
    states = ("nrem", "wake")
    metrics = ("q50", "q20", "iqr")
    fig, axes = plt.subplots(
        len(states), len(metrics), figsize=(15, 6), sharex=True, sharey="col"
    )
    for ri, state in enumerate(states):
        s = sub[sub["state"] == state]
        for ci_, metric in enumerate(metrics):
            ax = axes[ri, ci_]
            if s.empty:
                ax.set_title(f"{state} (no data)")
                continue
            for ch, grp in s.groupby("channel"):
                grp = grp.sort_values("bin_idx")
                ax.plot(grp["bin_center_s"] / 3600, grp[metric], lw=0.5, alpha=0.7)
            # Mean across channels per bin, overlaid as thick black line.
            mean_across = (
                s.groupby("bin_idx", sort=True)
                .agg(val=(metric, "mean"), t=("bin_center_s", "first"))
                .reset_index()
            )
            ax.plot(
                mean_across["t"] / 3600,
                mean_across["val"],
                color="black",
                lw=2.0,
                label="mean across channels",
            )
            ax.set_ylabel(f"{metric} (uV) [{state}]")
            if ri == 0:
                ax.set_title(metric)
            if ri == 0 and ci_ == 0:
                ax.legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time (h)")
    fig.suptitle(f"{subject} {probe} {structure}: per-bin distribution summaries")
    fig.tight_layout()
    out = FIG_DIR / f"timeseries_{subject}_{probe}_{structure}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    return out


def plot_q20_normalized(
    bins_df: pd.DataFrame, subject: str, probe: str, structure: str
) -> Path:
    """Plot (per-bin q20 - whole_q20) / whole_iqr vs time. Y-axis is dimensionless;
    a value of e.g. 0.5 means the bin's Q20 is half an IQR above the channel's typical Q20.
    """
    sub = bins_df[
        (bins_df["subject"] == subject)
        & (bins_df["probe"] == probe)
        & (bins_df["structure"] == structure)
    ].copy()
    sub["q20_norm"] = (sub["q20"] - sub["whole_q20"]) / sub["whole_iqr"]
    sub["abs_q20_norm"] = sub["q20_norm"].abs()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, state in zip(axes, ("nrem", "wake")):
        s = sub[sub["state"] == state]
        if s.empty:
            ax.set_title(f"{state} (no data)")
            continue
        for ch, grp in s.groupby("channel"):
            grp = grp.sort_values("bin_idx")
            ax.plot(grp["bin_center_s"] / 3600, grp["q20_norm"], lw=0.5, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhspan(-0.5, 0.5, color="grey", alpha=0.1, label="±0.5 IQR")
        # Mean of |q20_norm| across channels per bin, overlaid as thick black line.
        mean_abs = (
            s.groupby("bin_idx", sort=True)
            .agg(val=("abs_q20_norm", "mean"), t=("bin_center_s", "first"))
            .reset_index()
        )
        ax.plot(
            mean_abs["t"] / 3600,
            mean_abs["val"],
            color="black",
            lw=2.0,
            label="mean |q20_norm| across channels",
        )
        ax.set_ylabel(f"(q20 - whole_q20)/IQR  [{state}]")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time (h)")
    fig.suptitle(
        f"{subject} {probe} {structure}: Q20 drift normalized by whole-recording IQR"
    )
    fig.tight_layout()
    out = FIG_DIR / f"q20_normalized_{subject}_{probe}_{structure}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    return out


def plot_pctile_at_T(
    bins_df: pd.DataFrame, subject: str, probe: str, structure: str
) -> Path:
    """Per-bin percentile of the whole-recording Q20 within the bin's distribution.
    A flat line at 0.20 = no drift; swings = the global threshold sits at different bin-percentiles
    over time (e.g. Q15 in some bins, Q25 in others).
    """
    sub = bins_df[
        (bins_df["subject"] == subject)
        & (bins_df["probe"] == probe)
        & (bins_df["structure"] == structure)
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, state in zip(axes, ("nrem", "wake")):
        s = sub[sub["state"] == state]
        if s.empty:
            ax.set_title(f"{state} (no data)")
            continue
        for ch, grp in s.groupby("channel"):
            grp = grp.sort_values("bin_idx")
            ax.plot(grp["bin_center_s"] / 3600, grp["pctile_at_T"], lw=0.5, alpha=0.7)
        ax.axhline(
            GLOBAL_QUANTILE,
            color="black",
            lw=0.5,
            label=f"target Q={GLOBAL_QUANTILE:.2f}",
        )
        ax.set_ylabel(f"bin pctile at global Q20  [{state}]")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time (h)")
    fig.suptitle(
        f"{subject} {probe} {structure}: where the global Q20 threshold sits within each bin's distribution"
    )
    fig.tight_layout()
    out = FIG_DIR / f"pctile_at_T_{subject}_{probe}_{structure}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    return out


def plot_drift_vs_depth(
    drift_df: pd.DataFrame, subject: str, probe: str, structure: str
) -> Path:
    """Scatter of robust drift scalars vs. channel depth (y).

    Two rows: top = Q20-drift normalized by IQR; bottom = pctile-equivalent drift.
    Two columns: NREM, Wake.
    """
    sub = drift_df[
        (drift_df["subject"] == subject)
        & (drift_df["probe"] == probe)
        & (drift_df["structure"] == structure)
    ]
    states = ("nrem", "wake")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ci_, state in enumerate(states):
        s = sub[sub["state"] == state]
        if s.empty:
            for ri in (0, 1):
                axes[ri, ci_].set_title(f"{state} (no data)")
            continue
        s = s.sort_values("y")
        # top row: drift_scalar_robust (normalized by IQR)
        ax = axes[0, ci_]
        ax.plot(
            s["y"],
            s["drift_scalar_ptp"],
            "o-",
            color="tab:gray",
            ms=4,
            lw=0.5,
            alpha=0.6,
            label="ptp",
        )
        ax.plot(
            s["y"],
            s["drift_scalar_robust"],
            "o-",
            color="tab:red",
            ms=4,
            lw=0.8,
            label="Q90-Q10 (robust)",
        )
        ax.axhline(1.0, color="black", lw=0.4, ls=":")
        ax.set_ylabel("Q20 drift / whole IQR")
        ax.set_title(f"{state}")
        ax.legend(fontsize=7, loc="upper right")
        # bottom row: pctile-equivalent drift
        ax = axes[1, ci_]
        ax.plot(
            s["y"],
            s["pctile_drift_ptp"],
            "o-",
            color="tab:gray",
            ms=4,
            lw=0.5,
            alpha=0.6,
            label="ptp",
        )
        ax.plot(
            s["y"],
            s["pctile_drift_robust"],
            "o-",
            color="tab:blue",
            ms=4,
            lw=0.8,
            label="Q90-Q10 (robust)",
        )
        ax.set_ylabel("pctile-at-T drift")
        ax.set_xlabel("channel y (um from probe tip; larger = more dorsal)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{subject} {probe} {structure}: drift vs. channel depth")
    fig.tight_layout()
    out = FIG_DIR / f"drift_vs_depth_{subject}_{probe}_{structure}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    return out


for subject, probe, structure in spsl:
    has_data = (
        (bins_df["subject"] == subject)
        & (bins_df["probe"] == probe)
        & (bins_df["structure"] == structure)
    ).any()
    if not has_data:
        continue
    plot_triple_timeseries(bins_df, subject, probe, structure)
    plot_q20_normalized(bins_df, subject, probe, structure)
    plot_pctile_at_T(bins_df, subject, probe, structure)
    plot_drift_vs_depth(drift_df, subject, probe, structure)

# %% [markdown]
# ## Cohort summary
#
# Per-SPS distribution of drift scalars. Two summaries:
#
# - `drift_scalar_robust` = (Q90-Q10)/IQR_whole_recording of per-bin Q20, drift in absolute
#   units relative to within-channel spread.
# - `pctile_drift_robust` = Q90-Q10 of per-bin `pctile_at_T`, drift in
#   threshold-percentile-equivalent units (e.g. 0.10 means the threshold drifts across 10
#   percentile points across bins, e.g. from Q15 to Q25 over the recording).


# %%
def summary_table(drift_df: pd.DataFrame, metric: str, state: str) -> pd.DataFrame:
    sub = drift_df[drift_df["state"] == state]
    return (
        sub.groupby(["subject", "probe", "structure"])[metric]
        .describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "min", "max"]]
        .rename(columns={"50%": "median"})
        .round(3)
    )


def cohort_summary(drift_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        drift_df.groupby("state")[metric]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9])[
            ["count", "25%", "50%", "75%", "90%", "min", "max"]
        ]
        .rename(columns={"50%": "median"})
        .round(3)
    )


if not drift_df.empty:
    for state in ("nrem", "wake"):
        for metric in ("drift_scalar_robust", "pctile_drift_robust"):
            print(f"\n=== {metric}  [state={state}]  per-SPS distribution ===")
            print(summary_table(drift_df, metric, state).to_string())
    print("\n=== cohort-wide pooled across (combo, channel) ===")
    for metric in ("drift_scalar_robust", "pctile_drift_robust"):
        print(f"\n{metric}:")
        print(cohort_summary(drift_df, metric).to_string())
else:
    print("no drift scalars computed")


# %% [markdown]
# ## Span endpoints of `pctile_at_T`
#
# `pctile_drift_robust` reports the *width* (Q90 - Q10) of per-bin `pctile_at_T`, but
# not where along the [0, 1] axis that width sits. The span [Q10, Q90] (the 10th and
# 90th percentiles of per-bin `pctile_at_T`) answers that directly.
#
# Interpretation: the nominal Q20 threshold sits between these two quantiles of the
# local (per-bin) distribution across 80% of bins.


# %%
def span_endpoint_summary(drift_df: pd.DataFrame, state: str) -> pd.DataFrame:
    sub = drift_df[drift_df["state"] == state]
    return (
        sub.groupby(["subject", "probe", "structure"])[
            ["pctile_at_T_p10", "pctile_at_T_p90", "pctile_drift_robust"]
        ]
        .median()
        .round(3)
        .rename(
            columns={
                "pctile_at_T_p10": "Q10_of_pctile_at_T",
                "pctile_at_T_p90": "Q90_of_pctile_at_T",
                "pctile_drift_robust": "width",
            }
        )
    )


def cohort_endpoint_summary(drift_df: pd.DataFrame, state: str) -> pd.DataFrame:
    sub = drift_df[drift_df["state"] == state]
    out = (
        pd.DataFrame(
            {
                "Q10_of_pctile_at_T": sub["pctile_at_T_p10"].describe(
                    percentiles=[0.25, 0.5, 0.75]
                ),
                "Q90_of_pctile_at_T": sub["pctile_at_T_p90"].describe(
                    percentiles=[0.25, 0.5, 0.75]
                ),
            }
        )
        .T[["count", "min", "25%", "50%", "75%", "max"]]
        .rename(columns={"50%": "median"})
    )
    return out.round(3)


if not drift_df.empty:
    for state in ("nrem", "wake"):
        print(f"\n=== [{state}] median span [Q10, Q90] of pctile_at_T per SPS ===")
        print(
            "(the nominal Q20 threshold sits between Q10_of_pctile_at_T and "
            "Q90_of_pctile_at_T of the local distribution across 80% of bins)"
        )
        print(span_endpoint_summary(drift_df, state).to_string())

    for state in ("nrem", "wake"):
        print(
            f"\n=== [{state}] cohort-wide distribution of per-channel span endpoints ==="
        )
        print(cohort_endpoint_summary(drift_df, state).to_string())

# %%
