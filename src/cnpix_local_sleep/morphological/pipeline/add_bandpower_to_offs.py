"""Add bandpower statistics to aggregated OFF period parquet files.

This module adds per-OFF-period bandpower summary statistics (total, mean,
median, max power) to the LLAS, CLAS, and BLAS parquet files produced by
``aggregate_experiment_offs.do_experiment()``.

Run this AFTER ``aggregate_experiment_offs.do_experiment()`` has completed.

Each bandpower spec (band_name, bipolar, kind) and transform (log, zlog)
produces 4 columns (total, mean, median, max). With 2 bands and 2
transforms, this adds 16 columns total.
"""

from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from cnpix_local_sleep import files, hyp
from cnpix_local_sleep.morphological.mua import files as morphological_files


def add_interval_power_stats(
    offs: pd.DataFrame,
    pwr: xr.DataArray,
    col_suffix: str,
) -> pd.DataFrame:
    """Add aggregate power statistics during OFF periods to the OFF DataFrame.

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame containing 'start_time' and 'end_time' columns defining intervals.
    pwr : xr.DataArray
        Time series data array with a 'time' dimension.
    col_suffix : str
        Suffix to append to the new column names.
        For example, "bipolar_inst_log_delta".

    Returns
    -------
    offs: pd.DataFrame
        A copy of the input DataFrame with additional columns.
        For example, "total_bipolar_inst_log_delta", "mean_bipolar_inst_log_delta",
        "median_bipolar_inst_log_delta", "max_bipolar_inst_log_delta".
    """
    offs = offs.copy()
    time_values = pwr.time.values
    pwr_values = pwr.values

    cumulative_signal = np.concatenate(([0.0], np.cumsum(pwr_values)))

    start_idx = np.searchsorted(time_values, offs["start_time"].to_numpy(), side="left")
    end_idx = np.searchsorted(time_values, offs["end_time"].to_numpy(), side="right")

    interval_sums = cumulative_signal[end_idx] - cumulative_signal[start_idx]
    counts = end_idx - start_idx

    total_inst = np.full_like(interval_sums, np.nan, dtype=float)
    mean_inst = np.full_like(interval_sums, np.nan, dtype=float)
    median_inst = np.full_like(interval_sums, np.nan, dtype=float)
    max_inst = np.full_like(interval_sums, np.nan, dtype=float)

    valid_mask = counts > 0
    valid_indices = np.flatnonzero(valid_mask)
    total_inst[valid_mask] = interval_sums[valid_mask]
    mean_inst[valid_mask] = interval_sums[valid_mask] / counts[valid_mask]

    for idx in valid_indices:
        seg = pwr_values[start_idx[idx] : end_idx[idx]]
        median_inst[idx] = float(np.median(seg))
        max_inst[idx] = float(np.max(seg))

    offs[f"total_{col_suffix}"] = total_inst
    offs[f"mean_{col_suffix}"] = mean_inst
    offs[f"median_{col_suffix}"] = median_inst
    offs[f"max_{col_suffix}"] = max_inst
    return offs


def do_band(
    subject: str,
    probe: str,
    structure: str,
    band_name: str,
    bipolar: bool,
    kind: Literal["stft", "inst"],
    offs: pd.DataFrame,
) -> pd.DataFrame:
    pwr = xr.load_dataarray(
        files.get_structure_bandpower_path(
            subject, probe, structure, band_name, bipolar, kind
        )
    )
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    hg = hgs.pop("Full.Conservative").drop_states({"Artifact", "NoData"})
    pwr = pwr.isel(time=hg.covers_time(pwr["time"]))
    pwr = pwr.dropna("time")

    log_pwr = np.log10(pwr)
    zlog_pwr = xr.apply_ufunc(scipy.stats.zscore, log_pwr)

    for transform, transformed_pwr in ["log", log_pwr], ["zlog", zlog_pwr]:
        col_suffix = "bipolar_" if bipolar else "monopolar_"
        col_suffix += f"{kind}_{transform}_{band_name}"
        offs = add_interval_power_stats(offs, transformed_pwr, col_suffix)
    return offs


_BANDPOWER_SPECS = [("delta", True, "inst"), ("eta", True, "inst")]
"""Bandpower specifications to compute: (band_name, bipolar, kind)."""

_MERGE_KEYS = ["subject", "probe", "structure", "start_time", "end_time"]
"""Columns that uniquely identify an OFF period across parquet files."""


def _get_bandpower_columns(df: pd.DataFrame) -> list[str]:
    """Return column names added by bandpower processing."""
    prefixes = ("total_", "mean_", "median_", "max_")
    suffixes = tuple(f"_{band}" for band, _, _ in _BANDPOWER_SPECS)
    return [
        c
        for c in df.columns
        if any(c.startswith(p) for p in prefixes)
        and any(c.endswith(s) for s in suffixes)
    ]


def add_bandpower_columns(offs: pd.DataFrame) -> pd.DataFrame:
    """Add bandpower statistics to an OFF periods DataFrame.

    Groups OFFs by (subject, probe, structure), loads bandpower zarrs for each
    group, and computes per-OFF power statistics via ``do_band``.

    This is the reusable kernel behind :func:`do_experiment`; callers that
    already hold an OFF DataFrame (e.g. full-48h OFFs assembled in a notebook)
    can attach the same bandpower columns directly instead of round-tripping
    through the LLAS/CLAS/BLAS parquets.

    Parameters
    ----------
    offs : pd.DataFrame
        DataFrame of OFF periods with ``start_time`` and ``end_time`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with bandpower columns added.
    """
    # Drop existing bandpower columns for idempotency
    existing_bp_cols = _get_bandpower_columns(offs)
    if existing_bp_cols:
        offs = offs.drop(columns=existing_bp_cols)

    groups = offs.groupby(["subject", "probe", "structure"], observed=True)
    parts = []
    for (subject, probe, structure), group_df in groups:
        try:
            for band_name, bipolar, kind in _BANDPOWER_SPECS:
                print(
                    f"  {subject}, {probe}, {structure}: "
                    f"{band_name}, bipolar={bipolar}, kind={kind}"
                )
                group_df = do_band(
                    subject, probe, structure, band_name, bipolar, kind, group_df
                )
        except FileNotFoundError:
            print(
                f"  Warning: Bandpower file missing for {subject}, {probe}, "
                f"{structure}. Filling with NaN."
            )
        parts.append(group_df)

    return pd.concat(parts).sort_index()


def do_experiment():
    """Add bandpower statistics to LLAS, CLAS, and BLAS OFF parquet files.

    Computes bandpower columns on LLAS (the superset), then transfers them to
    CLAS and BLAS via merge to avoid redundant bandpower I/O.
    """
    # Step 1: Compute bandpower columns on LLAS (superset of all OFFs)
    llas_path = morphological_files.get_path("llas_offs.parquet")
    print(f"Loading {llas_path.name}...")
    llas = pd.read_parquet(llas_path)

    print("Adding bandpower columns to LLAS OFFs...")
    llas = add_bandpower_columns(llas)

    print(f"Saving {llas_path.name}...")
    llas.to_parquet(llas_path)

    # Step 2: Transfer bandpower columns to CLAS and BLAS via merge
    bp_cols = _get_bandpower_columns(llas)
    llas_bp = llas[_MERGE_KEYS + bp_cols]

    for name in ["clas", "blas"]:
        path = morphological_files.get_path(f"{name}_offs.parquet")
        print(f"Loading {path.name}...")
        offs = pd.read_parquet(path)

        # Drop existing bandpower columns for idempotency
        existing_bp_cols = _get_bandpower_columns(offs)
        if existing_bp_cols:
            offs = offs.drop(columns=existing_bp_cols)

        print(f"Merging bandpower columns into {name.upper()} OFFs...")
        offs = offs.merge(llas_bp, on=_MERGE_KEYS, how="left")

        print(f"Saving {path.name}...")
        offs.to_parquet(path)

    print("Done.")