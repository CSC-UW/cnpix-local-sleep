import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from cnpix_local_sleep import const, files, hyp
from cnpix_local_sleep import sps_conf


def aggregated_events_wide_to_long(
    evts_wide: pd.DataFrame, condition_cols: list[str] = None, drop: bool = True
) -> pd.DataFrame:
    if condition_cols is None:
        condition_cols = [c for c in const.CONDITIONS if c != "Full.Conservative"]
    evts_long = pd.DataFrame()
    for c in condition_cols:
        c_evts = evts_wide[evts_wide[c]].copy()
        c_evts["condition"] = c
        evts_long = pd.concat([evts_long, c_evts], ignore_index=True)
    if drop:
        evts_long = evts_long.drop(columns=condition_cols)
    return evts_long


def get_contrasts(
    c_means: pd.DataFrame,
    c_sums: pd.DataFrame = None,
    c_rates: pd.DataFrame = None,
    rate_columns: list[str] = None,
) -> pd.DataFrame:
    contrast_inputs = c_means
    if c_sums is not None:
        contrast_inputs = contrast_inputs.join(c_sums)
    if c_rates is not None:
        contrast_inputs = contrast_inputs.join(c_rates[rate_columns])

    contrast_dfs = {}
    for contrast, (condition_a, condition_b) in const.CONTRASTS.items():
        df = contrast_inputs.xs(condition_a, level="condition") - contrast_inputs.xs(
            condition_b, level="condition"
        )
        contrast_dfs[contrast] = df.add_suffix(f"_{contrast}")
    contrasts = pd.concat(list(contrast_dfs.values()), axis="columns")
    return contrasts


def aggregate_bandpowers(
    bipolar: bool,
    kind: Literal["stft", "inst"],
    band_names: list[str] = ["delta", "eta"],
    verbose: bool = False,
    spsl: list[tuple[str, str, str]] | None = None,
):
    """Per-timepoint log/zlog band power tagged by statistical condition.

    The z-score baseline is the whole clean ``Full.Conservative`` recording
    (minus ``Artifact``/``NoData``), computed independently per
    ``(subject, probe, structure)``, identical to the transform that annotates
    OFF periods in
    :mod:`cnpix_local_sleep.morphological.pipeline.add_bandpower_to_offs`.

    Parameters
    ----------
    spsl
        Explicit ``(subject, probe, structure)`` list to aggregate. Defaults to
        the finalized corticothalamic list for ``morphological`` (the historical
        behavior). Pass a cortical-only list to summarize just cortex.
    """
    res = pd.DataFrame()
    if spsl is None:
        spsl = sps_conf.get_finalized_corticothalamic_subject_probe_structure_list(
            method="morphological"
        )
    for subject, probe, structure in spsl:
        if verbose:
            print(f"Doing {subject}, {probe}, {structure}")
        hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
        hg = hgs.pop("Full.Conservative").drop_states({"Artifact", "NoData"})

        ds = xr.Dataset(
            {
                band_name: xr.load_dataarray(
                    files.get_structure_bandpower_path(
                        subject, probe, structure, band_name, bipolar, kind
                    )
                )
                for band_name in band_names
            }
        )
        ds = ds.isel(time=hg.covers_time(ds["time"]))
        ds = ds.dropna(dim="time")

        for band_name in band_names:
            ds[f"log_{band_name}"] = np.log10(ds[band_name])
            ds[f"zlog_{band_name}"] = xr.apply_ufunc(
                scipy.stats.zscore, ds[f"log_{band_name}"]
            )
            if (ds[f"log_{band_name}"] < 0).any():
                warnings.warn(
                    f"Low {band_name} for {subject}, {probe}, {structure}. Please investigate.",
                    UserWarning,
                )

        df = ds.to_dataframe().reset_index()
        df["subject"] = subject
        df["probe"] = probe
        df["structure"] = structure

        for c, hg in hgs.items():
            df[c] = hg.covers_time(df["time"])
        res = pd.concat([res, df], axis=0, ignore_index=True)

    for c in hgs.keys():
        res[c] = res[c].fillna(False)
    return res.drop(columns=["time"])