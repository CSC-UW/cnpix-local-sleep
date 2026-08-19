from pathlib import Path

import ecephys.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubplots as pp
import seaborn as sns
import xarray as xr

from cnpix_local_sleep import files, hyp, plots
from cnpix_local_sleep.files import get_subject_plot_dir
from cnpix_local_sleep.morphological.common import MorphologicalSourceConfig

NREM_CONDITIONS = [
    "Early.BSL.NREM",
    "Early.REC.NREM.Match",
    "Early.REC.NREM",
    "Late.REC.NREM",
]
NOD_CONDITIONS = ["Early.NOD", "Late.NOD"]

_VALUE_COLS = [
    "area",
    "median_duration",
    "span",
    "median_trace",
    "mad_trace",
    "min_trace",
    "center_of_mass_depth",
    "onset_slope",
    "onset_jitter",
    "onset_mad",
    "onset_r2",
    "offset_slope",
    "offset_jitter",
    "offset_mad",
    "offset_r2",
    "onset_offset_wedge",
]
_SMOOTHING = "20s"
_MAX_OBS = 20_000
_RANDOM_STATE = 0

_OFF_MARKER_SIZES = {
    "Early.BSL.NREM": 0.2,
    "Early.REC.NREM.Match": 0.2,
    "Early.REC.NREM": 0.2,
    "Late.REC.NREM": 0.2,
    "Early.NOD": 1.0,
    "Late.NOD": 1.0,
}
_PWR_MARKER_SIZE = 0.2

# Row configuration: maps variable name to display settings.
# - "source": "mean" (mean-smoothed OFFs), "sum" (sum-smoothed/normalized OFFs),
#              or "xarray" (power trace from an xarray DataArray).
# - "has_sem": whether SEM fill_between is available (only for "mean" source).
# - "nrem" / "nod": optional dicts with "ylim", "yticks", "yticklabels" overrides
#                    applied when the condition belongs to that group.
# For "xarray" sources, the DataArray is injected at runtime via the `xarray_data`
# argument to `_plot_condition_grid`.
ROW_CONFIGS = {
    "median_duration": {
        "source": "mean",
        "has_sem": True,
        "nrem": {
            "ylim": (0.0, 0.2),
            "yticks": [0.0, 0.1, 0.2],
            "yticklabels": ["0", "100", "200"],
        },
    },
    "span": {"source": "mean", "has_sem": True},
    "area": {
        "source": "sum",
        "nrem": {"yticks": [0, 100, 200]},
    },
    "delta": {
        "source": "xarray",
        "nrem": {"yticks": [0, 100, 200]},
    },
    "eta": {
        "source": "xarray",
        "nrem": {"yticks": [0, 100, 200]},
    },
    "median_trace": {"source": "mean", "has_sem": True},
    "mad_trace": {"source": "mean", "has_sem": True},
    "min_trace": {"source": "mean", "has_sem": True},
    "center_of_mass_depth": {"source": "mean", "has_sem": True},
    "onset_slope": {"source": "mean", "has_sem": True},
    "onset_jitter": {"source": "mean", "has_sem": True},
    "onset_mad": {"source": "mean", "has_sem": True},
    "onset_r2": {"source": "mean", "has_sem": True},
    "offset_slope": {"source": "mean", "has_sem": True},
    "offset_jitter": {"source": "mean", "has_sem": True},
    "offset_mad": {"source": "mean", "has_sem": True},
    "offset_r2": {"source": "mean", "has_sem": True},
    "onset_offset_wedge": {"source": "mean", "has_sem": True},
}

DEFAULT_ROWS = ["median_duration", "span", "area", "delta"]


def _prepare_data(
    subject: str,
    probe: str,
    structure: str,
    llas_all: pd.DataFrame,
) -> dict:
    """Load and precompute all data needed for condition grid plots."""
    # Filter OFFs to this subject/probe/structure
    cols = ["start_time"] + _VALUE_COLS
    mask = (
        (llas_all["subject"] == subject)
        & (llas_all["probe"] == probe)
        & (llas_all["structure"] == structure)
    )
    llas = (
        llas_all.loc[mask].sort_values("start_time").reset_index(drop=True).loc[:, cols]
    )

    # Load hypnograms
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    hg_full = hgs["Full.Conservative"].drop_states({"Artifact", "NoData"})

    # Load and normalize delta bandpower
    delta = xr.load_dataarray(
        files.get_structure_bandpower_path(
            subject, probe, structure, "delta", True, "inst"
        )
    )
    bsl_nrem_mask = hg_full.keep_states(["NREM"]).covers_time(delta.time)
    bsl_nrem_mean = delta.isel(time=bsl_nrem_mask).mean(dim="time")
    delta = (delta / bsl_nrem_mean) * 100.0

    # Load and normalize eta bandpower
    eta = xr.load_dataarray(
        files.get_structure_bandpower_path(
            subject, probe, structure, "eta", True, "inst"
        )
    )
    bsl_nrem_mask = hg_full.keep_states(["NREM"]).covers_time(eta.time)
    bsl_nrem_mean = eta.isel(time=bsl_nrem_mask).mean(dim="time")
    eta = (eta / bsl_nrem_mean) * 100.0

    # Downsample power traces for scatter plotting
    delta_smth = delta.rolling(time=1024).mean()
    delta_smth = delta_smth.isel(time=slice(None, None, 5))

    eta_smth = eta.rolling(time=1024).mean()
    eta_smth = eta_smth.isel(time=slice(None, None, 5))

    # Compute smoothed OFF traces
    llas_smoothed_mean = plots.get_smoothed_trace(
        llas, _VALUE_COLS, smoothing=_SMOOTHING, rolling_op="mean"
    )
    llas_smoothed_sem = plots.get_smoothed_trace(
        llas, _VALUE_COLS, smoothing=_SMOOTHING, rolling_op="sem"
    )
    llas_smoothed_sum = plots.get_smoothed_trace(
        llas,
        _VALUE_COLS,
        smoothing=_SMOOTHING,
        rolling_op="sum",
        fill_values={c: np.nan for c in _VALUE_COLS},
    )

    # Normalize sum trace to NREM baseline
    bsl_nrem_mask = hg_full.keep_states(["NREM"]).covers_time(
        llas_smoothed_sum["start_time"].values
    )
    bsl_nrem_mean = llas_smoothed_sum.loc[bsl_nrem_mask, _VALUE_COLS].mean()
    llas_smoothed_sum[_VALUE_COLS] = (
        llas_smoothed_sum[_VALUE_COLS] / bsl_nrem_mean
    ) * 100.0

    return dict(
        hgs=hgs,
        hg_full=hg_full,
        delta_smth=delta_smth,
        eta_smth=eta_smth,
        llas_smoothed_mean=llas_smoothed_mean,
        llas_smoothed_sem=llas_smoothed_sem,
        llas_smoothed_sum=llas_smoothed_sum,
    )


def _plot_condition_grid(
    conditions: list[str],
    hgs: dict,
    llas_smoothed_mean: pd.DataFrame,
    llas_smoothed_sem: pd.DataFrame,
    llas_smoothed_sum: pd.DataFrame,
    delta_smth: xr.DataArray,
    eta_smth: xr.DataArray,
    hg_full,
    save_path: Path,
    rows: list[str] | None = None,
    plot_sem: bool = False,
    show_labels: bool = True,
    show_titles: bool = True,
) -> None:
    """Plot a condition grid: rows are OFF properties + power, columns are conditions."""
    if rows is None:
        rows = DEFAULT_ROWS
    n_rows = len(rows)
    n_conditions = len(conditions)

    # Map xarray row names to their DataArrays
    xarray_data = {"delta": delta_smth, "eta": eta_smth}

    # Determine condition group for axis overrides
    is_nrem = set(conditions) <= set(NREM_CONDITIONS)
    is_nod = set(conditions) <= set(NOD_CONDITIONS)
    cnd_group = "nrem" if is_nrem else ("nod" if is_nod else None)

    # Collect y-data per row for quantile-based ylim
    row_ydata: dict[str, list[np.ndarray]] = {var: [] for var in rows}

    with pp.destination("figma"):
        fig, axes = plt.subplots(
            n_rows,
            n_conditions,
            figsize=(3.5 * n_conditions, 1.5 * n_rows),
            sharey="row",
            sharex="col",
            squeeze=False,
        )

        for col, condition in enumerate(conditions):
            cnd_hg = hgs[condition]
            if condition == "Early.NOD":
                cnd_hg = cnd_hg.keep_last(60 * 58)
            cnd_start = cnd_hg["start_time"].min()
            cnd_end = cnd_hg["end_time"].max()

            # Mask smoothed OFF data to this condition
            sum_mask = cnd_hg.covers_time(llas_smoothed_sum["start_time"].values)
            cnd_sum = llas_smoothed_sum.loc[sum_mask]

            mean_mask = cnd_hg.covers_time(llas_smoothed_mean["start_time"].values)
            cnd_mean = llas_smoothed_mean.loc[mean_mask]

            sem_mask = cnd_hg.covers_time(llas_smoothed_sem["start_time"].values)
            cnd_sem = llas_smoothed_sem.loc[sem_mask]

            # Downsample independently per source if needed
            if len(cnd_sum) > _MAX_OBS:
                cnd_sum = cnd_sum.sample(
                    n=_MAX_OBS, random_state=_RANDOM_STATE
                ).sort_values("start_time")
            if len(cnd_mean) > _MAX_OBS:
                cnd_mean = cnd_mean.sample(
                    n=_MAX_OBS, random_state=_RANDOM_STATE
                ).sort_values("start_time")
                cnd_sem = cnd_sem.sample(
                    n=_MAX_OBS, random_state=_RANDOM_STATE
                ).sort_values("start_time")

            marker_size = _OFF_MARKER_SIZES.get(condition, 0.2)

            for row_idx, var in enumerate(rows):
                ax = axes[row_idx, col]
                cfg = ROW_CONFIGS[var]
                source = cfg["source"]

                if source == "xarray":
                    da = xarray_data[var].sel(time=slice(cnd_start, cnd_end))
                    sns.scatterplot(
                        x=da.time.values,
                        y=da.values,
                        color="k",
                        s=_PWR_MARKER_SIZE,
                        ax=ax,
                    )
                    row_ydata[var].append(da.values)
                elif source == "sum":
                    sns.scatterplot(
                        data=cnd_sum,
                        x="start_time",
                        y=var,
                        color="k",
                        s=marker_size,
                        ax=ax,
                    )
                    row_ydata[var].append(cnd_sum[var].values)
                else:  # "mean"
                    sns.scatterplot(
                        data=cnd_mean,
                        x="start_time",
                        y=var,
                        color="k",
                        s=marker_size,
                        ax=ax,
                    )
                    if plot_sem and cfg.get("has_sem", False):
                        ax.fill_between(
                            cnd_mean["start_time"],
                            cnd_mean[var] - cnd_sem[var],
                            cnd_mean[var] + cnd_sem[var],
                            color="k",
                            alpha=0.2,
                        )
                    row_ydata[var].append(cnd_mean[var].values)

                ax.set(xlabel=None, ylabel=None, xmargin=0, xticks=[])

                # Apply per-condition-group axis overrides
                if cnd_group and cnd_group in cfg:
                    overrides = cfg[cnd_group]
                    if "ylim" in overrides:
                        ax.set_ylim(*overrides["ylim"])
                    if "yticks" in overrides:
                        ax.set_yticks(overrides["yticks"])
                    if "yticklabels" in overrides:
                        ax.set_yticklabels(overrides["yticklabels"])

                ecephys.plot.plot_hypnogram_overlay(
                    hgs["Full.Conservative"],
                    ax=ax,
                    state_colors=ecephys.plot.publication_colors,
                )

            if show_titles:
                axes[0, col].set_title(condition, fontsize=8)

        # Set quantile-based ylim for rows without explicit ylim
        for row_idx, var in enumerate(rows):
            cfg = ROW_CONFIGS[var]
            has_explicit_ylim = (
                cnd_group and cnd_group in cfg and "ylim" in cfg[cnd_group]
            )
            if not has_explicit_ylim:
                all_y = np.concatenate(row_ydata[var])
                if np.any(np.isfinite(all_y)):
                    lo, hi = np.nanpercentile(all_y, [1, 99])
                    axes[row_idx, 0].set_ylim(lo, hi)

        if show_labels:
            for row_idx, var in enumerate(rows):
                axes[row_idx, 0].set_ylabel(var)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_offs_vs_time(
    subject: str,
    probe: str,
    structure: str,
    variant: str,
    data: dict,
) -> None:
    """Generate NREM and NOD condition grid plots for one cortical structure.

    ``variant`` is the morphological variant (``"morphological"``); it is encoded into
    output filenames.
    """
    plot_dir = get_subject_plot_dir(subject)
    rows = [
        "onset_mad",
        "offset_mad",
        "onset_offset_wedge",
        "median_trace",
        "min_trace",
        "mad_trace",
        "area",
        "delta",
    ]

    _plot_condition_grid(
        NREM_CONDITIONS,
        save_path=plot_dir / f"{probe}.{structure}.{variant}.llas_timecourse_nrem.png",
        rows=rows,
        **data,
    )
    _plot_condition_grid(
        NOD_CONDITIONS,
        save_path=plot_dir / f"{probe}.{structure}.{variant}.llas_timecourse_nod.png",
        rows=rows,
        **data,
    )


def do_project(source_config: MorphologicalSourceConfig):
    """Plot OFF timecourses for every included structure of one morphological variant.

    ``source_config`` supplies both the on-disk ``method=`` path segment
    (via ``files_module``) and the inclusion list. No default; the caller
    must pass ``cnpix_local_sleep.morphological.mua.SOURCE_CONFIG``.
    """
    llas_all = pd.read_parquet(source_config.files_module.get_path("llas_offs.parquet"))
    spsl = source_config.get_subject_probe_structure_list(
        exclude_thalamus=True,
        exclude_striatum=True,
        exclude_other=True,
    )

    for subject, probe, structure in spsl:
        print(f"Plotting OFF timecourses for {subject} {probe} {structure}")
        data = _prepare_data(subject, probe, structure, llas_all)
        plot_offs_vs_time(subject, probe, structure, source_config.variant, data)