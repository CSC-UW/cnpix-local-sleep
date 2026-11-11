"""Manually annotated wake OFF periods over sleep deprivation.

Numbers for the last paragraph of the Supplementary Methods section "Agreement between
detectors and manual annotation".

The sleep-deprivation annotation is read on its image-stack grid, after the label QC
that the agreement scores use, for the structures those scores use. Within a
structure's detection rows, every labelled instance in the images up to the probe's
last labelled one is one OFF period. Its area, median duration, span and median MUA
are computed on the annotation grid as ``morphology.get_off_properties`` computes them
for detected OFF periods, with the MUA trace read and smoothed as detection reads it.
Only OFF periods whose samples are all scored Wake are kept.

Each OFF period is assigned to the first or second half of the probe's sleep
deprivation, by its first sample, and the scored Wake time of each half gives the rate
and cumulative area per structure. It is also assigned to the first hour of sleep
deprivation (``Early.NOD.Wake``) or to the last cumulative hour of scored Wake ending
at the probe's last labelled image, when it lies entirely in one of them. Delta power
at the OFF period's midpoint is read as for detected OFF periods.

Writes ``offs.parquet`` and ``rates.parquet`` into ``--out-dir`` and prints the pooled
correlations with delta power. ``annotated_wake_offs.R`` fits the mixed models from
those two files. ``--summary-only`` reprints the correlations without NFS.

Needs /Volumes/npx_nfs. Each worker holds one probe's MUA traces over the annotated
part of sleep deprivation; the 11 probes take about 20 min with 4 workers.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python \\
        ../cnpix-local-sleep/scripts/methods/annotated_wake_offs.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import zarr
from cnpix import stacks
from cnpix.evaluation import labels as cl

from cnpix_local_sleep import hyp, trace_io
from cnpix_local_sleep.evaluation import config, grid, head_to_head
from cnpix_local_sleep.morphological import correlation_stats
from cnpix_local_sleep.morphological.detection_opts import (
    get_mua_spatial_detection_opts,
)
from cnpix_local_sleep.morphological.mua import readers
from cnpix_local_sleep.morphological.pipeline import add_bandpower_to_offs

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "annotated_wake_offs"

FIRST_HOUR = "Early.NOD.Wake"
HOUR_S = 3600.0
MUA_MARGIN_S = 2.0
DASK_THREADS = 8
PROPERTIES = ["area", "median_duration", "span", "median_mua"]
DELTA = "centre_bipolar_inst_log_delta"
PAIR_KEYS = ["subject", "probe", "structure"]


def last_wake_hour_start(wake: pd.DataFrame, t_end: float) -> float:
    """Start of the last cumulative hour of scored Wake that ends at ``t_end``."""
    wake = wake[wake.start_time < t_end].sort_values("start_time")
    remaining = HOUR_S
    for start, end in zip(wake.start_time[::-1], wake.end_time[::-1], strict=True):
        end = min(end, t_end)
        if end - start >= remaining:
            return end - remaining
        remaining -= end - start
    return np.nan


def median_mua(flat, row, label, mua, mua_ix, grid_ix, row_to_col) -> pd.Series:
    """Per label, the median of the MUA samples that fall on its grid pixels."""
    lo = np.searchsorted(grid_ix, flat, "left")
    n = np.searchsorted(grid_ix, flat, "right") - lo
    pixel = np.repeat(np.arange(len(flat)), n)
    within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    values = mua[mua_ix[np.repeat(lo, n) + within], row_to_col[row][pixel]]
    return pd.Series(values).groupby(label[pixel]).median()


def probe_offs(
    subject: str, probe: str, structures: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotated wake OFF periods of one probe, and the scored Wake time of each half."""
    condition = config.SD_CONDITION
    ts = zarr.open(str(stacks.timestamps_path(subject, probe, condition)), mode="r")[:]
    labels, _ = cl.qc_and_fix_labels(
        cl.load_manual_labels(subject, probe, condition),
        contiguous=cl.chunk_contiguity(ts),
    )
    n_inspected = np.flatnonzero(labels.any(axis=(1, 2)))[-1] + 1
    labels, ts = labels[:n_inspected], ts[:n_inspected]
    n_rows, spc = labels.shape[1:]
    t = ts.reshape(-1)

    full = trace_io.open_preprocessed_traces_as_xarray(
        subject, probe, structure=None, condition=None, apply_detection_channel_mask=False
    )
    fs = float(full.attrs["fs"])
    pitch = float(np.median(np.diff(full.y.values)))
    row_depth, row_channel = full.y.values[::-1], full.channel.values[::-1]
    det_rows = {
        st: np.flatnonzero(
            head_to_head._mua_detection_row_masks(subject, probe, None, n_rows, st)[0]
        )
        for st in structures
    }

    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    sd = hgs[condition]
    wake_hg = sd.keep_states(["Wake"])
    wake = np.asarray(wake_hg.covers_time(t))
    in_first_hour = np.asarray(hgs[FIRST_HOUR].covers_time(t))
    midpoint = (sd.start_time.min() + sd.end_time.max()) / 2
    last_hour_start = last_wake_hour_start(wake_hg, t[-1] + 1 / fs)
    hours_overlap = last_hour_start < hgs[FIRST_HOUR].end_time.max()

    opts = get_mua_spatial_detection_opts()
    mua = readers.open_mua_traces_as_xarray(
        subject, probe, structure=None, condition=None, apply_detection_channel_mask=False
    )
    mua_channels = list(mua.channel.values)
    channels = sorted(
        {row_channel[r] for rows in det_rows.values() for r in rows}, key=mua_channels.index
    )
    col = {name: i for i, name in enumerate(channels)}
    i0, i1 = np.searchsorted(mua.time.values, [t[0] - MUA_MARGIN_S, t[-1] + MUA_MARGIN_S])
    mua = mua.isel(time=slice(int(i0), int(i1))).sel(channel=channels)
    mua.data = trace_io.smooth_dask_image(
        mua.data, opts["ndimage_filter_type"], opts["ndimage_filter_kwargs"]
    )
    with dask.config.set(scheduler="threads", num_workers=DASK_THREADS):
        mua_values = np.ascontiguousarray(mua.values)
    grid_ix = grid.times_to_grid_index(mua.time.values, t)
    mua_ix = np.flatnonzero(grid_ix >= 0)
    grid_ix = grid_ix[mua_ix]

    offs, rates = [], []
    for st, rows in det_rows.items():
        sub = labels[:, rows]
        img, row, sample = np.nonzero(sub)
        label = sub[img, row, sample]
        flat = img.astype(np.int64) * spc + sample
        px = pd.DataFrame(
            {
                "label": label,
                "t": t[flat],
                "row": row,
                "depth": row_depth[rows[row]],
                "wake": wake[flat],
                "first_hour": in_first_hour[flat],
            }
        )
        ev = px.groupby("label").agg(
            n_pixels=("t", "size"),
            start_time=("t", "min"),
            last_sample=("t", "max"),
            lo=("depth", "min"),
            hi=("depth", "max"),
            all_wake=("wake", "all"),
            all_first_hour=("first_hour", "all"),
        )
        ev["end_time"] = ev.last_sample + 1 / fs
        ev["area"] = ev.n_pixels * pitch / fs
        ev["median_duration"] = px.groupby(["label", "row"]).size().groupby("label").median() / fs
        ev["span"] = ev.hi - ev.lo
        row_to_col = np.array([col[row_channel[r]] for r in rows])
        ev["median_mua"] = median_mua(flat, row, label, mua_values, mua_ix, grid_ix, row_to_col)
        ev = ev[ev.all_wake].copy()
        ev["half"] = np.where(ev.start_time < midpoint, "first", "second")
        ev["hour"] = ""
        if not hours_overlap:
            ev.loc[ev.start_time >= last_hour_start, "hour"] = "last"
            ev.loc[ev.all_first_hour, "hour"] = "first"
        ev = add_bandpower_to_offs.do_band(subject, probe, st, "delta", True, "inst", ev)
        offs.append(ev.assign(subject=subject, probe=probe, structure=st))
        for half, in_half in (("first", t < midpoint), ("second", t >= midpoint)):
            minutes = (wake & in_half).sum() / fs / 60
            e = ev[ev.half == half]
            rates.append(
                {
                    "subject": subject,
                    "probe": probe,
                    "structure": st,
                    "half": half,
                    "wake_minutes": minutes,
                    "rate": len(e) / minutes,
                    "cumulative_area": e.area.sum() / minutes,
                }
            )
    cols = [*PAIR_KEYS, "half", "hour", *PROPERTIES, DELTA]
    return pd.concat(offs, ignore_index=True)[cols], pd.DataFrame(rates)


def compute(out_dir: Path, workers: int) -> None:
    by_probe: dict[tuple[str, str], list[str]] = {}
    for subject, probe, structure in head_to_head.labeled_cortical_structures("SD"):
        by_probe.setdefault((subject, probe), []).append(structure)
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(probe_offs, s, p, sts) for (s, p), sts in by_probe.items()]
        results = [f.result() for f in futures]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([r[0] for r in results], ignore_index=True).to_parquet(out_dir / "offs.parquet")
    pd.concat([r[1] for r in results], ignore_index=True).to_parquet(out_dir / "rates.parquet")


def report(offs: pd.DataFrame) -> None:
    pairs = offs[PAIR_KEYS].drop_duplicates()
    print(
        f"{len(offs)} annotated wake OFF periods in {len(pairs)} structures "
        f"of {pairs.subject.nunique()} rats"
    )
    print("Pooled Spearman correlation with delta power at the OFF midpoint:")
    for prop in PROPERTIES:
        per_pair = correlation_stats.compute_group_correlations(offs, prop, DELTA)
        per_pair = per_pair.dropna(subset=["rho"])
        m = correlation_stats.meta_analyze_correlations(per_pair)
        print(
            f"  {prop}: rho = {m['overall_rho']:.2f} [{m['ci_lo']:.2f}, {m['ci_hi']:.2f}], "
            f"p = {m['p_value']:.2g}, Bonferroni p = {min(1, len(PROPERTIES) * m['p_value']):.2g} "
            f"({len(per_pair)} structures, {int(per_pair.n.sum())} OFF periods)"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Probes computed in parallel; each holds its MUA traces in memory.",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Reprint the correlations from offs.parquet without reading NFS.",
    )
    args = ap.parse_args()
    if not args.summary_only:
        compute(args.out_dir, args.workers)
    report(pd.read_parquet(args.out_dir / "offs.parquet"))


if __name__ == "__main__":
    main()
