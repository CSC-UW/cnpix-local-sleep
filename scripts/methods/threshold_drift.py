"""Drift of a whole-recording NREM Q20 threshold across recording blocks.

Numbers for the Supplementary Methods paragraph on block-wise thresholding: the
CNPIX2-Segundo PPC example and the median drift across all channels of the included
subject-structure pairs.

Each pair's full 48 h MUA trace is read as ``morphological-offs detect-offs-full``
reads it (production reader, detection channel mask, detection-time median filter),
split into blocks at the zarr's time chunks (~5.5 min), and labelled with the
Full.Conservative hypnogram. Per channel, T is the 20th percentile of all the
recording's NREM samples. Per block, the fraction of that block's NREM samples below
T is the percentile of the block's own NREM distribution at which T sits. Blocks with
fewer than 5% NREM samples are dropped. A channel's drift is P90 - P10 of that
fraction across its blocks; the printed values are medians across channels.

Per-block fractions are cached as one parquet per pair in ``<out-dir>/per_block/``;
cached pairs are skipped. ``--summary-only`` re-summarises the cache without NFS.

Needs /Volumes/npx_nfs. Each worker holds one pair's full float64 trace in memory,
peaking at 25-65 GB (the 80-channel pairs are the largest). With 4 workers the 29
pairs take about 2 h; ``--summary-only`` takes under a minute.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python \\
        ../cnpix-local-sleep/scripts/methods/threshold_drift.py
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask
import dask.array
import numpy as np
import pandas as pd

from cnpix_local_sleep import atlas, hyp
from cnpix_local_sleep.morphological.detection_opts import (
    get_mua_spatial_detection_opts,
)
from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG, readers

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "threshold_drift"

STATE = "NREM"
QUANTILE = 0.20
MIN_STATE_FRAC = 0.05
EXAMPLE = ("CNPIX2-Segundo", "PPC")
DASK_THREADS = 8
PAIR_KEYS = ["subject", "probe", "structure"]


def part_path(out_dir: Path, subject: str, probe: str, structure: str) -> Path:
    return out_dir / "per_block" / f"{subject}_{probe}_{structure}.parquet"


def block_fractions(subject: str, probe: str, structure: str) -> pd.DataFrame:
    """Fraction of each block's NREM samples below the whole-recording NREM Q20."""
    opts = get_mua_spatial_detection_opts()
    lazy = readers.open_mua_traces_as_xarray(
        subject,
        probe,
        structure,
        condition=None,
        apply_detection_channel_mask=True,
        ndimage_filter_type=opts["ndimage_filter_type"],
        ndimage_filter_kwargs=opts["ndimage_filter_kwargs"],
    )
    bounds = readers.bin_boundaries_from_chunks(lazy)
    arr = np.empty(lazy.shape, dtype=lazy.dtype)
    with dask.config.set(scheduler="threads", num_workers=DASK_THREADS):
        dask.array.store(lazy.data, arr, lock=False)
    hg = hyp.load_statistical_condition_hypnograms(subject, probe)["Full.Conservative"]
    in_state = hg.get_states(lazy.time.values) == STATE
    thresh = np.array(
        [
            np.quantile(arr[in_state, k], QUANTILE, overwrite_input=True)
            for k in range(arr.shape[1])
        ]
    )
    channels = lazy.channel.values.astype(str)
    parts = []
    for block, (lo, hi) in enumerate(itertools.pairwise(bounds)):
        mask = in_state[lo:hi]
        n_state = int(mask.sum())
        if n_state == 0:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "probe": probe,
                    "structure": structure,
                    "block": block,
                    "block_len": int(hi - lo),
                    "n_state": n_state,
                    "channel": channels,
                    "frac_below": (arr[lo:hi][mask] < thresh).mean(axis=0),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def write_part(out_dir: Path, subject: str, probe: str, structure: str) -> None:
    df = block_fractions(subject, probe, structure)
    df.to_parquet(part_path(out_dir, subject, probe, structure), index=False)


def compute(out_dir: Path, pairs: list[tuple[str, str, str]], workers: int) -> None:
    (out_dir / "per_block").mkdir(parents=True, exist_ok=True)
    todo = [p for p in pairs if not part_path(out_dir, *p).exists()]
    print(f"{len(pairs) - len(todo)} of {len(pairs)} pairs cached", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futures = {ex.submit(write_part, out_dir, *p): p for p in todo}
        for f in as_completed(futures):
            print(" ".join(futures[f]), repr(f.exception() or "done"), flush=True)


def load_blocks(out_dir: Path, pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    paths = [part_path(out_dir, *p) for p in pairs]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No cached per-block fractions: {missing}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def per_channel(blocks: pd.DataFrame) -> pd.DataFrame:
    """P10, P90 and spread (P90 - P10) across blocks of each channel's fraction."""
    kept = blocks[blocks.n_state >= MIN_STATE_FRAC * blocks.block_len]
    g = kept.groupby([*PAIR_KEYS, "channel"])["frac_below"]
    chans = g.quantile([0.1, 0.9]).unstack().rename(columns={0.1: "p10", 0.9: "p90"})
    chans["spread"] = chans.p90 - chans.p10
    chans = chans.reset_index()
    chans["structure"] = chans.structure.map(atlas.consolidate_structure_label)
    return chans


def report(chans: pd.DataFrame) -> None:
    pairs = chans[["subject", "structure"]].drop_duplicates()
    print(
        f"{len(pairs)} subject-structure pairs, {pairs.subject.nunique()} subjects, "
        f"{len(chans)} channels"
    )
    ex = chans[(chans.subject == EXAMPLE[0]) & (chans.structure == EXAMPLE[1])]
    p10, p90, spread = (100 * ex[c].median() for c in ["p10", "p90", "spread"])
    print(
        f"{' '.join(EXAMPLE)} ({len(ex)} channels), medians across channels: "
        f"P10 = Q{p10:.2f}, P90 = Q{p90:.2f}, spread = {spread:.2f}"
    )
    print(f"All channels: median spread = {100 * chans.spread.median():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Pairs computed in parallel; each holds a full 48 h trace in memory.",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Summarise the cached per-block fractions without reading NFS.",
    )
    args = ap.parse_args()
    pairs = SOURCE_CONFIG.get_subject_probe_structure_list()
    if not args.summary_only:
        compute(args.out_dir, pairs, args.workers)
    chans = per_channel(load_blocks(args.out_dir, pairs))
    chans.to_parquet(args.out_dir / "per_channel.parquet", index=False)
    report(chans)


if __name__ == "__main__":
    main()
