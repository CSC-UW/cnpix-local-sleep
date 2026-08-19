"""Post-flight sanity check on ``detect-offs-full`` outputs.

For each SPS in ``quantile_thresholds.csv`` whose quantile is populated:
asserts the expected output files exist, counts MUA full-recording OFFs
in sleep states, and reports the CLAS pass-through fraction.

Writes a summary CSV under this repo's ``agent_docs/`` and exits non-zero if
any SPS is missing outputs or has zero OFFs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG
from cnpix_local_sleep.morphological.mua import files as mua_files
from cnpix_local_sleep import off_tables


CLAS = off_tables.NAMED_FILTERS["clas"]
SLEEP_STATES = ("NREM", "IS", "REM")


def _count_mua_full(subject: str, probe: str, structure: str) -> tuple[int, int, int]:
    """Return (n_total, n_nrem_like, n_clas_nrem_like)."""
    path = mua_files.get_full_offs_path(subject=subject, probe=probe, structure=structure)
    df = pd.read_parquet(path)
    nrem_like = df[df["state"].isin(SLEEP_STATES)] if "state" in df.columns else df
    n_total = len(df)
    n_nrem = len(nrem_like)
    mask = pd.Series(True, index=nrem_like.index)
    for col, (lo, hi) in CLAS.items():
        if col in nrem_like.columns:
            mask &= nrem_like[col].between(lo, hi)
    return n_total, n_nrem, int(mask.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "agent_docs"
        / "detect_offs_full_verification.csv",
        help="Output summary CSV path.",
    )
    args = ap.parse_args()

    df = SOURCE_CONFIG.load_quantile_thresholds()
    have = df["nrem_quantile_threshold"].notna() | df["wake_quantile_threshold"].notna()
    targets = df[have][["subject", "probe", "structure_acronym"]]

    rows = []
    missing_outputs: list[str] = []
    empty_results: list[str] = []

    for _, r in targets.iterrows():
        subj, pr, st = r["subject"], r["probe"], r["structure_acronym"]
        offs_path = mua_files.get_full_offs_path(subject=subj, probe=pr, structure=st)
        lbl_path = mua_files.get_full_off_label_indices_path(
            subject=subj, probe=pr, structure=st
        )
        nrem_zarr = mua_files.get_full_channel_thresholds_path(
            subject=subj, probe=pr, structure=st, threshold_type="nrem"
        )
        wake_zarr = mua_files.get_full_channel_thresholds_path(
            subject=subj, probe=pr, structure=st, threshold_type="wake"
        )

        have_offs = offs_path.exists()
        have_lbl = lbl_path.exists()
        have_nrem = nrem_zarr.exists()
        have_wake = wake_zarr.exists()
        all_ok = have_offs and have_lbl and have_nrem and have_wake
        key = f"{subj}/{pr}/{st}"

        if not all_ok:
            missing_outputs.append(key)
            rows.append(
                dict(
                    subject=subj, probe=pr, structure=st,
                    have_offs=have_offs, have_lbl=have_lbl,
                    have_nrem=have_nrem, have_wake=have_wake,
                    n_total=None, n_mua_nrem=None, n_mua_clas_nrem=None,
                    mua_clas_frac=None,
                )
            )
            continue

        n_total, n_mua_nrem, n_mua_clas_nrem = _count_mua_full(subj, pr, st)
        if n_mua_nrem == 0:
            empty_results.append(key)
        mua_clas_frac = (n_mua_clas_nrem / n_mua_nrem) if n_mua_nrem else float("nan")

        rows.append(
            dict(
                subject=subj, probe=pr, structure=st,
                have_offs=True, have_lbl=True, have_nrem=True, have_wake=True,
                n_total=n_total,
                n_mua_nrem=n_mua_nrem,
                n_mua_clas_nrem=n_mua_clas_nrem,
                mua_clas_frac=round(mua_clas_frac, 4),
            )
        )

    summary = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.to_string(index=False))

    print(f"\nWrote {args.out}")
    print(f"Targets: {len(targets)}")
    print(f"Missing outputs: {len(missing_outputs)}")
    print(f"Zero-NREM results: {len(empty_results)}")

    if missing_outputs:
        print("\nMissing outputs:")
        for k in missing_outputs:
            print(f"  {k}")
    if empty_results:
        print("\nZero-NREM results:")
        for k in empty_results:
            print(f"  {k}")

    return 1 if (missing_outputs or empty_results) else 0


if __name__ == "__main__":
    sys.exit(main())
