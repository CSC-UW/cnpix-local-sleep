"""Pre-flight check on morphological quantile_thresholds.csv coverage.

Reports cortex SPS coverage in the CSV and exercises the exact lookup
path that ``detect_offs_full`` uses (``SOURCE_CONFIG.get_quantile_threshold``).
Exit non-zero if any cortex row is missing a quantile unless
``--allow-incomplete``.
"""

from __future__ import annotations

import argparse
import sys

from cnpix_local_sleep import sps_conf
from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit 0 even if some cortex rows are missing quantiles.",
    )
    args = ap.parse_args()

    df = SOURCE_CONFIG.load_quantile_thresholds()
    cortex = sps_conf.get_subject_probe_structure_list(
        exclude_thalamus=True, exclude_striatum=True, exclude_other=True
    )
    cortex_keys = set(map(tuple, cortex))

    cortex_mask = df.apply(
        lambda r: (r["subject"], r["probe"], r["structure_acronym"]) in cortex_keys,
        axis=1,
    )
    cortex_df = df[cortex_mask]

    have_nrem = cortex_df["nrem_quantile_threshold"].notna()
    have_wake = cortex_df["wake_quantile_threshold"].notna()
    print(f"Cortex rows in CSV: {len(cortex_df)} / {len(cortex_keys)} expected")
    print(f"  both:    {(have_nrem & have_wake).sum()}")
    print(f"  NREM only: {(have_nrem & ~have_wake).sum()}")
    print(f"  Wake only: {(~have_nrem & have_wake).sum()}")
    print(f"  neither: {(~have_nrem & ~have_wake).sum()}")

    missing = cortex_df[~(have_nrem & have_wake)]
    if not missing.empty:
        print("\nCortex SPS with missing quantiles:")
        for _, r in missing.iterrows():
            print(
                f"  {r['subject']}/{r['probe']}/{r['structure_acronym']} "
                f"nrem={r['nrem_quantile_threshold']} wake={r['wake_quantile_threshold']}"
            )

    errors: list[str] = []
    smoke_conds = {"Early.BSL.NREM": "nrem", "Early.NOD.Wake": "wake"}
    for _, r in cortex_df.iterrows():
        for cond in smoke_conds:
            try:
                q = SOURCE_CONFIG.get_quantile_threshold(
                    r["subject"], r["probe"], r["structure_acronym"], cond
                )
                if not (0.0 <= q <= 1.0):
                    errors.append(
                        f"{r['subject']}/{r['probe']}/{r['structure_acronym']} "
                        f"{cond}: quantile {q} out of [0, 1]"
                    )
            except (KeyError, ValueError) as e:
                errors.append(
                    f"{r['subject']}/{r['probe']}/{r['structure_acronym']} {cond}: {e}"
                )

    complete = (have_nrem & have_wake).sum() == len(cortex_keys)
    if errors:
        print(f"\nLookup errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    if not complete and not args.allow_incomplete:
        print("\nFAIL: cortex coverage incomplete and --allow-incomplete not set.")
        return 1
    if any("out of" in e for e in errors):
        print("\nFAIL: out-of-range quantiles.")
        return 1
    print("\nOK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
