"""Validate banded (spatially-resolved) unit-based OFF detection.  

For each manually-labeled cortical (subject, probe, structure), run banded OFF
detection (:func:`cnpix_local_sleep.unit_based.banded.detect_structure_banded`) under a few
ablation configs and score the result, with the shared :mod:`cnpix_local_sleep.evaluation`
kernels, against:

  1. manual OFF labels (ground truth), and
  2. morphological spatial OFFs (same-modality cross-check, spatial-to-spatial).

Compute is bounded to the evaluation domain: detection runs only over the labeled
stack chunks (:func:`cnpix_local_sleep.unit_based.banded_eval.labeled_chunk_bouts`), not the
full 48 h. By default a small structure subset and the core configs run; pass
``--full`` / ``--limit`` to widen.

Examples
--------
    # core configs on 3 structures (default), write CSV + print summary
    python scripts/banded_unit_based_validation.py

    # one structure, every config
    python scripts/banded_unit_based_validation.py \
        --subject CNPIX12-Santiago --probe imec0 --structure M2 --full
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from cnpix_local_sleep.evaluation import banded_vs_morphological
from cnpix_local_sleep.evaluation.head_to_head import labeled_cortical_structures
from cnpix_local_sleep.unit_based import banded, banded_eval
from cnpix_local_sleep.unit_based import files as ub_files

# Named ablation configs. Each maps to detect_structure_banded kwargs. Keeping a
# curated list (not a full cross-product) caps runs at the distinct outcomes we
# actually want to compare.
CONFIGS: dict[str, dict] = {
    # Engine: sticky + cap0 is the hypothesised best; threshold is the baseline.
    "sticky_cap0_ladder": dict(
        algo="sticky",
        band_definition="fixed_tiled",
        band_sizes=[250.0, 500.0, None],
        tile_start="superficial",
        param_strategy="shared",
    ),
    "threshold_ladder": dict(
        algo="threshold",
        band_definition="fixed_tiled",
        band_sizes=[250.0, 500.0, None],
        tile_start="superficial",
        param_strategy="shared",
    ),
    # Band definition: single fine scale vs ladder vs greedy-FR.
    "sticky_cap0_250only": dict(
        algo="sticky",
        band_definition="fixed_tiled",
        band_sizes=[250.0, None],
        tile_start="superficial",
        param_strategy="shared",
    ),
    "sticky_cap0_greedy": dict(
        algo="sticky",
        band_definition="greedy_fr",
        param_strategy="shared",
    ),
    # Morphological duration cleaning: raise the pre-merge floor 30 -> 50 ms (measured
    # best single move on M2). Post-merge floors are swept separately (see --post-sweep).
    "sticky_cap0_greedy_pre50": dict(
        algo="sticky",
        band_definition="greedy_fr",
        param_strategy="shared",
        min_band_off_duration=0.05,
    ),
    # Two named points from the post-merge union sweep, for the diagnostic gallery:
    # post 50 ms (== pre floor, effectively no post-filter) and the F1-peak post 80 ms.
    "sticky_cap0_greedy_pre50_post50": dict(
        algo="sticky",
        band_definition="greedy_fr",
        param_strategy="shared",
        min_band_off_duration=0.05,
        min_merged_off_duration=0.05,
    ),
    "sticky_cap0_greedy_pre50_post80": dict(
        algo="sticky",
        band_definition="greedy_fr",
        param_strategy="shared",
        min_band_off_duration=0.05,
        min_merged_off_duration=0.08,
    ),
    "sticky_cap0_deep": dict(
        algo="sticky",
        band_definition="fixed_tiled",
        band_sizes=[250.0, 500.0, None],
        tile_start="deep",
        param_strategy="shared",
    ),
    # Parameters: adaptive per-unit cap vs shared cap0.
    "sticky_adaptive_perunit": dict(
        algo="sticky",
        band_definition="fixed_tiled",
        band_sizes=[250.0, 500.0, None],
        tile_start="superficial",
        param_strategy="adaptive",
        adaptive_scheme="per_unit",
    ),
}

DEFAULT_CONFIGS = [
    "sticky_cap0_ladder",
    "threshold_ladder",
    "sticky_cap0_250only",
    "sticky_cap0_greedy",
]

# Post-merge sweep (with --post-sweep): hold pre at 50 ms, sweep the post-merge floor on
# each merged OFF's UNION duration (full extent) from 50 -> 100 ms in 10 ms steps.
# post=50 ms (== pre floor) is effectively no post-filter (the union is always >= every
# constituent band-OFF). Detection runs ONCE; each threshold is a cheap filter + re-score.
POST_SWEEP_MS = list(range(50, 101, 10))

# LAS filter for the morphological cross-check (rasterized via its TRUE per-pixel masks).
MUA_FILTER = "clas"


def mua_raster_for(subject, probe, structure, eval_name):
    """Precompute the morphological true-mask raster for one structure (config-independent).

    Returns ``None`` (with a note) if it can't be built, so scoring/plotting fall back to
    computing it lazily.
    """
    try:
        cfg = banded_eval.config.EVAL_CONFIGS[eval_name]
        condition = cfg["condition"]
        manual, _ = banded_eval.labels.qc_and_fix_labels(
            banded_eval.labels.load_manual_labels(subject, probe, condition)
        )
        return banded_vs_morphological.rasterize_morphological_masks(
            subject,
            probe,
            structure,
            condition,
            manual.shape,
            filter_name=MUA_FILTER,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort; reported and skipped
        print(f"    morphological raster failed: {exc!r}")
        return None


def run_one(
    subject,
    probe,
    structure,
    config_name,
    cfg,
    eval_name,
    *,
    verbose=False,
    plots=True,
    max_plot_windows=3,
    fig_dir=None,
    mua_raster=None,
):
    """Detect + score one (structure, config); return a list of result rows.

    ``mua_raster`` is the precomputed morphological true-mask raster (structure-, not
    config-, dependent); reused across configs to avoid rebuilding the heavy raster.
    """
    nrem_bouts = banded_eval.labeled_chunk_bouts(subject, probe, eval_name)
    if not len(nrem_bouts) or float(nrem_bouts["duration"].sum()) <= 0:
        return []

    t0 = time.perf_counter()
    off_frame, infos, artifacts = banded.detect_structure_banded(
        subject,
        probe,
        structure,
        bouts_by_pass={eval_name: nrem_bouts},
        return_artifacts=True,
        verbose=verbose,
        **cfg,
    )
    detect_s = time.perf_counter() - t0
    if not len(off_frame):
        return []

    depth_lo = float(off_frame["lo"].min())
    depth_hi = float(off_frame["hi"].max())
    n_bands = int(infos.get(eval_name, {}).get("n_bands", 0))
    art = artifacts.get(eval_name, {})
    # Faithful union-of-band-boxes footprint (not the bounding box).
    score_kw = dict(
        depth_lo=depth_lo,
        depth_hi=depth_hi,
        off_df=art.get("off_df"),
        all_bands_on_off_df=art.get("all_bands_on_off_df"),
        footprint="union",
    )

    rows = []
    vm = banded_eval.evaluate_banded_vs_manual(
        subject,
        probe,
        structure,
        off_frame,
        eval_name,
        **score_kw,
    )
    vb = banded_vs_morphological.evaluate_banded_vs_morphological(
        subject,
        probe,
        structure,
        off_frame,
        eval_name,
        filter_name=MUA_FILTER,
        mua_raster=mua_raster,
        **score_kw,
    )
    for r in (vm, vb):
        r.update(
            {"config": config_name, "n_bands": n_bands, "detect_s": round(detect_s, 1)}
        )
        rows.append(r)

    # Diagnostic overlay plots (detections vs manual labels): stacked depth x time
    # panels per <=5 s window, so the result can be eyeballed, not just tabulated.
    if plots and fig_dir is not None:
        try:
            from cnpix_local_sleep.unit_based import banded_plots

            saved = banded_plots.do_structure_banded_plots(
                subject,
                probe,
                structure,
                off_frame,
                eval_name,
                depth_lo=depth_lo,
                depth_hi=depth_hi,
                config_name=config_name,
                off_df=art.get("off_df"),
                all_bands_on_off_df=art.get("all_bands_on_off_df"),
                metrics_row=vm,
                filter_name=MUA_FILTER,
                mua_raster=mua_raster,
                max_windows=max_plot_windows,
                out_dir=fig_dir,
            )
            if saved:
                print(f"    wrote {len(saved)} plot(s) -> {fig_dir}")
        except Exception as exc:  # noqa: BLE001 - plotting must not fail the run
            print(f"    PLOT FAILED: {exc!r}")
    return rows


def post_merge_sweep(
    subject,
    probe,
    structure,
    eval_name,
    post_ms_list,
    *,
    verbose=False,
    mua_raster=None,
):
    """Detect once (greedy + pre 50 ms), then sweep the post-merge UNION-duration floor.

    Efficient: detection + merge run a single time; each post threshold is just a
    ``union_duration >= post`` filter on the merged OFFs plus a re-score (no
    re-detection). ``mua_raster`` is the precomputed morphological true-mask raster, reused
    across all thresholds. Returns rows tagged ``config="pre50_post{ms}"`` / ``post_ms``.
    """
    bouts = banded_eval.labeled_chunk_bouts(subject, probe, eval_name)
    if not len(bouts) or float(bouts["duration"].sum()) <= 0:
        return []

    _, infos, artifacts = banded.detect_structure_banded(
        subject,
        probe,
        structure,
        algo="sticky",
        band_definition="greedy_fr",
        param_strategy="shared",
        min_band_off_duration=0.05,  # pre fixed at 50 ms; no post-filter at detect time
        bouts_by_pass={eval_name: bouts},
        return_artifacts=True,
        verbose=verbose,
    )
    art = artifacts.get(eval_name, {})
    off_df_full = art.get("off_df")
    all_bands = art.get("all_bands_on_off_df")
    if off_df_full is None or not len(off_df_full):
        return []
    n_bands = int(infos.get(eval_name, {}).get("n_bands", 0))

    rows = []
    for post_ms in post_ms_list:
        off_df = off_df_full[
            off_df_full["union_duration"] >= post_ms / 1000.0
        ].reset_index(drop=True)
        frame = banded.banded_on_off_df_to_off_frame(off_df)
        if not len(frame):
            continue
        score_kw = dict(
            depth_lo=float(frame["lo"].min()),
            depth_hi=float(frame["hi"].max()),
            off_df=off_df,
            all_bands_on_off_df=all_bands,
            footprint="union",
        )
        vm = banded_eval.evaluate_banded_vs_manual(
            subject,
            probe,
            structure,
            frame,
            eval_name,
            **score_kw,
        )
        vb = banded_vs_morphological.evaluate_banded_vs_morphological(
            subject,
            probe,
            structure,
            frame,
            eval_name,
            filter_name=MUA_FILTER,
            mua_raster=mua_raster,
            **score_kw,
        )
        for r in (vm, vb):
            r.update(
                {
                    "config": f"pre50_post{post_ms}",
                    "post_ms": post_ms,
                    "n_bands": n_bands,
                }
            )
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject")
    ap.add_argument("--probe")
    ap.add_argument("--structure")
    ap.add_argument(
        "--limit",
        type=int,
        default=3,
        help="max structures (ignored if --subject given)",
    )
    ap.add_argument("--full", action="store_true", help="run every config")
    ap.add_argument("--configs", nargs="+", help="explicit config names")
    ap.add_argument("--eval", default="NREM", choices=["NREM", "Wake"])
    ap.add_argument("--out", default=None, help="results CSV path")
    ap.add_argument(
        "--post-sweep",
        action="store_true",
        help="hold pre=50ms, sweep the post-merge union floor 50->100ms "
        "(detects once per structure)",
    )
    ap.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="skip the detection-vs-label overlay PNGs",
    )
    ap.add_argument(
        "--max-plot-windows",
        type=int,
        default=3,
        help="max overlay windows per (structure, config)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.configs:
        config_names = args.configs
    elif args.full:
        config_names = list(CONFIGS)
    else:
        config_names = DEFAULT_CONFIGS

    if args.subject and args.probe and args.structure:
        targets = [(args.subject, args.probe, args.structure)]
    else:
        targets = labeled_cortical_structures(args.eval)[: args.limit]

    import pathlib

    suffix = "post_sweep" if args.post_sweep else args.eval
    out = args.out or str(
        ub_files.get_experiment_dir("sticky")
        / "banded_validation"
        / f"results_{suffix}.csv"
    )
    fig_dir = str(pathlib.Path(out).parent / "figures") if args.plots else None

    all_rows = []
    if args.post_sweep:
        print(
            f"{len(targets)} structure(s), post-merge sweep {POST_SWEEP_MS} ms, "
            f"eval={args.eval}, mua={MUA_FILTER}"
        )
        for subject, probe, structure in targets:
            print(f"  {subject}/{probe}/{structure} :: post-sweep", flush=True)
            mua_raster = mua_raster_for(subject, probe, structure, args.eval)
            try:
                all_rows += post_merge_sweep(
                    subject,
                    probe,
                    structure,
                    args.eval,
                    POST_SWEEP_MS,
                    verbose=args.verbose,
                    mua_raster=mua_raster,
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"    FAILED: {exc!r}")
    else:
        print(
            f"{len(targets)} structure(s) x {len(config_names)} config(s), "
            f"eval={args.eval}, mua={MUA_FILTER}"
        )
        for subject, probe, structure in targets:
            # morphological raster is config-independent -> build once per structure.
            mua_raster = mua_raster_for(subject, probe, structure, args.eval)
            for config_name in config_names:
                cfg = CONFIGS[config_name]
                print(f"  {subject}/{probe}/{structure} :: {config_name}", flush=True)
                try:
                    all_rows += run_one(
                        subject,
                        probe,
                        structure,
                        config_name,
                        cfg,
                        args.eval,
                        mua_raster=mua_raster,
                        verbose=args.verbose,
                        plots=args.plots,
                        max_plot_windows=args.max_plot_windows,
                        fig_dir=fig_dir,
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"    FAILED: {exc!r}")

    df = pd.DataFrame(all_rows)
    if not len(df):
        print("No results.")
        return

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    # Console summary: F1 / IoU per (config, reference).
    cols = [
        "config",
        "post_ms",
        "structure",
        "reference",
        "F1",
        "IoU",
        "sensitivity",
        "precision",
        "n_off_events",
        "n_bands",
        "detect_s",
    ]
    show = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
