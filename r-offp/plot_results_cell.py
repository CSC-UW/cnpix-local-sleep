"""Emit individual violin panels sized to match a single cell of the grid.

Where ``plot_results.py`` emits full standalone panels (with axis labels, ticks,
and titles) and ``plot_results_grid.py`` assembles a curated subset into one
6x4 sheet, this script emits one bare panel per file at grid-cell dimensions:
each SVG occupies exactly the footprint one cell of ``cx_homeostasis_grid.svg``
would (an nrem cell -> 360x180 pt, a wake cell -> 180x180 pt), stripped of chrome
(no ticks labels / axis labels / title) in the same ``pp.destination("figma")``
style as the grid, with significance bars preserved. Drop one beside the grid in
Figma and it lines up.

It reuses the grid's cell geometry (:func:`plot_results_grid.grid_cell_figsize`)
and chrome-stripping (:func:`plot_results_grid._strip_chrome`), and the plotting
machinery from ``plot_results.py``. It is layout-independent: it iterates the
same ``entries`` as ``plot_results.py``, so it works for any plot YAML (the
cx_homeostasis panels, the bandpower panels, etc.), not only the fixed grid
layout.

By default it emits the adjusted (random effects removed) panels; pass ``--raw``
for the raw summarized values. Files are written next to the standalone panels as
``<panel>_adjusted_cell.svg`` / ``<panel>_cell.svg``.

Usage:
    python plot_results_cell.py config/plots/bandpower_homeostasis.yaml
    python plot_results_cell.py --raw config/plots/cx_homeostasis.yaml
    # Only the mean_zlog_delta / crossed_interaction / six entry:
    python plot_results_cell.py --response-var mean_zlog_delta \
        config/plots/bandpower_homeostasis.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import pubplots as pp

from plot_results import (
    NREM_CONDITIONS,
    WAKE_CONDITIONS,
    extract_significance_bars,
    get_output_path,
    load_adjusted_data,
    load_results,
    load_violin_data,
    plot_nrem_violin,
    plot_wake_violin,
    resolve_base_dir,
)
from plot_results_grid import _strip_chrome, grid_cell_figsize

logger = logging.getLogger(__name__)

# Panel name -> (drawer, which conditions gate drawing it).
PANELS: dict[str, dict] = {
    "nrem": {"drawer": plot_nrem_violin, "conditions": NREM_CONDITIONS},
    "wake": {"drawer": plot_wake_violin, "conditions": WAKE_CONDITIONS},
}


def emit_cell(
    df,
    response_var: str,
    panel: str,
    bars: list[dict],
    sig_config: dict,
    out_path: Path,
    plot_kind: str = "violin",
) -> None:
    """Draw one bare, grid-cell-sized panel and save it to ``out_path``.

    The figure canvas is fixed to the panel's grid-cell allocation
    (:func:`grid_cell_figsize`) under the active figma destination, so it is NOT
    saved with ``bbox_inches="tight"`` (that would resize the canvas and break the
    dimension match). Chrome is stripped to match a grid cell; significance bars
    (drawn with ``clip_on=False``) live in the top margin, as they do in the grid.
    """
    fig, ax = plt.subplots(figsize=pp.scale(*grid_cell_figsize(panel)))
    PANELS[panel]["drawer"](
        df,
        response_var,
        bars,
        sig_config,
        plot_kind=plot_kind,
        ax=ax,
    )
    _strip_chrome(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit individual violin panels sized to one grid cell."
    )
    parser.add_argument("plot_yaml", type=Path, help="Path to the plot YAML config.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base dir for resolving relative YAML paths (default: auto-detect).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Plot raw summarized values. Default is adjusted (random effects "
        "removed, from adjusted_data.csv).",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["violin", "box", "boxen"],
        default="violin",
        help="Per-cell plot kind (default: violin).",
    )
    parser.add_argument(
        "--response-var",
        default=None,
        help="If given, only emit cells for this response variable.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    with open(args.plot_yaml) as f:
        plot_config = yaml.safe_load(f)

    if args.base_dir is not None:
        base_dir = args.base_dir
    else:
        results_dir_name = plot_config.get("results_dir", "_output")
        base_dir = resolve_base_dir(args.plot_yaml, results_dir_name)
        if base_dir is None:
            logger.error(
                "Could not find '%s' directory. Use --base-dir.", results_dir_name
            )
            sys.exit(1)

    results_dir = base_dir / plot_config["results_dir"]
    data_dir = base_dir / plot_config["data_dir"]
    config_path = base_dir / plot_config["config"]
    output_dir = base_dir / plot_config["output_dir"]
    use_adjusted = not args.raw
    plot_kind = args.plot_kind
    sig_config = plot_config.get("significance", {})
    alpha = sig_config.get("alpha", 0.05)
    logger.info("Data: %s", "adjusted (RE removed)" if use_adjusted else "raw")

    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    entries = plot_config.get("entries", [])
    if not entries:
        logger.info("No entries to process.")
        return

    data_cache: dict[str, object] = {}
    n_generated = 0
    n_skipped = 0

    with pp.destination("figma"):
        for entry in entries:
            if len(entry) != 4:
                logger.error("Entry %r must be a 4-tuple; skipping.", entry)
                n_skipped += 1
                continue
            response_var, dataset, condition_set, model = entry
            if args.response_var and response_var != args.response_var:
                continue
            entry_label = f"[{response_var}, {dataset}, {condition_set}, {model}]"

            # Which panels does this condition set feed?
            cs_config = main_config.get("condition_sets", {}).get(condition_set)
            if cs_config is None:
                logger.error("Condition set '%s' undefined; skipping %s.",
                             condition_set, entry_label)
                n_skipped += 1
                continue
            cs_conditions = cs_config.get("conditions", [])
            posthoc_strs = cs_config.get("posthocs", [])
            panels_to_draw = [
                p for p, spec in PANELS.items()
                if any(c in spec["conditions"] for c in cs_conditions)
            ]

            results = load_results(
                results_dir, dataset, condition_set, response_var, model
            )
            if results is None:
                logger.warning("results.json not found for %s, skipping.", entry_label)
                n_skipped += 1
                continue
            bars = extract_significance_bars(results, posthoc_strs, alpha=alpha)

            if use_adjusted:
                df = load_adjusted_data(
                    results_dir, dataset, condition_set, response_var, model
                )
                if df is None:
                    logger.warning(
                        "adjusted_data.csv not found for %s; use --raw or run the "
                        "R analysis first. Skipping.", entry_label
                    )
                    n_skipped += 1
                    continue
            else:
                if dataset not in data_cache:
                    try:
                        data_cache[dataset] = load_violin_data(data_dir, dataset)
                    except FileNotFoundError:
                        logger.error("Parquet not found for %s; skipping.", entry_label)
                        n_skipped += 1
                        continue
                df = data_cache[dataset]

            if response_var not in df.columns:
                logger.error("'%s' not in data; skipping %s.", response_var, entry_label)
                n_skipped += 1
                continue

            suffix = "_adjusted_cell" if use_adjusted else "_cell"
            for panel in panels_to_draw:
                out_path = get_output_path(
                    output_dir,
                    dataset,
                    condition_set,
                    response_var,
                    model,
                    f"{panel}{suffix}",
                    plot_kind=plot_kind,
                )
                emit_cell(
                    df, response_var, panel, bars, sig_config, out_path,
                    plot_kind=plot_kind,
                )
                n_generated += 1

    print(
        f"\nGenerated {n_generated} cell plots, skipped {n_skipped} entries.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
