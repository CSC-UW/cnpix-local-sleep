"""Assemble the cx_homeostasis condition/significance plots into one 6x6 grid.

Where ``plot_results.py`` emits one file per (dataset, condition_set,
response_var, model, panel), this script composes a curated subset of those same
panels onto a single-page SVG for at-a-glance comparison across OFF classes and
measures:

- Rows (measure slots, top -> bottom): rate, MoM/mean duration, span,
  median_area, total_area_norm, median_median_trace. The duration and span rows
  use the median-aggregated response for LLAS/CLAS but the mean-aggregated analog
  (mean_median_duration / mean_span) for LLAS-exclusive, whose median forms show
  discretization striping there.
- Columns (left -> right): LLAS/Wake, LLAS/NREM, LLAS-exclusive/Wake,
  LLAS-exclusive/NREM, CLAS/Wake, CLAS/NREM.

Each cell's data/model is resolved unambiguously from the same plot YAML
(``config/plots/cx_homeostasis.yaml``) used by ``plot_results.py``: the single
entry whose (response_var, dataset) match and whose condition_set draws the
requested panel. If a cell resolves to zero or more than one viable entry, the
script raises rather than guessing.

The figure is intentionally bare (no ticks, tick labels, axis labels, or titles)
so it can be annotated in Figma. Significance bars are preserved.

Usage:
    python plot_results_grid.py config/plots/cx_homeostasis.yaml
    python plot_results_grid.py --plot-kind boxen config/plots/cx_homeostasis.yaml
    python plot_results_grid.py --plot-kind box --raw config/plots/cx_homeostasis.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import pubplots as pp

# Reuse the standalone plotting machinery verbatim (SPOT/DRY).
from plot_results import (
    NREM_CONDITIONS,
    WAKE_CONDITIONS,
    extract_significance_bars,
    load_adjusted_data,
    load_results,
    load_violin_data,
    plot_nrem_violin,
    plot_wake_violin,
    resolve_base_dir,
)

logger = logging.getLogger(__name__)

# Grid layout. Rows are measure slots (top -> bottom); columns are (dataset,
# panel) pairs (left -> right). Keep these two lists as the single source of the
# requested figure shape.
#
# A row entry is either a single response variable used for every column, or a
# {dataset: response_variable} mapping when the measure differs by OFF class. The
# duration and span rows use the mean-aggregated analogs for LLAS-exclusive
# (mean_median_duration / mean_span) because the median forms discretize into
# stripes there; LLAS and CLAS keep the median forms.
ROW_MEASURES: list[str | dict[str, str]] = [
    "rate",
    {
        "llas": "median_median_duration",
        "llas_exclusive": "mean_median_duration",
        "clas": "median_median_duration",
    },
    {
        "llas": "median_span",
        "llas_exclusive": "mean_span",
        "clas": "median_span",
    },
    "median_area",
    "total_area_norm",
    "median_median_trace",
]

COLUMNS: list[tuple[str, str]] = [
    ("llas", "wake"),
    ("llas", "nrem"),
    ("llas_exclusive", "wake"),
    ("llas_exclusive", "nrem"),
    ("clas", "wake"),
    ("clas", "nrem"),
]

#: Unscaled (width, height) of the whole grid figure, in inches. Passed through
#: ``pp.scale`` under the figma destination when the figure is built. Promoted to
#: a named constant so the per-cell geometry (:func:`grid_cell_figsize`) stays in
#: sync with the assembled grid.
GRID_FIGSIZE: tuple[float, float] = (6.0, 4.5)

#: Relative column widths by panel type, mirroring ``build_grid``'s
#: ``width_ratios``: wake panels (2 conditions) get half the width of nrem
#: panels (4 conditions).
PANEL_WIDTH_RATIO: dict[str, int] = {"wake": 1, "nrem": 2}


def grid_cell_figsize(
    panel: str,
    grid_figsize: tuple[float, float] = GRID_FIGSIZE,
    n_rows: int = len(ROW_MEASURES),
) -> tuple[float, float]:
    """Unscaled (width, height) in inches of one grid cell for ``panel``.

    Returns the cell's *allocation* (its figure-fraction times the total grid
    figsize) for the grid shape defined by ``COLUMNS``/``ROW_MEASURES``. This is
    deterministic (it ignores ``tight_layout`` margins and inter-cell gaps), so a
    standalone single-panel figure created at this figsize occupies exactly the
    footprint one grid cell would, and lines up when placed beside the grid.

    Feed the result through ``pp.scale`` under the same destination the grid uses
    (``figma``) to get the on-canvas size, matching the corresponding cell in
    ``cx_homeostasis_grid.svg``.
    """
    if panel not in PANEL_WIDTH_RATIO:
        raise ValueError(f"Unknown panel '{panel}' (expected 'wake' or 'nrem').")
    total_w, total_h = grid_figsize
    ratios = [PANEL_WIDTH_RATIO[p] for (_, p) in COLUMNS]
    cell_w = total_w * PANEL_WIDTH_RATIO[panel] / sum(ratios)
    cell_h = total_h / n_rows
    return (cell_w, cell_h)


def row_measure(row_spec: "str | dict[str, str]", dataset: str) -> str:
    """Resolve the response variable for a row slot in a given dataset column.

    A row is either a single response variable (used for every column) or a
    ``{dataset: response_variable}`` mapping. Raises with a clear message if a
    mapping omits a dataset that ``COLUMNS`` references.
    """
    if isinstance(row_spec, dict):
        if dataset not in row_spec:
            raise ValueError(
                f"Row {row_spec} has no measure for dataset '{dataset}'. "
                f"Add it, or make the row a single response-variable string."
            )
        return row_spec[dataset]
    return row_spec


def _condition_set_draws_panel(
    main_config: dict, condition_set: str, panel: str
) -> bool:
    """Whether a condition set contributes the requested (wake/nrem) panel.

    Mirrors ``plot_results.py``'s panel selection: a set draws the wake panel iff
    any of its conditions is a wake condition, and likewise for nrem. Thus the
    ``six`` set draws both, ``wake`` only wake, ``nrem`` only nrem.
    """
    cs_config = main_config.get("condition_sets", {}).get(condition_set)
    if cs_config is None:
        raise ValueError(
            f"Condition set '{condition_set}' not defined in main config."
        )
    conditions = cs_config.get("conditions", [])
    if panel == "wake":
        return any(c in WAKE_CONDITIONS for c in conditions)
    if panel == "nrem":
        return any(c in NREM_CONDITIONS for c in conditions)
    raise ValueError(f"Unknown panel '{panel}' (expected 'wake' or 'nrem').")


def resolve_cell(
    entries: list[list],
    main_config: dict,
    measure: str,
    dataset: str,
    panel: str,
) -> tuple[str, str]:
    """Resolve the unique (condition_set, model) feeding one grid cell.

    Filters the plot YAML entries to those matching (measure, dataset) whose
    condition_set draws ``panel``. Raises if zero or more than one survive --
    never guesses.
    """
    viable = [
        (cs, model)
        for (rv, ds, cs, model) in entries
        if rv == measure
        and ds == dataset
        and _condition_set_draws_panel(main_config, cs, panel)
    ]
    if len(viable) == 0:
        raise ValueError(
            f"No viable entry for cell (measure={measure}, dataset={dataset}, "
            f"panel={panel}). Add one to the plot YAML."
        )
    if len(viable) > 1:
        listed = ", ".join(f"[{cs}, {model}]" for cs, model in viable)
        raise ValueError(
            f"Ambiguous: {len(viable)} viable entries for cell "
            f"(measure={measure}, dataset={dataset}, panel={panel}): {listed}. "
            f"Refusing to guess."
        )
    return viable[0]


def _strip_chrome(ax) -> None:
    """Strip a cell down to its data: axis labels, title, legend, x ticks.

    The y ticks stay. Each grid row is a different measure on its own scale, so
    they are the only way to read a value off the figure; the x ticks are the
    same condition names in all 36 cells, so they go.
    """
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_title("")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def build_grid(
    entries: list[list],
    main_config: dict,
    data_dir: Path,
    results_dir: Path,
    sig_config: dict,
    alpha: float,
    plot_kind: str,
    use_adjusted: bool = True,
) -> plt.Figure:
    """Compose the 6x6 grid figure. Assumes an active pp.destination context.

    When ``use_adjusted`` is True (default), each cell plots the model's adjusted
    data (subject random effects removed, from ``adjusted_data.csv``); otherwise
    it plots the raw summarized-parquet values. Significance bars come from the
    model's ``results.json`` either way.
    """
    data_cache: dict[str, "object"] = {}

    # Wake panels have 2 conditions, NREM panels 4, so give wake columns half the
    # width of NREM columns (kept in sync with COLUMNS' panel types).
    width_ratios = [PANEL_WIDTH_RATIO[panel] for (_, panel) in COLUMNS]
    fig, axes = plt.subplots(
        len(ROW_MEASURES),
        len(COLUMNS),
        figsize=pp.scale(*GRID_FIGSIZE),
        gridspec_kw={"width_ratios": width_ratios},
    )

    for r, row_spec in enumerate(ROW_MEASURES):
        for c, (dataset, panel) in enumerate(COLUMNS):
            ax = axes[r, c]
            measure = row_measure(row_spec, dataset)
            condition_set, model = resolve_cell(
                entries, main_config, measure, dataset, panel
            )
            label = (
                f"(measure={measure}, dataset={dataset}, panel={panel}) -> "
                f"[{condition_set}, {model}]"
            )

            if use_adjusted:
                # Per-cell adjusted data (RE removed), specific to this fit.
                df = load_adjusted_data(
                    results_dir, dataset, condition_set, measure, model
                )
                if df is None:
                    raise FileNotFoundError(
                        f"adjusted_data.csv not found for {label}. "
                        f"Run the R analysis first, or pass --raw to plot raw "
                        f"values."
                    )
            else:
                if dataset not in data_cache:
                    data_cache[dataset] = load_violin_data(data_dir, dataset)
                df = data_cache[dataset]
            if measure not in df.columns:
                raise ValueError(
                    f"Response variable '{measure}' not in {dataset} data for {label}."
                )

            results = load_results(
                results_dir, dataset, condition_set, measure, model
            )
            if results is None:
                raise FileNotFoundError(
                    f"results.json not found for {label}. "
                    f"Run the R analysis first."
                )

            posthoc_strs = (
                main_config["condition_sets"][condition_set].get("posthocs", [])
            )
            bars = extract_significance_bars(results, posthoc_strs, alpha=alpha)

            plot_fn = plot_wake_violin if panel == "wake" else plot_nrem_violin
            plot_fn(
                df,
                measure,
                bars,
                sig_config,
                plot_kind=plot_kind,
                ax=ax,
            )
            _strip_chrome(ax)
            logger.info("Drew cell [%d,%d] %s", r, c, label)

    fig.tight_layout()
    # Reduce the gaps between subplots by 50% (relative to tight_layout's).
    fig.subplots_adjust(
        wspace=fig.subplotpars.wspace * 0.5,
        hspace=fig.subplotpars.hspace * 0.5,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble cx_homeostasis condition plots into a single "
        "6x6 grid SVG."
    )
    parser.add_argument("plot_yaml", type=Path, help="Path to the plot YAML config.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base dir for resolving relative YAML paths (default: auto-detect).",
    )
    parser.add_argument(
        "--offproj-output",
        action="store_true",
        default=False,
        help="Write the grid to the offproj NFS path instead of output_dir.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["violin", "box", "boxen"],
        default="violin",
        help="Per-cell plot kind (default: violin).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Plot raw summarized values. Default is adjusted (subject random "
        "effects removed, from adjusted_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output SVG path (default: "
        "<output_dir>/cx_homeostasis_grid.svg).",
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

    # Resolve base directory (same logic as plot_results.py).
    if args.base_dir is not None:
        base_dir = args.base_dir
    else:
        results_dir_name = plot_config.get("results_dir", "_output")
        base_dir = resolve_base_dir(args.plot_yaml, results_dir_name)
        if base_dir is None:
            logger.error("Could not find '%s' directory. Use --base-dir.", results_dir_name)
            sys.exit(1)

    results_dir = base_dir / plot_config["results_dir"]
    data_dir = base_dir / plot_config["data_dir"]
    config_path = base_dir / plot_config["config"]

    if args.offproj_output:
        from cnpix_local_sleep.morphological.mua import files as mua_files

        output_dir = mua_files.get_path(
            "placeholder",
            detection_mode="spatial",
            clade="Cx",
            enforce_in_schema=False,
        ).parent
        logger.info("Using cnpix_local_sleep output dir: %s", output_dir)
    else:
        output_dir = base_dir / plot_config["output_dir"]

    plot_kind = args.plot_kind
    use_adjusted = not args.raw
    logger.info("Data: %s", "adjusted (RE removed)" if use_adjusted else "raw")
    sig_config = plot_config.get("significance", {})
    alpha = sig_config.get("alpha", 0.05)

    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    entries = plot_config.get("entries", [])
    if not entries:
        logger.error("No entries in plot YAML; nothing to resolve.")
        sys.exit(1)

    # The figure is bare (annotated later in Figma), so always use the figma
    # destination regardless of the YAML 'destination' field.
    with pp.destination("figma"):
        fig = build_grid(
            entries,
            main_config,
            data_dir,
            results_dir,
            sig_config,
            alpha,
            plot_kind,
            use_adjusted=use_adjusted,
        )

        if args.output is not None:
            out_path = args.output
        else:
            kind_suffix = "" if plot_kind == "violin" else f"_{plot_kind}"
            raw_suffix = "" if use_adjusted else "_raw"
            out_path = (
                output_dir / f"cx_homeostasis_grid{kind_suffix}{raw_suffix}.svg"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    print(f"Saved grid to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
