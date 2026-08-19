"""Plot the locality analyses with significance bars (SVG).

A removable companion to ``plot_results.py`` for the locality pipeline. It reuses
every significance-bar / violin helper from ``plot_results.py`` unchanged and only
swaps in the locality conventions. Two plot types are driven from
``config/plots/locality.yaml``:

* request 1 (top-level ``entries``): the ``overlap_degree`` analysis: mean
  number of overlapping cortical structures per condition. The data come from
  ``summarized_locality_overlap_offs.parquet`` (not the
  ``summarized_<dataset>_offs`` pattern); the output subtree uses the analysis
  kind (``overlap_degree``) in the slot ``plot_results.py`` reserves for the
  condition set; conditions + posthocs are read from the ``overlap_degree`` block
  of ``config/locality.yaml``. Each six-condition fit yields a NREM panel (4
  conditions) and a wake panel (2 conditions).

* request 2 (``local_vs_overlapping`` section): the notebook's
  ``cross_structure_4b`` figure: two dodged violins (Local, Overlapping) per
  condition, with a per-condition significance bracket whose p-value is the
  pooled interaction model's within-condition simple effect (``interaction.simple``
  in ``interaction-<set>/<measure>/<model>/results.json``). The six conditions are
  read from ``condition_windows.six`` in ``config/locality.yaml``.

Delete this file plus ``config/plots/locality.yaml`` to remove locality
plotting entirely; ``plot_results.py`` is not touched.

Usage:
    python plot_locality_results.py config/plots/locality.yaml
    python plot_locality_results.py --boxplot config/plots/locality.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless SVG generation; must precede pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

import pubplots as pp

# Reuse the layer-agnostic plotter's machinery verbatim (no edits there).
import plot_results as pr

logger = logging.getLogger(__name__)

# y-axis label for the request-1 response variable (matches the notebook's
# "Mean # overlapping structures" subplot). Not in plot_results.LABEL_MAP.
OVERLAP_DEGREE_YLABEL = "Mean # overlapping structures"

# Two-color palette for overlap statuses (Local = tan, Overlapping = plum).
LOCALITY_PALETTE = {"Local": "#DC9D81", "Overlapping": "#5C3161"}
OVERLAP_ORDER = ["Local", "Overlapping"]


def load_locality_overlap_data(data_dir: Path, data_file: str) -> pd.DataFrame:
    """Load + filter the locality overlap-degree parquet for violin plotting.

    Mirrors ``plot_results.load_violin_data`` but reads an explicitly-named file
    (the locality parquet does not follow the ``summarized_<dataset>_offs``
    pattern) and filters only on the columns present (clade; the locality
    summary omits detection_mode/layer, being spatial/None/Cx by construction).
    """
    path = data_dir / data_file
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    mask = df["clade"] == "Cx"
    if "detection_mode" in df.columns:
        mask &= df["detection_mode"] == "spatial"
    if "layer" in df.columns:
        mask &= (df["layer"] == "None") | df["layer"].isna()
    result = df.loc[mask].copy()
    if hasattr(result["condition"], "cat"):
        result["condition"] = result["condition"].astype(str)
    return result


def _save_panels(
    df: pd.DataFrame,
    response_var: str,
    bars: list[dict],
    sig_config: dict,
    output_dir: Path,
    dataset: str,
    analysis_kind: str,
    model: str,
    plot_kind: str,
    condition_tick_labels: dict[str, str] | None,
    ylabel: str,
    nrem_tag: str,
    wake_tag: str,
) -> int:
    """Draw + save the NREM and wake violin panels; return the count saved."""
    n = 0
    fig, _ = pr.plot_nrem_violin(
        df, response_var, bars, sig_config, ylabel=ylabel,
        condition_tick_labels=condition_tick_labels, plot_kind=plot_kind,
    )
    nrem_path = pr.get_output_path(
        output_dir, dataset, analysis_kind, response_var, model, nrem_tag,
        plot_kind=plot_kind,
    )
    nrem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(nrem_path)
    pr.plt.close(fig)
    logger.info("Saved %s", nrem_path)
    n += 1

    fig, _ = pr.plot_wake_violin(
        df, response_var, bars, sig_config, ylabel=ylabel,
        condition_tick_labels=condition_tick_labels, plot_kind=plot_kind,
    )
    wake_path = pr.get_output_path(
        output_dir, dataset, analysis_kind, response_var, model, wake_tag,
        plot_kind=plot_kind,
    )
    wake_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(wake_path)
    pr.plt.close(fig)
    logger.info("Saved %s", wake_path)
    n += 1
    return n


# Request 2: within-condition Local-vs-Overlapping violins


def load_per_condition_data(data_dir: Path, data_file: str) -> pd.DataFrame:
    """Load the per-condition locality parquet (Local + Overlapping rows).

    Reuses the request-1 clade=="Cx" filter, then casts overlap_status to str so
    seaborn 0.13's dict-palette hue lookup works (categorical hue breaks it).
    """
    df = load_locality_overlap_data(data_dir, data_file)
    if hasattr(df["overlap_status"], "cat"):
        df["overlap_status"] = df["overlap_status"].astype(str)
    return df


def extract_simple_effect_bars(
    results: dict, conditions: list[str], alpha: float = 0.05
) -> dict[str, dict]:
    """Map each condition to its within-condition simple-effect bar descriptor.

    Reads ``interaction.simple`` (the pooled model's per-condition Overlapping -
    Local effects). These are computed unconditionally, so there is no
    main-effect gate; a condition is ``tested`` iff its contrast is present.
    """
    interaction = results.get("interaction", {})
    simple = interaction.get("simple") or {}
    contrasts = simple.get("contrasts", [])
    pvalues = simple.get("pvalues", [])
    bars: dict[str, dict] = {
        c: {"pvalue": None, "significant": False, "tested": False}
        for c in conditions
    }
    for c in conditions:
        key = f"Overlapping - Local | {c}"
        if key in contrasts:
            idx = contrasts.index(key)
            if idx < len(pvalues):
                p = pvalues[idx]
                bars[c] = {"pvalue": p, "significant": p < alpha, "tested": True}
    return bars


def draw_within_condition_bars(
    ax: "matplotlib.axes.Axes",
    conditions: list[str],
    bars: dict[str, dict],
    sig_config: dict,
    plot_df: pd.DataFrame,
    measure: str,
) -> None:
    """Draw one Local-vs-Overlapping bracket per condition over its dodged pair.

    For two dodged hue levels seaborn centers them at x +/- 0.2; each bracket
    spans that pair and sits just above the condition's data max. Significant
    contrasts are black, non-significant light grey, with a star label.
    """
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    gap = sig_config.get("bar_gap", 0.015) * y_range
    tip = sig_config.get("tip_length", 0.01) * y_range
    offset = 0.2
    new_top = y_max
    for i, cond in enumerate(conditions):
        bar = bars.get(cond, {})
        if not bar.get("tested"):
            continue
        vals = plot_df.loc[plot_df["condition"] == cond, measure]
        cmax = vals.max() if len(vals) else y_max
        y = cmax + 2 * gap
        if bar["significant"]:
            color = sig_config.get("significant_color", "black")
            lw = sig_config.get("significant_linewidth", 1.5)
        else:
            color = sig_config.get("nonsignificant_color", "0.7")
            lw = sig_config.get("nonsignificant_linewidth", 0.5)
        x_l, x_r = i - offset, i + offset
        ax.plot([x_l, x_r], [y, y], lw=lw, color=color, clip_on=False,
                solid_capstyle="butt")
        ax.plot([x_l, x_l], [y, y - tip], lw=lw, color=color, clip_on=False,
                solid_capstyle="butt")
        ax.plot([x_r, x_r], [y, y - tip], lw=lw, color=color, clip_on=False,
                solid_capstyle="butt")
        label = pr._pvalue_to_stars(bar["pvalue"])
        if label:
            ax.text(i, y, label, ha="center", va="baseline",
                    fontsize=sig_config.get("star_fontsize", "small"),
                    color=color, clip_on=False)
        new_top = max(new_top, y + 2 * gap)
    ax.set_ylim(y_min, new_top)


def plot_within_condition_violin(
    df: pd.DataFrame,
    measure: str,
    conditions: list[str],
    bars: dict[str, dict],
    sig_config: dict,
    ylabel: str | None = None,
    condition_tick_labels: dict[str, str] | None = None,
    plot_kind: str = "violin",
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Grouped Local/Overlapping violins per condition with within-pair brackets."""
    plot_df = df.loc[df["condition"].isin(conditions)].copy()
    fig, ax = plt.subplots(figsize=pp.scale(4.5, 2.2))

    common = dict(
        data=plot_df, x="condition", y=measure, order=conditions,
        hue="overlap_status", hue_order=OVERLAP_ORDER, palette=LOCALITY_PALETTE,
        dodge=True, ax=ax,
    )
    if plot_kind == "box":
        sns.boxplot(**common, fliersize=0)
    else:
        sns.violinplot(**common, inner="quart", cut=0)
    sns.stripplot(
        data=plot_df, x="condition", y=measure, order=conditions,
        hue="overlap_status", hue_order=OVERLAP_ORDER, dodge=True,
        palette="dark:black", size=2, ax=ax, legend=False,
    )

    if ylabel is None:
        ylabel = pr.LABEL_MAP.get(measure, measure)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(None)
    if condition_tick_labels:
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([condition_tick_labels.get(c, c) for c in conditions])

    # Legend outside the axes (the tall violins + brackets leave no clean corner).
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title=None, frameon=False,
              fontsize="small", loc="upper left", bbox_to_anchor=(1.0, 1.0))

    draw_within_condition_bars(ax, conditions, bars, sig_config, plot_df, measure)
    return fig, ax


def process_local_vs_overlapping(
    section: dict,
    main_config: dict,
    data_dir: Path,
    results_dir: Path,
    output_dir: Path,
    sig_config: dict,
    alpha: float,
    condition_tick_labels: dict[str, str] | None,
    plot_kind: str,
) -> tuple[int, int]:
    """Generate the request-2 within-condition violins; return (n_made, n_skip)."""
    conditions = (
        main_config.get("condition_windows", {}).get("six", {}).get("conditions")
    )
    if not conditions:
        logger.error("condition_windows.six.conditions missing from main config.")
        return 0, 1
    data_file = section.get("data_file")
    analysis_kind = section.get("analysis_kind", "interaction-six")
    df = load_per_condition_data(data_dir, data_file)

    n_made = n_skip = 0
    for entry in section.get("entries", []):
        if len(entry) != 3:
            logger.error("local_vs_overlapping entry %r must be "
                         "[measure, dataset, model]; skipping.", entry)
            n_skip += 1
            continue
        measure, dataset, model = entry
        label = f"[{measure}, {dataset}, {analysis_kind}, {model}]"
        if measure not in df.columns:
            logger.error("Measure '%s' not in data, skipping %s.", measure, label)
            n_skip += 1
            continue
        results = pr.load_results(
            results_dir, dataset, analysis_kind, measure, model
        )
        if results is None:
            logger.warning("results.json not found for %s, skipping.", label)
            n_skip += 1
            continue
        bars = extract_simple_effect_bars(results, conditions, alpha=alpha)
        fig, _ = plot_within_condition_violin(
            df, measure, conditions, bars, sig_config,
            condition_tick_labels=condition_tick_labels, plot_kind=plot_kind,
        )
        out_path = pr.get_output_path(
            output_dir, dataset, analysis_kind, measure, model,
            "within_condition", plot_kind=plot_kind,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")  # include the outside legend
        plt.close(fig)
        logger.info("Saved %s", out_path)
        n_made += 1
    return n_made, n_skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the locality overlap_degree analysis with "
        "significance bars."
    )
    parser.add_argument("plot_yaml", type=Path, help="Plot YAML config file.")
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="Base dir for relative paths (default: auto-detect).")
    parser.add_argument("--boxplot", action="store_true", default=False,
                        help="Use boxplots instead of violin plots.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
        results_dir_name = plot_config.get("results_dir", "_output_locality")
        base_dir = pr.resolve_base_dir(args.plot_yaml, results_dir_name)
        if base_dir is None:
            logger.error("Could not find '%s' directory. Use --base-dir.",
                         results_dir_name)
            sys.exit(1)

    results_dir = base_dir / plot_config["results_dir"]
    data_dir = base_dir / plot_config["data_dir"]
    config_path = base_dir / plot_config["config"]
    output_dir = base_dir / plot_config["output_dir"]
    data_file = plot_config.get("data_file")  # request 1 only; request 2 has its own
    plot_kind = "box" if args.boxplot else "violin"
    destination = plot_config.get("destination", "figma")
    sig_config = plot_config.get("significance", {})
    alpha = sig_config.get("alpha", 0.05)
    condition_tick_labels = plot_config.get("condition_tick_labels")

    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    # Conditions + post-hocs come from the overlap_degree block (it carries
    # both), so the six-condition fit feeds a NREM panel and a wake panel.
    od_block = main_config.get("overlap_degree", {})
    posthoc_strs = od_block.get("posthocs", [])

    entries = plot_config.get("entries", [])
    lvo_section = plot_config.get("local_vs_overlapping")
    if not entries and not lvo_section:
        logger.info("No entries or local_vs_overlapping section to process.")
        return

    n_generated = 0
    n_skipped = 0
    with pp.destination(destination):
        # Request 1: overlap_degree condition violins
        df = load_locality_overlap_data(data_dir, data_file) if entries else None
        for entry in entries:
            if len(entry) != 4:
                logger.error("Entry %r must be [response_var, dataset, "
                             "analysis_kind, model]; skipping.", entry)
                n_skipped += 1
                continue
            response_var, dataset, analysis_kind, model = entry
            entry_label = f"[{response_var}, {dataset}, {analysis_kind}, {model}]"

            if response_var not in df.columns:
                logger.error("Response variable '%s' not in data, skipping %s.",
                             response_var, entry_label)
                n_skipped += 1
                continue

            results = pr.load_results(
                results_dir, dataset, analysis_kind, response_var, model,
            )
            if results is None:
                logger.warning("results.json not found for %s, skipping.",
                               entry_label)
                n_skipped += 1
                continue

            bars = pr.extract_significance_bars(results, posthoc_strs, alpha=alpha)

            n_generated += _save_panels(
                df, response_var, bars, sig_config, output_dir, dataset,
                analysis_kind, model, plot_kind,
                condition_tick_labels, OVERLAP_DEGREE_YLABEL, "nrem", "wake",
            )

            # Adjusted panels (random effects removed), if the CSV exists.
            df_adj = pr.load_adjusted_data(
                results_dir, dataset, analysis_kind, response_var, model,
            )
            if df_adj is not None:
                n_generated += _save_panels(
                    df_adj, response_var, bars, sig_config, output_dir, dataset,
                    analysis_kind, model, plot_kind,
                    condition_tick_labels,
                    OVERLAP_DEGREE_YLABEL + " (RE removed)",
                    "nrem_adjusted", "wake_adjusted",
                )

        # Request 2: within-condition Local-vs-Overlapping violins
        if lvo_section:
            made, skipped = process_local_vs_overlapping(
                lvo_section, main_config, data_dir, results_dir, output_dir,
                sig_config, alpha, condition_tick_labels, plot_kind,
            )
            n_generated += made
            n_skipped += skipped

    print(f"\nGenerated {n_generated} plots, skipped {n_skipped} entries.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
