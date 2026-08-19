"""Generate violin/box plots with significance bars from mixed-effects model results.

Reads a YAML config listing
(response_variable, dataset, condition_set, model) 4-tuples, loads the
corresponding parquet data and JSON results, and produces publication-ready
plots with significance bars indicating post-hoc contrast results. The
condition_set (e.g. "six"/"nrem"/"wake") selects which fit feeds the panel
and which violin panel(s) are drawn.

Usage:
    python plot_results.py config/plots/cx_homeostasis.yaml
    python plot_results.py --boxplot config/plots/cx_homeostasis.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import cnpix_local_sleep.plots
import pandas as pd
import seaborn as sns
import yaml

import pubplots as pp

# Same contrast-matching as the summarizer and the table builders (SPOT).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import normalize_contrast  # noqa: E402

logger = logging.getLogger(__name__)

# Response variable -> y-axis label (consistent with R/plotting.R
# display_label_map and the notebook label_map).
LABEL_MAP: dict[str, str] = {
    "median_duration": "Median duration (s)",
    "mean_boxcox_duration": "Mean Box-Coxed duration (s^λ)",
    "median_median_duration": "MoM duration (s)",
    "mean_median_duration": "Mean MoM duration (s)",
    "mean_boxcox_median_duration": "Mean Box-Coxed median duration (s^λ)",
    "median_span": "Median span (µm)",
    "mean_span": "Mean span (µm)",
    "mean_boxcox_span": "Mean Box-Coxed span (µm^λ)",
    "median_span_rel2max": "Median relative span (frac)",
    "mean_boxcox_span_rel2max": "Mean Box-Coxed relative span (frac^λ)",
    "mean_grouped_boxcox_span_rel2max": (
        "Mean grouped Box-Coxed relative span (frac^λ)"
    ),
    "median_area": "Median area (s·µm)",
    "mean_area": "Mean area (s·µm)",
    "mean_boxcox_area": "Mean Box-Coxed area ((s·µm)^λ)",
    "median_area_rel2span": "Median relative area (s·frac)",
    "mean_boxcox_area_rel2span": "Mean Box-Coxed relative area ((s·frac)^λ)",
    "total_area": "Total area (s·µm)",
    "total_area_rel2span": "Total relative area (s·frac)",
    "rate": "Rate (Hz)",
    "total_area_norm": "Total area normalized (au)",
    "median_median_trace": "MoM trace (µV)",
    "mean_median_trace": "Mean MoM trace (µV)",
    "median_mad_trace": "Median MAD trace (µV)",
    "mean_mad_trace": "Mean MAD trace (µV)",
    "median_abs_onset_slope": "Median |onset slope| (s/µm)",
    "median_abs_offset_slope": "Median |offset slope| (s/µm)",
    "mean_onset_mad": "Mean onset MAD (s)",
    "mean_offset_mad": "Mean offset MAD (s)",
    "adj_mean_onset_mad": "Size-adjusted mean onset MAD (ms)",
    "adj_mean_offset_mad": "Size-adjusted mean offset MAD (ms)",
    "median_onset_jitter": "Median onset jitter (s)",
    "median_offset_jitter": "Median offset jitter (s)",
    # Band-power condition means (separable bandpower homeostasis pipeline).
    "mean_zlog_delta": "Mean zlog delta power (au)",
    "mean_zlog_eta": "Mean zlog eta power (au)",
    "mean_log_delta": "Mean log10 delta power",
    "mean_log_eta": "Mean log10 eta power",
}

# NREM conditions in display order (left-to-right on x-axis).
NREM_CONDITIONS: list[str] = [
    "Early.BSL.NREM",
    "Early.REC.NREM.Match",
    "Early.REC.NREM",
    "Late.REC.NREM",
]

# Wake conditions in display order.
WAKE_CONDITIONS: list[str] = [
    "Early.NOD.Wake",
    "Late.NOD.Wake",
]

# Normalized posthoc contrast strings belonging to the NREM subplot.
NREM_CONTRASTS: set[str] = {
    "Early.REC.NREM - Early.REC.NREM.Match",
    "Early.REC.NREM - Early.BSL.NREM",
    "Early.REC.NREM - Late.REC.NREM",
    "Early.BSL.NREM - Early.REC.NREM.Match",
}

# Normalized posthoc contrast string for the wake subplot.
WAKE_CONTRAST: str = "Late.NOD.Wake - Early.NOD.Wake"


# -------------------- Utility --------------------


# -------------------- Data loading --------------------


def summary_filename(dataset: str) -> str:
    """Filename of the summarized parquet for a dataset.

    The ``full48h_`` infix matches cnpix_local_sleep's ``export-full48h-offs`` output.
    """
    return f"summarized_full48h_{dataset}_offs.parquet"


def join_adjusted_edge_statistics(
    df: pd.DataFrame, data_dir: Path, dataset: str
) -> pd.DataFrame:
    """Left-join the size-adjusted edge columns when they have been exported.

    Python-side mirror of ``offp::join_adjusted_edge_statistics``. Onset/offset
    MAD depend mechanically on how many channels an event spans, so cnpix_local_sleep's
    ``off-analysis export-adjusted-edge-statistics`` re-estimates each cell mean
    by marginal standardization and writes
    ``summarized_full48h_<dataset>_edge_adjusted.parquet``. No-op when that file
    is absent, so nothing breaks if the companion export has not been run.
    """
    path = data_dir / f"summarized_full48h_{dataset}_edge_adjusted.parquet"
    if not path.exists():
        return df
    adjusted = pd.read_parquet(path)
    keys = ["subject", "probe", "structure", "condition"]
    keep = keys + [c for c in adjusted.columns if c.startswith("adj_")]
    return df.merge(adjusted[keep], on=keys, how="left")


def load_violin_data(data_dir: Path, dataset: str) -> pd.DataFrame:
    """Load and filter a summarized parquet file for violin plotting.

    Applies the standard cx_homeostasis filter: clade=="Cx",
    detection_mode=="spatial", layer=="None" or NaN.
    """
    path = data_dir / summary_filename(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = join_adjusted_edge_statistics(df, data_dir, dataset)

    # Restrict to cortical spatial layer-agnostic OFFs. Current cnpix_local_sleep output
    # omits layer/detection_mode (they are spatial/None/Cx by construction),
    # so filter only on columns present.
    mask = df["clade"] == "Cx"
    if "detection_mode" in df.columns:
        mask &= df["detection_mode"] == "spatial"
    if "layer" in df.columns:
        mask &= (df["layer"] == "None") | df["layer"].isna()
    result = df.loc[mask].copy()
    # Cast condition from categorical to str so that seaborn 0.13's
    # dict palette lookup works correctly (categorical hue breaks it).
    if hasattr(result["condition"], "cat"):
        result["condition"] = result["condition"].astype(str)
    return result


def load_results(
    results_dir: Path,
    dataset: str,
    condition_set: str,
    response_var: str,
    model: str,
) -> dict | None:
    """Load a results.json file, returning None if not found."""
    path = (
        results_dir
        / dataset
        / condition_set
        / response_var
        / model
        / "results.json"
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_adjusted_data(
    results_dir: Path,
    dataset: str,
    condition_set: str,
    response_var: str,
    model: str,
) -> pd.DataFrame | None:
    """Load adjusted data (random effects removed) from a CSV file.

    The CSV is produced by the R analysis pipeline alongside results.json.
    Returns None if the file doesn't exist.
    """
    path = (
        results_dir
        / dataset
        / condition_set
        / response_var
        / model
        / "adjusted_data.csv"
    )
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if hasattr(df["condition"], "cat"):
        df["condition"] = df["condition"].astype(str)
    return df


# -------------------- Significance bar extraction --------------------


def _parse_contrast_conditions(contrast_norm: str) -> tuple[str, str]:
    """Parse 'CondA - CondB' into (CondA, CondB)."""
    parts = contrast_norm.split(" - ", maxsplit=1)
    return parts[0].strip(), parts[1].strip()


def extract_significance_bars(
    results: dict,
    posthoc_strs: list[str],
    alpha: float = 0.05,
) -> list[dict]:
    """Extract significance bar descriptors from a results.json dict.

    Returns a list of dicts, each with keys: contrast_norm, significant,
    pvalue, tested.
    """
    main_effect = results.get("main_effect", {})
    is_significant = main_effect.get("significant", False)

    bars = []
    for raw_str in posthoc_strs:
        norm = normalize_contrast(raw_str)
        bar = {
            "contrast_norm": norm,
            "significant": False,
            "pvalue": None,
            "tested": False,
        }

        if not is_significant:
            bars.append(bar)
            continue

        posthoc = main_effect.get("posthoc")
        if posthoc is None:
            bars.append(bar)
            continue

        bar["tested"] = True

        # Find the index of this contrast in the JSON arrays.
        idx = None
        json_contrasts = posthoc.get("contrasts", {})
        if json_contrasts:
            json_norms = [normalize_contrast(c) for c in json_contrasts]
            if norm in json_norms:
                idx = json_norms.index(norm)

        if idx is None:
            # Positional fallback: match against YAML posthoc order.
            norm_posthocs = [normalize_contrast(s) for s in posthoc_strs]
            try:
                idx = norm_posthocs.index(norm)
            except ValueError:
                bar["tested"] = False
                bars.append(bar)
                continue

        pvalues = posthoc.get("pvalues", [])
        if idx < len(pvalues):
            bar["pvalue"] = pvalues[idx]
            bar["significant"] = pvalues[idx] < alpha
        else:
            bar["tested"] = False

        bars.append(bar)

    return bars


# -------------------- Significance bar drawing --------------------


def _pvalue_to_stars(p: float | None) -> str:
    """Convert a p-value to a significance label string.

    Returns "*" (p < 0.001), "" (p < 0.01), "*" (p < 0.05),
    "~" (p < 0.1, trend), or "" (not significant / None).
    """
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "~"
    return ""


def draw_significance_bars(
    ax: mpl.axes.Axes,
    bars: list[dict],
    conditions: list[str],
    sig_config: dict,
) -> None:
    """Draw significance bars (brackets) above a violin plot.

    Bars connect pairs of conditions on the x-axis. Significant contrasts
    are drawn in black; tested-but-non-significant contrasts in light grey.
    Untested contrasts are not drawn. A significance label (".", "*", "",
    or "*") is placed above each bar.
    """
    drawn_bars = []
    for bar in bars:
        if not bar["tested"]:
            continue
        cond_a, cond_b = _parse_contrast_conditions(bar["contrast_norm"])
        if cond_a not in conditions or cond_b not in conditions:
            logger.warning("Contrast condition not in x-axis: %s", bar["contrast_norm"])
            continue
        bar = dict(bar)  # copy so we don't mutate the original
        bar["x_left"] = min(conditions.index(cond_a), conditions.index(cond_b))
        bar["x_right"] = max(conditions.index(cond_a), conditions.index(cond_b))
        drawn_bars.append(bar)

    if not drawn_bars:
        return

    # Sort: shortest span first, then leftmost position (for ties).
    drawn_bars.sort(key=lambda b: (b["x_right"] - b["x_left"], b["x_left"]))

    # Compute y-positions above the data.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    bar_gap = sig_config.get("bar_gap", 0.015) * y_range
    tip_length = sig_config.get("tip_length", 0.01) * y_range
    bar_step = sig_config.get("bar_height", 0.02) * y_range + bar_gap

    y_base = y_max + bar_gap

    for i, bar in enumerate(drawn_bars):
        y = y_base + i * bar_step

        lw = sig_config.get("bar_linewidth", 1.0)
        if bar["significant"]:
            color = sig_config.get("significant_color", "black")
        else:
            color = sig_config.get("nonsignificant_color", "0.7")

        x_l = bar["x_left"]
        x_r = bar["x_right"]

        # Horizontal bar
        ax.plot(
            [x_l, x_r],
            [y, y],
            lw=lw,
            color=color,
            clip_on=False,
            solid_capstyle="butt",
        )
        # Left tip (downward)
        ax.plot(
            [x_l, x_l],
            [y, y - tip_length],
            lw=lw,
            color=color,
            clip_on=False,
            solid_capstyle="butt",
        )
        # Right tip (downward)
        ax.plot(
            [x_r, x_r],
            [y, y - tip_length],
            lw=lw,
            color=color,
            clip_on=False,
            solid_capstyle="butt",
        )

        # Significance label above the bar.
        label = _pvalue_to_stars(bar["pvalue"])
        if label:
            fontsize = sig_config.get("star_fontsize", "small")
            ax.text(
                (x_l + x_r) / 2,
                y,
                label,
                ha="center",
                va="baseline",
                fontsize=fontsize,
                color=color,
                clip_on=False,
            )

    # Expand y-axis to accommodate bars.
    new_y_max = y_base + len(drawn_bars) * bar_step + bar_gap
    ax.set_ylim(y_min, new_y_max)


# -------------------- Violin plot creation --------------------


def _plot_violin_with_bars(
    df: pd.DataFrame,
    response_var: str,
    conditions: list[str],
    bars: list[dict],
    sig_config: dict,
    figsize: tuple[float, float],
    ylabel: str | None = None,
    condition_tick_labels: dict[str, str] | None = None,
    plot_kind: str = "violin",
    ax: mpl.axes.Axes | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Create a violin or box plot with strip overlay and significance bars.

    If ``ax`` is given, the plot is drawn into it (and ``figsize`` is ignored);
    otherwise a new standalone figure of ``figsize`` is created. This lets
    callers assemble many panels into one gridded figure (see
    ``plot_results_grid.py``) while preserving the standalone behavior.
    """
    palette = cnpix_local_sleep.plots.get_condition_palette()
    plot_df = df.loc[df["condition"].isin(conditions)].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Violin/box edge linewidth and strip marker size, each 1 pt below the
    # pubplots figma defaults (violin lw pp.scale(1.25) ~= 3.9 pt; strip marker
    # size pp.scale(1.6) = 5 pt).
    violin_lw = pp.scale(1.25) - 1.0
    strip_size = pp.scale(1.6) - 1.0

    common_kwargs = dict(
        data=plot_df,
        x="condition",
        y=response_var,
        order=conditions,
        fill=True,
        dodge=False,
        hue="condition",
        palette=palette,
        linewidth=violin_lw,
        ax=ax,
    )
    if plot_kind == "box":
        sns.boxplot(**common_kwargs, fliersize=0)
    elif plot_kind == "boxen":
        sns.boxenplot(**common_kwargs, showfliers=False)
    else:
        sns.violinplot(**common_kwargs, inner="quart", cut=0)
    sns.stripplot(
        data=plot_df,
        x="condition",
        y=response_var,
        order=conditions,
        hue="condition",
        palette="dark:black",
        size=strip_size,
        ax=ax,
    )

    # Set labels.
    if ylabel is None:
        ylabel = LABEL_MAP.get(response_var, response_var)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(None)

    # Apply short tick labels if provided.
    if condition_tick_labels:
        tick_labels = [condition_tick_labels.get(c, c) for c in conditions]
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(tick_labels)

    # Draw significance bars.
    draw_significance_bars(ax, bars, conditions, sig_config)

    return fig, ax


def plot_nrem_violin(
    df: pd.DataFrame,
    response_var: str,
    bars: list[dict],
    sig_config: dict,
    ylabel: str | None = None,
    condition_tick_labels: dict[str, str] | None = None,
    plot_kind: str = "violin",
    ax: mpl.axes.Axes | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Create a 4-condition NREM violin/box plot with significance bars."""
    nrem_bars = [b for b in bars if b["contrast_norm"] in NREM_CONTRASTS]
    return _plot_violin_with_bars(
        df,
        response_var,
        NREM_CONDITIONS,
        nrem_bars,
        sig_config,
        figsize=pp.scale(2.5, 2),
        ylabel=ylabel,
        condition_tick_labels=condition_tick_labels,
        plot_kind=plot_kind,
        ax=ax,
    )


def plot_wake_violin(
    df: pd.DataFrame,
    response_var: str,
    bars: list[dict],
    sig_config: dict,
    ylabel: str | None = None,
    condition_tick_labels: dict[str, str] | None = None,
    plot_kind: str = "violin",
    ax: mpl.axes.Axes | None = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Create a 2-condition wake violin/box plot with significance bar."""
    wake_bars = [b for b in bars if b["contrast_norm"] == WAKE_CONTRAST]
    return _plot_violin_with_bars(
        df,
        response_var,
        WAKE_CONDITIONS,
        wake_bars,
        sig_config,
        figsize=pp.scale(1.5, 2),
        ylabel=ylabel,
        condition_tick_labels=condition_tick_labels,
        plot_kind=plot_kind,
        ax=ax,
    )


# -------------------- Output path --------------------


def get_output_path(
    output_dir: Path,
    dataset: str,
    condition_set: str,
    response_var: str,
    model: str,
    plot_type: str,
    plot_kind: str = "violin",
) -> Path:
    """Construct the output SVG path for a figure.

    Returns e.g.
    output_dir/llas/six/median_duration/crossed_interaction/nrem_violin.svg
    """
    return (
        output_dir
        / dataset
        / condition_set
        / response_var
        / model
        / f"{plot_type}_{plot_kind}.svg"
    )


# -------------------- Main --------------------


def resolve_base_dir(yaml_path: Path, results_dir_name: str) -> Path:
    """Walk up from the YAML file to find the directory containing results_dir."""
    base_dir = yaml_path.resolve().parent
    while base_dir != base_dir.parent:
        if (base_dir / results_dir_name).is_dir():
            return base_dir
        base_dir = base_dir.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate violin plots with significance bars from "
        "mixed-effects model results."
    )
    parser.add_argument(
        "plot_yaml",
        type=Path,
        help="Path to the plot YAML config file.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for resolving relative paths in the YAML. "
            "Defaults to auto-detection by walking up from the YAML file."
        ),
    )
    parser.add_argument(
        "--offproj-output",
        action="store_true",
        default=False,
        help=(
            "Write output figures to the offproj NFS path "
            "(.../method=morphological/detection_mode=spatial/clade=Cx/) "
            "instead of the output_dir specified in the YAML config."
        ),
    )
    parser.add_argument(
        "--boxplot",
        action="store_true",
        default=False,
        help="Use boxplots instead of violin plots.",
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

    # Resolve base directory.
    if args.base_dir is not None:
        base_dir = args.base_dir
    else:
        results_dir_name = plot_config.get("results_dir", "_output")
        base_dir = resolve_base_dir(args.plot_yaml, results_dir_name)
        if base_dir is None:
            logger.error(
                "Could not find '%s' directory. Use --base-dir.",
                results_dir_name,
            )
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
    plot_kind = "box" if args.boxplot else "violin"
    destination = plot_config.get("destination", "figma")
    sig_config = plot_config.get("significance", {})
    alpha = sig_config.get("alpha", 0.05)
    condition_tick_labels = plot_config.get("condition_tick_labels")

    # Load main config for per-dataset posthoc definitions.
    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    entries = plot_config.get("entries", [])
    if not entries:
        logger.info("No entries to process.")
        return

    # Cache loaded DataFrames by dataset.
    data_cache: dict[str, pd.DataFrame] = {}
    n_generated = 0
    n_skipped = 0

    for entry in entries:
        if len(entry) != 4:
            logger.error(
                "Entry %r must be [response_var, dataset, condition_set, model]; "
                "skipping.",
                entry,
            )
            n_skipped += 1
            continue
        response_var, dataset, condition_set, model = entry
        entry_label = f"[{response_var}, {dataset}, {condition_set}, {model}]"

        # Load violin data.
        if dataset not in data_cache:
            try:
                data_cache[dataset] = load_violin_data(data_dir, dataset)
            except FileNotFoundError:
                logger.error("Parquet not found for %s, skipping.", entry_label)
                n_skipped += 1
                continue

        df = data_cache[dataset]

        if response_var not in df.columns:
            logger.error(
                "Response variable '%s' not in data, skipping %s.",
                response_var,
                entry_label,
            )
            n_skipped += 1
            continue

        # Resolve the condition set: which conditions/contrasts this fit covers,
        # and hence which violin panel(s) to draw from it.
        cs_config = main_config.get("condition_sets", {}).get(condition_set)
        if cs_config is None:
            logger.error(
                "Condition set '%s' not defined in main config, skipping %s.",
                condition_set,
                entry_label,
            )
            n_skipped += 1
            continue
        cs_conditions = cs_config.get("conditions", [])
        posthoc_strs = cs_config.get("posthocs", [])
        draw_nrem = any(c in NREM_CONDITIONS for c in cs_conditions)
        draw_wake = any(c in WAKE_CONDITIONS for c in cs_conditions)

        # Load results.
        results = load_results(
            results_dir, dataset, condition_set, response_var, model
        )
        if results is None:
            logger.warning("results.json not found for %s, skipping.", entry_label)
            n_skipped += 1
            continue

        # Extract bars from this fit's posthocs.
        bars = extract_significance_bars(results, posthoc_strs, alpha=alpha)

        # Load adjusted data (may be None if not yet generated).
        df_adj = load_adjusted_data(
            results_dir, dataset, condition_set, response_var, model
        )
        if df_adj is None:
            logger.debug(
                "No adjusted_data.csv for %s, skipping adjusted plots.",
                entry_label,
            )

        # Generate plots. Which panels are drawn is determined by the condition
        # set: an `nrem` set draws only the NREM panel, a `wake` set only the
        # wake panel, and a `six` set draws both (from the single six-condition
        # fit, preserving the historical LLAS behavior).
        with pp.destination(destination):
            if draw_nrem:
                fig, ax = plot_nrem_violin(
                    df,
                    response_var,
                    bars,
                    sig_config,
                    condition_tick_labels=condition_tick_labels,
                    plot_kind=plot_kind,
                )
                nrem_path = get_output_path(
                    output_dir,
                    dataset,
                    condition_set,
                    response_var,
                    model,
                    "nrem",
                    plot_kind=plot_kind,
                )
                nrem_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(nrem_path)
                plt.close(fig)
                logger.info("Saved %s", nrem_path)
                n_generated += 1

            if draw_wake:
                fig, ax = plot_wake_violin(
                    df,
                    response_var,
                    bars,
                    sig_config,
                    condition_tick_labels=condition_tick_labels,
                    plot_kind=plot_kind,
                )
                wake_path = get_output_path(
                    output_dir,
                    dataset,
                    condition_set,
                    response_var,
                    model,
                    "wake",
                    plot_kind=plot_kind,
                )
                wake_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(wake_path)
                plt.close(fig)
                logger.info("Saved %s", wake_path)
                n_generated += 1

            # Adjusted plots (random effects removed).
            if df_adj is not None:
                adj_ylabel = LABEL_MAP.get(response_var, response_var) + " (RE removed)"

                if draw_nrem:
                    fig, ax = plot_nrem_violin(
                        df_adj,
                        response_var,
                        bars,
                        sig_config,
                        ylabel=adj_ylabel,
                        condition_tick_labels=condition_tick_labels,
                        plot_kind=plot_kind,
                    )
                    nrem_adj_path = get_output_path(
                        output_dir,
                        dataset,
                        condition_set,
                        response_var,
                        model,
                        "nrem_adjusted",
                        plot_kind=plot_kind,
                    )
                    nrem_adj_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(nrem_adj_path)
                    plt.close(fig)
                    logger.info("Saved %s", nrem_adj_path)
                    n_generated += 1

                if draw_wake:
                    fig, ax = plot_wake_violin(
                        df_adj,
                        response_var,
                        bars,
                        sig_config,
                        ylabel=adj_ylabel,
                        condition_tick_labels=condition_tick_labels,
                        plot_kind=plot_kind,
                    )
                    wake_adj_path = get_output_path(
                        output_dir,
                        dataset,
                        condition_set,
                        response_var,
                        model,
                        "wake_adjusted",
                        plot_kind=plot_kind,
                    )
                    wake_adj_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(wake_adj_path)
                    plt.close(fig)
                    logger.info("Saved %s", wake_adj_path)
                    n_generated += 1

    print(
        f"\nGenerated {n_generated} plots, skipped {n_skipped} entries.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()