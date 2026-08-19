"""Render Figma-ready NOD-to-NREM.Rebound transform-sensitivity grids.

The companion R script exports fitted marginal expectations and LRT p-values to
JSON. This renderer keeps the R model results intact while using pubplots' Figma
destination so SVG text remains editable Arial at its intended import size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pubplots as pp

CORR_ROOT = Path("_output_correlations")
SENSITIVITY_DIR = CORR_ROOT / "xtransform_sensitivity"
METRICS = ("rate", "total_area_norm")
OFF_TYPES = ("llas", "clas")
TRANSFORMS = ("raw", "rank", "log")
FIGSIZE_IN = (7.0, 3.5)
SUBJECT_MARKERS = (
    "o", "^", "s", "D", "P", "X", "*", "v", "<", ">", "h", "H", "p", "8", "+"
)


def padded_range(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.nanmin(values), np.nanmax(values)
    padding = (upper - lower) * 0.08
    if padding == 0:
        padding = max(abs(lower) * 0.08, 0.1)
    return lower - padding, upper + padding


def p_label(p_value: float) -> str:
    return "p<0.001" if p_value < 0.001 else f"p={p_value:.2g}"


def line_style(p_value: float) -> str:
    if p_value < 0.05:
        return "-"
    if p_value < 0.1:
        return "--"
    return (0, (1.5, 2.5))


def metric_labels(metric: str) -> tuple[str, str]:
    if metric == "rate":
        return (
            "Sleep deprivation OFF rate (Hz,\ntransformed)",
            "OFF rate NREM rebound (Hz)",
        )
    return (
        "Sleep deprivation total area norm. (au,\ntransformed)",
        "Total area norm. NREM rebound (au)",
    )


def panel_lookup(payload: dict) -> dict[tuple[str, str, str], dict]:
    lookup = {}
    for panel in payload["panels"]:
        key = (panel["metric"], panel["off_type"], panel["transform"])
        if key in lookup:
            raise ValueError(f"Duplicate publication panel: {key}")
        lookup[key] = panel
    expected = {(metric, off_type, transform)
                for metric in METRICS
                for off_type in OFF_TYPES
                for transform in TRANSFORMS}
    if lookup.keys() != expected:
        raise ValueError(f"Publication panels do not match the required grid: {lookup.keys()}")
    return lookup


def draw_panel(
    ax,
    panel: dict,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    structure_colors: dict[str, str],
    subject_markers: dict[str, str],
    x_label: str | None,
    y_label: str | None,
) -> None:
    data = pd.DataFrame(panel["data"])
    prediction = pd.DataFrame(panel["prediction"])

    ax.fill_between(
        prediction["x"], prediction["CI_low"], prediction["CI_high"],
        color="#595959", alpha=0.18, linewidth=0
    )
    ax.axhline(0, color="#a6a6a6", linewidth=pp.scale(0.35), zorder=0)
    ax.plot(
        prediction["x"], prediction["Predicted"], color="black",
        linewidth=pp.scale(0.55), linestyle=line_style(panel["pval"]), zorder=2
    )
    for subject, subject_data in data.groupby("subject", sort=False):
        colors = subject_data["structure"].map(structure_colors)
        if colors.isna().any():
            missing = sorted(subject_data.loc[colors.isna(), "structure"].unique())
            raise ValueError(f"Missing structure colors: {missing}")
        ax.scatter(
            subject_data["x"], subject_data["y"], marker=subject_markers[subject],
            c=colors, s=pp.scale(2.1) ** 2, linewidths=pp.scale(0.35),
            alpha=0.9, zorder=3
        )

    ax.set(xlim=xlim, ylim=ylim, xlabel=x_label, ylabel=y_label)
    ax.set_title(f"{panel['transform']} ({p_label(panel['pval'])})", fontsize=pp.scale(6))
    ax.tick_params(
        axis="both", labelsize=pp.scale(5), width=pp.scale(0.35),
        length=pp.scale(2), pad=pp.scale(1.75)
    )
    ax.xaxis.label.set_size(pp.scale(6))
    ax.yaxis.label.set_size(pp.scale(6))
    ax.xaxis.labelpad = pp.scale(2)
    ax.yaxis.labelpad = pp.scale(2)
    for spine in ax.spines.values():
        spine.set_linewidth(pp.scale(0.35))


def draw_legend(ax, handles: list[Line2D], title: str) -> None:
    ax.axis("off")
    legend = ax.legend(
        handles=handles, loc="upper left", ncol=3, frameon=False,
        fontsize=pp.scale(5), title=title, title_fontsize=pp.scale(6),
        handletextpad=pp.scale(0.25), columnspacing=pp.scale(0.5),
        borderaxespad=0, labelspacing=pp.scale(0.25), handlelength=pp.scale(0.8),
    )
    legend._legend_box.align = "left"


def render(payload: dict, output_path: Path) -> None:
    panels = panel_lookup(payload)
    structure_colors = payload["structure_colors"]
    subject_order = payload["subject_order"]
    if len(subject_order) > len(SUBJECT_MARKERS):
        raise ValueError("Add publication marker styles for the new subjects.")
    subject_markers = dict(zip(subject_order, SUBJECT_MARKERS[:len(subject_order)], strict=True))
    subject_labels = {
        subject: f"Subject{index}"
        for index, subject in enumerate(subject_order, start=1)
    }

    x_limits: dict[tuple[str, str, str], tuple[float, float]] = {}
    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for metric in METRICS:
        for off_type in OFF_TYPES:
            y_values = np.concatenate([
                pd.DataFrame(panels[(metric, off_type, transform)]["data"])["y"].to_numpy()
                for transform in TRANSFORMS
            ])
            y_limits[(metric, off_type)] = padded_range(y_values)
        for off_type in OFF_TYPES:
            for transform in TRANSFORMS:
                x_values = pd.DataFrame(
                    panels[(metric, off_type, transform)]["data"]
                )["x"].to_numpy()
                x_limits[(metric, off_type, transform)] = padded_range(x_values)

    with pp.destination("figma"):
        fig = plt.figure(figsize=pp.scale(*FIGSIZE_IN))
        fig.set_layout_engine("none")
        grid = fig.add_gridspec(
            3, 3, width_ratios=(1, 0.16, 1), height_ratios=(1, 1, 0.8),
            wspace=0, hspace=0.8
        )
        fig.subplots_adjust(left=0.045, right=1, top=0.94, bottom=0.07)

        for row, metric in enumerate(METRICS):
            x_label, y_label = metric_labels(metric)
            for off_index, off_type in enumerate(OFF_TYPES):
                triplet = grid[row, off_index * 2].subgridspec(1, 3, wspace=0.03)
                shared_y_axis = None
                for transform_index, transform in enumerate(TRANSFORMS):
                    ax = fig.add_subplot(triplet[0, transform_index], sharey=shared_y_axis)
                    draw_panel(
                        ax, panels[(metric, off_type, transform)],
                        x_limits[(metric, off_type, transform)], y_limits[(metric, off_type)],
                        structure_colors, subject_markers,
                        x_label if transform == "rank" else None,
                        y_label if transform == "raw" else None,
                    )
                    if shared_y_axis is None:
                        shared_y_axis = ax
                    else:
                        ax.tick_params(axis="y", left=False, labelleft=False)

        structure_handles = [
            Line2D([], [], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="none", markersize=pp.scale(3.6), label=structure)
            for structure, color in structure_colors.items()
        ]
        subject_handles = [
            Line2D([], [], marker=marker, color="black", linestyle="None",
                   markersize=pp.scale(1.2), label=subject_labels[subject])
            for subject, marker in subject_markers.items()
        ]
        draw_legend(fig.add_subplot(grid[2, 0]), structure_handles, "Structure")
        draw_legend(fig.add_subplot(grid[2, 2]), subject_handles, "Subject")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=SENSITIVITY_DIR,
        help="Directory holding publication_grid_<predictor>.json exports.",
    )
    args = parser.parse_args()
    for predictor in ("NOD", "NOD.Wake"):
        input_path = args.input_dir / f"publication_grid_{predictor}.json"
        output_path = args.input_dir / f"publication_grid_{predictor}.svg"
        with input_path.open() as file:
            payload = json.load(file)
        if payload["predictor"] != predictor:
            raise ValueError(f"{input_path} contains the wrong predictor.")
        render(payload, output_path)
        print(f"Saved Figma SVG to {output_path}")


if __name__ == "__main__":
    main()
