#!/usr/bin/env python
"""Assemble the replacement Supplementary Figure S2 draft.

The published S2b reports homeostatic changes in OFF-period onset/offset MAD. That
statistic depends on how many channels an OFF period spans for mechanical reasons, so
the published contrasts are partly contrasts in OFF *size*; the replacement reports
size-adjusted cell means instead.

This script exists because the previous assembled draft was made ad hoc and could not be
regenerated when the numbers changed. Every panel here is drawn from a versioned input
(``full48h_llas_offs`` is a Release asset rather than a committed file, fetched and cached
on first use -- see ``docs/DATA.md``):

===== ========================================================= ==================
panel  what                                                      source
===== ========================================================= ==================
  a    experimental protocol                                     placeholder, reused
                                                                 unchanged from the
                                                                 published figure
  b    the MAD floor and its release                             recomputed from
                                                                 ``full48h_llas_offs``
  c    one size-independent latent pair reproduces the curve     notebook cache
                                                                 ``mechanical_surface_fit.csv``
  d    the shared size-adjustment curve actually applied         refitted (~5 s)
  e    duration is *not* mechanical, so it is not adjusted for   notebook cache
                                                                 ``duration_invariance_simulated.csv``
                                                                 + recomputed observed
  f    size-adjusted ON/OFF transition synchrony                 r-offp ``_output``
===== ========================================================= ==================

Panels b-d are the case for adjusting for OFF size; e is the case against adjusting for
OFF duration; f is the result.

Panel f is drawn by the *same* ``plot_results`` functions that emit the standalone
panel SVGs, so the assembled draft and the Figma panels cannot drift apart.

Usage::

    python plot_s2_replacement.py                 # both figures, default paths
    python plot_s2_replacement.py --no-comparison # skip the review-only comparison

Prerequisites: ``off-analysis export-adjusted-edge-statistics``, ``renv::install('.')``,
the four ``adj_mean_*_mad`` analyses, and one execution of the validation notebook (for
the two cached simulation CSVs).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubplots as pp
import yaml

import plot_results as pr
from cnpix_local_sleep.morphological import edge_synchrony_validation as esv

logger = logging.getLogger(__name__)

# Constants (kept in sync with the validation notebook)

#: Validated categorical palette (dataviz skill: blue / orange / aqua / violet).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a19", "#8a8a80"

MS = 1e3  #: seconds -> milliseconds

#: Duration sweep of the trace-level simulation, and the channel counts it was run at.
#: Mirrors the notebook cells that write ``duration_invariance_simulated.csv``.
DURATIONS_MS = [40.0, 60.0, 90.0, 130.0, 190.0, 280.0]
DURATION_CHANNEL_COUNTS = [11, 16, 26, 41]

#: Unscaled figure size, in inches, passed through ``pp.scale`` under the figma
#: destination, the same convention ``plot_results_grid.py`` uses.
FIGSIZE = (7.2, 9.6)

#: Unscaled sizes for the standalone mechanism panels. Each keeps roughly the axes
#: aspect ratio it has inside the assembly (~1.5:1 per half-width slot), so a panel
#: dropped into Figma on its own looks like the one in the draft.
PANEL_FIGSIZE = {"b": (7.0, 2.4), "c": (3.5, 2.4), "d": (3.5, 2.4), "e": (7.0, 2.4)}


def sc(value: float) -> float:
    """Scale a font size, marker size or line width onto the figma canvas.

    The figma destination enlarges the canvas and the rcParams font sizes together,
    so anything this script sets by hand has to be enlarged with them or it renders
    as a speck. Call only inside a ``pp.destination`` block.
    """
    return pp.scale(value)

#: Panel f entries: (response_var, condition_set, axis label).
PANEL_F = [
    ("adj_mean_onset_mad", "nrem", "ON-transition MAD (ms)"),
    ("adj_mean_onset_mad", "wake", None),
    ("adj_mean_offset_mad", "nrem", "OFF-transition MAD (ms)"),
    ("adj_mean_offset_mad", "wake", None),
]

DATASET = "clas"
MODEL = "crossed_interaction"


# -------------------- Inputs --------------------


REPO_ROOT = Path(__file__).resolve().parents[1]


def notebook_outputs(repo_root: Path = REPO_ROOT) -> Path:
    """Directory holding the validation notebook's cached tables and figures."""
    return repo_root / "notebooks/figures/edge_synchrony_validation/outputs"


def load_events() -> pd.DataFrame:
    """The full-48h LLAS events, with ``n_channels`` and ``size_class``.

    Fetched from the Release and cached unless a copy is already in ``inst/extdata``.
    """
    return esv.load_events(
        "llas", columns=["condition", "onset_mad", "offset_mad", "onset_jitter",
                         "onset_slope", "offset_jitter", "offset_slope",
                         "median_duration"]
    )


# -------------------- Panels --------------------


def panel_a_placeholder(ax) -> None:
    """Reserve the protocol schematic, which is reused unchanged."""
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                      edgecolor=MUTED, linestyle="--", linewidth=sc(0.8))
    )
    ax.text(
        0.5, 0.5,
        "experimental protocol, reused unchanged from the published figure\n"
        "(not regenerated by this script)",
        transform=ax.transAxes, ha="center", va="center", fontsize=sc(6.5), color=MUTED,
    )


def panel_b_floor(ax_left, ax_right, nrem: pd.DataFrame) -> None:
    """The MAD floor: where the statistic is identically zero, and where it releases.

    Left: both edges' floors. Right: the onset edge decomposed into MAD, detrended
    jitter and the fitted depth ramp.
    """
    curve = esv.edge_floor_curve(nrem, "onset")
    offset_curve = esv.edge_floor_curve(nrem, "offset")
    forced = esv.mad_zero_forced_max_n()

    # Both edges are floored by the same four-channel structuring element, so the
    # left axis shows both; the right axis stays on the onset edge, where the
    # jitter/ramp decomposition is drawn.
    ax_left.plot(curve.n_channels, curve.p_mad_zero, "-o", ms=sc(2.5), lw=sc(1.0),
                 color=BLUE, label="ON-transition MAD")
    ax_left.plot(offset_curve.n_channels, offset_curve.p_mad_zero, "-o", ms=sc(2.5),
                 lw=sc(1.0), color=ORANGE, label="OFF-transition MAD")
    ax_left.axvspan(curve.n_channels.min() - 0.5, forced + 0.5, color=MUTED, alpha=0.12)
    ax_left.text(forced + 1, 0.92, f"forced zero\n(n $\\leq$ {forced})", fontsize=sc(6.0),
                 color=INK, va="top")
    ax_left.set(xlabel="channels spanned (n = span/20 + 1)",
                ylabel="P(MAD = 0)", xlim=(5, 45))
    ax_left.legend(loc="upper right", fontsize=sc(6.0))

    ax_right.plot(curve.n_channels, curve.mean_mad * MS, "-o", ms=sc(2.5), lw=sc(1.0),
                  color=BLUE, label="ON-transition MAD")
    ax_right.plot(curve.n_channels, curve.mean_jitter * MS, "-o", ms=sc(2.5), lw=sc(1.0),
                  color=ORANGE, label="jitter (detrended)")
    ax_right.plot(curve.n_channels, curve.mean_ramp * MS, "-o", ms=sc(2.5), lw=sc(1.0),
                  color=AQUA, label="|slope| $\\times$ span")
    ax_right.axvspan(curve.n_channels.min() - 0.5, forced + 0.5, color=MUTED,
                     alpha=0.12)
    ax_right.set(xlabel="channels spanned", ylabel="ms", xlim=(5, 45))
    ax_right.legend(loc="upper left", fontsize=sc(6.0))


def panel_c_mechanical(ax, outputs: Path) -> None:
    """One size-independent latent parameter pair reproduces the whole size curve."""
    fit = pd.read_csv(outputs / "mechanical_surface_fit.csv", index_col=0)
    grid = pd.read_csv(outputs / "mechanical_surface_grid.csv")
    winner = grid.sort_values("rmse").iloc[0]
    forced = esv.mad_zero_forced_max_n()

    ax.plot(fit.index, fit["observed"] * MS, "-o", ms=sc(3.5), lw=sc(1.2), color=BLUE,
            label="observed")
    ax.plot(fit.index, fit["predicted"] * MS, "--s", ms=sc(3.5), lw=sc(1.2), color=ORANGE,
            label="one size-independent\nlatent parameter pair")
    ax.axvspan(5, forced + 0.5, color=MUTED, alpha=0.12)
    ax.text(forced + 0.8, ax.get_ylim()[1] * 0.55, "floored,\nexcluded from fit",
            fontsize=sc(6.0), color=INK)
    ax.set(xlabel="channels spanned", ylabel="mean onset MAD (ms)")
    ax.legend(loc="lower right", fontsize=sc(6.0))
    ax.set_title(
        f"$\\sigma$ = {winner.sigma_ms:.0f} ms, slope SD = "
        f"{winner.slope_sd_us_per_um:.0f} $\\mu$s/$\\mu$m, "
        f"RMSE = {winner.rmse * MS:.2f} ms",
        fontsize=sc(6.5), color=MUTED, loc="left",
    )


def panel_e_duration(ax_sim, ax_obs, outputs: Path, nrem: pd.DataFrame) -> None:
    """Duration moves MAD in the data but not in the simulation, so it is a mediator."""
    simulated = pd.read_csv(outputs / "duration_invariance_simulated.csv", index_col=0)
    shades = dict(zip(DURATION_CHANNEL_COUNTS,
                      ["#bcd6f4", "#7aaeea", "#3a7fd0", "#12508f"], strict=True))

    for n_channels in DURATION_CHANNEL_COUNTS:
        ax_sim.plot(simulated.columns.astype(float), simulated.loc[n_channels],
                    "-o", ms=sc(2.5), lw=sc(1.0), color=shades[n_channels],
                    label=f"{n_channels} ch")

    bins = [0] + [d * 1e-3 for d in DURATIONS_MS[:-1]] + [np.inf]
    observed = (
        nrem[nrem["n_channels"].isin(DURATION_CHANNEL_COUNTS)]
        .assign(dur_bin=lambda d: pd.cut(d["median_duration"], bins))
        .groupby(["n_channels", "dur_bin"], observed=True)["onset_mad"]
        .agg(["mean", "size"])
    )
    for n_channels in DURATION_CHANNEL_COUNTS:
        block = observed.loc[n_channels]
        block = block[block["size"] >= 100]
        ax_obs.plot([b.mid * MS for b in block.index], block["mean"] * MS,
                    "-o", ms=sc(2.5), lw=sc(1.0), color=shades[n_channels],
                    label=f"{n_channels} ch")

    ax_sim.set_title("simulated: latent dispersion fixed", fontsize=sc(6.5), color=MUTED,
                     loc="left")
    ax_obs.set_title("observed: same events in the data", fontsize=sc(6.5), color=MUTED,
                     loc="left")
    for ax in (ax_sim, ax_obs):
        ax.set_xlabel("OFF median duration (ms)")
        ax.set_xlim(20, 300)
    ax_sim.set_ylabel("mean onset MAD (ms)")
    ax_obs.legend(fontsize=sc(6.0), ncols=2)


def fit_panel_d_curve(medium_large: pd.DataFrame):
    """Fit the shared size curve once; the ALS takes ~40 s, and both the assembled
    draft and the standalone panel d draw the same fit."""
    work = medium_large.copy()
    work["state"] = np.where(
        work["condition"].isin(esv.NREM_CONDITIONS), "nrem", "wake"
    )
    work["unit"] = work["combo"] + "@" + work["state"]
    work["cell"] = work["unit"] + "@" + work["condition"].astype(str)
    return work, esv.fit_shared_size_curve(work, "onset_mad", "unit", "cell")


def panel_d_shared_curve(ax, work: pd.DataFrame, curve) -> None:
    """The one size-adjustment curve, with each pair's own profile behind it."""
    residual = work["onset_mad"] - work.groupby("cell", observed=True)[
        "onset_mad"
    ].transform("mean")
    profile = (
        work.assign(residual=residual)
        .groupby(["unit", "n_channels"], observed=True)
        .agg(residual=("residual", "mean"), n_events=("residual", "size"))
        .reset_index()
    )
    profile = profile[profile["n_events"] >= 50]
    for unit, block in profile.groupby("unit", observed=True):
        amplitude = curve.unit_lambda.get(unit, np.nan)
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        ax.plot(block["n_channels"], block["residual"] / amplitude,
                color=MUTED, lw=sc(0.4), alpha=0.3)
    ax.plot(curve.curve.index, curve.curve.to_numpy(), color=BLUE, lw=sc(1.6),
            label="shared curve", zorder=5)

    well_sampled = curve.unit_shape_correlation.dropna()
    well_sampled = well_sampled[curve.unit_events.reindex(well_sampled.index) >= 60]
    # The curve carries the *shape* only: it is scaled to unit SD across events, so
    # the y axis is in SD of the curve, and each pair's amplitude (in ms per SD)
    # converts it back to milliseconds of MAD.
    ax.set(xlabel="channels spanned", ylabel="size effect, f(n)  (SD of curve)",
           xlim=(11, 80), ylim=(-2.5, 3.0))
    ax.legend(loc="lower right", fontsize=sc(6.0))
    amplitudes = curve.unit_lambda[curve.unit_events >= 60] * MS
    ax.set_title(
        f"grey: each pair's own profile, rescaled by its amplitude "
        f"({amplitudes.min():.1f}-{amplitudes.max():.1f} ms/SD); "
        f"median r = {well_sampled.median():.2f}",
        fontsize=sc(6.5), color=MUTED, loc="left",
    )


def panel_f_violins(axes, results_dir: Path, main_config: dict, sig_config: dict,
                    alpha: float, tick_labels: dict) -> None:
    """The result: size-adjusted transition synchrony, in the house panel style."""
    for ax, (response_var, condition_set, ylabel) in zip(axes, PANEL_F, strict=True):
        df = pr.load_adjusted_data(
            results_dir, DATASET, condition_set, response_var, MODEL
        )
        if df is None:
            raise FileNotFoundError(
                f"adjusted_data.csv missing for {response_var}/{condition_set}. "
                "Run the r-offp analyses first (see the module docstring)."
            )
        results = pr.load_results(
            results_dir, DATASET, condition_set, response_var, MODEL
        )
        if results is None:
            raise FileNotFoundError(
                f"results.json missing for {response_var}/{condition_set}."
            )
        posthoc_strs = main_config["condition_sets"][condition_set]["posthocs"]
        bars = pr.extract_significance_bars(results, posthoc_strs, alpha=alpha)
        draw = pr.plot_nrem_violin if condition_set == "nrem" else pr.plot_wake_violin
        draw(
            df, response_var, bars, sig_config,
            ylabel=ylabel or "", condition_tick_labels=tick_labels, ax=ax,
        )
        if ylabel is None:
            ax.set_ylabel(None)


# -------------------- Assembly --------------------


def label_panel(fig, ax, letter: str) -> None:
    """Bold panel letter at the top-left of an axes, in figure coordinates."""
    box = ax.get_position()
    # Anchor to the column, not to the axes, so the letter always clears the
    # rotated y-axis label rather than landing on it.
    x = 0.012 if box.x0 < 0.45 else 0.505
    fig.text(x, box.y1 + 0.012, letter, fontsize=sc(12.0), fontweight="bold",
             va="bottom", ha="left", color=INK)


def build_figure(nrem: pd.DataFrame, panel_d: tuple, outputs: Path, results_dir: Path,
                 main_config: dict, sig_config: dict, alpha: float,
                 tick_labels: dict) -> plt.Figure:
    """Compose the whole draft. Assumes an active ``pp.destination`` context."""
    fig = plt.figure(figsize=pp.scale(*FIGSIZE))
    grid = fig.add_gridspec(
        5, 6, height_ratios=[0.16, 1.0, 1.0, 1.0, 1.3], hspace=0.75, wspace=0.9,
        left=0.09, right=0.985, top=0.975, bottom=0.05,
    )

    ax_a = fig.add_subplot(grid[0, :])
    panel_a_placeholder(ax_a)

    ax_b1 = fig.add_subplot(grid[1, 0:3])
    ax_b2 = fig.add_subplot(grid[1, 3:6])
    panel_b_floor(ax_b1, ax_b2, nrem)

    ax_c = fig.add_subplot(grid[2, 0:3])
    panel_c_mechanical(ax_c, outputs)
    ax_d = fig.add_subplot(grid[2, 3:6])
    panel_d_shared_curve(ax_d, *panel_d)

    ax_e1 = fig.add_subplot(grid[3, 0:3])
    ax_e2 = fig.add_subplot(grid[3, 3:6], sharey=ax_e1)
    panel_e_duration(ax_e1, ax_e2, outputs, nrem)

    ax_f = [
        fig.add_subplot(grid[4, 0:2]), fig.add_subplot(grid[4, 2:3]),
        fig.add_subplot(grid[4, 3:5]), fig.add_subplot(grid[4, 5:6]),
    ]
    panel_f_violins(ax_f, results_dir, main_config, sig_config, alpha, tick_labels)
    for index, ax in enumerate(ax_f):
        # Only the wake panels' labels ("E.NOD.Wake") are long enough to collide.
        ax.tick_params(axis="x", labelrotation=30 if index % 2 else 0,
                       labelsize=sc(6.0))
        ax.tick_params(axis="y", labelsize=sc(6.0))

    for letter, ax in [("a", ax_a), ("b", ax_b1), ("c", ax_c), ("d", ax_d),
                       ("e", ax_e1), ("f", ax_f[0])]:
        label_panel(fig, ax, letter)
    return fig


def build_panel(letter: str, nrem: pd.DataFrame, outputs: Path,
                panel_d: tuple) -> plt.Figure:
    """One mechanism panel on its own canvas, for import as a standalone SVG.

    Draws through the *same* panel functions the assembly uses, so a panel exported
    on its own can never drift away from the same panel in the draft. Panel a is a
    placeholder and panel f's violins already ship from ``plot_results_cell.py``, so
    only b-e are offered here.
    """
    fig = plt.figure(figsize=pp.scale(*PANEL_FIGSIZE[letter]), layout="constrained")
    if letter == "b":
        left, right = fig.subplots(1, 2)
        panel_b_floor(left, right, nrem)
    elif letter == "c":
        panel_c_mechanical(fig.subplots(), outputs)
    elif letter == "d":
        panel_d_shared_curve(fig.subplots(), *panel_d)
    elif letter == "e":
        sim, obs = fig.subplots(1, 2, sharey=True)
        panel_e_duration(sim, obs, outputs, nrem)
    else:
        raise ValueError(f"no standalone builder for panel {letter!r}")
    return fig


def build_comparison(results_dir: Path, main_config: dict, sig_config: dict,
                     alpha: float, tick_labels: dict) -> plt.Figure:
    """Published vs adjusted, side by side. Review only, not for the supplement."""
    rows = [
        ("mean_onset_mad", "published ON (s)"),
        ("adj_mean_onset_mad", "adjusted ON (ms)"),
        ("mean_offset_mad", "published OFF (s)"),
        ("adj_mean_offset_mad", "adjusted OFF (ms)"),
    ]
    fig, axes = plt.subplots(
        len(rows), 2, figsize=pp.scale(5.4, 7.2),
        gridspec_kw={"width_ratios": [2, 1], "hspace": 0.75, "wspace": 0.45},
    )
    for row, (response_var, label) in enumerate(rows):
        for col, condition_set in enumerate(("nrem", "wake")):
            ax = axes[row, col]
            df = pr.load_adjusted_data(
                results_dir, DATASET, condition_set, response_var, MODEL
            )
            results = pr.load_results(
                results_dir, DATASET, condition_set, response_var, MODEL
            )
            if df is None or results is None:
                ax.set_axis_off()
                continue
            posthoc_strs = main_config["condition_sets"][condition_set]["posthocs"]
            bars = pr.extract_significance_bars(results, posthoc_strs, alpha=alpha)
            draw = pr.plot_nrem_violin if condition_set == "nrem" else pr.plot_wake_violin
            draw(df, response_var, bars, sig_config,
                 ylabel=label if col == 0 else "",
                 condition_tick_labels=tick_labels, ax=ax)
            ax.set_title(f"n = {len(df)} cells", fontsize=sc(6.5), color=MUTED, loc="left")
            ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Destination (default: docs/ms/figures_draft).")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip the review-only published-vs-adjusted figure.")
    parser.add_argument("--no-panels", action="store_true",
                        help="Skip the standalone per-panel SVGs (b-e).")
    parser.add_argument("--dpi", type=int, default=110,
                        help="Raster resolution. The canvas is ~22 x 30 in "
                             "(figma scaling), so 110 gives a page-sized PNG.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s: %(message)s")

    base_dir = Path(__file__).resolve().parent
    outputs = notebook_outputs()
    for required in ("mechanical_surface_fit.csv", "mechanical_surface_grid.csv",
                     "duration_invariance_simulated.csv"):
        if not (outputs / required).exists():
            logger.error("Missing notebook cache %s. Execute the validation notebook "
                         "first.", outputs / required)
            sys.exit(1)

    with open(base_dir / "config/plots/cx_homeostasis.yaml") as handle:
        plot_config = yaml.safe_load(handle)
    with open(base_dir / plot_config["config"]) as handle:
        main_config = yaml.safe_load(handle)
    sig_config = plot_config.get("significance", {})
    alpha = sig_config.get("alpha", 0.05)
    tick_labels = plot_config.get("condition_tick_labels", {})
    results_dir = base_dir / plot_config["results_dir"]

    output_dir = args.output_dir or (REPO_ROOT / "docs/ms/figures_draft")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading events ...")
    events = load_events()
    nrem = events[events["condition"].isin(esv.NREM_CONDITIONS)]
    logger.info("Fitting the shared size curve ...")
    panel_d = fit_panel_d_curve(events[events["size_class"] == "Medium+Large"])

    with pp.destination("figma"):
        logger.info("Building the assembled draft ...")
        fig = build_figure(nrem, panel_d, outputs, results_dir, main_config,
                           sig_config, alpha, tick_labels)
        path = output_dir / "S2_replacement_draft.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        logger.info("Saved %s (+ .pdf)", path)
        plt.close(fig)

        if not args.no_panels:
            panel_dir = output_dir / "S2_panels"
            panel_dir.mkdir(parents=True, exist_ok=True)
            for letter in PANEL_FIGSIZE:
                fig = build_panel(letter, nrem, outputs, panel_d)
                path = panel_dir / f"S2_panel_{letter}.svg"
                # As for the assembly: panel d's title is wider than a half-width
                # canvas, so let the bbox grow to hold it rather than clipping it.
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                logger.info("Saved %s", path)
                plt.close(fig)
            logger.info("Panel f's violins ship separately, from plot_results_cell.py: "
                        "_output/figures/%s/{nrem,wake}/adj_mean_*_mad/%s/",
                        DATASET, MODEL)

        if not args.no_comparison:
            logger.info("Building the review-only comparison ...")
            fig = build_comparison(results_dir, main_config, sig_config, alpha,
                                   tick_labels)
            path = output_dir / "S2_published_vs_adjusted.png"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            logger.info("Saved %s", path)
            plt.close(fig)


if __name__ == "__main__":
    main()
