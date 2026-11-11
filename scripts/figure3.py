"""Assemble manuscript Figure 3 (per-event and epoched OFF-delta correlations).

a,b  Single-recording scatter of each OFF property against log delta at the OFF
     midpoint, NREM and Wake (Subject 3, PPC).
c,d  Per-pair Spearman rho for the same four pairs, with random-effects pooled
     estimates, for All / Small / Medium + Large OFFs (Table S2a).
e-h  Epoched marginal and semipartial coefficients of OFF area on epoch delta
     (Table S2b), read from the added-value parquets.

Panels a-d recompute from the cached per-event parquet written by
``notebooks/figures/has_value.ipynb``; panels e-h read
``notebooks/figures/added_value/outputs/added_value_data/``. No NFS access.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python ../cnpix-local-sleep/scripts/figure3.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
import pubplots as pp
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from cnpix_local_sleep import plots
from cnpix_local_sleep.morphological import correlation_stats
from cnpix_local_sleep.morphological.pipeline import (
    export_manuscript_correlations as emc,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "figure3"
ADDED_VALUE_DATA = emc._default_added_value_dir()

FIG_W, FIG_H = 7.09, 8.86  # inches at final print size
PNG_DPI = 300

Y_COL = "centre_bipolar_inst_log_delta"
PROPERTIES = [p for p, _ in emc.PROPERTIES]
SCATTER_RECORDING = ("CNPIX12-Santiago", "imec1", "mPPC")
SCATTER_TITLE = "Subject 3, PPC"
COOKS_PERCENTILE = 0.995

_CAT = plots.get_category_palette()
SMALL, MEDLARGE, ALL = _CAT["LLAS"], _CAT["CLAS"], "black"
# Manuscript partition -> (category members, colour), drawn in this order.
PARTITIONS = {
    "All OFFs": (emc.SELECTIONS["All OFFs"], ALL),
    "Small": (emc.SELECTIONS["Small OFFs"], SMALL),
    "Medium + Large": (emc.SELECTIONS["Medium + Large"], MEDLARGE),
}
# Added-value tier names -> manuscript label and colour.
TIERS = {
    "Conservative (CLAS set)": ("Medium + Large", MEDLARGE),
    "LLAS-exclusive": ("Small", SMALL),
}
TIER_COLS = {
    "Conservative (CLAS set)": ("beta_cons_area_sr", "se_cons_area_sr"),
    "LLAS-exclusive": ("beta_llas_excl_area_sr", "se_llas_excl_area_sr"),
}

X_LABELS = {
    "area": "Area (s·µm)",
    "median_duration": "Median duration (ms)",
    "span": "Span (µm)",
    "median_trace": "Median MUA (µV)",
}
X_SCALE = {"median_duration": 1e3}
PAIR_TITLES = {
    "area": "δ vs. area",
    "median_duration": "δ vs. median\nduration",
    "span": "δ vs. span",
    "median_trace": "δ vs. median\nMUA",
}
DELTA_LABEL = r"Midpoint δ (log$_{10}$ µV$^2$)"

MATHTEXT_RC = {
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
}


# --------------------------------------------------------------------------- data


def load_offs(cache_path: Path | None = None) -> pd.DataFrame:
    cache_path = Path(cache_path) if cache_path else emc._default_cache_path()
    cols = sorted(
        set(emc.GROUP_COLS) | {"state", Y_COL} | set(PROPERTIES) | set(emc._filter_columns())
    )
    offs = pd.read_parquet(cache_path, columns=cols)
    offs = emc._assign_filter_category(offs)
    offs["state_class"] = emc._classify_state(offs)
    return offs[offs["state_class"].notna()].reset_index(drop=True)


def per_event_stats(offs: pd.DataFrame) -> dict:
    """(state, property, partition) -> (group correlations, meta-analysis)."""
    out = {}
    for name, (members, _c) in PARTITIONS.items():
        sel = offs[offs["category"].isin(members)]
        for state in emc.CONDITIONS:
            sub = sel[sel["state_class"] == state]
            for prop in PROPERTIES:
                gc = correlation_stats.compute_group_correlations(
                    sub, prop, Y_COL, group_cols=emc.GROUP_COLS
                )
                out[state, prop, name] = (gc, correlation_stats.meta_analyze_correlations(gc))
    return out


def load_added_value(prefix: str) -> dict:
    d = ADDED_VALUE_DATA
    return {
        "marginal": pd.read_parquet(d / f"{prefix}_area_marginal_pooled.parquet"),
        "partial": pd.read_parquet(d / f"{prefix}_area_semipartial_pooled.parquet"),
        "groups": pd.read_parquet(d / f"{prefix}_area_group_partial.parquet"),
    }


# ------------------------------------------------------------------------- layout


def add_axes(fig, left, top, width, height, **kw):
    """Axes placed in final-size inches measured from the figure's top-left."""
    return fig.add_axes(
        [left / FIG_W, 1 - (top + height) / FIG_H, width / FIG_W, height / FIG_H], **kw
    )


def fig_text(fig, x, y, s, **kw):
    """Figure text at final-size inches from the top-left."""
    return fig.text(x / FIG_W, 1 - y / FIG_H, s, **kw)


def panel_letter(fig, x, y, letter):
    fig_text(fig, x, y, letter, fontsize=pp.scale(12), fontweight="bold", ha="left",
             va="baseline")


def fmt(v, spec=".2f"):
    return format(v + 0.0, spec).replace("-", "\u2212")


def set_xticks(ax, xlim, step):
    """Ticks at multiples of ``step`` inside ``xlim``, with true minus signs."""
    ticks = np.arange(np.ceil(xlim[0] / step - 1e-9), np.floor(xlim[1] / step + 1e-9) + 1) * step
    ax.set_xlim(xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(t, "g") for t in ticks])


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------- a, b


def _cooks_keep(x, y, percentile):
    """Keep-mask dropping points above ``percentile`` of Cook's distance (simple OLS)."""
    n = len(x)
    if n <= 2:
        return np.ones(n, dtype=bool)
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    slope, intercept = np.polyfit(x, y, deg=1)
    resid = y - (slope * x + intercept)
    mse = np.sum(resid**2) / (n - 2)
    if sxx == 0 or mse == 0:
        return np.ones(n, dtype=bool)
    h = 1.0 / n + (x - xbar) ** 2 / sxx
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = np.where(h < 1, resid**2 / (2 * mse) * h / (1 - h) ** 2, np.inf)
    return cooks <= np.quantile(cooks, percentile)


def _density_contours(ax, xs, ys, cats, colors, xlim, ylim):
    gridsize, sigma = 120, 4.0
    xedges = np.linspace(*xlim, gridsize + 1)
    yedges = np.linspace(*ylim, gridsize + 1)
    xc, yc = 0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:])
    levels = np.geomspace(0.05, 0.9, 4)
    for cat, color in colors.items():
        m = cats == cat
        if m.sum() < 10:
            continue
        hist, _, _ = np.histogram2d(xs[m], ys[m], bins=[xedges, yedges])
        hist = gaussian_filter(hist, sigma=sigma)
        if hist.max() <= 0:
            continue
        ax.contour(xc, yc, (hist / hist.max()).T, levels=levels, colors=[color],
                   linewidths=pp.scale(0.6), zorder=3)


def scatter_panel(ax, sub, prop, rho, ylim):
    colors = {"Small": SMALL, "Medium + Large": MEDLARGE}
    alpha = {"Small": 0.12, "Medium + Large": 0.25}
    d = sub[[prop, Y_COL, "category"]].dropna()
    x = d[prop].to_numpy(float) * X_SCALE.get(prop, 1.0)
    y = d[Y_COL].to_numpy(float)
    cats = np.where(d["category"].astype(str) == "LLAS", "Small", "Medium + Large")
    keep = _cooks_keep(x, y, COOKS_PERCENTILE)
    x, y, cats = x[keep], y[keep], cats[keep]

    rgba = np.array([to_rgba(colors[c], alpha[c]) for c in cats])
    order = np.random.default_rng(0).permutation(len(x))
    ax.scatter(x[order], y[order], c=rgba[order], s=pp.scale(1) ** 2 * 0.25,
               linewidths=0, rasterized=True, zorder=1)

    xlim = (x.min(), x.max())
    _density_contours(ax, x, y, cats, colors, xlim, ylim)
    slope, intercept = np.polyfit(x, y, deg=1)
    ax.plot(xlim, np.polyval([slope, intercept], xlim), color="k", lw=pp.scale(0.8), zorder=4)
    pad = 0.03 * (xlim[1] - xlim[0])
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
    ax.set_ylim(ylim)
    ax.text(0.04, 0.97, f"ρ = {fmt(rho)}", transform=ax.transAxes, ha="left", va="top", zorder=5,
            bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.75})
    ax.set_xlabel(X_LABELS[prop])
    despine(ax)


def draw_scatter_rows(fig, offs, stats, tops, left=0.52, right=7.02, gap=0.13, height=1.1):
    subj, probe, struct = SCATTER_RECORDING
    rec = offs[(offs["subject"] == subj) & (offs["probe"] == probe) & (offs["structure"] == struct)]
    width = (right - left - 3 * gap) / 4
    for state, top, letter in zip(emc.CONDITIONS, tops, "ab"):
        sub = rec[rec["state_class"] == state]
        ylim = tuple(np.nanquantile(sub[Y_COL], [0.0005, 0.9995]))
        ylim = (ylim[0], ylim[1] + 0.12 * (ylim[1] - ylim[0]))
        panel_letter(fig, 0.0, top - 0.143, letter)
        head = fig_text(fig, left, top - 0.11, state, fontsize=pp.scale(7), fontweight="bold",
                        ha="left", va="baseline")
        fig.add_artist(matplotlib.text.Annotation(
            f"  ({SCATTER_TITLE})", xy=(1, 0), xycoords=head, va="bottom", ha="left",
            fontsize=pp.scale(7)))
        axes = []
        for i, prop in enumerate(PROPERTIES):
            ax = add_axes(fig, left + i * (width + gap), top, width, height,
                          sharey=axes[0] if axes else None)
            gc, _meta = stats[state, prop, "All OFFs"]
            row = gc[(gc["subject"] == subj) & (gc["probe"] == probe) & (gc["structure"] == struct)]
            scatter_panel(ax, sub, prop, float(row["rho"].iloc[0]), ylim)
            if i:
                ax.tick_params(labelleft=False)
            axes.append(ax)
        axes[0].set_ylabel(DELTA_LABEL)
        if state == "NREM":
            handles = [Line2D([0], [0], marker="o", ls="", color=c, ms=pp.scale(3), label=n)
                       for n, c in (("Small", SMALL), ("Medium + Large", MEDLARGE))]
            fig.legend(handles=handles, loc="lower right", ncols=2, frameon=False,
                       bbox_to_anchor=(right / FIG_W, 1 - (top - 0.04) / FIG_H),
                       handletextpad=0.1, columnspacing=1.0, borderaxespad=0)


# ----------------------------------------------------------------------- c, d

ROW_PITCH, CAT_DY = 5.0, 0.62
DIAMOND_TOP, DIAMOND_SPACING, DIAMOND_HW = -1.4 * ROW_PITCH, 3.6, 0.45 * ROW_PITCH


def forest_overlay(ax, entries, label_diamonds):
    """Recording-aligned overlay of the partition forests on one axes.

    ``entries`` is a list of ``(name, gc, meta, color)``. Each subject-structure pair
    occupies one slot; partitions are offset within it. Pooled diamonds are stacked
    below, labelled "Pooled" when ``label_diamonds``.
    """
    recs = sorted({k for _n, gc, _m, _c in entries
                   for k in gc[emc.GROUP_COLS].itertuples(index=False, name=None)})
    rank = {k: i for i, k in enumerate(recs)}
    base = np.arange(len(recs))[::-1] * ROW_PITCH
    offsets = (np.arange(len(entries)) - (len(entries) - 1) / 2.0) * CAT_DY
    lo, hi, dys = [], [], []
    for i, (_name, gc, meta, color) in enumerate(entries):
        keys = list(gc[emc.GROUP_COLS].itertuples(index=False, name=None))
        yy = np.array([base[rank[k]] for k in keys]) + offsets[i]
        rho, clo, chi = (gc[c].to_numpy() for c in ("rho", "ci_lo", "ci_hi"))
        ax.errorbar(rho, yy, xerr=[rho - clo, chi - rho], fmt="o", color=color, ecolor=color,
                    elinewidth=pp.scale(0.5), markersize=pp.scale(2.6), markeredgewidth=0,
                    capsize=0, alpha=0.8, zorder=2)
        yd = DIAMOND_TOP - i * DIAMOND_SPACING
        dys.append(yd)
        ax.fill([meta["ci_lo"], meta["overall_rho"], meta["ci_hi"], meta["overall_rho"]],
                [yd, yd + DIAMOND_HW, yd, yd - DIAMOND_HW], color=color, lw=0, zorder=5)
        lo += [clo.min(), meta["ci_lo"]]
        hi += [chi.max(), meta["ci_hi"]]
    ax.axvline(0, color="grey", ls="--", lw=pp.scale(0.6), zorder=0)
    xlo = max(-1.0, min(0.0, min(lo)) - 0.08)
    xhi = min(1.0, max(0.0, max(hi)) + 0.08)
    set_xticks(ax, (xlo, xhi), 0.5)
    ax.set_ylim(min(dys) - DIAMOND_HW - 1.5, base[0] + ROW_PITCH * 0.7)
    ax.tick_params(axis="y", length=0)
    if label_diamonds:
        ax.set_yticks([np.mean(dys)])
        ax.set_yticklabels(["Pooled"])
    else:
        ax.set_yticks([])
    return len(recs)


def draw_forests(fig, stats, top, height=2.45, left=0.66, right=7.02, gap=0.07, block_gap=0.24):
    width = (right - left - block_gap - 6 * gap) / 8
    for b, (state, letter) in enumerate(zip(emc.CONDITIONS, "cd")):
        x0 = left + b * (4 * width + 3 * gap + block_gap)
        panel_letter(fig, 0.0 if b == 0 else x0 - 0.2, top - 0.323, letter)
        fig_text(fig, x0 + 2 * width + 1.5 * gap, top - 0.29, state, fontsize=pp.scale(7),
                 fontweight="bold", ha="center", va="baseline")
        for i, prop in enumerate(PROPERTIES):
            ax = add_axes(fig, x0 + i * (width + gap), top, width, height)
            entries = [(n, *stats[state, prop, n], c) for n, (_m, c) in PARTITIONS.items()]
            n_rec = forest_overlay(ax, entries, label_diamonds=(b == 0 and i == 0))
            ax.set_title(PAIR_TITLES[prop], pad=pp.scale(3))
        fig_text(fig, x0 + 2 * width + 1.5 * gap, top + height + 0.2, "Spearman ρ",
                 ha="center", va="top")
    labels = {"Medium + Large": ("Medium", "+ Large")}
    for i, (name, (_m, c)) in enumerate(PARTITIONS.items()):
        y = top + 0.055 + 0.118 * i
        fig.add_artist(Line2D([0.065 / FIG_W], [1 - y / FIG_H], transform=fig.transFigure,
                              marker="o", ls="", color=c, ms=pp.scale(3)))
        for j, line in enumerate(labels.get(name, (name,))):
            fig_text(fig, 0.16, y + 0.097 * j, line, ha="left", va="center")
    fig_text(fig, 0.55, top + height / 2,
             f"Subject × structure pairs (n = {n_rec})", rotation=90, ha="left", va="center")


# ----------------------------------------------------------------------- e-h

BAND_LW, DOT_MS = 4.0, 2.2


def dumbbell(ax, av, xlim):
    """Marginal (open, upper) vs semipartial (filled, lower) pooled coefficient per tier."""
    dy = 0.18
    for i, (tier, (_label, c)) in enumerate(TIERS.items()):
        y = len(TIERS) - 1 - i
        m = av["marginal"].set_index("tier").loc[tier]
        p = av["partial"].set_index("tier").loc[tier]
        ax.plot([m["pooled_std_beta"], p["pooled_std_beta"]], [y + dy, y - dy],
                color=c, lw=pp.scale(1.0), alpha=0.45, zorder=1)
        for r, yy, filled in ((m, y + dy, False), (p, y - dy, True)):
            ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=c, lw=pp.scale(BAND_LW),
                    solid_capstyle="butt", alpha=0.55 if filled else 0.35, zorder=2)
            ax.plot([r["pooled_std_beta"]], [yy], marker="o", ls="", mfc=c if filled else "white",
                    mec=c, ms=pp.scale(DOT_MS), mew=pp.scale(0.6), zorder=3)
    ax.axvline(0, color="grey", ls="--", lw=pp.scale(0.6), zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, len(TIERS) - 0.4)
    ax.set_yticks(range(len(TIERS)))
    ax.set_yticklabels([lab.replace(" + ", "\n+ ") for lab, _c in TIERS.values()][::-1])
    for tl, (_lab, c) in zip(ax.get_yticklabels(), list(TIERS.values())[::-1]):
        tl.set_color(c)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Correlation with δ\n(pooled standardized β)")
    despine(ax)


def semipartial_forest(ax, av, tier):
    label, c = TIERS[tier]
    bcol, scol = TIER_COLS[tier]
    d = av["groups"].dropna(subset=[bcol, scol]).sort_values(bcol).reset_index(drop=True)
    y = np.arange(len(d))
    pr = av["partial"].set_index("tier").loc[tier]
    ax.errorbar(d[bcol], y, xerr=1.96 * d[scol], fmt="o", markersize=pp.scale(1.8),
                markeredgewidth=0, color=c, ecolor=c, elinewidth=pp.scale(0.5), capsize=0,
                alpha=0.85, zorder=3)
    yd, hw = -3.2, 1.6
    ax.fill([pr["ci_lo"], pr["pooled_std_beta"], pr["ci_hi"], pr["pooled_std_beta"]],
            [yd, yd + hw, yd, yd - hw], color=c, lw=0, zorder=5)
    ax.axvline(0, color="grey", ls="--", lw=pp.scale(0.6), zorder=0)
    ax.set_yticks([])
    ax.set_ylim(yd - hw - 0.8, len(d) + 0.5)
    ax.set_title(label, color=c, loc="left", pad=pp.scale(2))


def _shared_xlim(frames, pad):
    lo = min(f.min() for f in frames)
    hi = max(f.max() for f in frames)
    return min(0.0, lo) - pad, max(0.0, hi) + pad


def block_title(fig, x, y, state, subtitle):
    fig_text(fig, x, y, state, fontsize=pp.scale(7), fontweight="bold", ha="center",
             va="baseline")
    fig_text(fig, x, y + 0.10, subtitle, ha="center", va="baseline")


def draw_added_value(fig, top, height=0.9, dumbbell_w=0.78, forest_w=0.7, forest_gap=0.12):
    blocks = [("NREM", load_added_value("nrem"), 0.50, "ef"),
              ("Wake", load_added_value("wake"), 4.30, "gh")]
    for state, av, x0, letters in blocks:
        ax_e = add_axes(fig, x0, top, dumbbell_w, height)
        pooled = pd.concat([av["marginal"], av["partial"]])
        xlim = _shared_xlim([pooled["ci_lo"], pooled["ci_hi"]], 0.12)
        dumbbell(ax_e, av, xlim)
        set_xticks(ax_e, xlim, 0.5 if xlim[1] - xlim[0] > 0.8 else 0.25)
        handles = [Line2D([0], [0], marker="o", ls="", mfc=mfc, mec="k", mew=pp.scale(0.6),
                          ms=pp.scale(DOT_MS + 0.6), label=lab)
                   for lab, mfc in (("Marginal", "white"), ("Partial", "k"))]
        ax_e.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2,
                    frameon=False, handletextpad=0.05, columnspacing=0.6, borderaxespad=0.15)

        fx0 = x0 + dumbbell_w + 0.42
        frames = []
        for tier, (bcol, scol) in TIER_COLS.items():
            g = av["groups"].dropna(subset=[bcol, scol])
            frames += [g[bcol] - 1.96 * g[scol], g[bcol] + 1.96 * g[scol]]
        xlim = _shared_xlim(frames, 0.05)
        for j, tier in enumerate(TIERS):
            ax = add_axes(fig, fx0 + j * (forest_w + forest_gap), top, forest_w, height)
            semipartial_forest(ax, av, tier)
            set_xticks(ax, xlim, 0.5 if xlim[1] - xlim[0] > 0.8 else 0.25)
        fig_text(fig, fx0 - 0.08, top + height / 2, "Subject × structure pairs",
                 rotation=90, ha="right", va="center")
        fig_text(fig, fx0 + forest_w + forest_gap / 2, top + height + 0.18,
                 "Semipartial correlation", ha="center", va="top")

        block_title(fig, x0 + dumbbell_w / 2 - 0.2, top - 0.37, state, "OFF area vs. epoch δ")
        block_title(fig, fx0 + forest_w + forest_gap / 2, top - 0.37, state,
                    "OFF area vs. epoch δ")
        panel_letter(fig, 0.0 if state == "NREM" else x0 - 0.5, top - 0.383, letters[0])
        panel_letter(fig, fx0 - 0.40, top - 0.383, letters[1])


# ------------------------------------------------------------------------- main


def make_figure(offs: pd.DataFrame, stats: dict, out_dir: Path, stem: str = "figure3"):
    """Draw and save (SVG + PNG). Saving must happen inside the pubplots context, which
    supplies the fonts and ``svg.fonttype``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Axes are placed by hand, so pubplots' figure.autolayout must not run.
    with pp.destination("figma", style=MATHTEXT_RC), plt.rc_context({"figure.autolayout": False}):
        fig = plt.figure(figsize=pp.scale(FIG_W, FIG_H), layout="none")
        draw_scatter_rows(fig, offs, stats, tops=(0.30, 2.06))
        draw_forests(fig, stats, top=4.05)
        draw_added_value(fig, top=7.42)
        # The figma canvas is 300/96 larger than print size, so 96 dpi gives a 300 dpi
        # image at print size (raster elements in the SVG likewise).
        dpi = PNG_DPI / pp.scale(1.0)
        paths = out_dir / f"{stem}.svg", out_dir / f"{stem}.png"
        fig.savefig(paths[0], dpi=dpi)
        fig.savefig(paths[1], dpi=dpi, pil_kwargs={"dpi": (PNG_DPI, PNG_DPI)})
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    offs = load_offs()
    for p in make_figure(offs, per_event_stats(offs), args.out_dir):
        print(p)


if __name__ == "__main__":
    main()
