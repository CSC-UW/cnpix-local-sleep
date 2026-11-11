"""Assemble Supplementary Figure 6 (overlapping vs. local OFF size at matched duration).

a  Cohen's d, overlapping minus local, in duration, normalized span and area, per
   condition: from all OFFs of each subject-structure-condition cell (open) and
   within duration deciles (filled), pooled across cells by DerSimonian-Laird.
b  Mean normalized span of local and overlapping OFFs per duration decile (top) and
   the fraction of each group's OFFs falling in each decile (bottom).

Reads the event-level LLAS OFF table via ``cnpix_local_sleep.release_data`` (the
``r-offp/inst/extdata`` copy, else the GitHub Release asset). No NFS access. Prints
the numbers quoted in the caption and writes the plotted tables as parquet.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python ../cnpix-local-sleep/scripts/supp_figure6.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubplots as pp
from matplotlib.lines import Line2D

from cnpix_local_sleep import release_data
from cnpix_local_sleep.morphological.pipeline import cross_structure_offs as cso

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "supp_figure6"

sys.path.insert(0, str(REPO / "r-offp"))
from plot_locality_results import LOCALITY_PALETTE
from plot_results import NREM_CONDITIONS, WAKE_CONDITIONS

FIG_W, FIG_H = 7.09, 5.7  # inches at final print size
PNG_DPI = 300

CONDITIONS = NREM_CONDITIONS + WAKE_CONDITIONS
TICK = {
    "Early.BSL.NREM": "E.BSL",
    "Early.REC.NREM.Match": "E.Match",
    "Early.REC.NREM": "E.REC",
    "Late.REC.NREM": "L.REC",
    "Early.NOD.Wake": "E.NOD.Wake",
    "Late.NOD.Wake": "L.NOD.Wake",
}
QUANTITIES = {"duration": "Duration", "span_rel": "Normalized span", "area": "Area"}
KINDS = {"raw": ("All OFFs", "white"), "matched": ("Within duration deciles", "k")}
CELL = ["subject", "structure", "condition"]
N_DECILES = 10
MIN_PER_CELL = 20  # local and overlapping OFFs per cell
MIN_PER_DECILE = 5  # local and overlapping OFFs per decile
C_LOC, C_OVR = LOCALITY_PALETTE["Local"], LOCALITY_PALETTE["Overlapping"]


# --------------------------------------------------------------------------- data


def load_offs() -> pd.DataFrame:
    """Cortical OFFs, each flagged ``overlapping`` if any other cortical structure of
    the same subject has a temporally overlapping OFF in the same condition."""
    path = release_data.get_event_table_path("llas")
    cols = [
        "subject",
        "structure",
        "condition",
        "clade",
        "start_time",
        "end_time",
        "duration",
        "span",
        "max_span",
        "area",
    ]
    offs = pd.read_parquet(path, columns=cols)
    offs = offs[offs["clade"] == "Cx"]
    offs = offs.assign(span_rel=offs["span"] / offs["max_span"])
    parts = []
    for (_subj, cond), g in offs.groupby(["subject", "condition"], observed=True):
        by_struct = {
            (s, cond): d.sort_values("start_time")
            for s, d in g.groupby("structure", observed=True)
        }
        structs = [s for s, _c in by_struct]
        degree = cso.compute_overlap_degree(by_struct, structs, cond)
        parts += [by_struct[s, cond].assign(overlapping=degree[s] > 0) for s in structs]
    return pd.concat(parts, ignore_index=True)


def cohen_d(x_o: np.ndarray, x_l: np.ndarray) -> tuple[float, float]:
    """Cohen's d (pooled SD) of ``x_o`` minus ``x_l`` and its large-sample variance."""
    n_o, n_l = len(x_o), len(x_l)
    sp = np.sqrt(
        ((n_o - 1) * x_o.var(ddof=1) + (n_l - 1) * x_l.var(ddof=1)) / (n_o + n_l - 2)
    )
    d = (x_o.mean() - x_l.mean()) / sp if sp > 0 else np.nan
    v = (n_o + n_l) / (n_o * n_l) + d**2 / (2 * (n_o + n_l))
    return d, v


def split(d: pd.DataFrame, q: str) -> tuple[np.ndarray, np.ndarray]:
    """Column ``q`` of the overlapping and of the local OFFs."""
    return d.loc[d["overlapping"], q].to_numpy(), d.loc[~d["overlapping"], q].to_numpy()


def within_decile_d(d: pd.DataFrame, q: str) -> tuple[float, float, int]:
    """Decile ds averaged with weights n_o*n_l/(n_o+n_l), over deciles with at least
    ``MIN_PER_DECILE`` OFFs per group; returns (d, variance, number of deciles)."""
    ests, vars_, ws = [], [], []
    for _, g in d.groupby("dq"):
        x_o, x_l = split(g, q)
        if len(x_o) < MIN_PER_DECILE or len(x_l) < MIN_PER_DECILE:
            continue
        est, var = cohen_d(x_o, x_l)
        if not np.isfinite(est):
            continue
        ests.append(est)
        vars_.append(var)
        ws.append(len(x_o) * len(x_l) / (len(x_o) + len(x_l)))
    if not ests:
        return np.nan, np.nan, 0
    ests, vars_, ws = map(np.asarray, (ests, vars_, ws))
    return (
        np.sum(ws * ests) / ws.sum(),
        np.sum(ws**2 * vars_) / ws.sum() ** 2,
        len(ests),
    )


def cell_tables(offs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell effect sizes, and per-cell, per-decile span and composition."""
    effects, deciles = [], []
    for key, d in offs.groupby(CELL, observed=True):
        n_o, n_l = int(d["overlapping"].sum()), int((~d["overlapping"]).sum())
        if n_o < MIN_PER_CELL or n_l < MIN_PER_CELL:
            continue
        d = d.assign(
            dq=pd.qcut(d["duration"].rank(method="first"), N_DECILES, labels=False)
        )
        cell = dict(zip(CELL, key))
        rec = dict(cell, n=len(d), n_overlapping=n_o, n_local=n_l)
        for q in QUANTITIES:
            rec[f"raw_d_{q}"], rec[f"raw_var_d_{q}"] = cohen_d(*split(d, q))
            (
                rec[f"matched_d_{q}"],
                rec[f"matched_var_d_{q}"],
                rec[f"matched_bins_{q}"],
            ) = within_decile_d(d, q)
        effects.append(rec)
        for dq, g in d.groupby("dq"):
            x_o, x_l = split(g, "span_rel")
            deciles.append(
                dict(
                    cell,
                    dq=dq,
                    dur_med=g["duration"].median(),
                    span_local=x_l.mean() if len(x_l) >= MIN_PER_DECILE else np.nan,
                    span_over=x_o.mean() if len(x_o) >= MIN_PER_DECILE else np.nan,
                    frac_local=len(x_l) / n_l,
                    frac_over=len(x_o) / n_o,
                )
            )
    return pd.DataFrame(effects), pd.DataFrame(deciles)


def dersimonian_laird(y: np.ndarray, v: np.ndarray) -> dict:
    """Random-effects pooled estimate with a 95% CI, over finite effects."""
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    w = 1 / v
    q = np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1 / (v + tau2)
    est = np.sum(ws * y) / np.sum(ws)
    se = np.sqrt(1 / np.sum(ws))
    return {"d": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se, "k": k}


def pool(effects: pd.DataFrame) -> pd.DataFrame:
    """Pooled d per condition (and per state) x kind x quantity."""
    sets = [(c, [c]) for c in CONDITIONS] + [
        ("NREM", NREM_CONDITIONS),
        ("Wake", WAKE_CONDITIONS),
    ]
    rows = []
    for name, conds in sets:
        sub = effects[effects["condition"].isin(conds)]
        for kind in KINDS:
            for q in QUANTITIES:
                m = dersimonian_laird(
                    sub[f"{kind}_d_{q}"].to_numpy(), sub[f"{kind}_var_d_{q}"].to_numpy()
                )
                rows.append(dict(set=name, kind=kind, quantity=q, **m))
    return pd.DataFrame(rows)


def print_caption_numbers(effects, deciles, pooled):
    print(f"Cells with >= {MIN_PER_CELL} local and >= {MIN_PER_CELL} overlapping OFFs:")
    for cond in CONDITIONS:
        e = effects[effects["condition"] == cond]
        print(
            f"  {cond}: {len(e)} subject-structure pairs, {e['subject'].nunique()} "
            f"subjects, {e['structure'].nunique()} structures"
        )
    bins = effects[[f"matched_bins_{q}" for q in QUANTITIES]]
    print(f"Cells without a decile of >= {MIN_PER_DECILE} OFFs per group:")
    print(effects.loc[bins.eq(0).any(axis=1), CELL].to_string(index=False))
    ends = deciles[deciles["dq"].isin([0, N_DECILES - 1])]
    med = ends.groupby(["condition", "dq"])["dur_med"].median().unstack() * 1e3
    print("Median duration (ms) of the first and last decile, median over cells:")
    print(med.loc[CONDITIONS].round(1).to_string())
    rows = pooled[["d", "lo", "hi", "k"]].itertuples(index=False)
    pooled = pooled.assign(
        ci=[f"{d:.2f} [{lo:.2f}, {hi:.2f}] k={k}" for d, lo, hi, k in rows]
    )
    table = pooled.pivot(index=["set", "quantity"], columns="kind", values="ci")
    print("Pooled Cohen's d [95% CI], overlapping minus local:")
    rows = [(s, q) for s in CONDITIONS + ["NREM", "Wake"] for q in QUANTITIES]
    print(table.loc[rows, list(KINDS)].to_string())


# ------------------------------------------------------------------------- layout


def add_axes(fig, left, right, top, height):
    """Axes placed in final-size inches measured from the figure's top-left."""
    return fig.add_axes(
        [
            left / FIG_W,
            1 - (top + height) / FIG_H,
            (right - left) / FIG_W,
            height / FIG_H,
        ]
    )


def panel_letter(fig, x, y, letter):
    fig.text(
        x / FIG_W,
        1 - y / FIG_H,
        letter,
        fontweight="bold",
        fontsize=pp.scale(8),
        va="top",
        ha="left",
    )


def draw_pooled(fig, pooled, top=0.45, height=1.7, left=0.95, width=1.85, gap=0.2):
    panel_letter(fig, 0.0, 0.05, "a")
    lw = pp.scale(0.8)
    y = np.arange(len(CONDITIONS))
    for j, (q, title) in enumerate(QUANTITIES.items()):
        x0 = left + j * (width + gap)
        ax = add_axes(fig, x0, x0 + width, top, height)
        for i, (kind, (label, face)) in enumerate(KINDS.items()):
            sub = pooled[(pooled["kind"] == kind) & (pooled["quantity"] == q)]
            sub = sub.set_index("set").loc[CONDITIONS]
            yy = y + (i - 0.5) * 0.3
            ax.errorbar(
                sub["d"],
                yy,
                xerr=[sub["d"] - sub["lo"], sub["hi"] - sub["d"]],
                fmt="none",
                ecolor="k",
                elinewidth=lw,
                capsize=0,
                zorder=2,
            )
            ax.plot(
                sub["d"],
                yy,
                "o",
                mfc=face,
                mec="k",
                mew=lw,
                ms=pp.scale(3),
                ls="none",
                label=label,
                zorder=3,
            )
        ax.axvline(0, color="0.5", lw=pp.scale(0.5), ls=(0, (4, 3)), zorder=0)
        ax.set_ylim(len(CONDITIONS) - 0.4, -0.6)
        ax.set_yticks(y, [TICK[c] for c in CONDITIONS] if j == 0 else [])
        ax.set_xlim(-0.15, 0.85)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"])
        ax.set_title(title)
        ax.set_xlabel("Cohen's d, overlapping − local", labelpad=pp.scale(2))
        if j == len(QUANTITIES) - 1:
            ax.legend(
                loc="lower right",
                ncol=2,
                frameon=False,
                handletextpad=0.2,
                columnspacing=1.2,
                bbox_to_anchor=(1.0, 1.17),
                borderaxespad=0,
            )


def draw_deciles(
    fig,
    deciles,
    top=2.85,
    h_top=1.45,
    h_bot=0.75,
    gap_v=0.12,
    left=0.65,
    width=0.93,
    gap=0.15,
):
    panel_letter(fig, 0.0, 2.65, "b")
    lw, ms = pp.scale(0.8), pp.scale(1.8)
    x = np.arange(N_DECILES) + 1
    for j, cond in enumerate(CONDITIONS):
        x0 = left + j * (width + gap)
        ax_top = add_axes(fig, x0, x0 + width, top, h_top)
        ax_bot = add_axes(fig, x0, x0 + width, top + h_top + gap_v, h_bot)
        c = deciles[deciles["condition"] == cond]
        g = c.dropna(subset=["span_local", "span_over"]).groupby("dq")
        for key, col in [("span_local", C_LOC), ("span_over", C_OVR)]:
            m, se = g[key].mean(), g[key].sem()
            ax_top.fill_between(m.index + 1, m - se, m + se, color=col, alpha=0.3, lw=0)
            ax_top.plot(m.index + 1, m, "-o", color=col, lw=lw, ms=ms)
        ax_top.set_title(TICK[cond])
        ax_top.text(
            0.06,
            0.95,
            f"n = {c.groupby(['subject', 'structure']).ngroups}",
            transform=ax_top.transAxes,
            va="top",
            ha="left",
            fontsize=pp.scale(6),
        )
        gc = c.groupby("dq")
        w = 0.4
        ax_bot.bar(x - w / 2, gc["frac_local"].mean(), w, color=C_LOC, lw=0)
        ax_bot.bar(x + w / 2, gc["frac_over"].mean(), w, color=C_OVR, lw=0)
        ax_top.set_ylim(0.05, 0.92)
        ax_bot.set_ylim(0, 0.21)
        for a in (ax_top, ax_bot):
            a.set_xlim(0.3, N_DECILES + 0.7)
            a.set_xticks([1, 5, 10])
        ax_top.set_xticklabels([])
        ax_bot.set_xticklabels(["1", "5", "10"])
        if j == 0:
            ax_top.set_ylabel("Normalized span")
            ax_bot.set_ylabel("Fraction\nof OFFs")
            ax_top.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax_bot.set_yticks([0, 0.1, 0.2], ["0", "0.1", "0.2"])
        else:
            ax_top.set_yticks([0.2, 0.4, 0.6, 0.8], [])
            ax_bot.set_yticks([0, 0.1, 0.2], [])
    n = len(CONDITIONS)
    fig.text(
        (left + (n * width + (n - 1) * gap) / 2) / FIG_W,
        1 - (top + h_top + gap_v + h_bot + 0.27) / FIG_H,
        "Duration decile (within each subject, structure and condition)",
        ha="center",
        va="top",
    )
    handles = [
        Line2D([], [], color=col, lw=lw, marker="o", ms=ms) for col in (C_LOC, C_OVR)
    ]
    fig.legend(
        handles,
        ["Local", "Overlapping"],
        loc="upper right",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1 - 0.05 / FIG_W, 1 - 2.58 / FIG_H),
        borderaxespad=0,
    )


def make_figure(pooled, deciles, out_dir: Path, stem: str = "supp_figure6"):
    """Draw and save (SVG + PNG) inside the pubplots context, which supplies the fonts
    and ``svg.fonttype``."""
    with pp.destination("figma"), plt.rc_context({"figure.autolayout": False}):
        fig = plt.figure(figsize=pp.scale(FIG_W, FIG_H), layout="none")
        draw_pooled(fig, pooled)
        draw_deciles(fig, deciles)
        # The figma canvas is 300/96 larger than print size, so 96 dpi gives 300 dpi.
        dpi = PNG_DPI / pp.scale(1.0)
        paths = out_dir / f"{stem}.svg", out_dir / f"{stem}.png"
        fig.savefig(paths[0], dpi=dpi)
        fig.savefig(paths[1], dpi=dpi, pil_kwargs={"dpi": (PNG_DPI, PNG_DPI)})
    return paths


# ------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    effects, deciles = cell_tables(load_offs())
    pooled = pool(effects)
    print_caption_numbers(effects, deciles, pooled)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"pooled": pooled, "decile_cells": deciles}
    for name, df in tables.items():
        df.to_parquet(args.out_dir / f"supp_figure6_{name}.parquet", index=False)
    for p in make_figure(pooled, deciles, args.out_dir):
        print(p)


if __name__ == "__main__":
    main()
