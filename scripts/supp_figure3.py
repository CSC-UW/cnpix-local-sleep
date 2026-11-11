"""Draw Supplementary Figure 3 (wake OFF rate on the sleep deprivation and baseline days).

For each included subject-structure pair, the wake OFF rate (OFF count / wake time) in
the Early.NOD.Wake and Late.NOD.Wake windows ("SD" day) and in the wake that falls
within the same clock-time windows 24 h earlier ("BSL" day), for All (LLAS, left) and
Medium/large (CLAS, right) OFFs.

a  Per-pair rates in each window (grey) and their medians (black). Zero rates sit on a
   separate "0" row below an axis break.
b  Per-pair late - early differences on a symlog axis, with medians. Stars: two-sided
   Wilcoxon signed-rank tests against 0, Holm-corrected over the two days per column.

The per-pair table is built from the full-recording morphological ``offs.parquet``
files and the condition and whole-recording hypnograms, so it needs ``/Volumes/npx_nfs``
mounted. It is cached in ``--out-dir``; ``--plot-only`` redraws from that cache without
NFS. The numbers cited in the caption and the reply to referees are printed to stdout.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python ../cnpix-local-sleep/scripts/supp_figure3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import pubplots as pp
import scipy.stats as st
import wisc_ecephys_tools as wet
from matplotlib.ticker import FixedLocator, NullLocator
from statsmodels.stats.multitest import multipletests
from wisc_ecephys_tools.rats import exp_hgs

from cnpix_local_sleep import atlas, const, hyp, off_tables
from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "r-offp"))
from plot_results import _pvalue_to_stars

DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "supp_figure3"
STEM = "supp_figure3"
TABLE_NAME = "baseline_wake_control.parquet"

FIG_W, FIG_H = 4.6, 3.9  # inches at final print size
PNG_DPI = 300

DAY_S = 86400.0
PAIR = ["subject", "probe", "structure"]
LATE, EARLY = const.CONTRASTS["NOD.Incline"]
WINDOWS = [EARLY, LATE]
DAYS = [("BSL", "Baseline\nday", (0, 1)), ("SD", "Sleep\ndeprivation day", (2.4, 3.4))]
PANELS = [("llas", "All OFFs"), ("clas", "Medium/large OFFs")]
OFF_COLUMNS = ["start_time", "state", "span", "median_duration", "duration"]

ZERO_Y, BREAK_Y = 6e-5, 1.5e-4
YTICKS = [ZERO_Y, 1e-3, 1e-2, 1e-1, 1]
LINTHRESH = 1e-3
DTICKS = [-1, -0.1, -0.01, 0, 0.01, 0.1, 1]


# --------------------------------------------------------------------------- data


def whole_recording_wake(subject: str, probe: str):
    try:
        hg = hyp.load_whole_recording_hypnogram(subject, probe)
    except FileNotFoundError:
        # The offproj data project has no whole-recording hypnogram for CNPIX6-Eugene.
        print(f"{subject} {probe}: whole-recording hypnogram from the shared project")
        hg = exp_hgs.load_consolidated_hypnogram(
            wet.get_sglx_project("shared"),
            const.EXPERIMENT,
            subject,
            probe,
            simplify=True,
            fallback=True,
        )
    return hg.keep_states(["Wake"])


def pair_rows(subject: str, probe: str, structure: str) -> list[dict]:
    path = SOURCE_CONFIG.files_module.get_full_offs_path(subject, probe, structure)
    offs = pd.read_parquet(path, columns=OFF_COLUMNS).sort_values(
        "start_time", ignore_index=True
    )
    t = offs["start_time"].to_numpy()
    wake_off = (offs["state"] == "Wake").to_numpy()
    masks = {
        filt: off_tables.off_filter_mask(offs, filt).to_numpy() & wake_off
        for filt, _ in PANELS
    }
    hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
    whole = whole_recording_wake(subject, probe)
    rows = []
    for window in WINDOWS:
        sd = hgs[window].keep_states(["Wake"])
        w0, w1 = sd["start_time"].min(), sd["end_time"].max()
        for day, hg in (("SD", sd), ("BSL", whole.trim(w0 - DAY_S, w1 - DAY_S))):
            wake_s = float(hg["duration"].sum())
            covered = hg.covers_time(t)
            for filt, mask in masks.items():
                n = int((covered & mask).sum())
                rows.append(
                    {
                        "subject": subject,
                        "probe": probe,
                        "structure": atlas.consolidate_structure_label(structure),
                        "window": window,
                        "day": day,
                        "filt": filt,
                        "wake_s": wake_s,
                        "n": n,
                        "rate": n / wake_s if wake_s > 0 else np.nan,
                    }
                )
    return rows


def build_table() -> pd.DataFrame:
    rows = []
    for subject, probe, structure in SOURCE_CONFIG.get_subject_probe_structure_list():
        rows += pair_rows(subject, probe, structure)
        print("done", subject, probe, structure, flush=True)
    return pd.DataFrame(rows)


def paired(table: pd.DataFrame, filt: str, day: str, values: str = "rate"):
    """Pairs x [early, late] for one filter and day."""
    d = table[(table["filt"] == filt) & (table["day"] == day)]
    return d.pivot_table(index=PAIR, columns="window", values=values)[WINDOWS]


def late_minus_early(table: pd.DataFrame, filt: str, day: str) -> pd.Series:
    d = paired(table, filt, day)
    return d[LATE] - d[EARLY]


def holm_pvalues(table: pd.DataFrame, filt: str) -> tuple[list[float], np.ndarray]:
    """Wilcoxon p of the late - early differences per day, raw and Holm-corrected."""
    raw = [st.wilcoxon(late_minus_early(table, filt, day)).pvalue for day, _, _ in DAYS]
    return raw, multipletests(raw, method="holm")[1]


def report(table: pd.DataFrame) -> None:
    pairs = table[PAIR].drop_duplicates()
    print(f"\nn = {len(pairs)} pairs from {pairs['subject'].nunique()} rats")
    for day, _, _ in DAYS:
        minutes = paired(table, "llas", day, values="wake_s").median() / 60
        print(
            f"{day} median wake (min): early {minutes[EARLY]:.1f}, "
            f"late {minutes[LATE]:.1f}"
        )
    for filt, title in PANELS:
        raw, holm = holm_pvalues(table, filt)
        diffs = {day: late_minus_early(table, filt, day) for day, _, _ in DAYS}
        for (day, _, _), p, ph in zip(DAYS, raw, holm):
            dd = diffs[day]
            print(
                f"{title}, {day}: late > early in {(dd > 0).sum()}/{len(dd)}, "
                f"median {dd.median():+.4f} Hz, Wilcoxon p {p:.3g}, Holm p {ph:.3g}"
            )
        did = (diffs["SD"] - diffs["BSL"]).dropna()
        print(
            f"{title}, SD - BSL difference-in-differences: positive in "
            f"{(did > 0).sum()}/{len(did)}, median {did.median():+.4f} Hz, "
            f"Wilcoxon p {st.wilcoxon(did).pvalue:.3g}"
        )


# ------------------------------------------------------------------------ figure


def axis_break(ax):
    """Break marks on both side spines between the zero row and the log axis."""
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for x in (0, 1):
        for dy in (0.88, 1.0):
            ax.plot(
                [x - 0.025, x + 0.025],
                [BREAK_Y * dy / 1.06, BREAK_Y * dy * 1.06],
                color="k",
                lw=pp.scale(0.5),
                transform=tr,
                clip_on=False,
                zorder=5,
            )
        ax.plot(
            [x, x],
            [BREAK_Y * 0.88, BREAK_Y],
            color="white",
            lw=pp.scale(1.2),
            transform=tr,
            clip_on=False,
            zorder=4,
        )


def draw_column(ax, bx, table, filt, title, rng):
    _raw, holm = holm_pvalues(table, filt)
    lw_pair, lw_med = pp.scale(0.4), pp.scale(1.0)
    tr = mtransforms.blended_transform_factory(bx.transData, bx.transAxes)
    for (day, _label, xs), p in zip(DAYS, holm):
        d = paired(table, filt, day)
        for row in d.where(d > 0, ZERO_Y).values:
            ax.plot(xs, row, color="0.6", lw=lw_pair, zorder=1)
        med = d.median().values
        ax.plot(
            xs,
            np.where(med > 0, med, ZERO_Y),
            color="k",
            lw=lw_med,
            marker="o",
            ms=pp.scale(2.5),
            zorder=3,
        )
        xc, dd = np.mean(xs), late_minus_early(table, filt, day).values
        bx.scatter(
            xc + rng.uniform(-0.18, 0.18, len(dd)),
            dd,
            s=pp.scale(1.6) ** 2,
            color="k",
            lw=0,
            zorder=3,
        )
        bx.plot(
            [xc - 0.35, xc + 0.35],
            [np.median(dd)] * 2,
            color="k",
            lw=lw_med,
            solid_capstyle="butt",
            zorder=4,
        )
        stars = _pvalue_to_stars(p)
        bx.text(
            xc,
            0.86,
            stars or "n.s.",
            ha="center",
            va="baseline",
            transform=tr,
            fontsize=pp.scale(7) if stars else pp.scale(6),
            color="k" if stars else "0.5",
        )
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xticks([x for _, _, xs in DAYS for x in xs], [""] * 4)
    axis_break(ax)
    bx.axhline(0, color="0.5", lw=pp.scale(0.5), ls=(0, (4, 3)), zorder=0)
    bx.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    bx.set_xticks([np.mean(xs) for _, _, xs in DAYS], [label for _, label, _ in DAYS])
    for a in (ax, bx):
        a.set_xlim(-0.5, 3.9)


def make_figure(table: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Draw and save (SVG + PNG) inside the pubplots context, which supplies the fonts
    and ``svg.fonttype``."""
    assert table["rate"].notna().all()
    assert table.loc[table["rate"] > 0, "rate"].min() > BREAK_Y
    rng = np.random.default_rng(0)
    with pp.destination("figma"), plt.rc_context({"figure.autolayout": False}):
        fig, axs = plt.subplots(
            2, 2, figsize=pp.scale(FIG_W, FIG_H), sharey="row", height_ratios=[2.2, 1.3]
        )
        for (ax, bx), (filt, title) in zip(axs.T, PANELS):
            draw_column(ax, bx, table, filt, title, rng)
        axs[0, 0].set_ylabel("Wake OFF rate (Hz)")
        axs[0, 0].set_ylim(4e-5, 2.2)
        axs[0, 0].yaxis.set_major_locator(FixedLocator(YTICKS))
        axs[0, 0].yaxis.set_minor_locator(NullLocator())
        axs[0, 0].set_yticklabels(["0", "0.001", "0.01", "0.1", "1"])
        axs[1, 0].set_ylabel("Late − early (Hz)")
        axs[1, 0].set_ylim(-1.2, 1.2)
        axs[1, 0].yaxis.set_major_locator(FixedLocator(DTICKS))
        axs[1, 0].yaxis.set_minor_locator(NullLocator())
        axs[1, 0].set_yticklabels(["−1", "−0.1", "−0.01", "0", "0.01", "0.1", "1"])
        fig.subplots_adjust(
            left=0.12, right=0.975, bottom=0.09, top=0.95, wspace=0.12, hspace=0.32
        )
        # The figma canvas is 300/96 larger than print size, so 96 dpi gives a 300 dpi
        # image at print size.
        paths = out_dir / f"{STEM}.svg", out_dir / f"{STEM}.png"
        fig.savefig(paths[0])
        fig.savefig(
            paths[1],
            dpi=PNG_DPI / pp.scale(1.0),
            pil_kwargs={"dpi": (PNG_DPI, PNG_DPI)},
        )
    return paths


# -------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help=f"redraw from the cached {TABLE_NAME} instead of NFS",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / TABLE_NAME
    if args.plot_only:
        table = pd.read_parquet(table_path)
    else:
        table = build_table()
        table.to_parquet(table_path, index=False)
    report(table)
    for p in (table_path, *make_figure(table, args.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
