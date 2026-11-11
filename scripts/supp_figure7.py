"""Draw Supplementary Figure 7 (fraction of MUA samples below a fixed threshold).

For each included subject-probe-structure and each channel, a threshold is set at
the nominal quantile q of the channel's MUA envelope in a reference condition (the
wake quantile for wake, the NREM quantile for NREM), and the fraction of samples
below it is measured in a target condition. Each pair contributes the median of
that fraction across channels divided by q. One violin per comparison; stars are
two-sided Wilcoxon signed-rank tests of the log ratio against 0, Holm-corrected
across the five comparisons. Comparisons are labeled with condition boxes in
Figma, so the x tick labels are blank.

The per-pair table needs NFS (MUA traces and statistical condition hypnograms)
and takes about 75 min with an 80 GB memory peak on tononi-2. It is written to
``fixed_threshold_occupancy.parquet`` in the output directory; ``--plot-only``
redraws from that file.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python ../cnpix-local-sleep/scripts/supp_figure7.py
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
import seaborn as sns
from matplotlib.ticker import FixedLocator, NullLocator
from PIL import Image
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cnpix_local_sleep import hyp
from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG, readers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "r-offp"))
import plot_results as pr
from plot_locality_results import _restyle_violin_axis

DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "supp_figure7"
TABLE = "fixed_threshold_occupancy.parquet"
STEM = "supp_figure7"
WIDTH, HEIGHT = 2.9, 2.4
PNG_DPI = 300

# (contrast, state, target, reference), left to right
CONTRASTS = [
    ("NOD.Incline", "Wake", "Late.NOD.Wake", "Early.NOD.Wake"),
    ("NREM.BSL.Decline", "NREM", "Early.BSL.NREM", "Early.REC.NREM.Match"),
    ("NREM.Surge", "NREM", "Early.REC.NREM", "Early.BSL.NREM"),
    ("NREM.Rebound", "NREM", "Early.REC.NREM", "Early.REC.NREM.Match"),
    ("NREM.REC.Decline", "NREM", "Early.REC.NREM", "Late.REC.NREM"),
]
ORDER = [c for c, *_ in CONTRASTS]
QUANTILE_COLUMN = {"Wake": "wake_quantile_threshold", "NREM": "nrem_quantile_threshold"}


def compute_table() -> pd.DataFrame:
    thresholds = SOURCE_CONFIG.load_quantile_thresholds()
    thresholds = thresholds[thresholds["include"]]
    rows = []
    for r in thresholds.itertuples():
        subject, probe, structure = r.subject, r.probe, r.structure_acronym
        da = readers.open_mua_traces_as_xarray(
            subject,
            probe,
            structure,
            ndimage_filter_type="median",
            ndimage_filter_kwargs={"size": [15, 1]},
        )
        t = da.time.values
        hgs = hyp.load_statistical_condition_hypnograms(subject, probe)
        values = {}
        for _, state, target, reference in CONTRASTS:
            for cond in (target, reference):
                if cond not in values:
                    mask = hgs[cond].keep_states([state]).covers_time(t)
                    values[cond] = da.isel(time=np.flatnonzero(mask)).values
        for contrast, state, target, reference in CONTRASTS:
            q = float(getattr(r, QUANTILE_COLUMN[state]))
            ref_threshold = np.quantile(values[reference], q, axis=0)
            below = (values[target] < ref_threshold).mean(axis=0)
            rows.append(
                {
                    "subject": subject,
                    "probe": probe,
                    "structure": structure.replace("mPPC", "PPC"),
                    "contrast": contrast,
                    "target": target,
                    "reference": reference,
                    "q": q,
                    "fold": np.median(below) / q,
                }
            )
        print("done", subject, probe, structure, flush=True)
        del da, values
    return pd.DataFrame(rows)


def holm_pvalues(table: pd.DataFrame) -> dict[str, float]:
    pvals = [
        stats.wilcoxon(np.log(table.loc[table["contrast"] == c, "fold"])).pvalue
        for c in ORDER
    ]
    return dict(zip(ORDER, multipletests(pvals, method="holm")[1]))


def report(table: pd.DataFrame, p_holm: dict[str, float]) -> None:
    n_rats = table["subject"].nunique()
    n_pairs = len(table[["subject", "probe", "structure"]].drop_duplicates())
    print(f"n = {n_pairs} pairs from {n_rats} rats")
    for c in ORDER:
        fold = table.loc[table["contrast"] == c, "fold"]
        print(
            f"{c}: median fold {fold.median():.2f}, "
            f"{(fold > 1).sum()}/{len(fold)} pairs > 1, Holm p {p_holm[c]:.3g}"
        )


def plot(table: pd.DataFrame, p_holm: dict[str, float], out_dir: Path) -> None:
    np.random.seed(0)  # stripplot jitter
    with pp.destination("figma"):
        plt.rcParams["figure.autolayout"] = False
        fig, ax = plt.subplots(figsize=pp.scale(WIDTH, HEIGHT))
        sns.violinplot(
            data=table,
            x="contrast",
            y="fold",
            order=ORDER,
            color="white",
            linecolor="k",
            linewidth=pp.scale(1.25) - 1.0,
            inner="quart",
            cut=0,
            log_scale=True,
            ax=ax,
        )
        sns.stripplot(
            data=table,
            x="contrast",
            y="fold",
            order=ORDER,
            color="k",
            size=pp.scale(1.6) - 1.0,
            ax=ax,
        )
        _restyle_violin_axis(ax)
        ax.axhline(1, color="0.5", lw=pp.scale(0.5), ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(range(len(ORDER)), [""] * len(ORDER))
        ax.set_xlabel(None)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        for i, c in enumerate(ORDER):
            stars = pr._pvalue_to_stars(p_holm[c])
            ax.text(
                i,
                4.6,
                stars or "n.s.",
                ha="center",
                va="baseline",
                fontsize=pp.scale(7) if stars else pp.scale(6),
                color="k" if stars else "0.5",
            )
        ax.set_ylabel("Fraction of samples below\nfixed threshold / nominal quantile")
        ax.set_ylim(0.04, 6.5)
        ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 4]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        fig.subplots_adjust(left=0.2, right=0.985, bottom=0.17, top=0.985)
        fig.savefig(out_dir / f"{STEM}.svg")
        png = out_dir / f"{STEM}.png"
        fig.savefig(png, dpi=PNG_DPI / pp.scale(1.0))
    Image.open(png).save(png, dpi=(PNG_DPI, PNG_DPI))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--plot-only", action="store_true", help=f"redraw from {TABLE} (no NFS)"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        table = pd.read_parquet(args.out_dir / TABLE)
    else:
        table = compute_table()
        table.to_parquet(args.out_dir / TABLE, index=False)
    p_holm = holm_pvalues(table)
    report(table, p_holm)
    plot(table, p_holm, args.out_dir)


if __name__ == "__main__":
    main()
