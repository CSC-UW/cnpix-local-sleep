"""Draw Supplementary Figure 2 (detection borders and anatomical borders on two probes).

For each example probe, three 1-s snippets of the MUA envelope on every channel during
Early.BSL.NREM, with the detection borders (solid lines, dark bars) and anatomical
borders (dashed lines, light bars) of each included structure. The snippets are centred
on the three lowest points of the mean envelope across detection channels (50-ms running
mean) within the first 10 min of NREM in the condition, at least 30 s apart. The grey
scale spans the 1st to 99th percentile of the envelope on the detection channels over
those 10 min, separately for each probe.

Needs NFS (MUA traces, structure tables and the statistical condition hypnograms);
there is no cache. Prints the structures and snippet times of each probe.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python ../cnpix-local-sleep/scripts/supp_figure2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import ecephys.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubplots as pp
from cnpix.mua import files as cnpix_mua_files
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from cnpix_local_sleep import channel_anatomy, files, hyp, trace_io
from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "supp_figure2"
STEM = "supp_figure2"

EXAMPLES = [("CNPIX4-Doppio", "imec1"), ("CNPIX8-Allan", "imec0")]
CONDITION, CONDITION_LABEL = "Early.BSL.NREM", "early baseline NREM"
N_SNIP, SNIP_SEC, WINDOW_SEC, MIN_SEP_SEC, SMOOTH_SEC = 3, 1.0, 600, 30, 0.05

WIDTH, HEIGHT = 7.09, 8.6  # inches at final print size
PNG_DPI = 300
PITCH = 20.0
C_BORDER = "#1f77b4"
COLS = [(0.55, 3.0), (4.0, 6.45)]
TOP, ROW_H, ROW_GAP = 0.62, 2.05, 0.5


def compute_probe(subject, probe, included) -> dict:
    da = trace_io.scale_to_uV(
        trace_io.open_si_zarr_recording_as_xarray(
            cnpix_mua_files.get_mua_traces_path(subject, probe)
        )
    )
    hg = hyp.load_statistical_condition_hypnograms(subject, probe)[CONDITION]
    hg = hg.keep_states(["NREM"])
    t = da.time.values
    fs = 1 / np.median(np.diff(t[:10000]))
    idx = np.flatnonzero(hg.covers_time(t))[: int(WINDOW_SEC * fs)]
    v = da.isel(time=idx).values.astype(np.float32)
    y = da.y.values
    order = np.argsort(y)
    v, y = v[:, order], y[order]

    anat = channel_anatomy.load_structures(subject, probe)
    det = ecephys.utils.read_htsv(
        files.get_off_detection_structure_borders_path(subject, probe)
    ).set_index("acronym")
    borders = [
        {
            "acronym": a.acronym,
            "anat_lo": a.lo,
            "anat_hi": a.hi,
            "det_lo": det.loc[a.acronym].lo,
            "det_hi": det.loc[a.acronym].hi,
        }
        for a in anat.itertuples()
        if (subject, probe, a.acronym) in included
    ]
    ret = np.zeros(len(y), bool)
    for b in borders:
        ret |= (y >= b["det_lo"]) & (y <= b["det_hi"])

    m = (
        pd.Series(v[:, ret].mean(1))
        .rolling(int(SMOOTH_SEC * fs), center=True, min_periods=1)
        .mean()
        .values
    )
    n = int(SNIP_SEC * fs)
    half = n // 2
    centres = []
    for i in np.argsort(m):
        if i < half or i + half >= len(m):
            continue
        if all(abs(i - c) > MIN_SEP_SEC * fs for c in centres):
            centres.append(i)
        if len(centres) == N_SNIP:
            break
    starts = [c - half for c in sorted(centres)]
    assert all(np.diff(t[idx[i : i + n]]).max() < 1.5 / fs for i in starts), (
        "a snippet spans a gap between NREM bouts"
    )
    return {
        "subject": subject,
        "probe": probe,
        "y": y,
        "borders": borders,
        "fs": fs,
        "vlim": np.percentile(v[:, ret], [1, 99]),
        "snippets": [v[i : i + n] for i in starts],
        "t_into_condition": [t[idx[i]] - t[idx[0]] for i in starts],
    }


def compute() -> list[dict]:
    thr = SOURCE_CONFIG.load_quantile_thresholds()
    thr = thr[thr["include"]]
    included = set(zip(thr.subject, thr.probe, thr.structure_acronym))
    alias = {s: f"Subject{i}" for i, s in enumerate(sorted(set(thr.subject)), 1)}
    probes = []
    for subject, probe in EXAMPLES:
        P = compute_probe(subject, probe, included)
        P["alias"] = alias[subject]
        print(
            P["alias"],
            subject,
            probe,
            [b["acronym"] for b in P["borders"]],
            [round(x) for x in P["t_into_condition"]],
            flush=True,
        )
        probes.append(P)
    return probes


def frac(x0, x1, y0, h):
    """Axes rectangle from final-size inches measured from the figure's top-left."""
    return [x0 / WIDTH, 1 - (y0 + h) / HEIGHT, (x1 - x0) / WIDTH, h / HEIGHT]


def make_figure(probes, out_dir: Path):
    """Draw and save (SVG + PNG). Saving must happen inside the pubplots context, which
    supplies the fonts and ``svg.fonttype``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with pp.destination("figma"), plt.rc_context({"figure.autolayout": False}):
        fig = plt.figure(figsize=pp.scale(WIDTH, HEIGHT))
        lw_thin, lw = pp.scale(0.5), pp.scale(0.8)
        for j, (P, (x0, x1)) in enumerate(zip(probes, COLS)):
            y = P["y"]
            ylim = (y[0] - PITCH / 2, y[-1] + PITCH / 2)
            norm = Normalize(*P["vlim"])
            label_y = 1 - (TOP - 0.3) / HEIGHT
            fig.text(
                (x0 + x1) / 2 / WIDTH,
                label_y,
                P["alias"],
                ha="center",
                va="bottom",
                fontsize=pp.scale(7),
            )
            fig.text(
                (x0 - COLS[0][0] + 0.05) / WIDTH,
                label_y,
                "ab"[j],
                fontweight="bold",
                fontsize=pp.scale(7),
                va="bottom",
                ha="left",
            )
            for r, (snip, t0) in enumerate(zip(P["snippets"], P["t_into_condition"])):
                ax = fig.add_axes(frac(x0, x1, TOP + r * (ROW_H + ROW_GAP), ROW_H))
                ax.imshow(
                    snip.T,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    cmap="gist_yarg",
                    norm=norm,
                    extent=[0, snip.shape[0] / P["fs"], *ylim],
                )
                for b in P["borders"]:
                    for v in (b["det_lo"], b["det_hi"]):
                        ax.axhline(v, color=C_BORDER, lw=lw_thin)
                    for v in (b["anat_lo"], b["anat_hi"]):
                        ax.axhline(v, color=C_BORDER, lw=lw_thin, ls=(0, (4, 3)))
                ax.set_xlim(0, SNIP_SEC)
                ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
                ax.set_ylim(ylim)
                ax.set_yticks(np.arange(0, y[-1] + 1, 2000))
                ax.set_xlabel("Time (s)", labelpad=pp.scale(2))
                ax.set_ylabel("Position along probe (µm)")
                ax.set_title(f"{t0:.0f} s into {CONDITION_LABEL}", loc="right")
                bx = ax.inset_axes([1.02, 0, 0.12, 1], sharey=ax)
                bx.set_axis_off()
                for b in P["borders"]:
                    bx.add_patch(
                        plt.Rectangle(
                            (0, b["anat_lo"]),
                            0.35,
                            b["anat_hi"] - b["anat_lo"],
                            color=C_BORDER,
                            alpha=0.3,
                            lw=0,
                        )
                    )
                    bx.add_patch(
                        plt.Rectangle(
                            (0.45, b["det_lo"]),
                            0.35,
                            b["det_hi"] - b["det_lo"],
                            color=C_BORDER,
                            lw=0,
                        )
                    )
                    bx.text(
                        1.0,
                        (b["anat_lo"] + b["anat_hi"]) / 2,
                        b["acronym"],
                        va="center",
                        ha="left",
                        fontsize=pp.scale(6),
                    )
                bx.set_xlim(0, 1)
                for spine in ax.spines.values():
                    spine.set_linewidth(lw_thin)
            cbar_top = TOP + N_SNIP * ROW_H + (N_SNIP - 1) * ROW_GAP + 0.5
            cax = fig.add_axes(frac(x0 + 0.6, x1 - 0.6, cbar_top, 0.07))
            cb = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap="gist_yarg"),
                cax=cax,
                orientation="horizontal",
            )
            cb.set_label("MUA envelope (µV)", labelpad=pp.scale(2))
            cb.outline.set_linewidth(lw_thin)

        handles = [
            (
                Line2D([], [], color=C_BORDER, lw=lw),
                plt.Rectangle((0, 0), 1, 1, color=C_BORDER, lw=0),
            ),
            (
                Line2D([], [], color=C_BORDER, lw=lw, ls=(0, (4, 3))),
                plt.Rectangle((0, 0), 1, 1, color=C_BORDER, alpha=0.3, lw=0),
            ),
        ]
        fig.legend(
            handles,
            ["detection borders and extent", "anatomical borders and extent"],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1 - 0.04 / HEIGHT),
            borderaxespad=0,
            columnspacing=2.5,
            handlelength=3.5,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        )
        paths = out_dir / f"{STEM}.svg", out_dir / f"{STEM}.png"
        fig.savefig(paths[0])
        fig.savefig(
            paths[1],
            dpi=PNG_DPI / pp.scale(1.0),
            pil_kwargs={"dpi": (PNG_DPI, PNG_DPI)},
        )
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    for p in make_figure(compute(), args.out_dir):
        print(p)


if __name__ == "__main__":
    main()
