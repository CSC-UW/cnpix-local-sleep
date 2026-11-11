"""Draw Supplementary Figure 1 (morphological OFF detection, step by step).

One 6-s segment from the middle of the longest Early.REC.NREM bout of CNPIX8-Allan
imec0 M2, taken through the production detection:

a  Smoothed MUA envelope of the detection channels.
b  One channel against its NREM threshold, and the NREM samples of that channel in the
   threshold block that contains the segment.
c  Binary image of subthreshold samples, each channel at its own block threshold.
d  The cleaning operations of ``morphology.clean_binary_mask``, one per panel.
e  Connected components of the cleaned image, classified by the LLAS/CLAS filters.
f  The same components by median duration and span.
g  Admitted OFF periods outlined on the envelope.

Needs NFS (MUA traces, the NREM channel-thresholds zarr and the statistical condition
hypnograms). Prints the numbers quoted in the caption.

Usage (from gfys_workspace)::

    uv run --all-extras --group dev python ../cnpix-local-sleep/scripts/supp_figure1.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pubplots as pp
import scipy.ndimage
import xarray as xr
from matplotlib.colors import ListedColormap, PowerNorm, to_hex
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from cnpix_local_sleep import hyp, off_tables, plots
from cnpix_local_sleep.morphological import morphology
from cnpix_local_sleep.morphological.detection_opts import (
    get_mua_spatial_detection_opts,
)
from cnpix_local_sleep.morphological.mua import SOURCE_CONFIG, readers
from cnpix_local_sleep.morphological.mua import files as mua_files

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "notebooks" / "figures" / "outputs" / "supp_figure1"
STEM = "supp_figure1"

SUBJECT, PROBE, STRUCTURE = "CNPIX8-Allan", "imec0", "M2"
CONDITION = "Early.REC.NREM"
HALF_WINDOW, MARGIN = 3.0, 1.0  # s; the margin is labelled but not drawn
ZOOM = (1.4, 1.95)  # s, panel d

WIDTH, HEIGHT = 7.09, 8.85  # inches at final print size
PNG_DPI = 300
PITCH = 20.0
L, R_MAIN, R_SIDE, R = 0.55, 5.45, 5.75, 7.0
ROWS = {"a": 0.95, "b": 0.72, "c": 0.95, "d": 0.9, "e": 0.95, "g": 0.95}
GAPS = {"a": 0.3, "b": 0.54, "c": 0.5, "d": 0.66, "e": 0.54, "g": 0.5}

_CAT = plots.get_category_palette()
C_REJ, C_SMALL, C_LARGE = "#c4c4c4", _CAT["LLAS"], _CAT["CLAS"]
C_THR, C_LINK = "#d62728", "#1f77b4"
C_ADD, C_DEL = "#2c7fb8", "#f4a582"
C_MASK = "0.15"
REJECTED, SMALL, MEDLARGE = 1, 2, 3


# --------------------------------------------------------------------------- data


def cleaning_steps(opts) -> list[tuple[str, int, int]]:
    """(operation, structuring-element length, axis) in ``clean_binary_mask`` order."""
    return [
        ("closing", opts["n_channels_clean"], 1),
        ("opening", opts["n_samples_clean"], 0),
        ("opening", opts["n_channels_clean"], 1),
        ("closing", opts["n_channels_connect"], 1),
    ]


def admission_class(llas, clas):
    return np.where(clas, MEDLARGE, np.where(llas, SMALL, REJECTED))


def passes_without(props, filter_name, dropped):
    """``off_filter_mask`` with one criterion left out."""
    crit = off_tables.NAMED_FILTERS[filter_name]
    return np.logical_and.reduce(
        [props[c].between(*lim).to_numpy() for c, lim in crit.items() if c != dropped]
    )


def compute() -> dict:
    opts = get_mua_spatial_detection_opts()
    q_tab = SOURCE_CONFIG.load_quantile_thresholds().set_index(
        ["subject", "probe", "structure_acronym"]
    )
    q = float(q_tab.loc[(SUBJECT, PROBE, STRUCTURE), "nrem_quantile_threshold"])

    da = readers.open_mua_traces_as_xarray(
        SUBJECT,
        PROBE,
        STRUCTURE,
        ndimage_filter_type=opts["ndimage_filter_type"],
        ndimage_filter_kwargs=opts["ndimage_filter_kwargs"],
    )
    t, fs = da.time.values, da.attrs["fs"]
    hgs = hyp.load_statistical_condition_hypnograms(SUBJECT, PROBE)
    bout = hgs[CONDITION].keep_states(["NREM"]).sort_values("duration").iloc[-1]
    c = (bout.start_time + bout.end_time) / 2
    i0 = np.searchsorted(t, c - HALF_WINDOW - MARGIN)
    i1 = np.searchsorted(t, c + HALF_WINDOW + MARGIN)
    snip = da.isel(time=slice(i0, i1)).compute().sortby("y")
    vals, ts, y = snip.values, snip.time.values, snip.y.values

    thr = xr.open_dataarray(
        mua_files.get_full_channel_thresholds_path(SUBJECT, PROBE, STRUCTURE, "nrem"),
        engine="zarr",
    ).load()
    bb = np.asarray(thr.attrs["bin_boundaries"])
    b = np.searchsorted(bb, i0, side="right") - 1
    assert bb[b] <= i0 and i1 <= bb[b + 1], "snippet straddles a threshold block"
    thr_vec = thr.isel(bin=b).sortby("y").values

    steps = [vals < thr_vec[None, :]]
    for op, n, ax in cleaning_steps(opts):
        struct = np.ones((n, 1)) if ax == 0 else np.ones((1, n))
        steps.append(
            getattr(scipy.ndimage, f"binary_{op}")(
                steps[-1], structure=struct, iterations=1
            )
        )
    ref = morphology.clean_binary_mask(
        steps[0],
        n_samples_connect=opts["n_samples_connect"],
        n_samples_clean=opts["n_samples_clean"],
        n_channels_clean=opts["n_channels_clean"],
        n_channels_connect=opts["n_channels_connect"],
    )
    assert np.array_equal(ref, steps[-1]), (
        "stepwise cleaning diverges from clean_binary_mask"
    )

    lbl, nl = scipy.ndimage.label(steps[-1])
    lbl_ixs = scipy.ndimage.value_indices(lbl, ignore_value=0)
    props = morphology.get_off_properties(y, ts, fs, lbl_ixs, values=vals)
    labels = np.array(list(lbl_ixs.keys()))
    llas = off_tables.off_filter_mask(props, "llas").to_numpy()
    clas = off_tables.off_filter_mask(props, "clas").to_numpy()
    cls = np.zeros(nl + 1, int)
    cls[labels] = admission_class(llas, clas)
    cls_no_duration = admission_class(
        passes_without(props, "llas", "duration"),
        passes_without(props, "clas", "duration"),
    )

    ch = len(y) // 2
    blk_slice = slice(int(bb[b]), int(bb[b + 1]))
    blk = da.isel(time=blk_slice).sortby("y").isel(channel=ch).compute().values
    st_lbl = hgs["Full.Conservative"].get_states(t[blk_slice])

    return {
        "vals": vals,
        "ts": ts - (c - HALF_WINDOW),
        "y": y,
        "fs": fs,
        "opts": opts,
        "thr_vec": thr_vec,
        "steps": np.stack(steps),
        "lbl": lbl,
        "cls": cls,
        "cls_no_duration": cls_no_duration,
        "labels": labels,
        "span": props["span"].to_numpy(),
        "median_duration": props["median_duration"].to_numpy(),
        "ch": ch,
        "blk_nrem": blk[st_lbl == "NREM"],
        "q": q,
        "block_min": (bb[b + 1] - bb[b]) / fs / 60,
    }


def visible_components(D) -> np.ndarray:
    """Components with at least one pixel inside the drawn segment."""
    sel = (D["ts"] >= 0) & (D["ts"] <= 2 * HALF_WINDOW)
    return np.isin(D["labels"], np.unique(D["lbl"][sel]))


def print_caption_numbers(D) -> None:
    y, fs, opts = D["y"], D["fs"], D["opts"]
    pitch = np.unique(np.diff(y))
    q, thr, blk = D["q"], float(D["thr_vec"][D["ch"]]), D["blk_nrem"]
    print(f"channels: {len(y)}; pitch: {', '.join(f'{p:g}' for p in pitch)} µm")
    print(f"block: {D['block_min']:.2f} min; NREM samples in block: {blk.size}")
    print(
        f"q = {q:g}; threshold {thr:.4f} µV; Q{q * 100:g} of the block's NREM samples "
        f"{np.quantile(blk, q):.4f} µV; fraction below threshold "
        f"{(blk < thr).mean():.4f}"
    )
    for i, (op, n, ax) in enumerate(cleaning_steps(opts), start=1):
        size = f"{n / fs * 1e3:g} ms" if ax == 0 else f"{n * pitch[0]:g} µm"
        print(
            f"cleaning {i}: {'temporal' if ax == 0 else 'spatial'} {op}, {n} ({size})"
        )
    names = {REJECTED: "rejected", SMALL: "small", MEDLARGE: "medium/large"}
    cls = D["cls"][D["labels"]]
    vis = visible_components(D)
    for scope, m in (("all", np.ones_like(vis)), ("visible", vis)):
        counts = ", ".join(f"{names[k]} {(cls[m] == k).sum()}" for k in names)
        print(f"components ({scope}, {m.sum()}): {counts}")
    changed = cls != D["cls_no_duration"]
    durations = " / ".join(
        f"{crit['duration'][0] * 1e3:g}"
        for crit in (off_tables.llas_filters, off_tables.clas_filters)
    )
    print(
        f"total-duration rule (>= {durations} ms) changes the class of "
        f"{changed.sum()} components ({(changed & vis).sum()} visible)"
    )


# ------------------------------------------------------------------------- layout


def _row_tops():
    top_of, y_cur = {}, 0.0
    for k, h in ROWS.items():
        y_cur += GAPS[k]
        top_of[k] = y_cur
        y_cur += h
    assert y_cur + 0.35 <= HEIGHT, y_cur
    return top_of


TOP_OF = _row_tops()


def frac(x0, x1, y0, h):
    """Axes rectangle from final-size inches measured from the figure's top-left."""
    return [x0 / WIDTH, 1 - (y0 + h) / HEIGHT, (x1 - x0) / WIDTH, h / HEIGHT]


def panel_label(fig, s, row, x=0.05):
    fig.text(
        x / WIDTH,
        1 - (TOP_OF[row] - 0.13) / HEIGHT,
        s,
        fontweight="bold",
        fontsize=pp.scale(7),
        va="bottom",
        ha="left",
    )


def depth_axis(ax, label=True):
    ax.set_yticks([0, 400, 800])
    if label:
        ax.set_ylabel("Depth (µm)")


def time_axis(ax):
    ax.set_xlim(0, 2 * HALF_WINDOW)
    ax.set_xticks(np.arange(0, 2 * HALF_WINDOW + 0.01, 1))
    ax.set_xlabel("Time (s)", labelpad=pp.scale(2))


def make_figure(D, out_dir: Path):
    """Draw and save (SVG + PNG). Saving must happen inside the pubplots context, which
    supplies the fonts and ``svg.fonttype``."""
    ts, y, vals = D["ts"], D["y"], D["vals"]
    ch, q = D["ch"], D["q"]
    sel = (ts >= 0) & (ts <= 2 * HALF_WINDOW)
    tt = ts[sel]
    depth = y.max() - y
    col_order = np.argsort(depth)
    dmax = depth.max()
    env_norm = PowerNorm(gamma=2.0, vmin=0, vmax=np.quantile(vals[sel], 0.95))
    cls_img = D["cls"][D["lbl"]]
    vis = visible_components(D)
    vis_cls = D["cls"][D["labels"][vis]]
    n_by = {c: int((vis_cls == c).sum()) for c in (REJECTED, SMALL, MEDLARGE)}

    def as_img(a, rows=sel):
        return a[rows][:, col_order].T

    def show(ax, img, t=tt, **kw):
        dt = t[1] - t[0]
        ax.imshow(
            img,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=[t[0] - dt / 2, t[-1] + dt / 2, dmax + PITCH / 2, -PITCH / 2],
            **kw,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with pp.destination("figma"), plt.rc_context({"figure.autolayout": False}):
        fig = plt.figure(figsize=pp.scale(WIDTH, HEIGHT))
        lw_thin, lw = pp.scale(0.5), pp.scale(0.75)

        # a
        ax = fig.add_axes(frac(L, R_MAIN, TOP_OF["a"], ROWS["a"]))
        show(ax, as_img(vals), cmap="gist_yarg", norm=env_norm)
        ax.axhline(depth[ch], color=C_LINK, lw=lw_thin, ls=(0, (4, 3)))
        depth_axis(ax)
        time_axis(ax)
        ax.set_title(
            f"Smoothed MUA envelope, {STRUCTURE} detection channels", loc="left"
        )
        cax = fig.add_axes(
            frac(R_SIDE, R_SIDE + 0.07, TOP_OF["a"] + 0.15, ROWS["a"] - 0.3)
        )
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=env_norm, cmap="gist_yarg"),
            cax=cax,
            ticks=[5, 10, 15],
        )
        cb.set_label("µV")
        cb.outline.set_linewidth(lw_thin)
        panel_label(fig, "a", "a")

        # b
        ax_b = fig.add_axes(frac(L, R_MAIN, TOP_OF["b"], ROWS["b"]))
        tr, thr = vals[sel, ch], float(D["thr_vec"][ch])
        ymax = np.quantile(tr, 0.999) * 1.08
        ax_b.fill_between(
            tt, 0, ymax, where=tr < thr, color=C_THR, alpha=0.18, lw=0, step="mid"
        )
        ax_b.plot(tt, tr, color="k", lw=lw_thin)
        ax_b.axhline(thr, color=C_THR, lw=lw)
        ax_b.set_ylim(0, ymax)
        ax_b.set_ylabel("µV")
        time_axis(ax_b)
        ax_b.set_title("One channel (dashed line in a) and its threshold", loc="left")
        ax_h = fig.add_axes(frac(R_SIDE, R, TOP_OF["b"], ROWS["b"]), sharey=ax_b)
        bins = np.linspace(0, ymax, 70)
        blk = D["blk_nrem"]
        ax_h.hist(
            blk[blk >= thr], bins=bins, orientation="horizontal", color="0.6", lw=0
        )
        ax_h.hist(
            blk[blk < thr],
            bins=bins,
            orientation="horizontal",
            color=C_THR,
            alpha=0.5,
            lw=0,
        )
        ax_h.axhline(thr, color=C_THR, lw=lw)
        ax_h.xaxis.set_major_locator(NullLocator())
        ax_h.tick_params(labelleft=False)
        for s in ("top", "right", "bottom"):
            ax_h.spines[s].set_visible(False)
        ax_h.set_title(
            f"NREM samples of this channel,\n{D['block_min']:.1f}-min block", loc="left"
        )
        ax_h.text(
            0.98,
            thr,
            f"Q{q * 100:g}",
            color=C_THR,
            ha="right",
            va="bottom",
            transform=ax_h.get_yaxis_transform(),
            fontsize=pp.scale(5),
        )
        panel_label(fig, "b", "b")

        # c
        ax = fig.add_axes(frac(L, R_MAIN, TOP_OF["c"], ROWS["c"]))
        show(
            ax,
            as_img(D["steps"][0]),
            cmap=ListedColormap(["white", C_MASK]),
            vmin=0,
            vmax=1,
        )
        ax.add_patch(
            Rectangle(
                (ZOOM[0], -PITCH / 2),
                ZOOM[1] - ZOOM[0],
                dmax + PITCH,
                fill=False,
                ec=C_LINK,
                lw=lw,
                ls=(0, (3, 2)),
                zorder=5,
                clip_on=False,
            )
        )
        depth_axis(ax)
        time_axis(ax)
        ax.set_title(
            "Subthreshold samples (each channel thresholded at its own block quantile)",
            loc="left",
        )
        panel_label(fig, "c", "c")

        # d
        zsel = (ts >= ZOOM[0]) & (ts <= ZOOM[1])
        titles = ["Before cleaning (box in c)"]
        for i, (op, n, axis) in enumerate(cleaning_steps(D["opts"]), start=1):
            size = f"{n / D['fs'] * 1e3:g} ms" if axis == 0 else f"{n} ch"
            kind = "Temporal" if axis == 0 else "Spatial"
            titles.append(f"{i}. {kind} {op}, {size}")
        gap = 0.14
        pw = (R - L - gap * (len(titles) - 1)) / len(titles)
        zticks = np.arange(np.ceil(ZOOM[0] * 5) / 5, ZOOM[1] - 0.05, 0.2)
        steps = D["steps"]
        for i, title in enumerate(titles):
            x0 = L + i * (pw + gap)
            ax = fig.add_axes(frac(x0, x0 + pw, TOP_OF["d"], ROWS["d"]))
            cur = as_img(steps[i], zsel).astype(int)
            if i > 0:
                prev = as_img(steps[i - 1], zsel)
                cur = np.where(
                    cur.astype(bool) & ~prev,
                    2,
                    np.where(~cur.astype(bool) & prev, 3, cur),
                )
            show(
                ax,
                cur,
                t=ts[zsel],
                cmap=ListedColormap(["white", C_MASK, C_ADD, C_DEL]),
                vmin=0,
                vmax=3,
            )
            ax.set_xlim(ZOOM)
            ax.set_xticks(zticks)
            ax.set_xticklabels([f"{v:.1f}" for v in zticks])
            depth_axis(ax, label=i == 0)
            ax.set_xlabel("Time (s)", labelpad=pp.scale(2))
            if i > 0:
                ax.tick_params(labelleft=False)
            for s in ax.spines.values():
                s.set_edgecolor(C_LINK)
                s.set_linestyle((0, (3, 2)))
            ax.set_title(title, loc="left")
        fig.legend(
            [Rectangle((0, 0), 1, 1, color=c) for c in (C_ADD, C_DEL)],
            ["added by this step", "removed by this step"],
            loc="lower right",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(R / WIDTH, 1 - (TOP_OF["d"] - 0.2) / HEIGHT),
            borderaxespad=0,
            borderpad=0,
            handlelength=0.9,
            handleheight=0.9,
            columnspacing=1.2,
            handletextpad=0.4,
        )
        panel_label(fig, "d", "d")

        # e
        ax = fig.add_axes(frac(L, R_MAIN, TOP_OF["e"], ROWS["e"]))
        show(
            ax,
            as_img(cls_img),
            cmap=ListedColormap(["white", C_REJ, C_SMALL, C_LARGE]),
            vmin=0,
            vmax=3,
        )
        depth_axis(ax)
        time_axis(ax)
        ax.set_title(
            "Connected components of the cleaned image, classified by size", loc="left"
        )
        panel_label(fig, "e", "e")

        # f: criterion regions from the LLAS/CLAS filters (total duration not drawn)
        sc_top, sc_h = TOP_OF["e"] + 0.05, 1.45
        ax = fig.add_axes(frac(R_SIDE + 0.3, R, sc_top, sc_h))
        XL, YL = (15, 300), (40, 1300)
        md_s = off_tables.llas_filters["median_duration"][0] * 1e3
        md_l = off_tables.clas_filters["median_duration"][0] * 1e3
        span_s = off_tables.llas_filters["span"][0]
        span_l = off_tables.clas_filters["span"][0]
        ax.add_patch(
            Rectangle(
                (md_s, span_s), md_l - md_s, YL[1], color=C_SMALL, alpha=0.25, lw=0
            )
        )
        ax.add_patch(
            Rectangle(
                (md_l, span_s), XL[1], span_l - span_s, color=C_SMALL, alpha=0.25, lw=0
            )
        )
        ax.add_patch(
            Rectangle((md_l, span_l), XL[1], YL[1], color=C_LARGE, alpha=0.25, lw=0)
        )
        colors = [to_hex((C_REJ, C_SMALL, C_LARGE)[c - 1]) for c in vis_cls]
        ax.scatter(
            D["median_duration"][vis] * 1e3,
            D["span"][vis],
            c=colors,
            s=pp.scale(12),
            ec="k",
            lw=lw_thin,
            zorder=3,
            clip_on=False,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(XL)
        ax.set_ylim(YL)
        ax.set_xticks([25, 50, 100, 200], ["25", "50", "100", "200"])
        ax.set_yticks([100, 200, 400, 800], ["100", "200", "400", "800"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Median duration (ms)")
        ax.set_ylabel("Span (µm)")
        ax.set_title("Admission", loc="left")
        handles = [
            plt.Line2D([], [], ls="", marker="s", mfc=c, mec="none", ms=pp.scale(4.5))
            for c in (C_LARGE, C_SMALL, C_REJ)
        ]
        fig.legend(
            handles,
            [
                f"medium/large ({n_by[MEDLARGE]})",
                f"small ({n_by[SMALL]})",
                f"rejected ({n_by[REJECTED]})",
            ],
            loc="upper left",
            bbox_to_anchor=(R_SIDE / WIDTH, 1 - (sc_top + sc_h + 0.42) / HEIGHT),
            frameon=False,
            borderaxespad=0,
            borderpad=0,
            handletextpad=0.3,
            labelspacing=0.4,
        )
        panel_label(fig, "f", "e", x=R_SIDE - 0.05)

        # g
        ax = fig.add_axes(frac(L, R_MAIN, TOP_OF["g"], ROWS["g"]))
        show(ax, as_img(vals), cmap="gist_yarg", norm=env_norm)
        ax.contour(
            tt,
            depth[col_order],
            as_img(cls_img >= SMALL).astype(float),
            levels=[0.5],
            colors=["orange"],
            linewidths=lw,
        )
        ax.set_ylim(dmax + PITCH / 2, -PITCH / 2)
        depth_axis(ax)
        time_axis(ax)
        ax.set_title(
            "Admitted OFF periods (small and medium/large) outlined on the envelope",
            loc="left",
        )
        panel_label(fig, "g", "g")

        for a in fig.axes:
            for s in a.spines.values():
                s.set_linewidth(lw_thin)
        # The figma canvas is 300/96 larger than print size, so 96 dpi gives a 300 dpi
        # PNG at print size. The SVG keeps pubplots' savefig.dpi for its raster images.
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
    D = compute()
    print_caption_numbers(D)
    for p in make_figure(D, args.out_dir):
        print(p)


if __name__ == "__main__":
    main()
