"""Diagnostic plots: banded unit-based OFF detections vs evaluation labels. PROVISIONAL.

Per (subject, probe, structure, config) and per short (<=5 s) window inside the labeled
evaluation chunks, render three vertically-stacked depth x time panels -- manual labels
(top) / banded detection (middle) / morphological (bottom) -- sharing one time axis, so the
algorithm's output can be *seen* against the ground truth, not just read off metric
tables. The footprints are exactly the rasters the scorer uses
(:mod:`cnpix_local_sleep.unit_based.banded_eval`): the banded panel is the faithful
union-of-band-boxes footprint, restricted to the structure's channel rows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: save PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from cnpix_local_sleep.evaluation import banded_vs_morphological  # noqa: E402
from cnpix_local_sleep.evaluation import rasterize  # noqa: E402
from cnpix_local_sleep.unit_based import banded_eval  # noqa: E402

def _load_stack_ap_image(subject, probe, condition):
    """Lazy AP/MUA stack image (n_chunks, n_channels, spc) -- the depth x time activity
    image the manual labels were drawn on, in the SAME row space as the labels
    (``stack_row = (n-1) - channel_index``). Returns ``None`` (with a note) if unavailable.
    """
    try:
        import zarr
        from cnpix_local_sleep.evaluation import config as ev_config
        from cnpix_local_sleep.stacks import files as stk_files

        zpath = stk_files.get_sam3_off_stacks_ome_zarr_path(
            subject, probe, ev_config.stack_condition(condition)
        )
        return zarr.open(str(zpath), mode="r")["0"]["0"]  # series 0 = AP image, uint8
    except Exception as exc:  # noqa: BLE001 - underlay is best-effort
        print(f"  (no AP stack image to underlay: {exc!r})")
        return None


def _structure_rows(y_coords, row_mask):
    """(r0, r1, depths) for the contiguous stack-row slice of the structure.

    ``depths`` is the per-row depth (um) of stack rows ``r0:r1``; it is *descending*
    (row r0 is the deepest channel, since ``stack_row = (n-1) - channel_index`` and
    ``y_coords`` is ascending), so the panels render deep-at-top.
    """
    rows = np.where(row_mask)[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    depths = y_coords[(len(y_coords) - 1) - np.arange(r0, r1)]
    return r0, r1, depths


def _pick_windows(presence, chunks, r0, r1, win, max_windows):
    """Up to ``max_windows`` ``(chunk, s0, s1)`` windows ranked by OFF presence.

    ``presence`` is a ``(n_chunks, n_rows, spc)`` boolean of "any label/detection OFF
    here"; windows with no presence in the structure rows are skipped, so we never emit
    an empty panel.
    """
    spc = presence.shape[2]
    cands = []
    for c in np.sort(np.asarray(chunks)):
        c = int(c)
        for s0 in range(0, spc, win):
            s1 = min(s0 + win, spc)
            score = int(presence[c, r0:r1, s0:s1].sum())
            if score > 0:
                cands.append((score, c, s0, s1))
    cands.sort(reverse=True)
    return [(c, s0, s1) for _, c, s0, s1 in cands[:max_windows]]


def _metric_suffix(metrics_row):
    if not metrics_row:
        return ""

    def g(k):
        v = metrics_row.get(k)
        return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "n/a"

    return (
        f"  |  F1={g('F1')} IoU={g('IoU')} "
        f"sens={g('sensitivity')} prec={g('precision')}"
    )


def plot_window(
    arrays, panels, c, s0, s1, r0, r1, depths, dt, *, title, t0_abs, save_path
):
    """Render one window as stacked depth x time panels; save and return the path.

    ``panels`` is a list of ``{label, key, cmap, kind}`` dicts; ``kind="mask"`` panels
    binarise the array (label/detection footprints), ``kind="image"`` panels show the raw
    grayscale activity image (percentile-clipped for contrast).
    """
    win_dur = (s1 - s0) * dt
    # imshow extent = [left, right, bottom, top]; depths[0] (deep) at top.
    extent = [0.0, win_dur, float(depths[-1]), float(depths[0])]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 2.3 * len(panels) + 0.6), sharex=True, squeeze=False,
    )
    for ax, p in zip(axes[:, 0], panels):
        sl = np.asarray(arrays[p["key"]][c, r0:r1, s0:s1])
        if p["kind"] == "mask":
            ax.imshow(
                (sl > 0).astype(float), aspect="auto", origin="upper", extent=extent,
                cmap=p["cmap"], vmin=0, vmax=1, interpolation="nearest",
            )
        else:  # grayscale activity image -> percentile-clip the window for contrast
            vals = sl[np.isfinite(sl)]
            if vals.size and float(vals.max()) > float(vals.min()):
                vmin, vmax = np.percentile(vals, [1, 99])
                if vmax <= vmin:
                    vmin, vmax = float(vals.min()), float(vals.max())
            else:
                vmin, vmax = 0.0, 1.0
            ax.imshow(
                sl, aspect="auto", origin="upper", extent=extent, cmap=p["cmap"],
                vmin=vmin, vmax=vmax, interpolation="nearest",
            )
        ax.set_ylabel(f"{p['label']}\ndepth (um)")
    axes[-1, 0].set_xlabel(f"time (s)  [window starts at t={t0_abs:.1f}s]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    return save_path


def do_structure_banded_plots(
    subject: str,
    probe: str,
    structure: str,
    off_frame,
    eval_name: str,
    *,
    depth_lo: float,
    depth_hi: float,
    config_name: str,
    off_df=None,
    all_bands_on_off_df=None,
    metrics_row: dict | None = None,
    filter_name: str = "clas",
    mua_raster=None,
    max_windows: int = 3,
    window_s: float = 5.0,
    out_dir,
    manual_version: str = "latest",
) -> list[str]:
    """Emit up to ``max_windows`` stacked overlay PNGs for one (structure, config).

    Builds the same rasters the scorer uses -- manual (``_eval_geometry``), banded
    (union-of-band-boxes when ``off_df`` + ``all_bands_on_off_df`` are given, else the
    Off-frame bbox), and morphological as its true per-pixel masks (``filter_name``,
    e.g. ``"clas"``; pass a precomputed ``mua_raster`` to avoid rebuilding it) -- plus the
    actual AP/MUA stack image (the depth x time activity the manual labels were drawn
    on) as a bottom panel, over the labeled eval chunks. Picks the windows with the most
    label/detection activity in the structure's rows (<= ``window_s`` each), and saves one
    ``*_w{i}.png`` per window.
    """
    cfg, condition, manual, ts_flat, y_coords, chunks = banded_eval.eval_geometry(
        subject, probe, eval_name, manual_version=manual_version
    )
    row_mask = banded_eval.structure_row_mask(y_coords, depth_lo, depth_hi)
    n_chunks = manual.shape[0]
    spc = len(ts_flat) // n_chunks
    dt = float(ts_flat[1] - ts_flat[0])
    win = min(spc, max(1, int(round(window_s / dt))))

    if off_df is not None and all_bands_on_off_df is not None:
        banded = banded_eval.rasterize_banded_union(
            off_df, all_bands_on_off_df, ts_flat, y_coords, manual.shape,
            eval_chunks=chunks,
        )
    else:
        banded = rasterize.rasterize_offs(
            off_frame, ts_flat, y_coords, manual.shape, eval_chunks=chunks
        )

    # morphological as its TRUE per-pixel masks (not bounding boxes), same raster the
    # scorer uses. Reuse a precomputed raster when given (it is heavy to build).
    if mua_raster is None:
        mua_raster = banded_vs_morphological.rasterize_morphological_masks(
            subject, probe, structure, condition, manual.shape, filter_name=filter_name
        )
    mua = mua_raster

    r0, r1, depths = _structure_rows(y_coords, row_mask)
    presence = (manual > 0) | (banded > 0)  # rank windows by label/detection activity
    windows = _pick_windows(presence, chunks, r0, r1, win, max_windows)

    arrays = {"manual": manual, "banded": banded, "mua": mua}
    panels = [
        {"label": "manual", "key": "manual", "cmap": "Greens", "kind": "mask"},
        {"label": "banded", "key": "banded", "cmap": "Reds", "kind": "mask"},
        {"label": f"morphological ({filter_name})", "key": "mua", "cmap": "Blues",
         "kind": "mask"},
    ]
    # The actual depth x time activity image (the substrate the labels were drawn on),
    # underneath the label panels.
    trace = _load_stack_ap_image(subject, probe, condition)
    if trace is not None:
        arrays["trace"] = trace
        panels.append({"label": "MUA image", "key": "trace", "cmap": "gray", "kind": "image"})

    suff = _metric_suffix(metrics_row)
    paths = []
    for i, (c, s0, s1) in enumerate(windows):
        t0_abs = float(ts_flat[c * spc + s0])
        title = (
            f"{subject}/{probe}/{structure}  |  {config_name}  |  chunk {c}{suff}"
        )
        save_path = str(
            Path(out_dir) / f"{subject}_{probe}_{structure}_{config_name}_w{i}.png"
        )
        paths.append(
            plot_window(
                arrays, panels, c, s0, s1, r0, r1, depths, dt,
                title=title, t0_abs=t0_abs, save_path=save_path,
            )
        )
    return paths
