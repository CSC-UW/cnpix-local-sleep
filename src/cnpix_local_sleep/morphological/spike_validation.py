"""Sorted spikes inside morphological OFF periods, against the same footprints at random times.

Per structure, the structure's sorted units (any quality) are placed on the stack row
nearest their depth and their spikes counted per stack pixel over the scored images and
detection rows of an evaluation. Each relaxed (LLAS) OFF's spikes are counted over its true
pixel footprint, and over the same footprint (same rows, same per-row samples) moved to
:data:`K_SHIFTS` random positions within the scored images. An OFF is matched when any of
its pixels overlaps a manual label on the detection rows, and small when it meets the relaxed
but not the stringent (CLAS) criterion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage

from cnpix_local_sleep import const
from cnpix_local_sleep.evaluation import banded_vs_morphological, head_to_head
from cnpix_local_sleep.unit_based import banded, banded_eval

K_SHIFTS = 10
SEED = 20261109


def _spike_counts(trains, unit_rows, ts, chunks, rows):
    """``(n_rows_scored, n_chunks_scored * spc)`` spike counts per pixel ``[t_k, t_k + dt)``."""
    dt = float(np.median(np.diff(ts, axis=1)))
    flat = ts[chunks].ravel()
    assert np.all(np.diff(flat) > 0), "scored stack times must ascend"
    out = np.zeros((len(rows), flat.size), np.uint32)
    pos = {int(r): i for i, r in enumerate(rows)}
    for train, r in zip(trains, unit_rows):
        i = pos.get(int(r))
        if i is None or not len(train):
            continue
        s = train[(train >= flat[0]) & (train < flat[-1] + dt)]
        k = np.searchsorted(flat, s, side="right") - 1
        ok = (k >= 0) & (s - flat[k] < dt)
        out[i] += np.bincount(k[ok], minlength=flat.size).astype(np.uint32)
    return out


def _footprints(sub):
    """``{label: (chunk_i, row_i, sample)}`` for every instance in ``sub``, by label."""
    c, r, s = np.nonzero(sub)
    if not len(c):
        return {}
    lab = sub[c, r, s]
    order = np.argsort(lab, kind="stable")
    lab, c, r, s = lab[order], c[order], r[order], s[order]
    bounds = np.flatnonzero(np.diff(lab)) + 1
    return {
        int(L[0]): (ci, ri, si)
        for L, ci, ri, si in zip(
            np.split(lab, bounds), np.split(c, bounds), np.split(r, bounds), np.split(s, bounds)
        )
    }


def relaxed_off_spikes(subject: str, probe: str, structure: str, eval_name: str = "NREM") -> pd.DataFrame:
    """One row per relaxed OFF on the scored images: matched, small, observed spikes and
    the mean over the shifted footprints (``exp_mean``)."""
    bundle = banded.load_structure_inputs(subject, probe, structure)
    if bundle is None:
        return pd.DataFrame()
    _, condition, manual, ts_flat, y_coords, chunks = banded_eval.eval_geometry(
        subject, probe, eval_name
    )
    chunks = np.sort(chunks)
    ts = ts_flat.reshape(manual.shape[0], -1)
    spc = ts.shape[1]
    det_mask, _ = head_to_head._mua_detection_row_masks(
        subject, probe, condition, manual.shape[1], structure
    )
    rows = np.flatnonzero(det_mask)
    row_depth = y_coords[::-1]
    unit_rows = np.abs(row_depth[None, :] - np.asarray(bundle["depths"])[:, None]).argmin(1)
    counts = _spike_counts(bundle["trains"], unit_rows, ts, chunks, rows)
    ix = np.ix_(chunks, rows)
    relaxed, stringent = (
        banded_vs_morphological.rasterize_morphological_masks(
            subject, probe, structure, condition, manual.shape, filter_name=f
        )[ix]
        for f in ("llas", "clas")
    )
    labs = np.unique(relaxed)
    labs = labs[labs > 0]
    if not len(labs):
        return pd.DataFrame()
    hit = np.atleast_1d(scipy.ndimage.maximum(manual[ix] > 0, labels=relaxed, index=labs))
    matched = dict(zip(labs.tolist(), (hit > 0).tolist()))
    stringent_labels = set(np.unique(stringent).tolist()) - {0}
    rng = np.random.default_rng(SEED)
    n_flat = counts.shape[1]
    out = []
    for label, (ci, ri, si) in _footprints(relaxed).items():
        f = ci * spc + si
        off = f - f.min()
        span = int(off.max())
        obs = int(counts[ri, f].sum())
        if n_flat - span > 0:
            starts = rng.choice(np.arange(n_flat - span), size=K_SHIFTS, replace=True)
            exp_mean = float(np.mean([counts[ri, st + off].sum() for st in starts]))
        else:
            exp_mean = np.nan
        out.append({
            "subject": subject, "probe": probe, "structure": structure, "eval": eval_name,
            "label": label, "matched": matched[label], "small": label not in stringent_labels,
            "obs": obs, "exp_mean": exp_mean,
        })
    return pd.DataFrame(out)


def results_path(eval_name: str = "NREM") -> Path:
    """Where ``manual_annotation_sets.ipynb`` saves the :func:`relaxed_off_spikes` rows on NFS."""
    import wisc_ecephys_tools as wet

    return (
        wet.get_sglx_project("offproj").get_experiment_directory(const.EXPERIMENT)
        / f"morphological_off_spikes_full48h_{eval_name}.parquet"
    )
