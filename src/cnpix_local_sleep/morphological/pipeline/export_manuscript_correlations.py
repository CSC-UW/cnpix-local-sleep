"""Export the two correlation analyses (per-event and epoched-partial) into compact,
manuscript-ready summary tables the r-offp ``manuscript_tables.py`` generator reads.

Neither analysis is serialized in table-ready form by the source notebooks:

* Per-event (``notebooks/figures/has_value.ipynb``) computes, within each
  subject-probe-structure, the Spearman correlation between an OFF property and the
  log delta power at the OFF midpoint, then pools by a random-effects (DerSimonian-Laird)
  meta-analysis on Fisher-z. Only SVG figures + an in-notebook table exist. Here we
  recompute the pooled summary from the cached OFF+bandpower parquet (no NFS) using
  the same ``cnpix_local_sleep.morphological.correlation_stats`` functions the notebook uses.

* Epoched-partial (``notebooks/figures/added_value/incremental_added_value*.ipynb``)
  already serializes its primary 10-s pooled results as parquet; we simply read the
  manuscript-facing 2-tier "area" form (Medium+Large vs Small; no BLAS) and tidy it.

Manuscript OFF partitions map onto the per-event category selections as:
  All OFFs = "LLAS"; Medium + Large = "CLAS"; Small OFFs = "LLAS-exclusive".

Outputs (written into ``r-offp/inst/extdata/`` by default):
  ``manuscript_per_event_correlations.csv``          (delta; Table S2a)
  ``manuscript_per_event_correlations_by_band.csv``  (delta, eta, theta; Table S2c)
  ``manuscript_per_event_multivariable.csv``         (delta, all four properties jointly; Table S2d)
  ``manuscript_epoched_partial_correlations.csv``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from cnpix_local_sleep import files
from cnpix_local_sleep.morphological import correlation_stats
from cnpix_local_sleep import off_tables

# The four OFF properties the manuscript correlates with per-event power
# (has_value.ipynb). The power value is the log power at the OFF midpoint
# ("centre"); total and max depend on the number of samples the OFF spans.
PROPERTIES = [
    ("area", "OFF area"),
    ("median_duration", "OFF duration"),
    ("span", "OFF span"),
    ("median_trace", "Residual MUA"),
]
BANDS = ["delta", "eta", "theta"]


def xy_pairs(band: str = "delta") -> list[tuple[str, str, str]]:
    return [(x, f"centre_bipolar_inst_log_{band}", label) for x, label in PROPERTIES]


XY_PAIRS = xy_pairs("delta")
GROUP_COLS = ["subject", "probe", "structure"]
# Manuscript partition -> category-selection membership (nested LLAS >= CLAS >= BLAS).
SELECTIONS = {
    "All OFFs": ["LLAS", "CLAS", "BLAS"],       # selection "LLAS"
    "Medium + Large": ["CLAS", "BLAS"],          # selection "CLAS"
    "Small OFFs": ["LLAS"],                       # selection "LLAS-exclusive"
}
CONDITIONS = ["NREM", "Wake"]
BONFERRONI_N = len(XY_PAIRS)


def _repo_root() -> Path:
    # src/cnpix_local_sleep/morphological/pipeline/<this>.py -> parents[4] is the cnpix_local_sleep repo root.
    return Path(__file__).resolve().parents[4]


def _default_cache_path() -> Path:
    return (
        _repo_root() / "notebooks" / "figures" / "outputs" / "bandpower_vs_off"
        / "full48h_morphological_offs_with_bandpower.parquet"
    )


def _default_added_value_dir() -> Path:
    return (
        _repo_root() / "notebooks" / "figures" / "added_value" / "outputs"
        / "added_value_data"
    )




def _filter_columns() -> list[str]:
    cols = set()
    for filt in (off_tables.llas_filters, off_tables.clas_filters, off_tables.blas_filters):
        cols.update(filt.keys())
    return sorted(cols)


def _assign_filter_category(offs: pd.DataFrame) -> pd.DataFrame:
    """Label each OFF by the most restrictive nested tier it passes (BLAS >= CLAS >=
    LLAS); drop OFFs failing the loosest (LLAS) filters. Ported verbatim from
    has_value.ipynb so the per-event selection matches the notebook exactly."""

    def _passes(filters):
        mask = pd.Series(True, index=offs.index)
        for col, (lo, hi) in filters.items():
            mask &= offs[col].between(lo, hi)
        return mask

    llas_m = _passes(off_tables.llas_filters)
    clas_m = _passes(off_tables.clas_filters)
    blas_m = _passes(off_tables.blas_filters)
    offs = offs.loc[llas_m].copy()
    category = np.where(
        blas_m.loc[offs.index],
        "BLAS",
        np.where(clas_m.loc[offs.index], "CLAS", "LLAS"),
    )
    offs["category"] = pd.Categorical(category, categories=["LLAS", "CLAS", "BLAS"], ordered=True)
    return offs.reset_index(drop=True)


def _classify_state(offs: pd.DataFrame) -> pd.Series:
    state = offs["state"].astype("string")
    sc = pd.Series(pd.NA, index=offs.index, dtype="object")
    sc[state == "NREM"] = "NREM"
    sc[state == "Wake"] = "Wake"
    return sc


def export_per_event(
    cache_path: Path | None = None, verbose: bool = True, bands: list[str] = ("delta",)
) -> pd.DataFrame:
    """Pooled per-event correlations for each band in ``bands`` (``band`` column)."""
    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    pairs = [(band, *pair) for band in bands for pair in xy_pairs(band)]
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Per-event cache not found: {cache_path}. Rebuild it via has_value.ipynb "
            "(REBUILD_FULL48H_CACHE=True) first."
        )
    x_cols = {x for _b, x, _y, _l in pairs}
    y_cols = {y for _b, _x, y, _l in pairs}
    needed = sorted(set(GROUP_COLS) | {"state"} | set(_filter_columns()) | x_cols | y_cols)
    if verbose:
        print(f"[per-event] reading {len(needed)} columns from {cache_path.name} ...")
    offs = pd.read_parquet(cache_path, columns=needed)
    offs = _assign_filter_category(offs)
    offs["state_class"] = _classify_state(offs)
    offs = offs[offs["state_class"].notna()].reset_index(drop=True)

    rows = []
    for partition, members in SELECTIONS.items():
        sel = offs[offs["category"].isin(members)]
        for cond in CONDITIONS:
            sub = sel[sel["state_class"] == cond]
            for band, x_col, y_col, prop_label in pairs:
                gc = correlation_stats.compute_group_correlations(
                    sub, x_col, y_col, group_cols=GROUP_COLS, method="spearman"
                )
                if gc is None or len(gc) == 0:
                    continue
                meta = correlation_stats.meta_analyze_correlations(gc)
                loo = correlation_stats.leave_one_out(gc)
                loo_rho = loo["overall_rho"] if (loo is not None and len(loo)) else pd.Series(dtype=float)
                p = meta["p_value"]
                rows.append(
                    {
                        "band": band,
                        "partition": partition,
                        "condition": cond,
                        "property": prop_label,
                        "x_col": x_col,
                        "y_col": y_col,
                        "pooled_rho": meta["overall_rho"],
                        "ci_lo": meta["ci_lo"],
                        "ci_hi": meta["ci_hi"],
                        "p_value": p,
                        "p_bonferroni": min(p * BONFERRONI_N, 1.0),
                        "sig_bonferroni": bool(p < 0.05 / BONFERRONI_N),
                        "i_squared": meta["i_squared"],
                        "q_pvalue": meta["q_pvalue"],
                        "k": meta["k"],
                        "loo_rho_min": float(loo_rho.min()) if len(loo_rho) else np.nan,
                        "loo_rho_max": float(loo_rho.max()) if len(loo_rho) else np.nan,
                    }
                )
                if verbose:
                    print(f"  {band:5s} {partition:15s} {cond:5s} {prop_label:14s} "
                          f"rho={meta['overall_rho']:+.3f} p={p:.2e} k={meta['k']}")
    return pd.DataFrame(rows)


def _rank_z(a: np.ndarray) -> np.ndarray:
    r = pd.DataFrame(a).rank().to_numpy()
    return (r - r.mean(0)) / r.std(0, ddof=1)


def _semipartials(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS of ``y`` on the columns of ``x`` (both already standardized). The semipartial
    of predictor i is its joint coefficient times sqrt(1 - R2_i), R2_i from regressing
    x_i on the other predictors; its square is the increment in R2 over the others."""
    n, m = x.shape
    xc = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(xc, y, rcond=None)
    resid = y - xc @ beta
    rss = resid @ resid
    cov = rss / (n - m - 1) * np.linalg.inv(xc.T @ xc)
    shrink = 1.0 / np.sqrt(np.diag(np.linalg.inv(np.corrcoef(x, rowvar=False))))
    return {
        "r2_full": 1.0 - rss / np.sum((y - y.mean()) ** 2),
        "sr": beta[1:] * shrink,
        "se_sr": np.sqrt(np.diag(cov)[1:]) * shrink,
    }


def export_per_event_multivariable(
    cache_path: Path | None = None, verbose: bool = True, min_n: int = 30
) -> pd.DataFrame:
    """All OFFs: rank delta regressed on the ranks of all four properties within each
    subject-probe-structure. Per property, the pooled semipartial r (DerSimonian-Laird on
    the coefficient scale, regression variances) and the mean per-pair unique R2; per
    state, the mean full-model R2 and the shared R2 (full minus the four unique)."""
    cache_path = Path(cache_path) if cache_path else _default_cache_path()
    y_col = xy_pairs("delta")[0][1]
    x_cols = [x for x, _label in PROPERTIES]
    needed = sorted(set(GROUP_COLS) | {"state"} | set(_filter_columns()) | set(x_cols) | {y_col})
    offs = pd.read_parquet(cache_path, columns=needed)
    offs = _assign_filter_category(offs)
    offs["state_class"] = _classify_state(offs)
    offs = offs[offs["category"].isin(SELECTIONS["All OFFs"]) & offs["state_class"].notna()]

    rows = []
    for cond in CONDITIONS:
        sub = offs[offs["state_class"] == cond].dropna(subset=[*x_cols, y_col])
        fits = [
            _semipartials(_rank_z(g[x_cols].to_numpy()), _rank_z(g[[y_col]].to_numpy())[:, 0])
            for _key, g in sub.groupby(GROUP_COLS, observed=True)
            if len(g) >= min_n
        ]
        sr = np.array([f["sr"] for f in fits])
        se = np.array([f["se_sr"] for f in fits])
        r2_full = np.array([f["r2_full"] for f in fits])
        k = len(fits)
        for i, (x_col, prop_label) in enumerate(PROPERTIES):
            gc = correlation_stats.compute_group_correlations(
                sub, x_col, y_col, group_cols=GROUP_COLS, method="spearman"
            )
            dl = correlation_stats._dersimonian_laird(sr[:, i], se[:, i] ** 2)
            est, est_se = dl["overall_z"], dl["overall_z_se"]
            p = 2.0 * scipy.stats.norm.sf(abs(est / est_se))
            rows.append({
                "condition": cond, "term": prop_label, "x_col": x_col,
                "marginal_rho": correlation_stats.meta_analyze_correlations(gc)["overall_rho"],
                "semipartial_r": est, "ci_lo": est - 1.96 * est_se, "ci_hi": est + 1.96 * est_se,
                "p_value": p, "p_bonferroni": min(p * BONFERRONI_N, 1.0),
                "sig_bonferroni": bool(p < 0.05 / BONFERRONI_N),
                "r2": float(np.mean(sr[:, i] ** 2)),
                "n_same_sign": int(np.sum(np.sign(sr[:, i]) == np.sign(est))),
                "i_squared": dl["i_squared"], "k": k,
            })
            if verbose:
                print(f"  {cond:5s} {prop_label:14s} sr={est:+.3f} unique R2={rows[-1]['r2']:.3f} k={k}")
        rows.append({"condition": cond, "term": "Shared", "k": k,
                     "r2": float(np.mean(r2_full - np.sum(sr ** 2, axis=1)))})
        rows.append({"condition": cond, "term": "Full model", "k": k, "r2": float(np.mean(r2_full)),
                     "r2_q25": float(np.quantile(r2_full, 0.25)),
                     "r2_q75": float(np.quantile(r2_full, 0.75))})
    out = pd.DataFrame(rows)
    out["n_same_sign"] = out["n_same_sign"].astype("Int64")
    return out


def export_epoched_partial(added_value_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Tidy the manuscript-facing 2-tier 'area' pooled parquets (Medium+Large vs
    Small; no BLAS), across NREM/Wake and the marginal / semipartial models.

    ``semipartial`` is the reported partial quantity (the part correlation, = the
    joint coefficient rescaled by sqrt(1 - R2_i)); its square is that level's
    incremental R2, which is what makes it comparable with the marginal on one axis.
    The joint coefficient is deliberately not emitted -- it is a rescaling of the
    same estimate testing the same null, and carrying it here would put two partial
    conventions in one manuscript table. It remains on disk in
    ``*_area_partial_pooled.parquet`` if a sensitivity check needs it.
    """
    d = Path(added_value_dir) if added_value_dir else _default_added_value_dir()
    tier_relabel = {"Conservative (CLAS set)": "Medium + Large", "LLAS-exclusive": "Small OFFs"}
    frames = []
    for state in ["nrem", "wake"]:
        for model in ["marginal", "semipartial"]:
            fp = d / f"{state}_area_{model}_pooled.parquet"
            if not fp.exists():
                if verbose:
                    print(f"[epoched] missing {fp.name}; skipping")
                continue
            df = pd.read_parquet(fp)
            df = df.assign(
                condition=state.upper() if state == "nrem" else "Wake",
                model=model,
            )
            df["partition"] = df["tier"].map(tier_relabel).fillna(df["tier"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No epoched-partial 'area' pooled parquets under {d}")
    out = pd.concat(frames, ignore_index=True)
    cols = ["condition", "model", "partition", "pooled_std_beta", "ci_lo", "ci_hi",
            "p", "i_squared", "k"]
    return out[[c for c in cols if c in out.columns]]


def export_manuscript_correlations(
    output_dir: Path | None = None,
    cache_path: Path | None = None,
    added_value_dir: Path | None = None,
    do_per_event: bool = True,
    do_epoched: bool = True,
    verbose: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir) if output_dir else files.get_r_offp_extdata_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if do_per_event:
        pe = export_per_event(cache_path, verbose=verbose)
        pe_path = output_dir / "manuscript_per_event_correlations.csv"
        pe.to_csv(pe_path, index=False)
        written.append(pe_path)
        if verbose:
            print(f"[per-event] wrote {len(pe)} rows -> {pe_path}")
        pb = export_per_event(cache_path, verbose=False, bands=BANDS)
        pb_path = output_dir / "manuscript_per_event_correlations_by_band.csv"
        pb.to_csv(pb_path, index=False)
        written.append(pb_path)
        if verbose:
            print(f"[per-event] wrote {len(pb)} rows -> {pb_path}")
        mv = export_per_event_multivariable(cache_path, verbose=verbose)
        mv_path = output_dir / "manuscript_per_event_multivariable.csv"
        mv.to_csv(mv_path, index=False)
        written.append(mv_path)
        if verbose:
            print(f"[per-event] wrote {len(mv)} rows -> {mv_path}")
    if do_epoched:
        ep = export_epoched_partial(added_value_dir, verbose=verbose)
        ep_path = output_dir / "manuscript_epoched_partial_correlations.csv"
        ep.to_csv(ep_path, index=False)
        written.append(ep_path)
        if verbose:
            print(f"[epoched] wrote {len(ep)} rows -> {ep_path}")
    return written
