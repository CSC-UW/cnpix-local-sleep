"""Correlation analysis for OFF properties vs. bandpower measures.

Provides per-group Spearman/Pearson correlation computation, Fisher
z-transform utilities, and DerSimonian-Laird random-effects
meta-analysis of correlations across (subject, probe, structure) groups.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

# -------------------- Effect-size classification --------------------


def classify_effect_size(r: float) -> str:
    """Classify a correlation coefficient by magnitude.

    Uses Cohen's (1988) benchmarks for bivariate correlations:
    |r| < 0.1  -> negligible
    |r| < 0.3  -> small
    |r| < 0.5  -> moderate
    |r| >= 0.5 -> large
    """
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    if ar < 0.3:
        return "small"
    if ar < 0.5:
        return "moderate"
    return "large"


# -------------------- Per-group correlations --------------------


def compute_group_correlations(
    offs: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_cols: list[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.DataFrame:
    """Compute correlations between *x_col* and *y_col* per group.

    Parameters
    ----------
    offs
        DataFrame of individual OFF events.
    x_col, y_col
        Column names for the two variables to correlate.
    group_cols
        Columns defining groups.  Defaults to
        ``["subject", "probe", "structure"]``.
    method
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    DataFrame with one row per group and columns:
        *group_cols*, ``rho``, ``n``, ``z``, ``z_var``,
        ``ci_lo``, ``ci_hi``, ``effect_label``.
    """
    if group_cols is None:
        group_cols = ["subject", "probe", "structure"]

    corr_fn = scipy.stats.spearmanr if method == "spearman" else scipy.stats.pearsonr

    records = []
    for keys, grp in offs.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = grp[[x_col, y_col]].dropna()
        n = len(valid)
        if n < 4:
            continue
        result = corr_fn(valid[x_col], valid[y_col])
        rho = result.statistic if hasattr(result, "statistic") else result[0]
        z = np.arctanh(np.clip(rho, -0.9999, 0.9999))
        z_var = 1.0 / (n - 3)
        z_se = np.sqrt(z_var)
        ci_lo = np.tanh(z - 1.96 * z_se)
        ci_hi = np.tanh(z + 1.96 * z_se)
        records.append(
            dict(
                zip(group_cols, keys),
                rho=rho,
                n=n,
                z=z,
                z_var=z_var,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                effect_label=classify_effect_size(rho),
            )
        )

    return pd.DataFrame(records)


# DerSimonian-Laird random-effects meta-analysis


def _dersimonian_laird(z: np.ndarray, v: np.ndarray) -> dict:
    """Random-effects meta-analysis (DerSimonian & Laird, 1986).

    Parameters
    ----------
    z
        Fisher-z-transformed correlation per group.
    v
        Variance of each z_i, i.e. ``1 / (n_i - 3)``.

    Returns
    -------
    Dict with keys ``overall_z``, ``overall_z_se``, ``tau_squared``,
    ``q_stat``, ``q_pvalue``, ``i_squared``.
    """
    w = 1.0 / v
    z_fe = np.sum(w * z) / np.sum(w)
    q = np.sum(w * (z - z_fe) ** 2)
    k = len(z)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    w_re = 1.0 / (v + tau2)
    z_re = np.sum(w_re * z) / np.sum(w_re)
    z_re_se = 1.0 / np.sqrt(np.sum(w_re))
    i_squared = max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0
    q_pvalue = 1.0 - scipy.stats.chi2.cdf(q, k - 1)
    return dict(
        overall_z=z_re,
        overall_z_se=z_re_se,
        tau_squared=tau2,
        q_stat=q,
        q_pvalue=q_pvalue,
        i_squared=i_squared,
    )


def meta_analyze_correlations(
    group_corrs: pd.DataFrame,
) -> dict:
    """Random-effects meta-analysis of per-group correlations.

    Parameters
    ----------
    group_corrs
        Output of :func:`compute_group_correlations`.

    Returns
    -------
    Dict with ``overall_rho``, ``ci_lo``, ``ci_hi``, ``p_value``
    (two-sided z-test for rho != 0), ``tau_squared``, ``i_squared``,
    ``q_stat``, ``q_pvalue``, ``k`` (number of groups), and
    ``effect_label``.
    """
    z = group_corrs["z"].values
    v = group_corrs["z_var"].values
    dl = _dersimonian_laird(z, v)
    ci_lo_z = dl["overall_z"] - 1.96 * dl["overall_z_se"]
    ci_hi_z = dl["overall_z"] + 1.96 * dl["overall_z_se"]
    overall_rho = np.tanh(dl["overall_z"])
    # Two-sided z-test: is overall correlation different from zero?
    z_test = dl["overall_z"] / dl["overall_z_se"]
    p_value = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(z_test)))
    return dict(
        overall_rho=overall_rho,
        ci_lo=np.tanh(ci_lo_z),
        ci_hi=np.tanh(ci_hi_z),
        p_value=p_value,
        tau_squared=dl["tau_squared"],
        i_squared=dl["i_squared"],
        q_stat=dl["q_stat"],
        q_pvalue=dl["q_pvalue"],
        k=len(z),
        effect_label=classify_effect_size(overall_rho),
    )


# -------------------- Sensitivity analysis --------------------


def leave_one_out(group_corrs: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out sensitivity analysis for the meta-analysis.

    Re-runs the meta-analysis dropping each group in turn and reports
    the resulting overall rho.

    Returns
    -------
    DataFrame with one row per dropped group and columns:
        ``dropped`` (label string), ``overall_rho``, ``ci_lo``,
        ``ci_hi``.
    """
    id_cols = [c for c in group_corrs.columns if c in ("subject", "probe", "structure")]
    records = []
    for idx in group_corrs.index:
        subset = group_corrs.drop(idx)
        if len(subset) < 2:
            continue
        ma = meta_analyze_correlations(subset)
        label = " / ".join(str(group_corrs.loc[idx, c]) for c in id_cols)
        records.append(
            dict(
                dropped=label,
                overall_rho=ma["overall_rho"],
                ci_lo=ma["ci_lo"],
                ci_hi=ma["ci_hi"],
            )
        )
    return pd.DataFrame(records)


# -------------------- Formatting helpers --------------------


def format_meta_summary(
    meta: dict, x_col: str, y_col: str, method: str = "spearman"
) -> str:
    """Return a human-readable multi-line summary of a meta-analysis."""
    lines = [
        f"Meta-analysis: {x_col} vs {y_col} ({method})",
        f"  k = {meta['k']} groups",
        f"  Overall rho = {meta['overall_rho']:.3f}"
        f"  [{meta['ci_lo']:.3f}, {meta['ci_hi']:.3f}]"
        f"  ({meta['effect_label']})",
        f"  p (rho != 0) = {meta['p_value']:.2e}",
        f"  tau^2 = {meta['tau_squared']:.4f}",
        f"  I^2   = {meta['i_squared']:.1f}%",
        f"  Q     = {meta['q_stat']:.2f},  p(Q)  = {meta['q_pvalue']:.2e}",
    ]
    return "\n".join(lines)


# -------------------- Forest plot --------------------


def plot_forest(
    group_corrs: pd.DataFrame,
    meta: dict,
    title: str = "",
    ax: plt.Axes | None = None,
    plot_for_pub: bool = False
) -> plt.Axes:
    """Forest plot of per-group correlations with pooled estimate.

    Parameters
    ----------
    group_corrs
        Output of :func:`compute_group_correlations`.
    meta
        Output of :func:`meta_analyze_correlations`.
    title
        Plot title.
    ax
        Matplotlib axes to draw on.  Created if *None*.
    """
    id_cols = [c for c in ("subject", "probe", "structure") if c in group_corrs.columns]
    labels = [
        " / ".join(str(row[c]) for c in id_cols) for _, row in group_corrs.iterrows()
    ]

    k = len(group_corrs)
    if ax is None:
        fig_height = max(4, 0.35 * (k + 2))
        _, ax = plt.subplots(figsize=(4, fig_height), constrained_layout=True)

    y_positions = np.arange(k)[::-1]

    # Per-group CIs
    rhos = group_corrs["rho"].values
    ci_los = group_corrs["ci_lo"].values
    ci_his = group_corrs["ci_hi"].values
    ax.errorbar(
        rhos,
        y_positions,
        xerr=[np.maximum(0, rhos - ci_los), np.maximum(0, ci_his - rhos)],
        fmt="o",
        color="steelblue",
        ecolor="steelblue",
        elinewidth=1.2,
        markersize=4,
        capsize=2,
    )

    # Overall meta-analytic estimate (diamond)
    y_overall = -1.5
    diamond_hw = 0.4  # half-width in y
    diamond_x = [
        meta["ci_lo"],
        meta["overall_rho"],
        meta["ci_hi"],
        meta["overall_rho"],
    ]
    diamond_y = [
        y_overall,
        y_overall + diamond_hw,
        y_overall,
        y_overall - diamond_hw,
    ]
    ax.fill(diamond_x, diamond_y, color="firebrick", alpha=0.7)

    # Reference line at rho = 0
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    # Labels
    if plot_for_pub:
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.set_yticks(list(y_positions) + [y_overall])
        ax.set_yticklabels(labels + ["Overall"], fontsize=7)
    ax.set_xlabel("Spearman rho")
    if title:
        ax.set_title(title, fontsize=9)

    # Right-side annotations: rho [CI]
    if not plot_for_pub:
        for i, (rho, lo, hi, n) in enumerate(
            zip(rhos, ci_los, ci_his, group_corrs["n"].values)
        ):
            ax.text(
                1.0,
                y_positions[i],
                f" {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]  N={n}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=6,
                family="monospace",
            )

        ax.text(
            1.0,
            y_overall,
            f" {meta['overall_rho']:+.3f} [{meta['ci_lo']:+.3f}, {meta['ci_hi']:+.3f}]",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=6,
            fontweight="bold",
            family="monospace",
        )

    ax.set_ylim(y_overall - 1, y_positions[0] + 1)
    return ax