"""Figure generation for cross-structure OFF period analysis.

Produces all diagnostic figures for the cross-structure OFF relationship
pipeline step. Each ``plot_*`` function creates one figure type; the
``do_subject`` orchestrator calls them all and saves PNGs.
"""

import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, spearmanr

import cnpix_local_sleep.files
from cnpix_local_sleep import const, plots

_flare_cmap = sns.color_palette("flare", as_cmap=True)


def _get_plot_dir(subject: str, off_source: str | None = None) -> pathlib.Path:
    d = cnpix_local_sleep.files.get_subject_plot_dir(subject) / "cross_structure_offs"
    if off_source is not None:
        d = d / off_source
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_fig(fig: plt.Figure, path: pathlib.Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------- Individual plot functions --------------------


def plot_overlap_degree_bars(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    raw_counts: dict[str, dict[str, np.ndarray]],
    plot_dir: pathlib.Path,
    *,
    no_legend: bool = False,
) -> None:
    """Stacked bar chart of overlap degree by structure, conditions nested."""
    n_structures = len(structure_labels)
    max_overlap = n_structures - 1
    colors = _flare_cmap(np.linspace(0.1, 0.9, max_overlap + 1))

    n_conditions = len(conditions)
    group_width = 0.8
    bar_width = group_width / n_conditions
    offsets = np.linspace(
        -group_width / 2 + bar_width / 2,
        group_width / 2 - bar_width / 2,
        n_conditions,
    )
    x_positions = np.arange(n_structures)

    # Pre-compute fractions.
    fracs = np.zeros((n_structures, n_conditions, max_overlap + 1))
    for si, struct in enumerate(structure_labels):
        for ci, cond in enumerate(conditions):
            counts = raw_counts[cond][struct]
            for degree in range(max_overlap + 1):
                fracs[si, ci, degree] = (
                    np.mean(counts == degree) if len(counts) > 0 else 0.0
                )

    fig, ax = plt.subplots(figsize=(14, 5))
    overlap_handles = []
    for degree in range(max_overlap + 1):
        label = "Local (0)" if degree == 0 else f"Overlap w/ {degree}"
        for si in range(n_structures):
            for ci in range(n_conditions):
                x = x_positions[si] + offsets[ci]
                bottom = fracs[si, ci, :degree].sum()
                ax.bar(
                    x,
                    fracs[si, ci, degree],
                    width=bar_width * 0.9,
                    bottom=bottom,
                    color=colors[degree],
                    edgecolor="white",
                    linewidth=0.3,
                    label=label if (ci == 0 and si == 0) else "_nolegend_",
                )
        overlap_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[degree]))

    for si in range(n_structures):
        for ci, cond in enumerate(conditions):
            x = x_positions[si] + offsets[ci]
            ax.text(
                x,
                -0.04,
                cond,
                ha="center",
                va="top",
                fontsize=5,
                rotation=45,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(structure_labels, fontsize=10)
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.set_ylabel("Fraction of OFFs")
    ax.set_ylim(0, 1)

    if not no_legend:
        overlap_legend_labels = [
            "Local (0)" if d == 0 else f"Overlap w/ {d}"
            for d in range(max_overlap + 1)
        ]
        ax.legend(
            overlap_handles,
            overlap_legend_labels,
            loc="upper right",
            fontsize=7,
            title="# overlapping structures",
        )
    fig.suptitle(f"{subject} | Local vs Global OFFs", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, plot_dir / "overlap_degree_bars.png")


def plot_pairwise_overlap_heatmaps(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    pairwise_df: pd.DataFrame,
    plot_dir: pathlib.Path,
) -> None:
    """Heatmap of pairwise overlap fractions, one subplot per condition."""
    n = len(structure_labels)
    n_cols = min(3, len(conditions))
    n_rows = int(np.ceil(len(conditions) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False
    )

    for idx, cond in enumerate(conditions):
        ax = axes[idx // n_cols, idx % n_cols]
        mat = np.full((n, n), np.nan)
        for i, sa in enumerate(structure_labels):
            n_a = len(offs_dict[(sa, cond)])
            mat[i, i] = 1.0
            for j, sb in enumerate(structure_labels):
                if i == j:
                    continue
                # Find overlaps where sa is reference.
                sub = pairwise_df[
                    (pairwise_df["condition"] == cond)
                    & (
                        ((pairwise_df["struct_a"] == sa) & (pairwise_df["struct_b"] == sb))
                    )
                ]
                sub_rev = pairwise_df[
                    (pairwise_df["condition"] == cond)
                    & (
                        ((pairwise_df["struct_a"] == sb) & (pairwise_df["struct_b"] == sa))
                    )
                ]
                if not sub.empty:
                    n_with = sub["index_a"].nunique()
                elif not sub_rev.empty:
                    n_with = sub_rev["index_b"].nunique()
                else:
                    n_with = 0
                mat[i, j] = n_with / n_a if n_a > 0 else 0

        sns.heatmap(
            mat,
            xticklabels=structure_labels,
            yticklabels=structure_labels,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Fraction overlapping"},
        )
        ax.set_title(cond, fontsize=9)
        ax.set_ylabel("Reference structure")
        ax.set_xlabel("Other structure")

    for idx in range(len(conditions), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        f"{subject} | Fraction of OFFs overlapping with another structure",
        fontsize=13,
    )
    fig.tight_layout()
    _save_fig(fig, plot_dir / "pairwise_overlap_heatmaps.png")


def plot_pairwise_overlap_details(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    pairwise_df: pd.DataFrame,
    plot_dir: pathlib.Path,
) -> None:
    """Histograms of overlap duration, fraction, onset/offset lag for each pair."""
    structure_pairs = list(itertools.combinations(structure_labels, 2))
    cond = conditions[0]
    hist_color = "steelblue"

    fig, axes = plt.subplots(
        len(structure_pairs), 4, figsize=(22, 3 * len(structure_pairs)), squeeze=False
    )

    for row, (sa, sb) in enumerate(structure_pairs):
        ov = pairwise_df[
            (pairwise_df["condition"] == cond)
            & (pairwise_df["struct_a"] == sa)
            & (pairwise_df["struct_b"] == sb)
        ]
        if ov.empty:
            for col in range(4):
                axes[row, col].text(
                    0.5, 0.5, "No overlaps", ha="center", va="center",
                    transform=axes[row, col].transAxes,
                )
            axes[row, 0].set_title(f"{sa} vs {sb}")
            continue

        # Col 0: Overlap duration.
        median_dur = ov["overlap_duration"].median() * 1000
        axes[row, 0].hist(
            ov["overlap_duration"] * 1000, bins=50, color=hist_color, alpha=0.7
        )
        axes[row, 0].set_xlabel("Overlap duration (ms)")
        axes[row, 0].set_ylabel("Count")
        axes[row, 0].set_title(
            f"{sa} vs {sb}: overlap duration (median={median_dur:.1f}ms)"
        )

        # Col 1: Overlap fraction.
        axes[row, 1].hist(ov["fraction_of_a"], bins=30, color=hist_color, alpha=0.7)
        axes[row, 1].set_xlabel(f"Overlap / duration({sa})")
        axes[row, 1].set_ylabel("Count")
        axes[row, 1].set_title(f"{sa} vs {sb}: overlap fraction")
        axes[row, 1].set_xlim(0, 1)

        # Col 2: Onset lag.
        median_lag = ov["onset_lag"].median() * 1000
        axes[row, 2].hist(ov["onset_lag"] * 1000, bins=50, color=hist_color, alpha=0.7)
        axes[row, 2].axvline(0, color="k", ls="--", lw=0.8)
        axes[row, 2].set_xlabel(f"Onset lag {sb} - {sa} (ms)")
        axes[row, 2].set_ylabel("Count")
        axes[row, 2].set_title(
            f"{sa} vs {sb}: onset lag (median={median_lag:.1f}ms)"
        )

        # Col 3: Offset lag.
        median_off_lag = ov["offset_lag"].median() * 1000
        axes[row, 3].hist(
            ov["offset_lag"] * 1000, bins=50, color=hist_color, alpha=0.7
        )
        axes[row, 3].axvline(0, color="k", ls="--", lw=0.8)
        axes[row, 3].set_xlabel(f"Offset lag {sb} - {sa} (ms)")
        axes[row, 3].set_ylabel("Count")
        axes[row, 3].set_title(
            f"{sa} vs {sb}: offset lag (median={median_off_lag:.1f}ms)"
        )

    fig.suptitle(f"{subject} | {cond} | Pairwise overlap details", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, plot_dir / "pairwise_overlap_details.png")


def plot_cross_correlograms(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    peths_df: pd.DataFrame,
    peth_window: float,
    peth_bin_size: float,
    plot_dir: pathlib.Path,
) -> None:
    """Full pairwise cross-correlogram matrix, one figure per condition."""
    n_structures = len(structure_labels)

    for cond in conditions:
        fig, axes = plt.subplots(
            n_structures,
            n_structures,
            figsize=(2.5 * n_structures, 2 * n_structures),
            sharex=True,
            sharey=False,
        )
        if n_structures == 1:
            axes = np.array([[axes]])

        cond_peths = peths_df[peths_df["condition"] == cond]

        for i, ref_struct in enumerate(structure_labels):
            for j, tgt_struct in enumerate(structure_labels):
                ax = axes[i, j]
                sub = cond_peths[
                    (cond_peths["ref_structure"] == ref_struct)
                    & (cond_peths["tgt_structure"] == tgt_struct)
                ]
                if sub.empty:
                    ax.set_visible(False)
                    continue

                color = "gray" if i == j else "steelblue"
                ax.bar(
                    sub["bin_center"].values * 1000,
                    sub["rate"].values,
                    width=peth_bin_size * 1000 * 0.9,
                    color=color,
                    alpha=0.7,
                )
                ax.axvline(0, color="k", ls="--", lw=0.5)

                if j == 0:
                    ax.set_ylabel(ref_struct, fontsize=8)
                if i == 0:
                    ax.set_title(tgt_struct, fontsize=8)
                ax.tick_params(labelsize=6)

        fig.suptitle(
            f"{subject} | {cond} | OFF onset cross-correlograms",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        cond_safe = cond.replace(".", "-")
        _save_fig(fig, plot_dir / f"cross_correlograms_{cond_safe}.png")


def plot_local_vs_overlapping_properties(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    raw_counts: dict[str, dict[str, np.ndarray]],
    plot_dir: pathlib.Path,
    *,
    no_legend: bool = False,
) -> None:
    """Split violin plots: local vs overlapping OFFs for OFF properties."""
    cond = conditions[0]
    annotated_dfs = []
    for struct in structure_labels:
        struct_offs = offs_dict[(struct, cond)].copy()
        if struct_offs.empty:
            continue
        struct_offs["n_overlapping_structures"] = raw_counts[cond][struct]
        struct_offs["is_local"] = raw_counts[cond][struct] == 0
        annotated_dfs.append(struct_offs)

    if not annotated_dfs:
        return

    annotated_offs = pd.concat(annotated_dfs, ignore_index=True)

    properties_to_compare = ["median_duration", "span", "area"]
    available_props = [p for p in properties_to_compare if p in annotated_offs.columns]
    if not available_props:
        return

    max_overlap = len(structure_labels) - 1
    flare_vals = np.linspace(0.1, 0.9, max_overlap + 1)
    overlap_status_palette = {
        "Local": _flare_cmap(flare_vals[0]),
        "Overlapping": _flare_cmap(flare_vals[-1]),
    }

    fig, axes = plt.subplots(
        1, len(available_props), figsize=(5 * len(available_props), 4), squeeze=False
    )

    for col_idx, prop in enumerate(available_props):
        ax = axes[0, col_idx]
        plot_df = annotated_offs[["structure", "is_local", prop]].dropna().copy()
        plot_df["overlap_status"] = plot_df["is_local"].map(
            {True: "Local", False: "Overlapping"}
        )
        if hasattr(plot_df["structure"], "cat"):
            plot_df["structure"] = plot_df["structure"].cat.remove_unused_categories()
        structure_order = (
            plot_df["structure"].cat.categories.tolist()
            if hasattr(plot_df["structure"], "cat")
            else sorted(plot_df["structure"].unique())
        )
        sns.violinplot(
            data=plot_df,
            x="structure",
            y=prop,
            hue="overlap_status",
            split=True,
            inner="quart",
            cut=0,
            ax=ax,
            palette=overlap_status_palette,
            order=structure_order,
            legend=not no_legend,
        )
        if prop in ("area", "median_duration"):
            ax.set_yscale("log")
        ax.set_title(prop)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"{subject} | {cond} | OFF properties: Local vs Overlapping", fontsize=12
    )
    fig.tight_layout()
    _save_fig(fig, plot_dir / "local_vs_overlapping_properties.png")


def plot_blas_overlap_enrichment(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    raw_counts: dict[str, dict[str, np.ndarray]],
    plot_dir: pathlib.Path,
) -> None:
    """Test whether BLAS OFFs are more likely to overlap than non-BLAS OFFs."""
    cond = conditions[0]

    annotated_dfs = []
    for struct in structure_labels:
        struct_offs = offs_dict[(struct, cond)].copy()
        if struct_offs.empty:
            continue
        struct_offs["is_local"] = raw_counts[cond][struct] == 0
        annotated_dfs.append(struct_offs)

    if not annotated_dfs:
        return

    annotated_offs = pd.concat(annotated_dfs, ignore_index=True)
    if "category" not in annotated_offs.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_data = []

    for struct in structure_labels:
        struct_offs = annotated_offs[annotated_offs["structure"] == struct]
        if struct_offs.empty:
            continue

        is_blas = struct_offs["category"] == "BLAS"
        n_blas = is_blas.sum()
        n_non_blas = (~is_blas).sum()
        if n_blas == 0 or n_non_blas == 0:
            continue

        frac_blas_overlap = (~struct_offs.loc[is_blas, "is_local"]).mean()
        frac_non_blas_overlap = (~struct_offs.loc[~is_blas, "is_local"]).mean()

        table = [
            [
                int((~struct_offs.loc[is_blas, "is_local"]).sum()),
                int(struct_offs.loc[is_blas, "is_local"].sum()),
            ],
            [
                int((~struct_offs.loc[~is_blas, "is_local"]).sum()),
                int(struct_offs.loc[~is_blas, "is_local"].sum()),
            ],
        ]
        odds_ratio, p_value = fisher_exact(table)

        bar_data.append(
            {
                "structure": struct,
                "group": "BLAS",
                "frac_overlap": frac_blas_overlap,
                "n": n_blas,
                "OR": odds_ratio,
                "p": p_value,
            }
        )
        bar_data.append(
            {
                "structure": struct,
                "group": "non-BLAS",
                "frac_overlap": frac_non_blas_overlap,
                "n": n_non_blas,
            }
        )

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        sns.barplot(
            data=bar_df,
            x="structure",
            y="frac_overlap",
            hue="group",
            ax=ax,
        )
        ax.set_ylabel("Fraction overlapping")
        ax.set_title(f"BLAS vs non-BLAS overlap enrichment ({cond})")

        # Annotate with odds ratio.
        for i, struct in enumerate(structure_labels):
            blas_row = bar_df[
                (bar_df["structure"] == struct) & (bar_df["group"] == "BLAS")
            ]
            if not blas_row.empty and "OR" in blas_row.columns:
                row = blas_row.iloc[0]
                ax.text(
                    i,
                    1.01,
                    f"OR={row['OR']:.1f}, p={row['p']:.1e}",
                    ha="center",
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )

    fig.suptitle(f"{subject} | BLAS overlap enrichment", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, plot_dir / "blas_overlap_enrichment.png")


def plot_properties_vs_overlap_degree(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    raw_counts: dict[str, dict[str, np.ndarray]],
    plot_dir: pathlib.Path,
) -> None:
    """Box plots of span_rel2max and duration vs overlap degree."""
    cond = conditions[0]

    annotated_dfs = []
    for struct in structure_labels:
        struct_offs = offs_dict[(struct, cond)].copy()
        if struct_offs.empty:
            continue
        struct_offs["n_overlapping_structures"] = raw_counts[cond][struct]
        annotated_dfs.append(struct_offs)

    if not annotated_dfs:
        return

    plot_offs = pd.concat(annotated_dfs, ignore_index=True)
    degree_values = sorted(plot_offs["n_overlapping_structures"].unique())
    plot_offs["overlap_degree"] = pd.Categorical(
        plot_offs["n_overlapping_structures"].astype(str),
        categories=[str(d) for d in degree_values],
        ordered=True,
    )
    if hasattr(plot_offs["structure"], "cat"):
        plot_offs["structure"] = plot_offs["structure"].cat.remove_unused_categories()
    structure_order = (
        plot_offs["structure"].cat.categories.tolist()
        if hasattr(plot_offs["structure"], "cat")
        else sorted(plot_offs["structure"].unique())
    )

    n_degrees = len(degree_values)
    flare_palette = [
        _flare_cmap(v) for v in np.linspace(0.1, 0.9, n_degrees)
    ]
    degree_palette = {str(d): flare_palette[i] for i, d in enumerate(degree_values)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: span_rel2max vs overlap degree.
    if "span_rel2max" in plot_offs.columns:
        ax = axes[0]
        plot_df = plot_offs.dropna(subset=["span_rel2max"])
        sns.boxplot(
            data=plot_df,
            x="structure",
            y="span_rel2max",
            hue="overlap_degree",
            hue_order=[str(d) for d in degree_values],
            order=structure_order,
            ax=ax,
            fliersize=0,
            palette=degree_palette,
            legend=False,
        )
        ax.set_xlabel("Structure")
        ax.set_ylabel("span_rel2max")
        ax.set_title("Spatial span vs overlap degree")

        for struct in structure_order:
            sdf = plot_offs[plot_offs["structure"] == struct].dropna(
                subset=["span_rel2max"]
            )
            if not sdf.empty:
                rho, p = spearmanr(
                    sdf["n_overlapping_structures"], sdf["span_rel2max"]
                )
                print(
                    f"{struct}: span_rel2max vs overlap: rho={rho:.3f}, p={p:.2e}"
                )
    else:
        axes[0].set_visible(False)

    # Right: duration vs overlap degree.
    ax = axes[1]
    sns.boxplot(
        data=plot_offs,
        x="structure",
        y="duration",
        hue="overlap_degree",
        hue_order=[str(d) for d in degree_values],
        order=structure_order,
        ax=ax,
        fliersize=0,
        palette=degree_palette,
        legend=False,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Structure")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Duration vs overlap degree")

    for struct in structure_order:
        sdf = plot_offs[plot_offs["structure"] == struct]
        if not sdf.empty:
            rho, p = spearmanr(sdf["n_overlapping_structures"], sdf["duration"])
            print(f"{struct}: duration vs overlap: rho={rho:.3f}, p={p:.2e}")

    fig.suptitle(f"{subject} | {cond} | OFF properties vs overlap", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, plot_dir / "properties_vs_overlap_degree.png")


def plot_jitter_null(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    jitter_df: pd.DataFrame,
    plot_dir: pathlib.Path,
    *,
    no_legend: bool = False,
) -> None:
    """Observed vs null distribution histograms per structure."""
    for cond in conditions:
        cond_jitter = jitter_df[jitter_df["condition"] == cond]
        if cond_jitter.empty:
            continue

        structs_present = [
            s for s in structure_labels if s in cond_jitter["structure"].values
        ]
        if not structs_present:
            continue

        fig, axes = plt.subplots(
            1, len(structs_present), figsize=(3 * len(structs_present), 3), sharey=True
        )
        if len(structs_present) == 1:
            axes = [axes]

        for ax, struct in zip(axes, structs_present):
            struct_data = cond_jitter[cond_jitter["structure"] == struct]
            null_arr = struct_data["null_overlap_frac"].values
            obs = struct_data["observed_overlap_frac"].iloc[0]

            ax.hist(null_arr, bins=20, color="gray", alpha=0.6, label="Null")
            ax.axvline(obs, color="red", lw=2, label=f"Observed ({obs:.2f})")
            ax.set_title(struct, fontsize=9)
            ax.set_xlabel("Overlap fraction")
            if ax is axes[0]:
                ax.set_ylabel("Shuffle count")
            if not no_legend:
                ax.legend(fontsize=6)

        n_shuffles = cond_jitter["shuffle_index"].nunique()
        fig.suptitle(
            f"{subject} | {cond} | Jitter control (n={n_shuffles} shuffles)",
            fontsize=12,
        )
        fig.tight_layout()
        cond_safe = cond.replace(".", "-")
        _save_fig(fig, plot_dir / f"jitter_null_{cond_safe}.png")


def plot_condition_comparison(
    subject: str,
    structure_labels: list[str],
    conditions: list[str],
    comparison_df: pd.DataFrame,
    plot_dir: pathlib.Path,
    *,
    no_legend: bool = False,
) -> None:
    """Grouped bar charts: overlap fraction and mean degree across conditions."""
    palette = plots.get_condition_palette()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fraction with any overlap.
    ax = axes[0]
    pivot = comparison_df.pivot(
        index="structure", columns="condition", values="frac_any_overlap"
    )
    ordered_conditions = [c for c in const.CORE_CONDITIONS if c in pivot.columns]
    pivot = pivot.reindex(index=structure_labels, columns=ordered_conditions)
    pivot.plot(
        kind="bar",
        ax=ax,
        color=[palette[c] for c in ordered_conditions],
        legend=not no_legend,
    )
    ax.set_ylabel("Fraction overlapping")
    ax.set_title("OFFs overlapping with >= 1 other structure")
    ax.tick_params(axis="x", rotation=45)
    if not no_legend:
        ax.legend(fontsize=7, bbox_to_anchor=(1.0, 1.0))

    # Right: mean overlap degree.
    ax = axes[1]
    pivot = comparison_df.pivot(
        index="structure", columns="condition", values="mean_overlap_degree"
    )
    ordered_conditions = [c for c in const.CORE_CONDITIONS if c in pivot.columns]
    pivot = pivot.reindex(index=structure_labels, columns=ordered_conditions)
    pivot.plot(
        kind="bar",
        ax=ax,
        color=[palette[c] for c in ordered_conditions],
        legend=not no_legend,
    )
    ax.set_ylabel("Mean # overlapping structures")
    ax.set_title("Mean overlap degree")
    ax.tick_params(axis="x", rotation=45)
    if not no_legend:
        ax.legend(fontsize=7, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(f"{subject} | Condition comparison", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, plot_dir / "condition_comparison.png")


# -------------------- Orchestrator --------------------


def do_subject(
    subject: str,
    *,
    offs_dict: dict[tuple[str, str], pd.DataFrame],
    structure_labels: list[str],
    conditions: list[str],
    condition_durations: dict[str, float],
    raw_counts: dict[str, dict[str, np.ndarray]],
    pairwise_df: pd.DataFrame,
    peths_df: pd.DataFrame,
    jitter_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
    peth_window: float = 0.5,
    peth_bin_size: float = 0.01,
    no_jitter: bool = False,
    no_legend: bool = False,
    off_source: str | None = None,
) -> None:
    """Generate and save all cross-structure OFF figures for a subject.

    Called by ``cross_structure_offs.do_subject()`` with pre-computed data.

    Args:
        subject: Subject name.
        offs_dict: OFFs keyed by (structure, condition).
        structure_labels: Ordered list of structure names.
        conditions: Conditions analyzed.
        condition_durations: Duration of each condition in seconds.
        raw_counts: ``{condition: {structure: overlap_count_array}}``.
        pairwise_df: Pairwise overlap DataFrame.
        peths_df: PETH DataFrame.
        jitter_df: Jitter null DataFrame.
        comparison_df: Condition comparison DataFrame.
        baselines_df: Chance baselines DataFrame.
        peth_window: PETH half-window in seconds.
        peth_bin_size: PETH bin width in seconds.
        no_jitter: If True, skip jitter null plot.
        no_legend: If True, suppress legends on all plots.
        off_source: OFF source key; when given, figures are saved under a
            source-specific subdirectory so different sources don't overwrite.
    """
    plot_dir = _get_plot_dir(subject, off_source)

    plot_overlap_degree_bars(
        subject, structure_labels, conditions, raw_counts, plot_dir,
        no_legend=no_legend,
    )

    plot_pairwise_overlap_heatmaps(
        subject, structure_labels, conditions, offs_dict, pairwise_df, plot_dir
    )

    plot_pairwise_overlap_details(
        subject, structure_labels, conditions, pairwise_df, plot_dir
    )

    plot_cross_correlograms(
        subject,
        structure_labels,
        conditions,
        peths_df,
        peth_window,
        peth_bin_size,
        plot_dir,
    )

    plot_local_vs_overlapping_properties(
        subject, structure_labels, conditions, offs_dict, raw_counts, plot_dir,
        no_legend=no_legend,
    )

    plot_blas_overlap_enrichment(
        subject, structure_labels, conditions, offs_dict, raw_counts, plot_dir
    )

    plot_properties_vs_overlap_degree(
        subject, structure_labels, conditions, offs_dict, raw_counts, plot_dir
    )

    if not no_jitter:
        plot_jitter_null(
            subject, structure_labels, conditions, jitter_df, plot_dir,
            no_legend=no_legend,
        )

    plot_condition_comparison(
        subject, structure_labels, conditions, comparison_df, plot_dir,
        no_legend=no_legend,
    )

    print(f"Saved figures to {plot_dir}")