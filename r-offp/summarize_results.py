"""Produce a compact summary table from r-offp mixed-effects model results.

Reads a YAML file specifying
(response_variable, dataset, condition_set, model) 4-tuples and outputs a
markdown table (to stdout) and a CSV file summarizing the main effects and
posthoc contrasts for each entry.

Usage:
    python summarize_results.py config/summary_tables/example.yaml
"""

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


def format_pvalue(p):
    """Format a p-value with significance indicator."""
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return "p < 0.01 **"
    if p < 0.05:
        return "p < 0.05 *"
    if p < 0.1:
        return f"p = {p:.3f} ."
    return f"p = {p:.3f}"


def format_main_effect(main_effect):
    """Format the main effect cell: p-value and Cohen's f2.

    ``cohens_f2`` may be absent/None (e.g. an engine that reports no effect-size
    analog, or a non-converged fit); fall back to a bare p-value cell.
    """
    p = main_effect["pval"]
    p_str = format_pvalue(p)
    if main_effect["significant"]:
        f2 = main_effect.get("cohens_f2")
        if f2 is not None:
            return f"{p_str} [f\u00b2={f2:.2f}]"
    return f"p={p_str}"


def format_posthoc_cell(posthoc, contrast_idx):
    """Format a single posthoc contrast cell: p-value, Cohen's d, direction.

    ``cohens_d`` may be absent/None; the p-value and direction still render.
    """
    p = posthoc["pvalues"][contrast_idx]
    est = posthoc["estimates"][contrast_idx]
    direction = "+" if est > 0 else "\u2212"
    p_str = format_pvalue(p)
    d_list = posthoc.get("cohens_d")
    d = d_list[contrast_idx] if d_list is not None else None
    d_str = f" [d={d:.2f}]" if d is not None else ""
    return f"{p_str}{d_str} {direction}"


def normalize_contrast(contrast_str):
    """Normalize a contrast string for matching.

    Strips ` = 0`, ` == 0`, and extra whitespace so that contrast labels
    from JSON (which omit ` = 0`) match the YAML posthoc definitions
    (which include ` = 0`).
    """
    s = contrast_str.strip()
    for suffix in (" = 0", " == 0"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Collapse internal whitespace
    return " ".join(s.split())


def load_results(results_dir, dataset, condition_set, response_var, model):
    """Load results.json for a (dataset, condition_set, response_var, model)
    analysis.
    """
    path = (
        results_dir
        / dataset
        / condition_set
        / response_var
        / model
        / "results.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_condition_set_posthocs(config, condition_set):
    """Get the list of posthoc contrast strings for a condition set from the
    main config YAML.
    """
    cs = config.get("condition_sets", {}).get(condition_set)
    if cs is None:
        return []
    return cs.get("posthocs", [])


def build_table(summary_config, base_dir):
    """Build the summary table rows and column headers.

    Returns (headers, rows) where each row is a list of cell strings.
    """
    results_dir = base_dir / summary_config["results_dir"]
    config_path = base_dir / summary_config["config"]
    with open(config_path) as f:
        main_config = yaml.safe_load(f)

    contrast_labels = summary_config.get("contrast_labels", {})
    entries = summary_config["entries"]

    # Collect all unique posthoc contrasts across all condition sets referenced,
    # maintaining the order they first appear.
    all_contrasts = OrderedDict()
    for response_var, dataset, condition_set, model in entries:
        for posthoc_str in get_condition_set_posthocs(main_config, condition_set):
            norm = normalize_contrast(posthoc_str)
            if norm not in all_contrasts:
                label = contrast_labels.get(posthoc_str, posthoc_str)
                all_contrasts[norm] = label

    # Build column headers
    headers = ["Response", "Dataset", "Set", "Model", "Main effect"]
    headers.extend(all_contrasts.values())

    # Build rows
    rows = []
    for response_var, dataset, condition_set, model in entries:
        result = load_results(
            results_dir, dataset, condition_set, response_var, model
        )
        main_effect = result["main_effect"]

        row = [
            response_var,
            dataset.upper(),
            condition_set,
            model,
            format_main_effect(main_effect),
        ]

        # Determine which contrasts this condition set defines
        cs_posthocs = get_condition_set_posthocs(main_config, condition_set)
        cs_contrast_norms = [normalize_contrast(s) for s in cs_posthocs]

        for norm_contrast in all_contrasts:
            if norm_contrast not in cs_contrast_norms:
                row.append("N/A")
            elif not main_effect["significant"] or "posthoc" not in main_effect:
                row.append("\u2014")
            else:
                posthoc = main_effect["posthoc"]
                # Find the index of this contrast in the posthoc arrays.
                # Try matching against the JSON contrasts field first;
                # fall back to positional matching against the YAML order.
                idx = None
                if posthoc.get("contrasts"):
                    json_norms = [
                        normalize_contrast(c) for c in posthoc["contrasts"]
                    ]
                    if norm_contrast in json_norms:
                        idx = json_norms.index(norm_contrast)

                if idx is None:
                    # Positional fallback: match against YAML posthoc order
                    try:
                        idx = cs_contrast_norms.index(norm_contrast)
                    except ValueError:
                        row.append("N/A")
                        continue

                if idx < len(posthoc.get("pvalues", [])):
                    row.append(format_posthoc_cell(posthoc, idx))
                else:
                    row.append("\u2014")

        rows.append(row)

    return headers, rows


def format_markdown_table(headers, rows):
    """Format headers and rows as a GitHub-flavored markdown table."""
    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(cells):
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines = []
    lines.append(format_row(headers))
    lines.append(
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    )
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def write_csv(headers, rows, output_path):
    """Write the table as a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize r-offp model results into a compact table."
    )
    parser.add_argument(
        "summary_yaml",
        type=Path,
        help="Path to the summary YAML config file.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for resolving relative paths in the YAML. "
            "Defaults to the directory containing the YAML file's parent "
            "(i.e. the r-offp root)."
        ),
    )
    args = parser.parse_args()

    with open(args.summary_yaml) as f:
        summary_config = yaml.safe_load(f)

    # Default base_dir: parent of the YAML file's directory
    # e.g. if YAML is config/summary_tables/example.yaml, base_dir is .
    # This assumes the YAML is inside r-offp/config/summary_tables/
    if args.base_dir is not None:
        base_dir = args.base_dir
    else:
        base_dir = args.summary_yaml.resolve().parent
        # Walk up to find the directory containing results_dir
        results_dir_name = summary_config.get("results_dir", "_output")
        while base_dir != base_dir.parent:
            if (base_dir / results_dir_name).is_dir():
                break
            base_dir = base_dir.parent
        else:
            print(
                f"Error: Could not find '{results_dir_name}' directory.",
                file=sys.stderr,
            )
            sys.exit(1)

    headers, rows = build_table(summary_config, base_dir)

    # Output markdown to stdout
    print(format_markdown_table(headers, rows))

    # Output CSV alongside the input YAML, as <stem>.csv.
    csv_path = args.summary_yaml.with_suffix(".csv")
    write_csv(headers, rows, csv_path)
    print(f"\nCSV written to: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
