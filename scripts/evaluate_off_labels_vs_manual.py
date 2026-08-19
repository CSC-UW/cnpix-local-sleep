"""Score SAM3 model + full-48h morphological OFF detections against manual labels.

Lives here rather than in ``samoffs`` because it needs *both* sides: ``samoffs``
must not depend on ``cnpix_local_sleep``, but ``cnpix_local_sleep`` may depend on ``samoffs``. This
script is therefore excluded from the publication repository, which carries no
SAM3 dependency.

Runs both evaluations over every (subject, probe) pair with manual labels and
writes a tidy per-pair results table plus a pooled/macro summary. See
``cnpix_local_sleep.evaluation`` for the metric definitions and the per-condition
image-selection convention (NREM: labeled chunks only; Wake: all chunks).

Usage::

    python evaluate_off_labels_vs_manual.py [--out-dir DIR]

Outputs (default ``docs/reports/``):
    off_label_method_comparison.per_pair.csv
    off_label_method_comparison.summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cnpix_local_sleep.evaluation import config
from cnpix_local_sleep.morphological import full48h_eval
from samoffs import model_eval

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "reports"


def run_all() -> pd.DataFrame:
    """Run model + full-48h morphological evals for NREM and Wake; return per-pair rows."""
    frames: list[pd.DataFrame] = []
    for eval_name in ("NREM", "Wake"):
        cfg = config.EVAL_CONFIGS[eval_name]
        print(f"\n=== {eval_name}: SAM3 model vs manual ===")
        frames.append(model_eval.evaluate_all_models(eval_name))
        print(f"\n=== {eval_name}: full-48h morphological vs manual ({cfg['filters']}) ===")
        frames.append(full48h_eval.evaluate_all_full48h(eval_name))
    return pd.concat(frames, ignore_index=True)


def summarize(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Pooled (pixel-weighted) and macro (per-pair-mean) sens/spec per group."""
    grp_cols = ["eval", "source", "filter_name"]
    rows = []
    for keys, g in per_pair.groupby(grp_cols, dropna=False):
        tp, fp, fn, tn = (int(g[c].sum()) for c in ("TP", "FP", "FN", "TN"))
        rows.append(
            {
                "eval": keys[0],
                "source": keys[1],
                "filter_name": keys[2],
                "n_pairs": int(len(g)),
                "pooled_sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
                "pooled_specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
                "pooled_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "pooled_IoU": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
                "macro_sensitivity": float(g["sensitivity"].mean()),
                "macro_specificity": float(g["specificity"].mean()),
                "macro_precision": float(g["precision"].mean()),
                "macro_IoU": float(g["IoU"].mean()),
                "event_sensitivity_macro": float(g["event_sensitivity"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["eval", "source", "filter_name"]).reset_index(
        drop=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_pair = run_all()
    summary = summarize(per_pair)

    per_pair_path = args.out_dir / "off_label_method_comparison.per_pair.csv"
    summary_path = args.out_dir / "off_label_method_comparison.summary.csv"
    per_pair.to_csv(per_pair_path, index=False)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n\n===================== SUMMARY =====================")
    print(
        summary[
            [
                "eval",
                "source",
                "filter_name",
                "n_pairs",
                "pooled_sensitivity",
                "pooled_specificity",
                "macro_sensitivity",
                "macro_specificity",
                "pooled_IoU",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {per_pair_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
