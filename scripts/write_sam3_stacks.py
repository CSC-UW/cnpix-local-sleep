"""Generate SAM3 OFF-annotation image stacks for the NOD cohort.

Script-based replacement for ``notebooks/sam3/write_ome_off_stacks.ipynb``. For
each ``(subject, probe)`` in the cohort and each requested condition it writes an
OME-Zarr image stack (AP / spikes / LFP / structure borders) to

    <offproj>/<experiment>/{subject}/method=sam3/probe={probe}/
        condition={condition}/off_stacks.ome.zarr
        condition={condition}/timestamps.zarr

via :func:`cnpix_local_sleep.stacks.write.do_subject_probe` (which, after the writer was
realigned with the ``method=sam3`` migration, targets exactly the layout that
``cnpix_local_sleep.evaluation`` and the OffViewer read from). Consumers download
these OME-Zarr directories directly; no packaged copies are produced.

These stacks are what get manually annotated in napari for SAM3 finetuning /
evaluation. The whole-recording ``processed_ap.zarr`` (v1) is condition-agnostic
on disk -- the condition only selects timepoints via its hypnogram -- so every
condition reuses the same preprocessed inputs that produced the existing
``Early.REC.NREM`` / ``Late.NOD.Wake`` stacks. No new preprocessing is required.

The cohort defaults to ``get_subject_probe_list(method="annotation-grid")``,
which is exactly the 26 ``(subject, probe)`` pairs that already carry
``Early.REC.NREM`` + ``Late.NOD.Wake`` stacks. Existing stacks are skipped unless ``--overwrite`` is passed,
so the script is safe to re-run and resume.

Run through the workspace venv so editable sibling packages are used:

Examples
--------
    # Dry run: show every (pair, condition) target and what would happen.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --dry-run

    # Default: every annotation condition across the whole cohort; existing
    # stacks are skipped, so this is a no-op resume when all exist.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py

    # Just the most important condition.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --conditions Early.REC.NREM.Match

    # One subject, one probe, quick 3-chunk smoke test.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --subjects CNPIX15-Claude --probes imec0 \
        --conditions Early.REC.NREM.Match --max-chunks 3
"""

from __future__ import annotations

import argparse
import time
import traceback

import cnpix_local_sleep.stacks.files as stk_files
from cnpix_local_sleep.sps_conf import get_subject_probe_list
from cnpix_local_sleep.stacks import write

# Conditions for stack generation, most-important first. All are valid keys of
# ``load_statistical_condition_hypnograms``; the stack is stored under
# ``condition=<key>`` and holds exactly that hypnogram's timepoints. NOD
# annotation stacks are the wake-only windows (``*.NOD.Wake``), never the mixed
# ``Early.NOD``/``Late.NOD`` (cnpix-local-sleep/docs/reports/2026-09-15_sam3_stack_condition_audit.md).
DEFAULT_CONDITIONS: tuple[str, ...] = (
    "Early.REC.NREM.Match",
    "Late.REC.NREM",
    "Early.BSL.NREM",
    "Early.NOD.Wake",
    "Early.REC.NREM",
    "Late.NOD.Wake",
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Conditions to generate (default: %(default)s).",
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Restrict to these subjects (exact match). Default: whole cohort.",
    )
    p.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help="Restrict to these probes (e.g. imec0). Default: all probes.",
    )
    p.add_argument(
        "--method",
        default="annotation-grid",
        help="Inclusion method for the cohort pair list (default: %(default)s; "
        "matches the 26 pairs that already have stacks).",
    )
    p.add_argument(
        "--ap-type",
        default="v1",
        choices=["v1", "v3"],
        help="Preprocessed AP version to read (default: %(default)s).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite stacks that already exist (default: skip them).",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap chunks per stack (smoke-testing only; produces partial stacks).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the (pair, condition) targets and exit without writing.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    pairs = get_subject_probe_list(method=args.method)
    if args.subjects is not None:
        pairs = [(s, p) for (s, p) in pairs if s in set(args.subjects)]
    if args.probes is not None:
        pairs = [(s, p) for (s, p) in pairs if p in set(args.probes)]

    if not pairs:
        raise SystemExit("No (subject, probe) pairs matched the given filters.")

    n_targets = len(pairs) * len(args.conditions)
    print(
        f"Cohort: {len(pairs)} pair(s) x {len(args.conditions)} condition(s) "
        f"= {n_targets} target(s)"
    )
    print(f"Conditions (in order): {args.conditions}")
    print("Stacks written under: method=sam3/probe=<probe>/condition=<condition>/")
    print()

    if args.dry_run:
        for condition in args.conditions:
            for subject, probe in pairs:
                savedir = stk_files.get_sam3_savedir_path(
                    subject, probe, condition, None
                )
                exists = savedir.exists() and any(savedir.iterdir())
                state = "EXISTS (would skip)" if exists else "would write"
                if args.overwrite and exists:
                    state = "EXISTS (would overwrite)"
                print(f"  {condition:22s} {subject}/{probe}: {state} -> {savedir}")
        print("\nDry run complete; nothing written.")
        return

    # Outer loop over conditions so the highest-priority condition finishes for
    # the whole cohort before the next one begins.
    statuses: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str, str, str]] = []
    t0 = time.time()
    for condition in args.conditions:
        for subject, probe in pairs:
            tag = f"{subject}/{probe}/{condition}"
            start = time.time()
            try:
                print(f"Starting stack: {tag}")
                write.do_subject_probe(
                    subject=subject,
                    probe=probe,
                    condition=condition,
                    structure_acronym=None,
                    ap_type=args.ap_type,
                    overwrite=args.overwrite,
                    max_chunks=args.max_chunks,
                )
                elapsed = time.time() - start
                statuses.append((condition, subject, probe))
                print(f"Completed {tag} in {elapsed:.1f}s\n")
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                failures.append((condition, subject, probe, repr(exc)))
                print(f"FAILED {tag}: {exc}")
                traceback.print_exc()
                print()

    total = time.time() - t0
    print("=" * 70)
    print(f"Done in {total / 60:.1f} min. "
          f"{len(statuses)} ok, {len(failures)} failed.")
    if failures:
        print("\nFailures:")
        for condition, subject, probe, err in failures:
            print(f"  {condition} {subject}/{probe}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
