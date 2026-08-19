"""Generate SAM3 OFF-annotation image stacks (+ tarballs) for the NOD cohort.

Script-based replacement for ``notebooks/sam3/write_ome_off_stacks.ipynb``. For
each ``(subject, probe)`` in the cohort and each requested condition it writes an
OME-Zarr image stack (AP / spikes / LFP / structure borders) to

    <offproj>/<experiment>/{subject}/method=sam3/probe={probe}/
        condition={condition}/off_stacks.ome.zarr
        condition={condition}/timestamps.zarr

via :func:`cnpix_local_sleep.stacks.write.do_subject_probe` (which, after the writer was
realigned with the ``method=sam3`` migration, targets exactly the layout that
``cnpix_local_sleep.evaluation`` and the OffViewer read from), then optionally tars each
``condition=`` directory into

    <offproj>/<experiment>/sam_stack_tarballs/{subject}_{probe}_{condition}.tar.gz

These stacks are what get manually annotated in napari for SAM3 finetuning /
evaluation. The whole-recording ``processed_ap.zarr`` (v1) is condition-agnostic
on disk -- the condition only selects timepoints via its hypnogram -- so every
condition reuses the same preprocessed inputs that produced the existing
``Early.REC.NREM`` / ``Late.NOD`` stacks. No new preprocessing is required.

The cohort defaults to ``get_subject_probe_list(method="annotation-grid")``,
which is exactly the 26 ``(subject, probe)`` pairs that already carry
``Early.REC.NREM`` + ``Late.NOD`` stacks. Existing stacks are skipped unless ``--overwrite`` is passed,
so the script is safe to re-run and resume.

Run through the workspace venv so editable sibling packages are used:

Examples
--------
    # Dry run: show every (pair, condition) target and what would happen.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --dry-run

    # Default: the four requested conditions across the whole cohort, tarred.
    # Early.REC.NREM.Match is generated first (highest priority).
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py

    # Just the most important condition.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --conditions Early.REC.NREM.Match

    # One subject, one probe, no tarballs, quick 3-chunk smoke test.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --subjects CNPIX15-Claude --probes imec0 \
        --conditions Early.REC.NREM.Match --no-tar --max-chunks 3

    # Only (re)build tarballs from stacks that already exist on disk.
    uv run --project gfys_workspace python cnpix-local-sleep/scripts/write_sam3_stacks.py \
        --tar-only
"""

from __future__ import annotations

import argparse
import subprocess
import time
import traceback

import wisc_ecephys_tools as wet

import cnpix_local_sleep as op
import cnpix_local_sleep.stacks.files as stk_files
from cnpix_local_sleep.sps_conf import get_subject_probe_list
from cnpix_local_sleep.stacks import write

# The four conditions requested for new stack generation, most-important first.
# All are valid keys of ``load_statistical_condition_hypnograms`` and, like the
# existing ``Late.NOD`` stacks, are stored under a ``condition=<name>`` directory
# that matches the hypnogram key verbatim.
DEFAULT_CONDITIONS: tuple[str, ...] = (
    "Early.REC.NREM.Match",
    "Late.REC.NREM",
    "Early.BSL.NREM",
    "Early.NOD",
)


def get_tarball_dir():
    """Directory holding ``{subject}_{probe}_{condition}.tar.gz`` shipping tarballs."""
    return (
        wet.get_sglx_project("offproj").get_experiment_directory(op.EXPERIMENT)
        / "sam_stack_tarballs"
    )


def tar_stack(subject: str, probe: str, condition: str, overwrite: bool) -> str:
    """Tar the ``method=sam3`` ``condition=`` stack directory for one recording.

    Returns a short status string: ``"tarred"``, ``"tar-skip-exists"``, or
    ``"tar-skip-no-stack"``.
    """
    savedir = stk_files.get_sam3_savedir_path(subject, probe, condition, None)
    if not savedir.exists() or not any(savedir.iterdir()):
        print(f"  [tar] no stack at {savedir}; skipping tarball")
        return "tar-skip-no-stack"

    tarball = get_tarball_dir() / f"{subject}_{probe}_{condition}.tar.gz"
    tarball.parent.mkdir(parents=True, exist_ok=True)
    if tarball.exists() and not overwrite:
        print(f"  [tar] exists, skipping: {tarball}")
        return "tar-skip-exists"

    # ``-C savedir.parent`` so the archive contains ``condition=<condition>/...``.
    subprocess.run(
        ["tar", "-czf", str(tarball), "-C", str(savedir.parent), savedir.name],
        check=True,
    )
    print(f"  [tar] created {tarball}")
    return "tarred"


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
        help="Rewrite stacks/tarballs that already exist (default: skip them).",
    )
    p.add_argument(
        "--no-tar",
        action="store_true",
        help="Write stacks only; do not create tarballs.",
    )
    p.add_argument(
        "--tar-only",
        action="store_true",
        help="Do not write stacks; only tar stacks that already exist on disk.",
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
    if not args.no_tar and not args.dry_run:
        print(f"Tarballs written under: {get_tarball_dir()}")
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
    statuses: list[tuple[str, str, str, str]] = []
    failures: list[tuple[str, str, str, str]] = []
    t0 = time.time()
    for condition in args.conditions:
        for subject, probe in pairs:
            tag = f"{subject}/{probe}/{condition}"
            start = time.time()
            try:
                if not args.tar_only:
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
                tar_status = (
                    "tar-skipped"
                    if args.no_tar
                    else tar_stack(subject, probe, condition, args.overwrite)
                )
                elapsed = time.time() - start
                statuses.append((condition, subject, probe, tar_status))
                print(f"Completed {tag} ({tar_status}) in {elapsed:.1f}s\n")
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
