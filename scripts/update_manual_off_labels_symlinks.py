"""Maintain ``manual_off_labels.npz`` -> highest-version symlinks under offproj_s3.

For every directory under
``<offproj_s3>/<experiment>/{subject}/probe={probe}/condition={condition}/``
that contains one or more ``manual_off_labels_v{N}.npz`` files, this script
ensures a sibling ``manual_off_labels.npz`` symlink (relative target) points to
the highest-N version present.

Behavior per directory:

- No ``v*`` files -> skip.
- Symlink missing -> create.
- Symlink present and points to the right target -> no-op.
- Symlink present but points to an older version -> retarget atomically
  (``unlink`` + ``symlink_to``).
- A *regular file* (not a symlink) named ``manual_off_labels.npz`` is present
  -> refuse. Print a warning and leave it alone, on the theory that real label
  data may have been placed there by hand.

Usage::

    python update_manual_off_labels_symlinks.py            # dry-run by default
    python update_manual_off_labels_symlinks.py --apply    # actually create/update

Idempotent: re-running on an already-current tree produces only ``OK`` rows.
This is a maintenance utility, not a one-shot migration; it should remain in
``cnpix-local-sleep/scripts/`` and be re-run after new manual-label versions are added.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import wisc_ecephys_tools as wet

from cnpix_local_sleep import const


VERSION_RE = re.compile(r"^manual_off_labels_v(\d+)\.npz$")
SYMLINK_NAME = "manual_off_labels.npz"


def get_experiment_root() -> Path:
    return wet.get_sglx_project("offproj_s3").get_experiment_directory(
        const.EXPERIMENT
    )


def find_label_dirs(root: Path) -> dict[Path, list[tuple[int, str]]]:
    """Return {dir: sorted [(version, filename), ...]} for every dir that has any v*.npz."""
    by_dir: dict[Path, list[tuple[int, str]]] = {}
    for p in root.glob("*/probe=*/condition=*/manual_off_labels_v*.npz"):
        if p.is_symlink():
            continue
        m = VERSION_RE.match(p.name)
        if m is None:
            continue
        version = int(m.group(1))
        by_dir.setdefault(p.parent, []).append((version, p.name))
    for entries in by_dir.values():
        entries.sort()
    return by_dir


def reconcile_dir(
    dir_path: Path,
    versions: list[tuple[int, str]],
    *,
    dry_run: bool,
) -> str:
    """Return a short status string describing the action taken (or planned)."""
    target_version, target_name = versions[-1]
    link_path = dir_path / SYMLINK_NAME

    if link_path.is_symlink():
        current_target = Path(link_path.readlink()).name  # readlink() value, parsed leaf
        if current_target == target_name:
            return f"OK   {link_path} -> {target_name}"
        action = f"UPDATE {link_path} : {current_target} -> {target_name}"
        if dry_run:
            return f"WOULD {action}"
        link_path.unlink()
        link_path.symlink_to(target_name)
        return action

    if link_path.exists():
        # Regular file (or directory): do not clobber.
        return f"SKIP (regular file at {link_path})"

    # Does not exist: create.
    action = f"CREATE {link_path} -> {target_name}"
    if dry_run:
        return f"WOULD {action}"
    link_path.symlink_to(target_name)
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create/update symlinks. Without this flag the script runs in dry-run mode.",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    root = get_experiment_root()
    if not root.is_dir():
        raise SystemExit(f"Experiment root does not exist: {root}")

    print(f"Experiment root: {root}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}\n")

    by_dir = find_label_dirs(root)

    counts = {"created": 0, "updated": 0, "ok": 0, "skipped": 0}
    for dir_path in sorted(by_dir):
        result = reconcile_dir(dir_path, by_dir[dir_path], dry_run=dry_run)
        print(f"  {result}")
        if result.startswith("OK"):
            counts["ok"] += 1
        elif "SKIP" in result:
            counts["skipped"] += 1
        elif "CREATE" in result:
            counts["created"] += 1
        elif "UPDATE" in result:
            counts["updated"] += 1

    print(
        f"\nSummary over {len(by_dir)} dir(s) with manual labels: "
        f"{counts['created']} created, {counts['updated']} updated, "
        f"{counts['ok']} already current, {counts['skipped']} skipped."
    )


if __name__ == "__main__":
    main()
