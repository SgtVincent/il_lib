#!/usr/bin/env python3
"""
Find run directories under ROOT_DIR (default: outputs) where a 'ckpt' subfolder
exists but is empty. By default the script performs a dry-run and only prints
what would be removed. Pass --delete to actually remove the run directories.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import re
from typing import Iterable, Tuple


def find_ckpt_dirs(root: Path) -> Iterable[Path]:
    yield from root.rglob("ckpt")


def is_dir_empty(path: Path) -> bool:
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True


def remove_empty_parents(root: Path) -> list[Path]:
    """Remove empty directories under root, excluding root itself.
    Returns the list of directories removed.
    """
    removed: list[Path] = []
    # Walk the tree depth-first, so children are removed before parents
    for d in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir() and d != root:
            try:
                d.rmdir()
                removed.append(d)
            except OSError:
                # Not empty or not removable
                continue
    return removed


def hh_dir_only_trainlog(hh_dir: Path, to_delete_dirs: set[Path]) -> bool:
    """Return True if hh_dir only contains a 'train.log' file and directories
    that are in `to_delete_dirs` (i.e., run directories to be removed).
    """
    has_non_train_file = False
    for child in hh_dir.iterdir():
        if child.is_file():
            if child.name != "train.log":
                has_non_train_file = True
                break
        elif child.is_dir():
            # allow .hydra dir or dirs slated for deletion; other dirs prevent removal
            if child.name == ".hydra":
                continue
            if child not in to_delete_dirs:
                return False
    return (not has_non_train_file)


def remove_hh_dirs_with_only_trainlog(root: Path, to_delete_dirs: set[Path], delete: bool = False, verbose: bool = False) -> list[Path]:
    """Remove or list hh directories that would contain only a train.log after
    removing the `to_delete_dirs` set. Returns list of hh dirs removed (or would be removed).
    """
    removed: list[Path] = []
    # Date dirs are children of root
    hh_pattern = re.compile(r"^\d{2}-\d{2}-\d{2}$")
    for date_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for hh_dir in sorted([h for h in date_dir.iterdir() if h.is_dir() and hh_pattern.match(h.name)]):
            if hh_dir_only_trainlog(hh_dir, to_delete_dirs):
                if delete:
                    try:
                        shutil.rmtree(hh_dir)
                        removed.append(hh_dir)
                    except Exception:
                        continue
                else:
                    removed.append(hh_dir)
    return removed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove run directories with empty ckpt folders"
    )
    parser.add_argument("--root", default="outputs", help="Root outputs directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--delete", action="store_true", help="Actually delete runs (default: dry-run)")
    group.add_argument("--dry-run", action="store_true", help="Explicit dry-run (do not delete). Mutually exclusive with --delete")
    parser.add_argument("--verbose", action="store_true", help="Print more info")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"Root directory '{root}' not found")
        return 2

    empty_ckpts: list[Tuple[Path, Path]] = []
    for ckpt in find_ckpt_dirs(root):
        if not ckpt.is_dir():
            continue
        if is_dir_empty(ckpt):
            empty_ckpts.append((ckpt, ckpt.parent))

    if not empty_ckpts:
        print("No empty ckpt folders found.")

    run_dirs = [run for ckpt, run in empty_ckpts]
    for ckpt, run in empty_ckpts:
        if args.delete:
            print(f"Removing run directory: {run} (empty ckpt at {ckpt})")
            shutil.rmtree(run)
        else:
            print(f"[DRY-RUN] Would remove run directory: {run} (empty ckpt at {ckpt})")

    # Compute which parent directories would become empty after removing
    # these run dirs, and report them.  Also detect hh directories to remove.
    # This block runs both in dry-run and delete mode; actions are gated below.
    run_dirs_set = set(run_dirs)
    simulated_parents: set[Path] = set()

    def would_be_empty_after_removal(p: Path) -> bool:
        # Check if p has any children or files that are NOT going to be removed
        for child in p.iterdir():
            if child.is_dir():
                if child in run_dirs_set:
                    continue
                if child in simulated_parents:
                    continue
                return False
            else:
                # File exists, so p would not be empty
                return False
        return True

    if not args.delete:
        for run in run_dirs:
            p = run.parent
            while p != root and p not in simulated_parents:
                if would_be_empty_after_removal(p):
                    simulated_parents.add(p)
                    p = p.parent
                else:
                    break
        if simulated_parents:
            if args.verbose:
                for p in sorted(simulated_parents):
                    print(f"[DRY-RUN] Would remove parent directory: {p}")
            else:
                print(f"[DRY-RUN] Would remove parent directories: {len(simulated_parents)}")
    # Also simulate or perform removal of hh directories that only contain train.log
    # after removing the planned run dirs.
    to_delete_dirs = set(run_dirs)
    hh_dirs = remove_hh_dirs_with_only_trainlog(root, to_delete_dirs, delete=args.delete, verbose=args.verbose)
    if args.delete:
        if hh_dirs:
            if args.verbose:
                for p in hh_dirs:
                    print(f"Removed hh directory containing only train.log: {p}")
            else:
                print(f"Removed hh directories: {len(hh_dirs)}")
    else:
        if hh_dirs:
            if args.verbose:
                for p in hh_dirs:
                    print(f"[DRY-RUN] Would remove hh directory containing only train.log: {p}")
            else:
                print(f"[DRY-RUN] Would remove hh directories: {len(hh_dirs)}")

    # If we're simulating parents, include the hh_dirs parents in the simulated_parents
    # so we show date dirs that would become empty too.
    if not args.delete and hh_dirs:
        for hh in hh_dirs:
            p = hh.parent
            while p != root and p not in simulated_parents:
                # if p would be empty after hh removal, add it
                try:
                    # if p only contains files allowed or dirs in run_dirs_set then consider it
                    if would_be_empty_after_removal(p):
                        simulated_parents.add(p)
                        p = p.parent
                    else:
                        break
                except Exception:
                    break
        # after including hh parents, print updated simulated_parents if any
        if simulated_parents:
            if args.verbose:
                for p in sorted(simulated_parents):
                    print(f"[DRY-RUN] Would remove parent directory: {p}")
            else:
                print(f"[DRY-RUN] Would remove parent directories: {len(simulated_parents)}")

    print(f"Found empty ckpt folders: {len(empty_ckpts)}")
    if args.delete:
        removed_parents = remove_empty_parents(root)
        print(f"Removed run directories: {len(empty_ckpts)}")
        if removed_parents:
            if args.verbose:
                for p in removed_parents:
                    print(f"Removed empty parent directory: {p}")
            else:
                print(f"Removed additional empty parent directories: {len(removed_parents)}")
    else:
        print("No directories changed (dry-run). Rerun with --delete to remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
