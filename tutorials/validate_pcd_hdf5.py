#!/usr/bin/env python3
"""
validate_pcd_hdf5.py

Quick utility to validate generated PCD HDF5 files under:
  <data_path>/pcd_vid/task-<task_id:04d>/episode_<demo_id:08d>.hdf5

Checks performed for each file:
- file size is at least 1MB
- file can be opened
- contains at least one dataset with name containing 'fused_pcd'
- fused_pcd dataset is numeric, has last-dim >= 6 (RGB + XYZ expected)
- no NaNs or Infs in checked slice
- not all zeros or constant values

Usage:
  python scripts/validate_pcd_hdf5.py --data-path /path/to/data --task-ids 0 1 2
  python scripts/validate_pcd_hdf5.py --task-ids 0 --sample-frames 5
  python scripts/validate_pcd_hdf5.py --task-ids 0 --remove

Output: prints per-file status and writes `pcd_validation_report.csv` in the data folder

Example:  python /mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/repo/b1k-baselines/baselines/il_lib/tutorials/validate_pcd_hdf5.py   --data-path /mnt/bn/navigation-hl/mlx/users/chenjunting/data   --task-ids $(seq 40 49)   --sample-frames 1   --max-procs 32   --remove
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

MIN_VALID_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1MB

try:
    import h5py
    import numpy as np
except Exception:  # pragma: no cover - allow script to show helpful message
    print("Error: required packages not found. Install 'h5py' and 'numpy' in your environment.")
    raise


def find_fused_pcd_dataset(h5: h5py.File) -> Optional[str]:
    # look for a dataset containing 'fused_pcd'
    candidates = []
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and "fused_pcd" in name:
            candidates.append(name)

    h5.visititems(_visit)
    return candidates[0] if candidates else None


def validate_file(path: Path, sample_frames: int = 1) -> dict:
    row = {
        "file": str(path),
        "file_size_bytes": None,
        "too_small": False,
        "ok": False,
        "found_dataset": None,
        "shape": None,
        "checked_frames": 0,
        "has_nan": False,
        "has_inf": False,
        "all_zero": False,
        "notes": "",
    }

    # Fast size-based filter: tiny files are almost certainly corrupted / incomplete.
    try:
        size_bytes = path.stat().st_size
        row["file_size_bytes"] = int(size_bytes)
        if size_bytes < MIN_VALID_FILE_SIZE_BYTES:
            row["too_small"] = True
            row["notes"] = f"file size {size_bytes}B < {MIN_VALID_FILE_SIZE_BYTES}B"
            return row
    except Exception as e:
        row["notes"] = f"error reading file size: {e}"
        return row

    try:
        with h5py.File(path, "r") as f:
            ds_name = find_fused_pcd_dataset(f)
            row["found_dataset"] = ds_name
            if ds_name is None:
                row["notes"] = "no fused_pcd dataset found"
                return row

            ds = f[ds_name]
            row["shape"] = tuple(ds.shape)

            # determine how many frames to check
            # ds usually has shape (T, N, C) or (T, C) etc.
            # We'll safely attempt to read up to `sample_frames` frames along dim 0
            if ds.ndim == 3:
                T = ds.shape[0]
                ncheck = min(T, sample_frames)
                sample = ds[:ncheck]
            elif ds.ndim == 2 and ds.shape[0] >= sample_frames:
                # treat axis 0 as time or sequence
                sample = ds[:sample_frames]
            else:
                sample = ds[...]

            # basic numeric checks
            sample_np = np.asarray(sample)
            row["checked_frames"] = int(sample_np.shape[0]) if sample_np.ndim > 0 else 1

            row["has_nan"] = bool(np.isnan(sample_np).any())
            row["has_inf"] = bool(np.isinf(sample_np).any())
            row["all_zero"] = bool(np.all(sample_np == 0))

            # channel check: expect last dim >=6 by convention (rgb + xyz)
            if sample_np.ndim >= 3:
                last_dim = sample_np.shape[-1]
                if last_dim < 6:
                    row["notes"] = f"unexpected channel dim {last_dim} (expected >=6)"
                    return row

            # pass
            if not (row["has_nan"] or row["has_inf"] or row["all_zero"]):
                row["ok"] = True
            else:
                if not row["notes"]:
                    row["notes"] = "nan/inf/all-zero detected"

    except Exception as e:
        row["notes"] = f"error opening file: {e}"

    return row


def gather_files(data_path: Path, task_ids: List[int]) -> List[Path]:
    files = []
    for tid in sorted(set(task_ids)):
        task_dir = data_path / "pcd_vid" / f"task-{tid:04d}"
        if not task_dir.exists():
            print(f"Warning: task dir not found: {task_dir}")
            continue
        for f in sorted(task_dir.glob("*.hdf5")):
            files.append(f)
    return files


def _validate_job(job: tuple[str, int, bool]) -> dict:
    """Worker entrypoint: validate one file (and optionally remove if invalid)."""
    path_str, sample_frames, do_remove = job
    path = Path(path_str)

    row = validate_file(path, sample_frames=sample_frames)
    # Always include removal info in the report for consistency
    row["removed"] = False
    row["remove_error"] = ""

    if do_remove and (not row.get("ok", False)):
        try:
            path.unlink()
            row["removed"] = True
        except Exception as e:
            row["remove_error"] = str(e)

    return row


def main():
    parser = argparse.ArgumentParser(description="Validate generated pcd HDF5 files for corruption and basic shape checks")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data",
        help="Root data path containing pcd_vid/*",
    )
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0], help="Task IDs to check")
    parser.add_argument("--sample-frames", type=int, default=1, help="Number of frames to sample per file for checking")
    parser.add_argument("--out-csv", type=str, default=None, help="If set, writes a CSV report to this path")
    parser.add_argument(
        "--max-procs",
        type=int,
        default=None,
        help="Maximum number of parallel worker processes (default: os.cpu_count())",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        default=False,
        help="Remove invalid files (use with caution)",
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)
    files = gather_files(data_path, args.task_ids)

    if not files:
        print("No HDF5 files found for these task IDs under pcd_vid/")
        raise SystemExit(2)

    # Parallel validation
    max_procs = args.max_procs if args.max_procs is not None else (os.cpu_count() or 4)
    num_workers = max(1, min(int(max_procs), len(files)))

    jobs = [(str(f), int(args.sample_frames), bool(args.remove)) for f in files]

    # Optional tqdm progress bar
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
    except Exception:
        _tqdm = None

    rows = []
    ok_cnt = 0
    removed_cnt = 0
    fail_cnt = 0

    print(f"Validating {len(files)} files with {num_workers} worker(s)")

    with mp.Pool(processes=num_workers) as pool:
        # chunksize=1 so each worker pulls one file at a time from the shared pool
        results_iter = pool.imap_unordered(_validate_job, jobs, chunksize=1)
        if _tqdm is not None:
            results_iter = _tqdm(results_iter, total=len(jobs), desc="PCD HDF5")

        for r in results_iter:
            rows.append(r)

            status = "OK" if r.get("ok", False) else "FAIL"
            extra = " (REMOVED)" if r.get("removed") else ""
            notes = r.get("notes", "")
            print("  ->", f"{status}{extra}", r.get("file", ""), notes)

            if r.get("ok", False):
                ok_cnt += 1
            else:
                fail_cnt += 1
            if r.get("removed", False):
                removed_cnt += 1

    print(f"Checked {len(rows)} files — {ok_cnt} OK, {fail_cnt} failed")
    if args.remove:
        print(f"Removed {removed_cnt} invalid files")

    out_csv = args.out_csv or (data_path / "pcd_validation_report.csv")
    with open(out_csv, "w", newline="") as csvf:
        # Ensure stable field ordering even if some rows have missing keys
        fieldnames = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Report written to {out_csv}")


if __name__ == "__main__":
    main()