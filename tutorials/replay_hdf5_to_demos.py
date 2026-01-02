#!/usr/bin/env python3
"""Replay a single BEHAVIOR-1K raw HDF5 into 2025-challenge-demos outputs.

This is a thin wrapper around BEHAVIOR-1K's off-the-shelf replay script:
  BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py

Why this wrapper exists:
- The official replay script expects the raw file to already live at:
    <data_folder>/2025-challenge-rawdata/task-XXXX/episode_YYYYYYYY.hdf5
- In practice you often have a raw .hdf5 somewhere else (private dumps, scratch dirs).

This wrapper stages (symlink/copy) the raw file into the expected layout, then runs replay.

Run this inside the `behavior` conda env.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_BEHAVIOR1K_ROOT_CANDIDATES = [
    Path(os.environ.get("BEHAVIOR_1K_ROOT", "")) if os.environ.get("BEHAVIOR_1K_ROOT") else None,
    Path("/home/ubuntu/ove_repo/BEHAVIOR-1K"),
]


def _resolve_behavior1k_root(user_value: str | None) -> Path:
    if user_value:
        p = Path(user_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--behavior1k_root does not exist: {p}")
        return p

    for candidate in DEFAULT_BEHAVIOR1K_ROOT_CANDIDATES:
        if candidate is None:
            continue
        try:
            p = candidate.expanduser().resolve()
        except Exception:
            continue
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not auto-detect BEHAVIOR-1K repo root. "
        "Pass --behavior1k_root or set BEHAVIOR_1K_ROOT."
    )


def _resolve_replay_obs_py(behavior1k_root: Path, user_value: str | None) -> Path:
    if user_value:
        p = Path(user_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--replay_obs_py not found: {p}")
        return p

    p = behavior1k_root / "OmniGibson" / "scripts" / "learning" / "replay_obs.py"
    if not p.exists():
        raise FileNotFoundError(f"Expected replay script not found at: {p}")
    return p


def _import_task_id(task_name: str) -> int:
    try:
        from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
    except Exception as e:
        raise RuntimeError(
            "Failed to import omnigibson. Make sure you are in the correct environment (e.g. `conda activate behavior`) "
            "and OmniGibson is installed."
        ) from e

    if task_name not in TASK_NAMES_TO_INDICES:
        raise KeyError(
            f"Unknown task_name={task_name!r}. If this is a new task, update the mapping in "
            "omnigibson.learning.utils.eval_utils.TASK_NAMES_TO_INDICES."
        )

    return int(TASK_NAMES_TO_INDICES[task_name])


def _safe_symlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        return True
    except Exception:
        return False


def _stage_raw_hdf5(
    raw_hdf5: Path,
    data_folder: Path,
    task_id: int,
    demo_id: int,
    stage_mode: str,
    overwrite: bool,
) -> Path:
    raw_hdf5 = raw_hdf5.expanduser().resolve()
    if not raw_hdf5.exists():
        raise FileNotFoundError(f"raw HDF5 not found: {raw_hdf5}")

    dst = data_folder / "2025-challenge-rawdata" / f"task-{task_id:04d}" / f"episode_{demo_id:08d}.hdf5"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return dst
        dst.unlink()

    if stage_mode == "symlink":
        ok = _safe_symlink(raw_hdf5, dst)
        if ok:
            return dst
        # Fall back to copying if symlink fails (e.g. perms, filesystem constraints)
        shutil.copy2(raw_hdf5, dst)
        return dst

    if stage_mode == "copy":
        shutil.copy2(raw_hdf5, dst)
        return dst

    if stage_mode == "hardlink":
        try:
            os.link(raw_hdf5, dst)
        except Exception:
            shutil.copy2(raw_hdf5, dst)
        return dst

    raise ValueError(f"Unknown --stage_mode: {stage_mode}")


def _expected_outputs(data_folder: Path, task_id: int, demo_id: int, want_rgbd: bool, want_seg: bool, want_low_dim: bool):
    expected: list[Path] = []

    if want_rgbd or want_seg:
        try:
            from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
        except Exception:
            ROBOT_CAMERA_NAMES = {"R1Pro": {"left_wrist": "", "right_wrist": "", "head": ""}}

        camera_ids = list(ROBOT_CAMERA_NAMES.get("R1Pro", {}).keys()) or ["left_wrist", "right_wrist", "head"]

        if want_rgbd:
            for cam in camera_ids:
                expected.append(
                    data_folder
                    / "2025-challenge-demos"
                    / "videos"
                    / f"task-{task_id:04d}"
                    / f"observation.images.rgb.{cam}"
                    / f"episode_{demo_id:08d}.mp4"
                )
                expected.append(
                    data_folder
                    / "2025-challenge-demos"
                    / "videos"
                    / f"task-{task_id:04d}"
                    / f"observation.images.depth.{cam}"
                    / f"episode_{demo_id:08d}.mp4"
                )

        if want_seg:
            for cam in camera_ids:
                expected.append(
                    data_folder
                    / "2025-challenge-demos"
                    / "videos"
                    / f"task-{task_id:04d}"
                    / f"observation.images.seg_instance_id.{cam}"
                    / f"episode_{demo_id:08d}.mp4"
                )

    if want_low_dim:
        expected.append(
            data_folder
            / "2025-challenge-demos"
            / "data"
            / f"task-{task_id:04d}"
            / f"episode_{demo_id:08d}.parquet"
        )
        expected.append(
            data_folder
            / "2025-challenge-demos"
            / "meta"
            / "episodes"
            / f"task-{task_id:04d}"
            / f"episode_{demo_id:08d}.json"
        )

    return expected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage a raw HDF5 into 2025-challenge-rawdata layout and run BEHAVIOR-1K replay_obs.py"
    )
    parser.add_argument("--raw_hdf5", type=str, required=True, help="Path to the input raw .hdf5")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (must be in OmniGibson mapping)")
    parser.add_argument("--demo_id", type=int, required=True, help="Demo ID (episode index to write)")
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Output root folder that will contain 2025-challenge-rawdata/ and 2025-challenge-demos/",
    )

    parser.add_argument("--behavior1k_root", type=str, default=None, help="Path to BEHAVIOR-1K repo root")
    parser.add_argument("--replay_obs_py", type=str, default=None, help="Path to replay_obs.py (optional override)")

    parser.add_argument(
        "--omnigibson_data_path",
        type=str,
        default=None,
        help="Sets OMNIGIBSON_DATA_PATH for the replay subprocess (should contain 2025-challenge-task-instances/)",
    )

    parser.add_argument(
        "--stage_mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="How to stage raw into 2025-challenge-rawdata (symlink recommended)",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing staged raw + outputs (default: true)",
    )

    parser.add_argument(
        "--skip_if_complete",
        action="store_true",
        help="Skip running replay if expected outputs already exist",
    )

    # What to generate (defaults match 'as in 2025-challenge-demos')
    parser.add_argument("--rgbd", action=argparse.BooleanOptionalAction, default=True, help="Generate rgb+depth videos")
    parser.add_argument("--seg", action=argparse.BooleanOptionalAction, default=True, help="Generate seg_instance_id videos")
    parser.add_argument("--low_dim", action=argparse.BooleanOptionalAction, default=True, help="Generate parquet + per-episode meta")

    # Extra derivatives
    parser.add_argument("--bbox", action="store_true", help="Generate bbox videos (head camera only; requires rgbd+seg)")
    parser.add_argument("--pcd_gt", action="store_true", help="Generate fused point cloud from ground-truth RGBD")
    parser.add_argument("--pcd_vid", action="store_true", help="Generate fused point cloud from RGBD videos")
    parser.add_argument(
        "--offline_rgbd",
        action="store_true",
        help="Encode RGBD videos offline after replaying (sometimes more stable)",
    )

    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set OMNIGIBSON_HEADLESS=1 for the replay subprocess (default: true)",
    )

    args = parser.parse_args()

    behavior1k_root = _resolve_behavior1k_root(args.behavior1k_root)
    replay_obs_py = _resolve_replay_obs_py(behavior1k_root, args.replay_obs_py)

    data_folder = Path(args.data_folder).expanduser().resolve()
    data_folder.mkdir(parents=True, exist_ok=True)

    task_id = _import_task_id(args.task_name)

    expected = _expected_outputs(
        data_folder=data_folder,
        task_id=task_id,
        demo_id=args.demo_id,
        want_rgbd=args.rgbd,
        want_seg=args.seg,
        want_low_dim=args.low_dim,
    )

    if args.skip_if_complete and expected and all(p.exists() for p in expected):
        print(f"[skip] outputs already exist for task-{task_id:04d} episode_{args.demo_id:08d}")
        return 0

    staged = _stage_raw_hdf5(
        raw_hdf5=Path(args.raw_hdf5),
        data_folder=data_folder,
        task_id=task_id,
        demo_id=args.demo_id,
        stage_mode=args.stage_mode,
        overwrite=args.overwrite,
    )
    print(f"[stage] {staged}")

    cmd = [
        sys.executable,
        str(replay_obs_py),
        "--data_folder",
        str(data_folder),
        "--task_name",
        args.task_name,
        "--demo_id",
        str(args.demo_id),
    ]
    if args.low_dim:
        cmd.append("--low_dim")
    if args.rgbd:
        cmd.append("--rgbd")
    if args.seg:
        cmd.append("--seg")
    if args.bbox:
        cmd.append("--bbox")
    if args.pcd_gt:
        cmd.append("--pcd_gt")
    if args.pcd_vid:
        cmd.append("--pcd_vid")
    if args.offline_rgbd:
        cmd.append("--offline_rgbd")

    env = os.environ.copy()
    if args.headless:
        env.setdefault("OMNIGIBSON_HEADLESS", "1")
    if args.omnigibson_data_path:
        env["OMNIGIBSON_DATA_PATH"] = str(Path(args.omnigibson_data_path).expanduser().resolve())

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    missing = [p for p in expected if not p.exists()]
    if missing:
        print("[warn] replay finished but some expected outputs are missing:")
        for p in missing[:30]:
            print("  -", p)
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")
        return 2

    print("[ok] replay outputs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
