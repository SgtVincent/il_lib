import argparse
import os
from pathlib import Path
import re
import sys
import json
import numpy as np
import h5py
import pandas as pd
import torch as th
from tqdm import tqdm
import threading
import queue as _queue
import subprocess
from multiprocessing import Process, Queue

# Ensure omnigibson can be imported if it's in the python path
from omnigibson.learning.utils.obs_utils import (
    ROBOT_CAMERA_NAMES,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    OBS_LOADER_MAP,
    CAMERA_INTRINSICS,
    process_fused_point_cloud,
    makedirs_with_mode,
)

def custom_rgbd_vid_to_pcd(
    data_folder: str,
    task_id: int,
    demo_id: int,
    episode_id: int,
    robot_camera_names=None,
    downsample_ratio: int = 4,
    pcd_range=(-0.2, 1.0, -1.0, 1.0, -0.2, 1.5),
    pcd_num_points: int = 4096,
    batch_size: int = 500,
    use_fps: bool = False,
):
    if robot_camera_names is None:
        robot_camera_names = ROBOT_CAMERA_NAMES["R1Pro"]

    output_dir = os.path.join(data_folder, "pcd_vid", f"task-{task_id:04d}")
    makedirs_with_mode(output_dir)

    # create a new hdf5 file to store the point cloud data
    with h5py.File(f"{output_dir}/episode_{demo_id:08d}.hdf5", "w") as out_f:
        in_f = pd.read_parquet(
            f"{data_folder}/2025-challenge-demos/data/task-{task_id:04d}/episode_{demo_id:08d}.parquet"
        )
        cam_rel_poses = th.from_numpy(np.array(in_f["observation.cam_rel_poses"].tolist(), dtype=np.float32))
        data_size = cam_rel_poses.shape[0]
        fused_pcd_dset = out_f.create_dataset(
            f"data/demo_{episode_id}/robot_r1::fused_pcd",
            shape=(data_size, pcd_num_points, 6),
            compression="lzf",
        )
        # helper: small prefetch iterator to overlap I/O with processing
        class PrefetchIterator:
            def __init__(self, it, maxsize=1):
                self._it = it
                self._queue = _queue.Queue(maxsize=maxsize)
                self._sentinel = object()
                self._thread = threading.Thread(target=self._prefetch)
                self._thread.daemon = True
                self._thread.start()

            def _prefetch(self):
                try:
                    for item in self._it:
                        self._queue.put(item)
                except Exception:
                    pass
                finally:
                    # signal EOF
                    try:
                        self._queue.put(self._sentinel)
                    except Exception:
                        pass

            def __iter__(self):
                return self

            def __next__(self):
                item = self._queue.get()
                if item is self._sentinel:
                    raise StopIteration
                return item

        # get observation loaders
        obs_loaders = {}
        for camera_id, robot_camera_name in robot_camera_names.items():
            resolution = HEAD_RESOLUTION if camera_id == "head" else WRIST_RESOLUTION
            keys = ["rgb", "depth_linear"]
            for key in keys:
                kwargs = {}
                if key == "seg_semantic_id":
                    with open(
                        f"{data_folder}/2025-challenge-demos/meta/episodes/task-{task_id:04d}/episode_{demo_id:08d}.json"
                    ) as f:
                        kwargs["id_list"] = th.tensor(json.load(f)[f"{robot_camera_name}::unique_ins_ids"])
                obs_loaders[f"{robot_camera_name}::{key}"] = PrefetchIterator(
                    iter(
                        OBS_LOADER_MAP[key](
                            data_path=f"{data_folder}/2025-challenge-demos",
                            task_id=task_id,
                            demo_id=f"{demo_id:08d}",
                            camera_id=camera_id,
                            output_size=(resolution[0] // downsample_ratio, resolution[1] // downsample_ratio),
                            batch_size=batch_size,
                            stride=batch_size,
                            **kwargs,
                        )
                    ),
                    maxsize=1,
                )

        # We batch process every batch_size frames
        # Use tqdm for progress bar
        for i in tqdm(range(0, data_size, batch_size), desc=f"Task {task_id} Demo {demo_id}", leave=False):
            obs = dict()  # to store rgbd and pass into process_fused_point_cloud
            obs["cam_rel_poses"] = cam_rel_poses[i : i + batch_size]
            # get all camera intrinsics
            camera_intrinsics = {}
            for camera_id, robot_camera_name in robot_camera_names.items():
                # Calculate the downsampled camera intrinsics
                camera_intrinsics[robot_camera_name] = (
                    th.from_numpy(CAMERA_INTRINSICS["R1Pro"][camera_id]) / downsample_ratio
                )
                camera_intrinsics[robot_camera_name][-1, -1] = 1.0
                obs[f"{robot_camera_name}::rgb"] = next(obs_loaders[f"{robot_camera_name}::rgb"]).movedim(-3, -1)
                obs[f"{robot_camera_name}::depth_linear"] = next(obs_loaders[f"{robot_camera_name}::depth_linear"])
            # process the fused point cloud
            pcd = process_fused_point_cloud(
                obs=obs,
                camera_intrinsics=camera_intrinsics,
                pcd_range=pcd_range,
                pcd_num_points=pcd_num_points,
                use_fps=use_fps,
                verbose=False, # Disable verbose to avoid spamming
            )
            # handle tuple return (pcd, extras)
            if isinstance(pcd, (tuple, list)):
                pcd_tensor = pcd[0]
            else:
                pcd_tensor = pcd
            fused_pcd_dset[i : i + batch_size] = pcd_tensor.cpu()

    # notify progress (one job finished) via optional progress_queue in globals
    try:
        pq = globals().get("PROGRESS_QUEUE", None)
        if pq is not None and hasattr(pq, "put"):
            pq.put(1)
    except Exception:
        pass

def _process_job(job):
    task_id, demo_id, data_path = job
    try:
        # print(f"Processing Task {task_id}, Demo {demo_id}") # Handled by tqdm
        custom_rgbd_vid_to_pcd(
            data_folder=data_path,
            task_id=task_id,
            demo_id=demo_id,
            episode_id=0,
            pcd_range=(-0.2, 1.5, -1.5, 1.5, 0.2, 1.5),
            batch_size=1000,
            use_fps=True,
        )
        return (task_id, demo_id, None)
    except Exception as e:
        return (task_id, demo_id, str(e))

def main():
    parser = argparse.ArgumentParser(description="Generate colored point clouds for a batch of episodes.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data",
        help="Path to the dataset root (e.g. /path/to/data)",
    )
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=None,
        help="List of task IDs to process. If omitted, defaults to 0-49.",
    )
    parser.add_argument(
        "--max_procs",
        type=int,
        default=1,
        help="Maximum number of parallel processes (defaults to number of CPUs).",
    )
    
    args = parser.parse_args()

    # If no task ids provided, default to range 0-49
    if args.task_ids is None:
        task_ids = list(range(0, 50))
    else:
        task_ids = sorted(list(set(args.task_ids)))

    print(f"Processing task IDs: {task_ids}")
    print(f"Data path: {args.data_path}")

    # Build job list (task_id, demo_id)
    jobs = []
    for task_id in task_ids:
        task_str = f"task-{task_id:04d}"
        episodes_dir = Path(args.data_path) / "2025-challenge-demos" / "meta" / "episodes" / task_str

        if not episodes_dir.exists():
            print(f"Directory not found: {episodes_dir}")
            continue

        episode_files = sorted(list(episodes_dir.glob("episode_*.json")))
        if not episode_files:
            print(f"No episode files found in {episodes_dir}")
            continue

        for episode_file in episode_files:
            match = re.search(r"episode_(\d+).json", episode_file.name)
            if not match:
                continue
            demo_id = int(match.group(1))

            # Check if pcd already exists
            out_pcd = Path(args.data_path) / "pcd_vid" / task_str / f"episode_{demo_id:08d}.hdf5"
            if out_pcd.exists():
                print(f"Skipping Task {task_id}, Demo {demo_id}: pcd already exists at {out_pcd}")
                continue

            jobs.append((task_id, demo_id, args.data_path))

    if not jobs:
        print("No jobs to process. Exiting.")
        return

    print(f"Total jobs to process: {len(jobs)}")

    # Determine max processes
    max_procs = args.max_procs if args.max_procs is not None else os.cpu_count() or 4

    # Run jobs in parallel using per-process workers so we can bind each process to a specific GPU.
    # Partition jobs across `num_workers` processes and assign GPUs round-robin if available.
    num_workers = min(max_procs, len(jobs))

    # Detect available GPUs via nvidia-smi (count GPUs). Fallback to CPU-only if detection fails.
    num_gpus = 0
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        lines = [l for l in smi.stdout.splitlines() if l.strip()]
        num_gpus = len(lines)
    except Exception:
        num_gpus = 0

    # Partition jobs round-robin across workers
    worker_jobs = [[] for _ in range(num_workers)]
    for idx, job in enumerate(jobs):
        worker_jobs[idx % num_workers].append(job)

    progress_queue = Queue()

    # Listener thread to update main progress bar
    from tqdm import tqdm as _tqdm
    progress_bar = _tqdm(total=len(jobs), desc="Jobs")
    errors = []

    def _listener():
        done = 0
        while done < len(jobs):
            item = progress_queue.get()
            if isinstance(item, int):
                progress_bar.update(item)
                done += item
            elif isinstance(item, tuple) and item and item[0] == "err":
                # ('err', task_id, demo_id, err_str)
                _, t, d, err_str = item
                errors.append((t, d, err_str))
                progress_bar.update(1)
                done += 1
            else:
                # generic increment
                progress_bar.update(1)
                done += 1

    listener_thread = threading.Thread(target=_listener, daemon=True)
    listener_thread.start()

    def worker_main(job_list, data_path, gpu_id, worker_id, progress_q):
        # Bind this process to a single GPU (if available)
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # make the progress queue available to helpers via globals (custom function reads it)
        globals()["PROGRESS_QUEUE"] = progress_q
        for job in job_list:
            try:
                task_id, demo_id, _ = job
                custom_rgbd_vid_to_pcd(
                    data_folder=data_path,
                    task_id=task_id,
                    demo_id=demo_id,
                    episode_id=0,
                    pcd_range=(-0.2, 1.5, -1.5, 1.5, 0.2, 1.5),
                    batch_size=1000,
                    use_fps=True,
                )
            except Exception as e:
                # send an error tuple to the listener and continue
                try:
                    try:
                        t_id, d_id, _ = job
                    except Exception:
                        t_id = None
                        d_id = None
                    progress_q.put(("err", t_id, d_id, str(e)))
                except Exception:
                    pass

    processes = []
    for wid in range(num_workers):
        gpu_id = wid % num_gpus if num_gpus > 0 else None
        p = Process(target=worker_main, args=(worker_jobs[wid], args.data_path, gpu_id, wid, progress_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # wait for listener to consume all progress updates
    listener_thread.join()

    progress_bar.close()

    if errors:
        print("The following jobs failed:")
        for t, d, err in errors:
            print(f"Task {t}, Demo {d}: {err}")

if __name__ == "__main__":
    main()
