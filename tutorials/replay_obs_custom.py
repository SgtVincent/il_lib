#!/usr/bin/env python3
"""
replay_obs_custom.py

A lightweight variant of `replay_obs.py` adapted to a custom dataset layout where each
episode is stored in its own folder named like:

    <task_name>_<raw_episode_id>/
        <task_name>.hdf5
        <task_name>.json
        running_args.json

Differences vs original:
- Accepts `--demo_id` as a string (either the raw numeric id like "17507..." or the full
  folder name like "attach_a_camera_to_a_tripod_17507...").
- Automatically discovers the HDF5 file inside the episode folder and uses the trailing
  raw episode id as-is for all output filenames (no zero-padding).
- Outputs processed artifacts under `<out_root>/task-<task_id:04d>/...` (default
  out_root is `demos`, created in the current working directory by default).

Usage examples:
  python replay_obs_custom.py --data_folder /path/to/attach_a_camera_to_a_tripod \
      --task_name attach_a_camera_to_a_tripod --demo_id attach_a_camera_to_a_tripod_17507... \
      --rgbd --seg --low_dim

"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Optional, Tuple

# Avoid expensive torch.compile / inductor compilation during dataset generation.
# OmniGibson uses torch.compile in some utility decorators at import time.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

# If running on a headless machine, default OmniGibson to headless mode unless the
# user explicitly set OMNIGIBSON_HEADLESS.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")

import h5py
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import pandas as pd
import torch as th
import torch.nn.functional as F

from omnigibson.envs import DataPlaybackWrapper
from omnigibson.sensors import VisionSensor
from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    TASK_INDICES_TO_NAMES,
    ROBOT_CAMERA_NAMES,
    CAMERA_INTRINSICS,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    process_fused_point_cloud,
    write_video,
    instance_id_to_instance,
    instance_to_bbox,
    rgbd_vid_to_pcd,
    OBS_LOADER_MAP,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

# module logger
log = create_module_logger(module_name="replay_obs_custom")
log.setLevel(20)

# keep small viewer size
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

FLUSH_EVERY_N_STEPS = 500


class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def _process_obs(self, obs, info):
        robot = self.env.robots[0]
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            assert camera_name.split("::")[1] in robot.sensors, f"Camera {camera_name} not found in robot sensors"
            if f"{camera_name}::seg_semantic" in obs:
                # remove seg semantic map (Alternatively, change this line to store seg semantic instead)
                obs.pop(f"{camera_name}::seg_semantic")
            if f"{camera_name}::seg_instance_id" in obs:
                # move seg instance maps to cpu
                obs[f"{camera_name}::seg_instance_id"] = obs[f"{camera_name}::seg_instance_id"].cpu()
            # store camera pose
            cam_pose = robot.sensors[camera_name.split("::")[1]].get_position_orientation()
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        return obs

    def postprocess_traj_group(self, traj_grp):
        log.info(f"Postprocessing trajectory group {traj_grp.name}")
        traj_grp.attrs["robot_type"] = "R1Pro"
        traj_grp.attrs["task_obs_keys"] = self.env.task.low_dim_obs_keys
        traj_grp.attrs["ins_id_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)

        camera_names = set(ROBOT_CAMERA_NAMES["R1Pro"].values())
        for name in self.env.robots[0].sensors:
            if f"robot_r1::{name}" in camera_names:
                unique_ins_ids = set()
                # batch process to avoid memory issues
                for i in range(0, traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"].shape[0], FLUSH_EVERY_N_STEPS):
                    unique_ins_ids.update(
                        th.unique(
                            th.from_numpy(
                                traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"][i : i + FLUSH_EVERY_N_STEPS]
                            )
                        )
                        .to(th.uint32)
                        .tolist()
                    )
                traj_grp.attrs[f"robot_r1::{name}::unique_ins_ids"] = list(unique_ins_ids)
        log.info(f"Postprocessing trajectory group {traj_grp.name} done")


# Helper: locate the custom-style HDF5 inside the dataset folder
def find_episode_hdf5(data_folder: str, task_name: str, demo_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Finds an HDF5 file for the given task and demo identifier.

    Returns (hdf5_path, episode_folder_name) or (None, None) if not found.

    The function supports inputs like:
      demo_id = "1750769844991256"
      demo_id = "attach_a_camera_to_a_tripod_1750769844991256"

    and looks for patterns like:
      <data_folder>/<task_name>_<raw_id>/<any>.hdf5
      <data_folder>/<any_dir_containing_raw_id>/<any>.hdf5
      <data_folder>/*.hdf5 (containing raw id)
    """
    # candidate: if demo_id itself is a folder under data_folder
    candidate_folder = os.path.join(data_folder, demo_id)
    if os.path.isdir(candidate_folder):
        # prefer a file named <task_name>.hdf5 but accept any .hdf5 inside
        task_h5 = os.path.join(candidate_folder, f"{task_name}.hdf5")
        if os.path.exists(task_h5):
            return task_h5, demo_id
        for f in os.listdir(candidate_folder):
            if f.endswith(".hdf5"):
                return os.path.join(candidate_folder, f), demo_id

    # extract the raw trailing id if present
    raw_id = demo_id.split("_")[-1]

    # candidate: <data_folder>/<task_name>_<raw_id>/
    candidate_folder = os.path.join(data_folder, f"{task_name}_{raw_id}")
    if os.path.isdir(candidate_folder):
        for f in os.listdir(candidate_folder):
            if f.endswith(".hdf5"):
                return os.path.join(candidate_folder, f), f"{task_name}_{raw_id}"

    # scan top-level entries for a folder containing raw_id
    for entry in os.listdir(data_folder):
        entry_path = os.path.join(data_folder, entry)
        if os.path.isdir(entry_path) and raw_id in entry:
            for f in os.listdir(entry_path):
                if f.endswith(".hdf5"):
                    return os.path.join(entry_path, f), entry

    # scan for hdf5 files directly in data_folder
    for f in os.listdir(data_folder):
        if f.endswith(".hdf5") and raw_id in f:
            return os.path.join(data_folder, f), raw_id

    return None, None


def resolve_out_root(provided_out_root: str, data_folder: str) -> str:
    """Resolve the final out_root directory.

    Behavior:
    - If the user provided the default value ("demos"), prefer the canonical
      dataset location at `gm.DATA_PATH/2025-challenge-demos` if it exists.
    - Otherwise return an absolute path for the given `provided_out_root`.
    """
    if provided_out_root == "demos":
        candidate = None
        try:
            candidate = os.path.join(gm.DATA_PATH, "2025-challenge-demos")
        except Exception:
            candidate = None
        if candidate and os.path.exists(candidate):
            return candidate
        # fallback to local ./demos
        return os.path.abspath(provided_out_root)
    return provided_out_root if os.path.isabs(provided_out_root) else os.path.abspath(provided_out_root)


def replay_hdf5_file(
    data_folder: str,
    task_id: int,
    demo_id_raw: str,
    input_hdf5_path: Optional[str] = None,
    camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    generate_rgbd: bool = False,
    generate_seg: bool = False,
    generate_bbox: bool = False,
    flush_every_n_steps: int = 500,
    offline_rgbd: bool = False,
    out_root: str = "demos",
) -> int:
    """Replays a single HDF5 file and saves videos + replayed hdf5.

    Differences from original:
    - `demo_id_raw` is kept as a string and used for output filenames: ``episode_<demo_id_raw>.*``
    - `input_hdf5_path` can be provided (recommended for custom dataset). If None, falls back
      to the original path layout (may raise if not found).

    Returns the internal `episode_id` index inside the HDF5 file (demo_{episode_id}).
    """
    if generate_bbox:
        assert generate_rgbd and generate_seg, "Bounding box data requires rgb and segmentation data"

    # where to put the replayed file
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)

    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False

    modalities = []
    if generate_rgbd:
        modalities += ["rgb", "depth_linear"]
    if generate_seg:
        modalities += ["seg_semantic", "seg_instance_id"]

    robot_sensor_config = {
        "VisionSensor": {
            "modalities": modalities,
            "sensor_kwargs": {
                "image_height": WRIST_RESOLUTION[0],
                "image_width": WRIST_RESOLUTION[1],
            },
        },
    }

    additional_wrapper_configs = []

    # try to find the full scene file using same logic as original (by task name)
    task_name = TASK_INDICES_TO_NAMES[task_id]
    task_scene_file_folder = os.path.join(os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name)
    full_scene_file = None
    if os.path.exists(task_scene_file_folder):
        for file in os.listdir(task_scene_file_folder):
            if file.endswith(".json") and "partial_rooms" not in file:
                full_scene_file = os.path.join(task_scene_file_folder, file)
                break
    assert full_scene_file is not None, f"No full scene file found in {task_scene_file_folder}"

    # load optional load_room_instances mapping from a metadata csv (same as original behavior)
    load_room_instances = None
    try:
        with open(f"{gm.DATA_PATH}/2025-challenge-task-instances/metadata/B50_task_misc.csv", newline="", encoding="utf-8") as f:
            task_misc_csv = csv.reader(f, delimiter=",", quotechar='"')
            for row in task_misc_csv:
                if task_name in row[1]:
                    load_room_instances = row[2].strip().split("\n")
                    break
    except FileNotFoundError:
        log.warning("B50_task_misc.csv not found; proceeding without load_room_instances optimization")

    # determine input hdf5 path
    if input_hdf5_path is None:
        # try to resolve using the older format (episode_<08d>.hdf5 under 2025-challenge-rawdata)
        try:
            int_demo = int(demo_id_raw)
            candidate = os.path.join(data_folder, f"2025-challenge-rawdata/task-{task_id:04d}/episode_{int_demo:08d}.hdf5")
            if os.path.exists(candidate):
                input_hdf5_path = candidate
            else:
                raise FileNotFoundError
        except Exception:
            raise FileNotFoundError(
                "Input HDF5 not provided and can't be located under the legacy layout. Provide the episode folder or HDF5 via --data_folder and --demo_id."
            )

    out_replayed_path = os.path.join(replay_dir, f"episode_{demo_id_raw}.hdf5")

    env = BehaviorDataPlaybackWrapper.create_from_hdf5(
        input_path=input_hdf5_path,
        output_path=out_replayed_path,
        compression={"compression": "lzf"},
        robot_obs_modalities=["proprio"],
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=dict(),
        n_render_iterations=3,
        flush_every_n_traj=1,
        flush_every_n_steps=flush_every_n_steps,
        additional_wrapper_configs=additional_wrapper_configs,
        full_scene_file=full_scene_file,
        include_robot_control=False,
        include_contacts=False,
        load_room_instances=load_room_instances,
    )

    # Modify head camera if rgbd generation requested
    if generate_rgbd:
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].horizontal_aperture = 40.0
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_height = HEAD_RESOLUTION[0]
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_width = HEAD_RESOLUTION[1]
        env.load_observation_space()

    # choose the episode index (index inside the HDF5 file)
    num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
    episode_id = num_samples.index(max(num_samples))
    log.info(f" >>> Replaying demo folder '{demo_id_raw}', internal episode index {episode_id}")

    # prepare video writers
    video_writers = dict()
    # Resolve out_root as an absolute path; for relative paths, resolve relative to the current working directory.
    # Do NOT place out_root under data_folder by default.
    if os.path.isabs(out_root):
        demos_root = out_root
    else:
        demos_root = os.path.abspath(out_root)
    makedirs_with_mode(demos_root)
    log.info(f"Outputs will be written to {demos_root}")
    if generate_rgbd:
        for camera_id, camera_name in camera_names.items():
            rgb_dir = os.path.join(demos_root, "videos", f"task-{task_id:04d}", f"observation.images.rgb.{camera_id}")
            depth_dir = os.path.join(demos_root, "videos", f"task-{task_id:04d}", f"observation.images.depth.{camera_id}")
            makedirs_with_mode(rgb_dir)
            makedirs_with_mode(depth_dir)
            resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION
            video_writers[f"{camera_name}::rgb"] = create_video_writer(
                fpath=f"{rgb_dir}/episode_{demo_id_raw}.mp4",
                resolution=resolution,
                codec_name="libx265",
                pix_fmt="yuv420p",
                stream_options={"x265-params": "log-level=none"},
            )
            video_writers[f"{camera_name}::depth_linear"] = create_video_writer(
                fpath=f"{depth_dir}/episode_{demo_id_raw}.mp4",
                resolution=resolution,
                codec_name="libx265",
                pix_fmt="yuv420p10le",
                stream_options={"x265-params": "lossless=1:log-level=none"},
            )

    env.playback_episode(
        episode_id=episode_id,
        record_data=True,
        video_writers=video_writers if not offline_rgbd else None,
    )

    # offline writing of videos if requested
    for camera_id, camera_name in camera_names.items():
        resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION
        if generate_rgbd and offline_rgbd:
            write_video(
                env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"],
                video_writer=video_writers[f"{camera_name}::rgb"],
                batch_size=flush_every_n_steps,
                mode="rgb",
            )
            log.info(f"Saved rgb video for {camera_name}")
            write_video(
                env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::depth_linear"],
                video_writer=video_writers[f"{camera_name}::depth_linear"],
                batch_size=flush_every_n_steps,
                mode="depth",
            )
            log.info(f"Saved depth video for {camera_name}")
        if generate_seg:
            seg_dir = os.path.join(demos_root, "videos", f"task-{task_id:04d}", f"observation.images.seg_instance_id.{camera_id}")
            makedirs_with_mode(seg_dir)
            video_writers[f"{camera_name}::seg_instance_id"] = create_video_writer(
                fpath=f"{seg_dir}/episode_{demo_id_raw}.mp4",
                resolution=resolution,
                codec_name="libx265",
                pix_fmt="yuv420p",
                stream_options={"x265-params": "log-level=none"},
            )
            ins_id_ids = env.hdf5_file[f"data/demo_{episode_id}"].attrs[f"{camera_name}::unique_ins_ids"]
            write_video(
                env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"],
                video_writer=video_writers[f"{camera_name}::seg_instance_id"],
                batch_size=flush_every_n_steps,
                mode="seg",
                seg_ids=ins_id_ids,
            )
            log.info(f"Saved seg video for {camera_name}")
        if generate_bbox:
            if "zed" in camera_name:
                bbox_dir = os.path.join(demos_root, "videos", f"task-{task_id:04d}", f"observation.images.bbox.{camera_id}")
                makedirs_with_mode(bbox_dir)
                video_writers[f"{camera_name}::bbox"] = create_video_writer(
                    fpath=f"{bbox_dir}/episode_{demo_id_raw}.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p",
                    stream_options={"x265-params": "log-level=none"},
                )
                task_relevant_objs = None
                try:
                    with open(f"{gm.DATA_PATH}/2025-challenge-task-instances/metadata/B50_object_instance_ID.csv", newline="", encoding="utf-8") as f:
                        oi_csv = csv.reader(f, delimiter=",", quotechar='"')
                        for row in oi_csv:
                            if row[0] == task_name:
                                task_relevant_objs = row[1] + row[2]
                                break
                except FileNotFoundError:
                    log.error("B50_object_instance_ID.csv not found; cannot generate bbox videos without metadata")
                    raise
                instance_id_mapping = json.loads(env.hdf5_file[f"data/demo_{episode_id}"].attrs["ins_id_mapping"])
                instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                unique_ins_ids = env.hdf5_file[f"data/demo_{episode_id}"].attrs[f"{camera_name}::unique_ins_ids"]
                for i in range(
                    0,
                    env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"].shape[0],
                    flush_every_n_steps,
                ):
                    instance_seg, instance_mapping = instance_id_to_instance(
                        th.from_numpy(
                            env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"][i : i + flush_every_n_steps]
                        ),
                        instance_id_mapping,
                        unique_ins_ids,
                    )
                    instance_mapping = {k: v for k, v in instance_mapping.items() if v in task_relevant_objs}
                    bbox = instance_to_bbox(instance_seg, instance_mapping, set(instance_mapping.keys()))
                    write_video(
                        th.from_numpy(
                            env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"][i : i + flush_every_n_steps]
                        ),
                        video_writer=video_writers[f"{camera_name}::bbox"],
                        batch_size=flush_every_n_steps,
                        mode="bbox",
                        bbox=bbox,
                        instance_mapping=instance_mapping,
                        task_relevant_objects=task_relevant_objs,
                    )
                log.info(f"Saved bbox video for {camera_name}")

    # Close all video writers
    for container, stream in video_writers.values():
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    log.info("Playback complete. Saving data...")
    env.save_data()

    log.info(f"Successfully processed episode_{demo_id_raw}")
    return episode_id


def generate_low_dim_data(
    data_folder: str,
    task_id: int,
    demo_id_raw: str,
    episode_id: int,
    out_root: str = "demos",
):
    """Post-process the replayed low-dimensional data to parquet using the raw episode id in filenames.

    The `out_root` indicates the root folder where demo outputs are written. By default `out_root` is
    resolved relative to the current working directory (not under `data_folder`)."""
    # resolve demos root as absolute path (do NOT place under data_folder by default)
    demos_root = out_root if os.path.isabs(out_root) else os.path.abspath(out_root)
    demos_data_dir = os.path.join(demos_root, "data")
    demos_meta_dir = os.path.join(demos_root, "meta", "episodes")
    makedirs_with_mode(demos_root)
    log.info(f"Low-dim data will be written to {demos_root}")
    makedirs_with_mode(f"{demos_data_dir}/task-{task_id:04d}")
    makedirs_with_mode(f"{demos_meta_dir}/task-{task_id:04d}")
    replayed_path = f"{data_folder}/replayed/episode_{demo_id_raw}.hdf5"
    with h5py.File(replayed_path, "r") as replayed_f:
        actions = np.array(replayed_f["data"][f"demo_{episode_id}"]["action"][:], dtype=np.float32)
        proprio = np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"]["robot_r1::proprio"][:], dtype=np.float32)
        task_info = np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"]["task::low_dim"][:], dtype=np.float32)
        cam_rel_poses = np.array(
            replayed_f["data"][f"demo_{episode_id}"]["obs"]["robot_r1::cam_rel_poses"][:], dtype=np.float32
        )
        assert (
            actions.shape[0] == proprio.shape[0] == task_info.shape[0]
        ), "Action, proprio, and task-info must have the same length"
        T = len(actions)

        # attempt to convert demo_id_raw to int for episode_index if possible
        try:
            demo_index_int = int(demo_id_raw)
        except Exception:
            demo_index_int = 0

        data = {
            "index": np.arange(T, dtype=np.int64),
            "episode_index": np.zeros(T, dtype=np.int64) + demo_index_int,
            "task_index": np.zeros(T, dtype=np.int64) + task_id,
            "timestamp": np.arange(T, dtype=np.float64) / 30.0,
            "observation.state": list(proprio),
            "observation.cam_rel_poses": list(cam_rel_poses),
            "action": list(actions),
            "observation.task_info": list(task_info),
        }
        df = pd.DataFrame(data)
        df.to_parquet(f"{demos_data_dir}/task-{task_id:04d}/episode_{demo_id_raw}.parquet", index=False)

        task_metadata = {}
        for attr_name in replayed_f["data"].attrs:
            if isinstance(replayed_f["data"].attrs[attr_name], np.int64):
                task_metadata[attr_name] = int(replayed_f["data"].attrs[attr_name])
            elif isinstance(replayed_f["data"].attrs[attr_name], np.ndarray):
                task_metadata[attr_name] = replayed_f["data"].attrs[attr_name].tolist()
            else:
                task_metadata[attr_name] = replayed_f["data"].attrs[attr_name]
        for attr_name in replayed_f["data"][f"demo_{episode_id}"].attrs:
            if isinstance(replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name], np.int64):
                task_metadata[attr_name] = int(replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name])
            elif isinstance(replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name], np.ndarray):
                task_metadata[attr_name] = replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name].tolist()
            else:
                task_metadata[attr_name] = replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name]
        with open(f"{demos_meta_dir}/task-{task_id:04d}/episode_{demo_id_raw}.json", "w") as f:
            json.dump(task_metadata, f, indent=4)
    log.info(f"Successfully processed {replayed_path}")


def rgbd_gt_to_pcd(
    data_folder: str,
    task_id: int,
    demo_id_raw: str,
    episode_id: int,
    robot_camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    downsample_ratio: int = 4,
    pcd_range = (-0.2, 1.0, -1.0, 1.0, -0.2, 1.5),
    pcd_num_points: int = 4096,
    batch_size: int = 500,
    use_fps: bool = False,
    out_root: Optional[str] = None,
):
    log.info(f"Generating point cloud data from RGBD for {demo_id_raw} (episode index {episode_id})")
    # output directory prefers out_root when provided (keeps original dataset structure)
    if out_root:
        output_dir = os.path.join(out_root, "pcd_gt", f"task-{task_id:04d}")
    else:
        output_dir = os.path.join(data_folder, "pcd_gt", f"task-{task_id:04d}")
    makedirs_with_mode(output_dir)

    with h5py.File(f"{data_folder}/replayed/episode_{demo_id_raw}.hdf5", "r") as in_f:
        with h5py.File(f"{output_dir}/episode_{demo_id_raw}.hdf5", "w") as out_f:
            data = in_f["data"][f"demo_{episode_id}"]["obs"]
            data_size = data["robot_r1::cam_rel_poses"].shape[0]
            fused_pcd_dset = out_f.create_dataset(
                f"data/demo_{episode_id}/robot_r1::fused_pcd",
                shape=(data_size, pcd_num_points, 6),
                compression="lzf",
            )
            for i in range(0, data_size, batch_size):
                log.info(f"Processing batch {i} of {data_size}...")
                obs = dict()
                obs["cam_rel_poses"] = th.from_numpy(data["robot_r1::cam_rel_poses"][i : i + batch_size])
                camera_intrinsics = {}
                for camera_id, robot_camera_name in robot_camera_names.items():
                    resolution = HEAD_RESOLUTION if camera_id == "head" else WRIST_RESOLUTION
                    camera_intrinsics[robot_camera_name] = (
                        th.from_numpy(CAMERA_INTRINSICS["R1Pro"][camera_id]) / downsample_ratio
                    )
                    camera_intrinsics[robot_camera_name][-1, -1] = 1.0
                    obs[f"{robot_camera_name}::rgb"] = F.interpolate(
                        th.from_numpy(data[f"{robot_camera_name}::rgb"][i : i + batch_size, :, :, :3]).movedim(-1, -3),
                        size=(resolution[0] // downsample_ratio, resolution[1] // downsample_ratio),
                        mode="nearest-exact",
                    ).movedim(-3, -1)
                    obs[f"{robot_camera_name}::depth_linear"] = F.interpolate(
                        th.from_numpy(data[f"{robot_camera_name}::depth_linear"][i : i + batch_size]).unsqueeze(0),
                        size=(resolution[0] // downsample_ratio, resolution[1] // downsample_ratio),
                        mode="nearest-exact",
                    ).squeeze(0)

                pcd = process_fused_point_cloud(
                    obs=obs,
                    camera_intrinsics=camera_intrinsics,
                    pcd_range=pcd_range,
                    pcd_num_points=pcd_num_points,
                    use_fps=use_fps,
                    verbose=True,
                )
                log.info("Saving point cloud data...")
                fused_pcd_dset[i : i + batch_size] = pcd.cpu()

    log.info("Point cloud data saved!")


def rgbd_vid_to_pcd_custom(
    out_root: str,
    task_id: int,
    demo_id_raw: str,
    episode_id: int,
    robot_camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    downsample_ratio: int = 4,
    pcd_range: Tuple[float, float, float, float, float, float] = (
        -0.2,
        1.0,
        -1.0,
        1.0,
        -0.2,
        1.5,
    ),
    pcd_num_points: int = 4096,
    batch_size: int = 500,
    use_fps: bool = False,
):
    """Generate point cloud data from compressed RGBD videos under the specified out_root.

    This reads parquets under ``<out_root>/data/...`` and writes HDF5 outputs to ``<out_root>/pcd_vid/...``.
    """
    log.info(f"Generating point cloud data from RGBD videos for {demo_id_raw} in {out_root}")
    output_dir = os.path.join(out_root, "pcd_vid", f"task-{task_id:04d}")
    makedirs_with_mode(output_dir)

    # locate parquet file (support both raw id name and zero-padded 08d name)
    parquet_candidates = [
        os.path.join(out_root, "data", f"task-{task_id:04d}", f"episode_{demo_id_raw}.parquet")
    ]
    try:
        demo_int = int(demo_id_raw)
        parquet_candidates.append(os.path.join(out_root, "data", f"task-{task_id:04d}", f"episode_{demo_int:08d}.parquet"))
    except Exception:
        pass

    parquet_file = None
    for p in parquet_candidates:
        if os.path.exists(p):
            parquet_file = p
            break
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet file found for demo {demo_id_raw} under {out_root}/data/task-{task_id:04d}")

    in_f = pd.read_parquet(parquet_file)

    obs_loaders = {}
    total_n_points_before_downsample = 0
    for camera_id, robot_camera_name in robot_camera_names.items():
        resolution = HEAD_RESOLUTION if camera_id == "head" else WRIST_RESOLUTION
        output_size = (resolution[0] // downsample_ratio, resolution[1] // downsample_ratio)
        total_n_points_before_downsample += output_size[0] * output_size[1]
        keys = ["rgb", "depth_linear"]
        for key in keys:
            kwargs = {}
            if key == "seg_semantic_id":
                meta_path = os.path.join(out_root, "meta", "episodes", f"task-{task_id:04d}", f"episode_{demo_id_raw}.json")
                with open(meta_path) as f:
                    kwargs["id_list"] = th.tensor(json.load(f)[f"{robot_camera_name}::unique_ins_ids"])
            obs_loaders[f"{robot_camera_name}::{key}"] = iter(
                OBS_LOADER_MAP[key](
                    data_path=out_root,
                    task_id=task_id,
                    demo_id=demo_id_raw,
                    camera_id=camera_id,
                    output_size=output_size,
                    batch_size=batch_size,
                    stride=batch_size,
                    **kwargs,
                )
            )

    cam_rel_poses = th.from_numpy(np.array(in_f["observation.cam_rel_poses"].tolist(), dtype=np.float32))
    data_size = cam_rel_poses.shape[0]
    with h5py.File(os.path.join(output_dir, f"episode_{demo_id_raw}.hdf5"), "w") as out_f:
        fused_pcd_dset = out_f.create_dataset(
            f"data/demo_{episode_id}/robot_r1::fused_pcd",
            shape=(data_size, pcd_num_points if pcd_num_points is not None else total_n_points_before_downsample, 6),
            compression="lzf",
        )
        # We batch process every batch_size frames
        for i in range(0, data_size, batch_size):
            log.info(f"Processing batch {i} of {data_size}...")
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
                verbose=True,
            )
            log.info("Saving point cloud data...")
            fused_pcd_dset[i : i + batch_size] = pcd.cpu()

    log.info("Point cloud data saved!")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos (custom dataset layout)")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder (episode parent directory)")
    parser.add_argument("--task_name", type=str, required=True, help="Task name to process, e.g., attach_a_camera_to_a_tripod")
    parser.add_argument(
        "--demo_id",
        type=str,
        required=True,
        help="Demo folder name or raw id (e.g., attach_a_camera_to_a_tripod_17507... or 17507...)",
    )
    parser.add_argument("--out_root", type=str, default="demos", help="Output root folder path (default: ./demos). By default this will be created in the current working directory (not under --data_folder). Pass an absolute path to use a specific location.")

    parser.add_argument("--low_dim", action="store_true", help="Include this flag to generate low dimensional data")
    parser.add_argument("--rgbd", action="store_true", help="Include this flag to generate rgbd videos")
    parser.add_argument("--offline_rgbd", action="store_true", help="Whether rgbd videos should be generated offline")
    parser.add_argument("--pcd_gt", action="store_true", help="Include this flag to generate point cloud data from ground truth RGBD")
    parser.add_argument("--pcd_vid", action="store_true", help="Include this flag to generate point cloud data from RGBD videos")
    parser.add_argument("--seg", action="store_true", help="Include this flag to generate segmentation maps")
    parser.add_argument("--bbox", action="store_true", help="Include this flag to generate bounding box data")
    parser.add_argument(
        "--keep_replayed",
        action="store_true",
        help="Keep the intermediate replayed HDF5 under <data_folder>/replayed instead of deleting it at the end.",
    )

    args = parser.parse_args()

    task_id = TASK_NAMES_TO_INDICES[args.task_name]

    # find the input HDF5 file for this episode using custom layout
    input_hdf5_path, episode_folder = find_episode_hdf5(args.data_folder, args.task_name, args.demo_id)
    if input_hdf5_path is None:
        log.error(
            f"Could not find HDF5 for demo '{args.demo_id}' under {args.data_folder}. Try passing full episode folder name or ensure the HDF5 exists."
        )
        sys.exit(1)

    # derive the raw episode id from the folder name if possible
    if episode_folder is not None and "_" in episode_folder:
        episode_raw = episode_folder.split("_")[-1]
    else:
        episode_raw = args.demo_id.split("_")[-1]

    # Resolve output root (prefer canonical '2025-challenge-demos' under gm.DATA_PATH when available)
    resolved_out_root = resolve_out_root(args.out_root, args.data_folder)
    log.info(f"Resolved output root: {resolved_out_root}")

    # replay hdf5 (required for any outputs that depend on replayed keys like cam_rel_poses)
    needs_replay = args.rgbd or args.seg or args.bbox or args.low_dim or args.pcd_gt
    if args.offline_rgbd and not args.rgbd:
        raise ValueError("--offline_rgbd requires --rgbd")

    episode_index = None
    if needs_replay:
        episode_index = replay_hdf5_file(
            data_folder=args.data_folder,
            task_id=task_id,
            demo_id_raw=episode_raw,
            input_hdf5_path=input_hdf5_path,
            generate_rgbd=args.rgbd,
            generate_seg=args.seg,
            generate_bbox=args.bbox,
            flush_every_n_steps=FLUSH_EVERY_N_STEPS,
            offline_rgbd=args.offline_rgbd,
            out_root=resolved_out_root,
        )
    else:
        episode_index = 0

    if args.low_dim:
        generate_low_dim_data(
            data_folder=args.data_folder,
            task_id=task_id,
            demo_id_raw=episode_raw,
            episode_id=episode_index,
            out_root=resolved_out_root,
        )

    if args.pcd_gt or args.pcd_vid:
        cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "omnigibson/learning/configs/task/behavior.yaml")
        try:
            import yaml

            with open(cfg_file) as f:
                pcd_range = tuple(yaml.safe_load(f)["pcd_range"])
        except Exception:
            pcd_range = (-0.2, 1.0, -1.0, 1.0, -0.2, 1.5)

        if args.pcd_gt:
            rgbd_gt_to_pcd(
                data_folder=args.data_folder,
                task_id=task_id,
                demo_id_raw=episode_raw,
                episode_id=episode_index,
                robot_camera_names=ROBOT_CAMERA_NAMES["R1Pro"],
                pcd_range=pcd_range,
                downsample_ratio=4,
                pcd_num_points=4096,
                batch_size=1000,
                use_fps=True,
                out_root=resolved_out_root,
            )
        if args.pcd_vid:
            rgbd_vid_to_pcd_custom(
                out_root=resolved_out_root,
                task_id=task_id,
                demo_id_raw=episode_raw,
                episode_id=episode_index,
                robot_camera_names=ROBOT_CAMERA_NAMES["R1Pro"],
                pcd_range=pcd_range,
                downsample_ratio=4,
                pcd_num_points=4096,
                batch_size=1000,
                use_fps=True,
            )

    # remove replayed hdf5 to free up storage unless explicitly requested to keep it
    if not args.keep_replayed:
        try:
            os.remove(f"{args.data_folder}/replayed/episode_{episode_raw}.hdf5")
        except FileNotFoundError:
            log.warning(f"File {args.data_folder}/replayed/episode_{episode_raw}.hdf5 not found")
    else:
        log.info(f"Keeping replayed file at {args.data_folder}/replayed/episode_{episode_raw}.hdf5")

    log.info("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
