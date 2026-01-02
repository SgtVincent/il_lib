"""
Process private BEHAVIOR-1K data from custom folder structure to match 2025-challenge-demos format.

This script processes raw data episodes from the Behavior1K_100h_ByteDance folder structure:
    <task_name>/<task_name>_<timestamp>/
        - <task_name>.hdf5  (raw demo data)
        - <task_name>.json  (skill annotations)
        - running_args.json (task info)

And converts them to 2025-challenge-demos structure:
    annotations/task-XXXX/episode_XXXXXXXX.json
    data/task-XXXX/episode_XXXXXXXX.parquet
    meta/episodes/task-XXXX/episode_XXXXXXXX.json
    videos/task-XXXX/<modality>/episode_XXXXXXXX.mp4

Usage:
    python process_private_data.py --input_path /path/to/Behavior1K_100h_ByteDance --output_path /path/to/output
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Task name to task index mapping (from 2025-challenge-demos/meta/tasks.jsonl)
TASK_NAME_TO_INDEX = {
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}


DEFAULT_REPLAY_OBS_PATH = "/home/ubuntu/ove_repo/BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py"

# Matches 2025-challenge-demos/video folder naming
DEFAULT_VIDEO_KEYS = [
    "observation.images.rgb.left_wrist",
    "observation.images.rgb.right_wrist",
    "observation.images.rgb.head",
    "observation.images.depth.left_wrist",
    "observation.images.depth.right_wrist",
    "observation.images.depth.head",
    "observation.images.seg_instance_id.left_wrist",
    "observation.images.seg_instance_id.right_wrist",
    "observation.images.seg_instance_id.head",
]


def makedirs_with_mode(path: str, mode: int = 0o775):
    """Create directory with specific permissions."""
    os.makedirs(path, mode=mode, exist_ok=True)


def normalize_task_name(task_name: str) -> str:
    """Normalize task name to match the format in TASK_NAME_TO_INDEX."""
    # Replace spaces with underscores and convert to lowercase
    return task_name.strip().replace(" ", "_").replace("-", "_").lower()


def discover_episodes(input_path: str) -> Dict[str, List[Tuple[str, str, dict]]]:
    """
    Discover all episodes in the input directory.
    
    Returns:
        Dict mapping task_name -> list of (episode_folder_path, hdf5_path, running_args)
    """
    episodes_by_task = defaultdict(list)
    input_path = Path(input_path)
    
    # Iterate through all task folders
    for task_folder in input_path.iterdir():
        if not task_folder.is_dir():
            continue
            
        task_name = task_folder.name
        
        # Iterate through all episode folders within the task
        for episode_folder in task_folder.iterdir():
            if not episode_folder.is_dir():
                continue
                
            # Check for required files
            hdf5_file = episode_folder / f"{task_name}.hdf5"
            json_file = episode_folder / f"{task_name}.json"
            running_args_file = episode_folder / "running_args.json"
            
            if not hdf5_file.exists():
                print(f"Warning: HDF5 file not found in {episode_folder}, skipping...")
                continue
                
            # Load running_args if exists
            running_args = {}
            if running_args_file.exists():
                with open(running_args_file, 'r') as f:
                    running_args = json.load(f)
                    
            episodes_by_task[task_name].append((
                str(episode_folder),
                str(hdf5_file),
                str(json_file) if json_file.exists() else None,
                running_args
            ))
            
    return episodes_by_task


def _expected_video_paths(output_path: str, task_id: int, episode_id: int, video_keys: List[str]) -> List[str]:
    episode_name = f"episode_{episode_id:08d}.mp4"
    task_folder = f"task-{task_id:04d}"
    return [os.path.join(output_path, "videos", task_folder, key, episode_name) for key in video_keys]


def _episode_outputs_complete(
    *,
    output_parquet: str,
    output_annotation: str,
    output_meta: str,
    videos_enabled: bool,
    expected_video_paths: List[str],
) -> bool:
    if not (os.path.exists(output_parquet) and os.path.exists(output_annotation) and os.path.exists(output_meta)):
        return False
    if videos_enabled:
        return all(os.path.exists(p) for p in expected_video_paths)
    return True


def generate_videos_with_replay_obs(
    *,
    input_hdf5_path: str,
    output_path: str,
    task_name: str,
    task_id: int,
    episode_id: int,
    replay_obs_path: str,
    omnigibson_data_path: Optional[str],
    omnigibson_headless: bool,
    keep_tmp: bool,
) -> None:
    """Generate MP4 videos by replaying the raw HDF5 with OmniGibson (see replay_obs.py).

    Notes:
        - Private raw HDF5s do not include visual observations, so we must replay to record RGB/Depth/Seg.
        - replay_obs.py expects input at: <data_folder>/2025-challenge-rawdata/task-XXXX/episode_YYYYYYYY.hdf5
        - It outputs videos to: <data_folder>/2025-challenge-demos/videos/task-XXXX/<video_key>/episode_YYYYYYYY.mp4
    """

    if not os.path.exists(replay_obs_path):
        raise FileNotFoundError(
            f"replay_obs.py not found at {replay_obs_path}. "
            f"Pass --replay_obs_path to point to OmniGibson/scripts/learning/replay_obs.py"
        )

    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="replay_obs_", dir=output_path)
        if not keep_tmp
        else None
    )
    tmp_root = tmp_ctx.name if tmp_ctx is not None else os.path.join(output_path, "_replay_obs_tmp")
    try:
        if tmp_ctx is None:
            # keep_tmp=True path
            makedirs_with_mode(tmp_root)

        raw_dir = os.path.join(tmp_root, "2025-challenge-rawdata", f"task-{task_id:04d}")
        makedirs_with_mode(raw_dir)
        raw_hdf5_dst = os.path.join(raw_dir, f"episode_{episode_id:08d}.hdf5")
        shutil.copy2(input_hdf5_path, raw_hdf5_dst)

        cmd = [
            sys.executable,
            replay_obs_path,
            "--data_folder",
            tmp_root,
            "--task_name",
            task_name,
            "--demo_id",
            str(episode_id),
            "--rgbd",
            "--seg",
        ]

        env = os.environ.copy()
        if omnigibson_data_path:
            env["OMNIGIBSON_DATA_PATH"] = omnigibson_data_path
        if omnigibson_headless:
            # Avoid needing an X server; still requires a functional GPU/driver
            env.setdefault("OMNIGIBSON_HEADLESS", "True")

        subprocess.run(cmd, check=True, env=env)

        src_videos_task = os.path.join(tmp_root, "2025-challenge-demos", "videos", f"task-{task_id:04d}")
        if not os.path.exists(src_videos_task):
            raise RuntimeError(
                "Replay finished but no videos were produced. "
                "Check Isaac / OmniGibson rendering setup and replay_obs.py logs."
            )

        dst_videos_task = os.path.join(output_path, "videos", f"task-{task_id:04d}")
        makedirs_with_mode(dst_videos_task)
        shutil.copytree(src_videos_task, dst_videos_task, dirs_exist_ok=True)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def process_hdf5_to_parquet(
    hdf5_path: str,
    output_parquet_path: str,
    annotation_json_path: Optional[str] = None,
    episode_id: Optional[int] = None,
    task_id: Optional[int] = None,
) -> Tuple[int, dict, int]:
    """
    Process HDF5 file and convert to parquet format.
    
    Returns:
        Tuple of (num_samples, episode_stats, offset)
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo_grp = f['data']['demo_0']
        
        # Get the data arrays
        actions = demo_grp['action'][:]  # (T-1, 23)
        states = demo_grp['state'][:]     # (T, state_dim)
        num_samples = actions.shape[0]
        
        # Determine valid duration from annotations if available
        valid_start = 0
        valid_end = num_samples
        if annotation_json_path and os.path.exists(annotation_json_path):
            with open(annotation_json_path, 'r') as ann_f:
                ann_data = json.load(ann_f)
                if 'meta_data' in ann_data and 'valid_duration' in ann_data['meta_data']:
                    valid_start, valid_end = ann_data['meta_data']['valid_duration']
                    valid_end = min(valid_end, num_samples)
        
        offset = valid_start  # Store offset for annotation adjustment
        
        # Slice to valid duration
        actions = actions[valid_start:valid_end]
        states = states[valid_start:valid_end + 1]  # states has one more element
        num_samples = actions.shape[0]
        
        # Build dataframe
        # Note: The parquet format requires specific columns
        # We'll create a simplified version that can be extended
        
        data = {
            'index': list(range(num_samples)),
            'episode_index': [int(episode_id) if episode_id is not None else 0] * num_samples,
            'timestamp': [float(i) / 30.0 for i in range(num_samples)],  # Assuming 30 fps
        }

        if task_id is not None:
            data['task_index'] = [int(task_id)] * num_samples
        
        # Add action columns
        data['action'] = [actions[i].tolist() for i in range(num_samples)]
        
        # Add state/observation columns
        # The state format may vary - we'll store the first 256 as observation.state
        state_dim = min(256, states.shape[1])
        data['observation.state'] = [states[i, :state_dim].tolist() for i in range(num_samples)]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to parquet
        makedirs_with_mode(os.path.dirname(output_parquet_path))
        df.to_parquet(output_parquet_path, engine='pyarrow')
        
        # Compute episode stats
        episode_stats = {
            'length': num_samples,
            'distance_traveled': 0.0,  # Would need base position to compute
            'left_eef_displacement': 0.0,
            'right_eef_displacement': 0.0,
        }
        
        return num_samples, episode_stats, offset


def adjust_frame_durations(annotation: dict, offset: int) -> dict:
    """
    Adjust frame durations in annotations to account for valid_duration offset.
    
    Args:
        annotation: The annotation dict
        offset: The frame offset to subtract from all frame durations
        
    Returns:
        Modified annotation dict with adjusted frame durations
    """
    annotation = annotation.copy()
    
    # Adjust skill annotations
    if 'skill_annotation' in annotation:
        for skill in annotation['skill_annotation']:
            if 'frame_duration' in skill:
                start, end = skill['frame_duration']
                skill['frame_duration'] = [max(0, start - offset), max(0, end - offset)]
    
    # Adjust primitive annotations
    if 'primitive_annotation' in annotation:
        for primitive in annotation['primitive_annotation']:
            if 'frame_duration' in primitive:
                start, end = primitive['frame_duration']
                primitive['frame_duration'] = [max(0, start - offset), max(0, end - offset)]
    
    # Adjust meta_data
    if 'meta_data' in annotation:
        if 'valid_duration' in annotation['meta_data']:
            start, end = annotation['meta_data']['valid_duration']
            annotation['meta_data']['valid_duration'] = [0, end - start]
        if 'task_duration' in annotation['meta_data']:
            valid_duration = annotation['meta_data'].get('valid_duration', [0, 0])
            annotation['meta_data']['task_duration'] = valid_duration[1] - valid_duration[0]
    
    return annotation


def process_annotation(
    source_json_path: str,
    output_annotation_path: str,
    task_name: str,
    offset: int = 0
) -> None:
    """Process and copy the annotation JSON file."""
    if source_json_path and os.path.exists(source_json_path):
        # Read and potentially modify the annotation
        with open(source_json_path, 'r') as f:
            annotation = json.load(f)
            
        # Ensure consistent format
        if 'task_name' not in annotation:
            annotation['task_name'] = task_name.replace('_', ' ')
        
        # Adjust frame durations if there's an offset
        if offset > 0:
            annotation = adjust_frame_durations(annotation, offset)
            
        makedirs_with_mode(os.path.dirname(output_annotation_path))
        with open(output_annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)
    else:
        # Create minimal annotation
        annotation = {
            'task_name': task_name.replace('_', ' '),
            'data_folder': '',
            'meta_data': {
                'task_duration': 0,
                'valid_duration': [0, 0]
            },
            'skill_annotation': [],
            'primitive_annotation': []
        }
        makedirs_with_mode(os.path.dirname(output_annotation_path))
        with open(output_annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)


def create_meta_episode_json(
    output_meta_path: str,
    num_samples: int,
    task_name: str
) -> None:
    """Create a minimal meta episode JSON file."""
    meta_data = {
        'n_episodes': 1,
        'n_steps': num_samples,
        'num_samples': num_samples,
        'robot_type': 'R1Pro',
        'task_obs_keys': [],
    }
    
    makedirs_with_mode(os.path.dirname(output_meta_path))
    with open(output_meta_path, 'w') as f:
        json.dump(meta_data, f, indent=4)


def process_single_episode(
    task_name: str,
    task_id: int,
    episode_id: int,
    episode_folder: str,
    hdf5_path: str,
    annotation_path: Optional[str],
    output_path: str,
    skip_existing: bool,
    videos: bool,
    replay_obs_path: str,
    omnigibson_data_path: Optional[str],
    omnigibson_headless: bool,
    keep_replay_tmp: bool,
    video_keys: List[str],
) -> Optional[dict]:
    """
    Process a single episode.
    
    Returns:
        Episode stats dict or None if skipped/failed
    """
    episode_name = f"episode_{episode_id:08d}"
    task_folder = f"task-{task_id:04d}"
    
    # Output paths
    output_parquet = os.path.join(output_path, "data", task_folder, f"{episode_name}.parquet")
    output_annotation = os.path.join(output_path, "annotations", task_folder, f"{episode_name}.json")
    output_meta = os.path.join(output_path, "meta", "episodes", task_folder, f"{episode_name}.json")

    expected_video_paths = _expected_video_paths(output_path, task_id, episode_id, video_keys)
    
    # Skip only if all expected outputs exist
    if skip_existing and _episode_outputs_complete(
        output_parquet=output_parquet,
        output_annotation=output_annotation,
        output_meta=output_meta,
        videos_enabled=videos,
        expected_video_paths=expected_video_paths,
    ):
        print(f"  Skipping {episode_name} (already exists / complete)")
        return None
    
    try:
        # Process HDF5 to parquet
        num_samples, stats, offset = process_hdf5_to_parquet(
            hdf5_path,
            output_parquet,
            annotation_path,
            episode_id=episode_id,
            task_id=task_id,
        )
        
        # Process annotation (with offset adjustment)
        process_annotation(annotation_path, output_annotation, task_name, offset=offset)
        
        # Create meta episode JSON
        create_meta_episode_json(output_meta, num_samples, task_name)

        videos_generated = False
        # Generate videos via OmniGibson replay (raw HDF5 does not contain visual obs)
        if videos:
            try:
                generate_videos_with_replay_obs(
                    input_hdf5_path=hdf5_path,
                    output_path=output_path,
                    task_name=task_name,
                    task_id=task_id,
                    episode_id=episode_id,
                    replay_obs_path=replay_obs_path,
                    omnigibson_data_path=omnigibson_data_path,
                    omnigibson_headless=omnigibson_headless,
                    keep_tmp=keep_replay_tmp,
                )
                videos_generated = True
            except Exception as e:
                # Non-fatal: keep low-dim parquet + annotations, but warn that videos are missing.
                print(f"  Warning: failed to generate videos for {episode_name}: {e}")
        
        stats['episode_index'] = episode_id
        stats['task_name'] = task_name
        stats['task_id'] = task_id
        stats['videos_generated'] = videos_generated
        
        return stats
        
    except Exception as e:
        print(f"  Error processing {episode_folder}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process private BEHAVIOR-1K data to 2025-challenge-demos format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/Behavior1K_100h_ByteDance",
        help="Path to the input data folder (Behavior1K_100h_ByteDance)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/Behavior1K_100h_processed",
        help="Path to the output folder (will mimic 2025-challenge-demos structure)"
    )
    parser.add_argument(
        "--task_mapping_file",
        type=str,
        default=None,
        help="Optional path to tasks.jsonl for custom task ID mapping"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip episodes that have already been processed (default: overwrite)"
    )
    parser.add_argument(
        "--videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate MP4 videos under videos/ by replaying in OmniGibson (default: enabled)"
    )
    parser.add_argument(
        "--replay_obs_path",
        type=str,
        default=DEFAULT_REPLAY_OBS_PATH,
        help="Path to OmniGibson/scripts/learning/replay_obs.py"
    )
    parser.add_argument(
        "--omnigibson_data_path",
        type=str,
        default=None,
        help="Optional override for OMNIGIBSON_DATA_PATH (defaults to BEHAVIOR-1K/OmniGibson/datasets)"
    )
    parser.add_argument(
        "--omnigibson_headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson replay in headless mode (default: enabled)"
    )
    parser.add_argument(
        "--keep_replay_tmp",
        action="store_true",
        default=False,
        help="Keep temporary replay folders (debugging only; uses extra disk)"
    )
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=None,
        help="Only process specific task names (default: all)"
    )
    parser.add_argument(
        "--max_episodes_per_task",
        type=int,
        default=None,
        help="Maximum number of episodes to process per task"
    )
    
    args = parser.parse_args()
    
    # Load custom task mapping if provided
    task_name_to_index = TASK_NAME_TO_INDEX.copy()
    if args.task_mapping_file and os.path.exists(args.task_mapping_file):
        with open(args.task_mapping_file, 'r') as f:
            for line in f:
                task_info = json.loads(line)
                task_name_to_index[task_info['task_name']] = task_info['task_index']
    
    # Discover all episodes
    print(f"Discovering episodes in {args.input_path}...")
    episodes_by_task = discover_episodes(args.input_path)
    
    if not episodes_by_task:
        print("No episodes found!")
        return
    
    print(f"Found {len(episodes_by_task)} tasks with episodes")
    
    # Filter tasks if specified
    if args.task_names:
        filtered_tasks = {}
        for task_name in args.task_names:
            normalized_name = normalize_task_name(task_name)
            if normalized_name in episodes_by_task:
                filtered_tasks[normalized_name] = episodes_by_task[normalized_name]
            elif task_name in episodes_by_task:
                filtered_tasks[task_name] = episodes_by_task[task_name]
        episodes_by_task = filtered_tasks
    
    # Create output directory structure
    makedirs_with_mode(os.path.join(args.output_path, "annotations"))
    makedirs_with_mode(os.path.join(args.output_path, "data"))
    makedirs_with_mode(os.path.join(args.output_path, "meta", "episodes"))
    makedirs_with_mode(os.path.join(args.output_path, "videos"))
    
    # Track all episode stats for summary
    all_episodes = []
    task_stats = defaultdict(lambda: {'count': 0, 'total_frames': 0})
    
    # Track task ID assignments for new tasks
    max_existing_task_id = max(task_name_to_index.values()) if task_name_to_index else -1
    
    # Process each task
    for task_name, episodes in sorted(episodes_by_task.items()):
        # Get or assign task ID
        normalized_task_name = normalize_task_name(task_name)
        if normalized_task_name in task_name_to_index:
            task_id = task_name_to_index[normalized_task_name]
        elif task_name in task_name_to_index:
            task_id = task_name_to_index[task_name]
        else:
            # Assign new task ID
            max_existing_task_id += 1
            task_id = max_existing_task_id
            task_name_to_index[normalized_task_name] = task_id
            print(f"Assigning new task ID {task_id} to task '{task_name}'")
        
        print(f"\nProcessing task: {task_name} (ID: {task_id})")
        print(f"  Found {len(episodes)} episodes")
        
        # Limit episodes if specified
        if args.max_episodes_per_task:
            episodes = episodes[:args.max_episodes_per_task]
        
        # Sort episodes by episode ID
        episodes = sorted(episodes, key=lambda x: int(Path(x[0]).name.split('_')[-1]))
        
        # Process each episode
        for episode_folder, hdf5_path, annotation_path, running_args in tqdm(
            episodes, desc=f"  Task {task_id}", leave=False
        ):
            folder_name = Path(episode_folder).name
            episode_id = int(folder_name.split('_')[-1])
            stats = process_single_episode(
                task_name=normalized_task_name,
                task_id=task_id,
                episode_id=episode_id,
                episode_folder=episode_folder,
                hdf5_path=hdf5_path,
                annotation_path=annotation_path,
                output_path=args.output_path,
                skip_existing=args.skip_existing,
                videos=args.videos,
                replay_obs_path=args.replay_obs_path,
                omnigibson_data_path=args.omnigibson_data_path,
                omnigibson_headless=args.omnigibson_headless,
                keep_replay_tmp=args.keep_replay_tmp,
                video_keys=DEFAULT_VIDEO_KEYS,
            )
            
            if stats:
                all_episodes.append(stats)
                task_stats[task_id]['count'] += 1
                task_stats[task_id]['total_frames'] += stats['length']
    
    # Create meta files
    print("\nCreating meta files...")
    
    # Create tasks.jsonl
    tasks_jsonl_path = os.path.join(args.output_path, "meta", "tasks.jsonl")
    with open(tasks_jsonl_path, 'w') as f:
        for task_name, task_idx in sorted(task_name_to_index.items(), key=lambda x: x[1]):
            if task_idx in task_stats:
                task_info = {
                    "task_index": task_idx,
                    "task_name": task_name,
                    "task": task_name.replace('_', ' ')
                }
                f.write(json.dumps(task_info) + '\n')
    
    # Create episodes.jsonl
    episodes_jsonl_path = os.path.join(args.output_path, "meta", "episodes.jsonl")
    with open(episodes_jsonl_path, 'w') as f:
        for stats in all_episodes:
            episode_info = {
                "episode_index": stats['episode_index'],
                "tasks": [stats['task_name'].replace('_', ' ')],
                "length": stats['length'],
                "distance_traveled": stats.get('distance_traveled', 0.0),
                "left_eef_displacement": stats.get('left_eef_displacement', 0.0),
                "right_eef_displacement": stats.get('right_eef_displacement', 0.0)
            }
            f.write(json.dumps(episode_info) + '\n')
    
    # Create info.json
    total_episodes = len(all_episodes)
    total_frames = sum(s['length'] for s in all_episodes)
    total_videos = sum((len(DEFAULT_VIDEO_KEYS) if s.get('videos_generated') else 0) for s in all_episodes)
    info_json_path = os.path.join(args.output_path, "meta", "info.json")
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_stats),
        "total_videos": total_videos,
        "chunks_size": 10000,
        "fps": 30,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "metainfo_path": "meta/episodes/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [23], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            # Minimal compatibility: our parquet stores 256-dim state (first 256 dims)
            "observation.state": {"dtype": "float32", "shape": [256], "names": None},
            # Videos are represented as "video" features in the official dataset
            **({k: "video" for k in DEFAULT_VIDEO_KEYS} if args.videos else {}),
        },
    }
    with open(info_json_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total tasks processed: {len(task_stats)}")
    print(f"Total episodes processed: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"\nOutput saved to: {args.output_path}")
    print("\nTask Summary:")
    for task_id in sorted(task_stats.keys()):
        stats = task_stats[task_id]
        # Find task name
        task_name = [k for k, v in task_name_to_index.items() if v == task_id][0]
        print(f"  Task {task_id:4d} ({task_name}): {stats['count']} episodes, {stats['total_frames']} frames")


if __name__ == "__main__":
    main()
