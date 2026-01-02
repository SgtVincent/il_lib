# Replay raw HDF5 into `2025-challenge-demos` (OmniGibson)

This note documents the **off-the-shelf** replay pipeline in BEHAVIOR-1K that converts a raw trajectory HDF5 into the same on-disk format as the official **`2025-challenge-demos`** dataset:

- RGB videos (3 cameras)
- Depth videos (3 cameras)
- Instance-seg videos (3 cameras)
- Low-dim parquet (actions, proprio, cam poses, privileged task info)
- Per-episode JSON metadata

The authoritative entrypoint is:

- `BEHAVIOR-1K/OmniGibson/scripts/learning/replay_obs.py`

## Prerequisites

- Use the repo’s conda env:
  - `conda activate behavior`
- OmniGibson + Isaac Sim must be installed and runnable on your machine.
- You typically need a GPU node (even in headless mode).
- Point OmniGibson to the dataset folder containing `2025-challenge-task-instances/...`:
  - `export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K/datasets`
- For servers / clusters (recommended):
  - `export OMNIGIBSON_HEADLESS=1`

## Expected input layout (raw)

`replay_obs.py` expects the raw HDF5 to be placed under:

```
<DATA_FOLDER>/
  2025-challenge-rawdata/
    task-0035/
      episode_00000123.hdf5
```

Where:
- `task-0035` is the numeric task id (from `omnigibson.learning.utils.eval_utils.TASK_NAMES_TO_INDICES`)
- `episode_00000123.hdf5` is the demo id you pass as `--demo_id`

## What gets written (demos)

When you run replay, outputs are written under:

```
<DATA_FOLDER>/
  2025-challenge-demos/
    data/task-0035/episode_00000123.parquet
    meta/episodes/task-0035/episode_00000123.json
    videos/task-0035/observation.images.rgb.head/episode_00000123.mp4
    videos/task-0035/observation.images.rgb.left_wrist/episode_00000123.mp4
    videos/task-0035/observation.images.rgb.right_wrist/episode_00000123.mp4
    videos/task-0035/observation.images.depth.*/*.mp4
    videos/task-0035/observation.images.seg_instance_id.*/*.mp4
```

Notes:
- Depth is encoded as x265 lossless with `pix_fmt=yuv420p10le` in this script.
- The script also creates `<DATA_FOLDER>/replayed/episode_00000123.hdf5` temporarily and deletes it at the end.

Important:
- `replay_obs.py` generates **per-episode** artifacts (videos / parquet / `meta/episodes/...`). It does **not** generate dataset-level files like `meta/info.json`.
  - If you need a full HuggingFace / LeRobot-style dataset folder, either copy `meta/info.json` from an official `2025-challenge-demos` release and edit counts, or use the converter in `tutorials/process_private_data.py` to build `meta/info.json` for your output directory.

## Option A: Run the official replay script directly

From the BEHAVIOR-1K repo root:

```bash
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K/datasets

python OmniGibson/scripts/learning/replay_obs.py \
  --data_folder <DATA_FOLDER> \
  --task_name attach_a_camera_to_a_tripod \
  --demo_id 123 \
  --low_dim --rgbd --seg
```

Optional flags:
- `--offline_rgbd`: encode rgb/depth videos after replay finishes (sometimes more stable)
- `--bbox`: also generate bbox videos (head camera only; requires `--rgbd --seg`)
- `--pcd_gt`: generate fused point cloud from ground-truth RGBD into `<DATA_FOLDER>/pcd_gt/...`
- `--pcd_vid`: generate fused point cloud from RGBD videos into `<DATA_FOLDER>/pcd_vid/...`

## Option B: Use the wrapper in this repo (recommended for arbitrary raw paths)

This repo provides a thin wrapper that:
1) Computes `task_id` from `task_name`
2) Stages your raw `.hdf5` into the required `2025-challenge-rawdata/task-XXXX/episode_YYYYYYYY.hdf5` path (symlink by default)
3) Invokes the official `replay_obs.py`

Example:

```bash
conda activate behavior

export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K/datasets

python tutorials/replay_hdf5_to_demos.py \
  --raw_hdf5 /abs/path/to/some_episode.hdf5 \
  --task_name attach_a_camera_to_a_tripod \
  --demo_id 123 \
  --data_folder /abs/path/to/output_root \
  --behavior1k_root /abs/path/to/BEHAVIOR-1K
```

Useful options:
- `--stage_mode {symlink,copy,hardlink}` (default: `symlink`)
- `--overwrite/--no-overwrite` (default: overwrite)
- `--skip_if_complete` (skip if expected outputs exist)
- `--omnigibson_data_path /path/to/BEHAVIOR-1K/datasets` (sets `OMNIGIBSON_DATA_PATH` for the replay subprocess)
- `--no-rgbd` / `--no-seg` / `--no-low_dim` (disable some outputs)

## SLURM example

See `tutorials/replay_hdf5_to_demos.sbatch.sh`.

```bash
sbatch tutorials/replay_hdf5_to_demos.sbatch.sh \
  --raw_hdf5 /abs/path/to/some_episode.hdf5 \
  --task_name attach_a_camera_to_a_tripod \
  --demo_id 123 \
  --data_folder /abs/path/to/output_root \
  --behavior1k_root /abs/path/to/BEHAVIOR-1K
```

## Common failure modes

- `GLFW initialization failed` / segfaults / “No device could be created”:
  - This almost always means Isaac / rendering can’t initialize (wrong node type, missing GPU, driver mismatch, etc.).
  - Try `OMNIGIBSON_HEADLESS=1` and run on a GPU node.
- `No B50_task_misc.csv file found`:
  - Your `OMNIGIBSON_DATA_PATH` is wrong or your datasets are incomplete. Make sure `2025-challenge-task-instances/metadata/B50_task_misc.csv` exists.
- `No full scene file found in ... joylo/sampled_task/<task_name>`:
  - Your BEHAVIOR-1K checkout is missing the sampled task scene assets for that task.
