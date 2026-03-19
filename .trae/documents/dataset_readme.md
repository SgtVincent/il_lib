# Dataset Description

* **Homepage:** \[More Information Needed]

* **Paper:** \[More Information Needed]

* **License:** mit

## Dataset Structure

[meta/info.json](meta/info.json):

```json
{
    "codebase_version": "v2.1",
    "robot_type": "R1Pro",
    "total_episodes": 10000,
    "total_frames": 119094660,
    "total_tasks": 50,
    "total_videos": 90000,
    "chunks_size": 10000,
    "fps": 30,
    "splits": {
        "train": "0:10000"
    },
    "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
    "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
    "metainfo_path": "meta/episodes/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
    "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
    "features": {
        "observation.images.rgb.left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.rgb.right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.rgb.head": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 720,
                "video.width": 720,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.depth.left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "depth"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p16le",
                "video.is_depth_map": true,
                "has_audio": false
            }
        },
        "observation.images.depth.right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "depth"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p16le",
                "video.is_depth_map": true,
                "has_audio": false
            }
        },
        "observation.images.depth.head": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "depth"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 720,
                "video.width": 720,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p16le",
                "video.is_depth_map": true,
                "has_audio": false
            }
        },
        "observation.images.seg_instance_id.left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.seg_instance_id.right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.seg_instance_id.head": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 720,
                "video.width": 720,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                23
            ],
            "names": null
        },
        "timestamp": {
            "dtype": "float64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "observation.cam_rel_poses": {
            "dtype": "float32",
            "shape": [
                21
            ],
            "names": null
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                256
            ],
            "names": null
        },
        "observation.task_info": {
            "dtype": "float32",
            "shape": [
                null
            ],
            "names": null
        }
    }
}
```

## Annotated Video

Generate an overlayed mp4 by combining `videos/` with `annotations/` (skill + primitive segments).

### Single Video

```bash
python tutorials/annotate_video.py \
  --dataset_path DATASETS/behavior/2025-challenge-demos \
  --task_id 0 \
  --episode_id 10 \
  --camera_id head \
  --output_path /tmp/annotated_video.mp4
```

### Parallel (50 Tasks)

```bash
python tutorials/annotate_videos_parallel.py \
  --dataset_path DATASETS/behavior/2025-challenge-demos \
  --num_video_per_task 2 \
  --seed 0 \
  --output_folder /tmp/annotated_videos \
  --camera_id head \
  --workers 16
```

Outputs are saved as:

- `${output_folder}/task-XXXX/episode_YYYYYYYY_${camera_id}.mp4`
