import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm


def draw_text_with_background(img, text, position, font_scale=0.8, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def draw_progress_bar(img, progress, position, width, height, color=(0, 255, 0), bg_color=(50, 50, 50)):
    x, y = position
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    fill_width = int(width * progress)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)


def _safe_progress(start, end, frame_idx):
    try:
        start = int(start)
        end = int(end)
    except Exception:
        return 0.0
    if end <= start:
        return 0.0
    return (frame_idx - start) / (end - start)


def _safe_desc(ann, key):
    val = ann.get(key)
    if isinstance(val, list):
        if not val:
            return "N/A"
        first = val[0]
        return str(first) if first is not None else "N/A"
    if isinstance(val, str):
        return val if val else "N/A"
    if val is None:
        return "N/A"
    return str(val)


def annotate_frame(frame, frame_idx, skill_anns, primitive_anns):
    active_skill = None
    skill_progress = 0
    for ann in skill_anns:
        frame_duration = ann.get("frame_duration")
        if not frame_duration or len(frame_duration) < 2:
            continue
        start, end = frame_duration[0], frame_duration[1]
        if int(start) <= frame_idx < int(end):
            active_skill = ann
            skill_progress = _safe_progress(start, end, frame_idx)
            break

    active_primitive = None
    prim_progress = 0
    for ann in primitive_anns:
        frame_duration = ann.get("frame_duration")
        if not frame_duration or len(frame_duration) < 2:
            continue
        start, end = frame_duration[0], frame_duration[1]
        if int(start) <= frame_idx < int(end):
            active_primitive = ann
            prim_progress = _safe_progress(start, end, frame_idx)
            break

    if active_skill:
        desc = _safe_desc(active_skill, "skill_description")
        draw_text_with_background(frame, f"Skill: {desc}", (20, 40), bg_color=(0, 0, 150))
        draw_progress_bar(frame, skill_progress, (20, 50), 200, 10, color=(0, 150, 255))

    if active_primitive:
        desc = _safe_desc(active_primitive, "primitive_description")
        draw_text_with_background(frame, f"Primitive: {desc}", (20, 90), bg_color=(0, 100, 0))
        draw_progress_bar(frame, prim_progress, (20, 100), 200, 10, color=(50, 205, 50))

    return frame


def _list_task_names(dataset_path):
    ann_root = os.path.join(dataset_path, "annotations")
    if not os.path.isdir(ann_root):
        return []
    task_names = []
    for name in os.listdir(ann_root):
        if name.startswith("task-") and os.path.isdir(os.path.join(ann_root, name)):
            task_names.append(name)
    return sorted(task_names)


def _list_valid_episodes(dataset_path, task_name, camera_id):
    ann_dir = os.path.join(dataset_path, "annotations", task_name)
    vid_dir = os.path.join(dataset_path, "videos", task_name, f"observation.images.rgb.{camera_id}")
    if not os.path.isdir(ann_dir) or not os.path.isdir(vid_dir):
        return []

    ann_eps = set()
    for fn in os.listdir(ann_dir):
        if fn.endswith(".json") and fn.startswith("episode_"):
            ann_eps.add(fn[:-5])

    vid_eps = set()
    for fn in os.listdir(vid_dir):
        if fn.endswith(".mp4") and fn.startswith("episode_"):
            vid_eps.add(fn[:-4])

    return sorted(ann_eps & vid_eps)


def _annotate_one(dataset_path, task_name, episode_name, camera_id, output_path, overwrite):
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        if os.path.exists(output_path) and not overwrite:
            return {"status": "skipped_exists", "output_path": output_path}

        annotation_path = os.path.join(dataset_path, "annotations", task_name, f"{episode_name}.json")
        video_path = os.path.join(dataset_path, "videos", task_name, f"observation.images.rgb.{camera_id}", f"{episode_name}.mp4")

        if not os.path.exists(annotation_path):
            return {"status": "missing_annotation", "output_path": output_path, "annotation_path": annotation_path}
        if not os.path.exists(video_path):
            return {"status": "missing_video", "output_path": output_path, "video_path": video_path}

        with open(annotation_path, "r") as f:
            annotations = json.load(f)

        skill_annotations = annotations.get("skill_annotation", [])
        primitive_annotations = annotations.get("primitive_annotation", [])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "video_open_failed", "output_path": output_path, "video_path": video_path}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        written = 0
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(annotate_frame(frame, frame_idx, skill_annotations, primitive_annotations))
            written += 1

        cap.release()
        out.release()
        return {"status": "ok", "output_path": output_path, "frames": written}
    except Exception as e:
        return {
            "status": "exception",
            "output_path": output_path,
            "task_name": task_name,
            "episode_name": episode_name,
            "camera_id": camera_id,
            "error": repr(e),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel annotate videos for multiple tasks.")
    parser.add_argument("--dataset_path", type=str, default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos")
    parser.add_argument("--num_video_per_task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_folder", type=str, default="./outputs/annotated_videos/")
    parser.add_argument("--camera_id", type=str, default="head")
    parser.add_argument("--max_tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_names = _list_task_names(args.dataset_path)
    if task_names:
        task_names = task_names[: args.max_tasks]
    else:
        task_names = [f"task-{i:04d}" for i in range(args.max_tasks)]

    jobs = []
    per_task_stats = []
    for task_name in task_names:
        episodes = _list_valid_episodes(args.dataset_path, task_name, args.camera_id)
        if not episodes:
            per_task_stats.append((task_name, 0, 0))
            continue

        rng = random.Random(args.seed + int(task_name.split("-")[1]))
        k = min(args.num_video_per_task, len(episodes))
        chosen = rng.sample(episodes, k=k)
        per_task_stats.append((task_name, len(episodes), k))

        for episode_name in chosen:
            out_path = os.path.join(args.output_folder, task_name, f"{episode_name}_{args.camera_id}.mp4")
            jobs.append((task_name, episode_name, out_path))

    if not jobs:
        print("No jobs to run. Check dataset_path / camera_id / annotations+videos layout.")
        for task_name, total_eps, sampled in per_task_stats:
            print(f"{task_name}: episodes={total_eps}, sampled={sampled}")
        exit(0)

    os.makedirs(args.output_folder, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for task_name, episode_name, out_path in jobs:
            futures.append(
                ex.submit(_annotate_one, args.dataset_path, task_name, episode_name, args.camera_id, out_path, args.overwrite)
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"status": "exception", "error": repr(e)})

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped_exists")
    failed = [r for r in results if r.get("status") not in ("ok", "skipped_exists")]

    print(f"Done. ok={ok}, skipped_exists={skipped}, failed={len(failed)}")
    if failed:
        for r in failed[:20]:
            print(r)
        if len(failed) > 20:
            print(f"... ({len(failed) - 20} more failures)")
