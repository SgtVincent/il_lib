#!/usr/bin/env python3

import argparse
import collections
import glob
import json
import os
import re
import sys


def _first_str(v):
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _get_text_from_ann(ann: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in ann:
            s = _first_str(ann[k])
            if s:
                return s
    return None


def _load_tasks_jsonl(dataset_root: str) -> dict[int, dict]:
    fn = os.path.join(dataset_root, "meta", "tasks.jsonl")
    if not os.path.isfile(fn):
        return {}
    out = {}
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("task_index")
            if isinstance(idx, int):
                out[idx] = obj
    return out


def _object_id_to_phrase(obj_id: str | None) -> str | None:
    if obj_id is None:
        return None
    s = str(obj_id).strip().replace("-", "_")
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        return None
    parts = [p for p in s.split("_") if p]
    if not parts:
        return None
    had_digit_suffix = False
    while parts and re.fullmatch(r"\d+", parts[-1]):
        had_digit_suffix = True
        parts.pop()
    if had_digit_suffix and len(parts) >= 3 and re.fullmatch(r"[a-z]{5,10}", parts[-1]):
        parts.pop()
    if len(parts) >= 2 and re.fullmatch(r"[a-z]{5,10}", parts[-1]):
        prev = parts[-2].lower()
        if prev in {
            "floor",
            "floors",
            "fridge",
            "cabinet",
            "bookcase",
            "drawer",
            "door",
            "sink",
            "table",
            "desk",
            "shelf",
            "sofa",
            "bed",
            "nightstand",
            "hallstand",
            "countertop",
            "patio",
            "lawn",
            "tree",
            "box",
            "basket",
            "trunk",
        }:
            parts.pop()
    if parts and parts[0].lower() == "floors":
        parts[0] = "floor"
    phrase = " ".join(parts).strip().lower()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase if phrase else None


def _format_obj_list(objs: list[str]) -> str:
    objs = [o for o in objs if o]
    if not objs:
        return ""
    if len(objs) == 1:
        return f"the {objs[0]}"
    return " and ".join([f"the {o}" for o in objs])


def _flatten_ids(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out.extend(_flatten_ids(y))
        return out
    return [str(x)]


def _spatial_prefix_to_phrase(x) -> str:
    toks = []
    for s in _flatten_ids(x):
        s = str(s).strip()
        if not s:
            continue
        if s == "low_level":
            continue
        if s.endswith("_door"):
            continue
        toks.append(s)
    if not toks:
        return ""
    phrase = toks[0].replace("_", " ").strip().lower()
    phrase = re.sub(r"\s+", " ", phrase)
    if phrase.startswith("[") and phrase.endswith("]"):
        return ""
    return phrase


def _extract_object_ids(ann: dict) -> tuple[list[str], list]:
    manipulating = ann.get("manipulating_object_id", [])
    manipulating_ids = _flatten_ids(manipulating)

    obj = ann.get("object_id", [])
    if obj is None:
        obj = []
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        obj_ids = obj[0]
    elif isinstance(obj, list):
        obj_ids = obj
    else:
        obj_ids = [obj]
    return manipulating_ids, obj_ids


def _primitive_to_caption(ann: dict, primitive_text: str) -> str:
    verb = primitive_text.strip().lower()
    spatial = _spatial_prefix_to_phrase(ann.get("spatial_prefix", []))

    manipulating_ids, object_ids = _extract_object_ids(ann)
    manipulating = [_object_id_to_phrase(x) for x in manipulating_ids]
    object_groups = object_ids if isinstance(object_ids, list) else [object_ids]
    all_obj_ids = _flatten_ids(object_groups)

    if verb in {"open door", "close door"}:
        for x in all_obj_ids:
            phrase = _object_id_to_phrase(x)
            if not phrase:
                continue
            if any(k in phrase for k in ("fridge", "cabinet", "drawer", "door")):
                return f"{verb.split()[0]} the {phrase}".strip()

    m = re.search(r"\b(from|on|in|into|onto|to|under|over|inside)\s*$", verb)
    if m:
        prep = m.group(1)
        action = verb[: m.start(1)].strip()
    elif spatial:
        prep = spatial
        action = verb
    else:
        prep = None
        action = verb

    if prep and len(object_groups) >= 2:
        target_ids = _flatten_ids(object_groups[-1])
        mains_ids = _flatten_ids(object_groups[:-1])
        target = _object_id_to_phrase(target_ids[0]) if target_ids else None
        mains = manipulating if manipulating else [_object_id_to_phrase(x) for x in mains_ids]
        mains = [x for x in mains if x]
        mains = list(dict.fromkeys(mains))
        if not mains and target:
            return f"{action} {prep} the {target}".strip()
        if not target:
            return f"{action} {_format_obj_list(mains)}".strip()
        return f"{action} {_format_obj_list(mains)} {prep} the {target}".strip()

    if prep and len(object_groups) == 1 and not manipulating:
        target_ids = _flatten_ids(object_groups[0])
        target = _object_id_to_phrase(target_ids[0]) if target_ids else None
        if target:
            return f"{action} {prep} the {target}".strip()

    mains_ids = _flatten_ids(object_groups)
    mains = manipulating if manipulating else [_object_id_to_phrase(x) for x in mains_ids[:1]]
    mains = [x for x in mains if x]
    mains = list(dict.fromkeys(mains))
    if mains:
        return f"{action} {_format_obj_list(mains)}".strip()
    return action.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_path",
        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos",
    )
    p.add_argument(
        "--output_path",
        default=os.path.join(os.path.dirname(__file__), "b1k_subtask_supervision.json"),
    )
    p.add_argument("--annotation_level", choices=["primitive", "skill"], default="primitive")
    p.add_argument("--max_tasks", type=int, default=50)
    args = p.parse_args()

    try:
        import tqdm

        progress = tqdm.tqdm
    except Exception:
        progress = lambda x: x

    tasks = _load_tasks_jsonl(args.dataset_path)
    out = {}

    for task_id in progress(range(args.max_tasks)):
        task_info = tasks.get(task_id, {})
        task_name = task_info.get("task_name") or f"task_{task_id:04d}"
        task_description = task_info.get("task") or task_name

        task_dir = os.path.join(args.dataset_path, "annotations", f"task-{task_id:04d}")
        if not os.path.isdir(task_dir):
            continue

        caption_votes_by_idx: dict[int, collections.Counter] = {}
        max_idx = -1
        for fn in glob.glob(os.path.join(task_dir, "*.json")):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Warning: failed to load {fn}: {e}", file=sys.stderr)
                continue

            if args.annotation_level == "primitive":
                anns = data.get("primitive_annotation", [])
                keys = ("primitive_description", "primitive_name", "primitive_label", "primitive")
                idx_key = "primitive_idx"
            else:
                anns = data.get("skill_annotation", [])
                keys = ("skill_description", "skill_name", "skill_label", "skill")
                idx_key = "skill_idx"

            if not isinstance(anns, list):
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                idx = ann.get(idx_key, None)
                if not isinstance(idx, int):
                    continue
                primitive_text = _get_text_from_ann(ann, keys)
                if primitive_text is None:
                    continue
                caption = _primitive_to_caption(ann, primitive_text)
                if not caption:
                    continue

                max_idx = max(max_idx, idx)
                counter = caption_votes_by_idx.get(idx)
                if counter is None:
                    counter = collections.Counter()
                    caption_votes_by_idx[idx] = counter
                counter[caption] += 1

        if max_idx < 0:
            continue
        primitive_captions = []
        for idx in range(max_idx + 1):
            counter = caption_votes_by_idx.get(idx)
            if not counter:
                primitive_captions.append("")
                continue
            primitive_captions.append(counter.most_common(1)[0][0])

        out[task_name] = {
            "task_description": task_description,
            "primitive_captions": primitive_captions,
        }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {len(out)} tasks to: {args.output_path}")


if __name__ == "__main__":
    main()
