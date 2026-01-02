#!/usr/bin/env python3
"""
collect_task_skills.py

Quick utility to collect the skills used by each task (50 tasks) from the
challenge dataset annotations and write two CSVs into the script directory by default:

- `task_skills.csv`         : rows = tasks, columns = task_id,task_name,num_skills,skills (semicolon-separated)
- `task_skills_matrix.csv`  : binary matrix where columns are unique skills across tasks

Usage example:
    python collect_task_skills.py --dataset_path /path/to/2025-challenge-demos --output_dir /path/to/tutorials
"""

import os
import json
import glob
import csv
import argparse
import sys

# Import task mapping directly from OmniGibson
# sys.path.insert(0, "/home/ubuntu/ove_repo/BEHAVIOR-1K/OmniGibson")
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
import tqdm


def get_skill_from_ann(ann: dict) -> str | None:
    """Return a human-readable skill string if available in the annotation dict."""
    for key in ("skill_description", "skill_name", "skill_label", "skill", "skill_id"):
        if key in ann:
            v = ann[key]
            if isinstance(v, list):
                v = v[0] if v else None
            if v is None:
                continue
            # If skill_id is numeric, prefix so the value is still informative
            if key == "skill_id" and not isinstance(v, str):
                return f"skill_id:{v}"
            return str(v).strip()
    return None


def collect_skills_for_task(dataset_root: str, task_id: int) -> set:
    """Scan all annotation files for a task and return a set of skill strings."""
    task_dir = os.path.join(dataset_root, "annotations", f"task-{task_id:04d}")
    skills = set()
    if not os.path.isdir(task_dir):
        return skills
    for fn in glob.glob(os.path.join(task_dir, "*.json")):
        try:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {fn}: {e}", file=sys.stderr)
            continue
        skill_anns = data.get("skill_annotation", [])
        if not isinstance(skill_anns, list):
            continue
        for ann in skill_anns:
            name = get_skill_from_ann(ann)
            if name:
                skills.add(name)
    return skills


def main():
    p = argparse.ArgumentParser(description="Collect skills per task and write CSVs")
    p.add_argument("--dataset_path", default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos", 
                   help="Path to dataset root containing `annotations/`")
    p.add_argument("--output_dir", default=os.path.dirname(__file__), help="Where to write CSV outputs (defaults to this script dir)")
    args = p.parse_args()

    # Resolve dataset path via common defaults if not provided
    dataset_path = args.dataset_path

    # Use imported task mapping
    tasks = sorted(TASK_NAMES_TO_INDICES.items(), key=lambda kv: kv[1])  # list of (task_name, task_id)

    results = {}
    for task_name, task_id in tqdm.tqdm(tasks):
        skills = collect_skills_for_task(dataset_path, task_id)
        results[(task_name, task_id)] = skills

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Write per-task skills list CSV
    tasks_file = os.path.join(args.output_dir, "task_skills.csv")
    with open(tasks_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "task_name", "num_skills", "skills_semicolon_separated"])
        for (tname, tid), skills in sorted(results.items(), key=lambda kv: kv[0][1]):
            skills_list = sorted(skills)
            writer.writerow([tid, tname, len(skills_list), "; ".join(skills_list)])

    # 2) Write a binary matrix (tasks x skills)
    all_skills = sorted({s for sset in results.values() for s in sset})
    matrix_file = os.path.join(args.output_dir, "task_skills_matrix.csv")
    with open(matrix_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "task_name"] + all_skills)
        for (tname, tid), skills in sorted(results.items(), key=lambda kv: kv[0][1]):
            row = [tid, tname] + ["1" if s in skills else "0" for s in all_skills]
            writer.writerow(row)

    print("Done. Wrote:")
    print(" - ", tasks_file)
    print(" - ", matrix_file)


if __name__ == "__main__":
    main()
