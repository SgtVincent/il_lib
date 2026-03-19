import os
import sys
import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


_IL_LIB_ROOT = os.path.dirname(os.path.dirname(__file__))
if _IL_LIB_ROOT not in sys.path:
    sys.path.insert(0, _IL_LIB_ROOT)


@dataclass
class Stats:
    count: int
    mean: List[float]
    std: List[float]
    min: List[float]
    max: List[float]


def _import_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _to_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.as_tensor(x)


def _flatten_action(actions: Any, action_keys: Optional[List[str]] = None) -> torch.Tensor:
    if isinstance(actions, dict):
        if action_keys is None:
            keys = list(actions.keys())
        else:
            keys = list(action_keys)
        parts = [_to_tensor(actions[k]) for k in keys]
        return torch.cat(parts, dim=-1)
    return _to_tensor(actions)


def _iter_samples(dataset: Any, max_samples: int) -> Iterable[Dict[str, Any]]:
    if hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__"):
        it = iter(dataset)
        for _ in range(max_samples):
            yield next(it)
        return
    n = len(dataset)
    m = min(n, max_samples)
    for i in range(m):
        yield dataset[i]


def _compute_stats(x: torch.Tensor) -> Stats:
    if x.ndim != 2:
        raise ValueError(f"Expected (N,A), got {tuple(x.shape)}")
    x = x.float()
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
    return Stats(
        count=int(x.shape[0]),
        mean=mean.cpu().tolist(),
        std=std.cpu().tolist(),
        min=mn.cpu().tolist(),
        max=mx.cpu().tolist(),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--task_name", type=str, default="dummy")
    p.add_argument("--dataset_class", type=str, default="il_lib.datas.dataset.SyntheticBehaviorDataset")
    p.add_argument("--robot_type", type=str, default="R1Pro")
    p.add_argument("--obs_window_size", type=int, default=2)
    p.add_argument("--ctx_len", type=int, default=16)
    p.add_argument("--max_num_demos", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_keys", type=str, default="base,torso,left_arm,left_gripper,right_arm,right_gripper")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    Dataset = _import_class(args.dataset_class)
    action_keys = [k.strip() for k in args.action_keys.split(",") if k.strip()]

    demo_keys = Dataset.get_all_demo_keys(args.data_dir, args.task_name)
    demo_keys = list(demo_keys)[: int(args.max_num_demos)]
    dataset = Dataset(
        data_path=args.data_dir,
        demo_keys=demo_keys,
        robot_type=args.robot_type,
        obs_window_size=int(args.obs_window_size),
        ctx_len=int(args.ctx_len),
        seed=int(args.seed),
        visual_obs_types=["pcd"],
    )

    samples = []
    for sample in _iter_samples(dataset, int(args.max_samples)):
        a = _flatten_action(sample["actions"], action_keys=action_keys)
        if a.ndim == 3:
            a = a.reshape(-1, a.shape[-1])
        elif a.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported actions shape: {tuple(a.shape)}")
        samples.append(a)

    x = torch.cat(samples, dim=0)
    stats = _compute_stats(x)

    out = {
        "dataset_class": args.dataset_class,
        "data_dir": args.data_dir,
        "task_name": args.task_name,
        "robot_type": args.robot_type,
        "action_keys": action_keys,
        "action_dim": int(x.shape[-1]),
        "stats": asdict(stats),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
