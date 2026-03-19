import argparse
import importlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from omegaconf import OmegaConf


_DATAMODULE_KEYS = {
    "_target_",
    "data_path",
    "task_name",
    "batch_size",
    "val_batch_size",
    "val_split_ratio",
    "dataloader_num_workers",
    "max_num_demos",
    "seed",
    "dataset_class",
    "skill_name",
}


def _load_records(path: str, max_cases: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            out.append(json.loads(line))
            if max_cases is not None and len(out) >= int(max_cases):
                break
    return out


def _group_by_demo(records: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        if "demo_key" not in r or "index" not in r:
            continue
        dk = int(r["demo_key"])
        groups.setdefault(dk, []).append(r)
    for dk in groups:
        groups[dk] = sorted(groups[dk], key=lambda x: int(x["index"]))
    return groups


def _load_dataset_class(path: str):
    module_path, class_name = str(path).rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _build_dataset(
    *,
    DatasetClass,
    data_path: str,
    demo_key: int,
    seed: int,
    dataset_kwargs: Dict[str, Any],
):
    kwargs = dict(dataset_kwargs)
    kwargs.pop("shuffle", None)
    return DatasetClass(
        data_path=os.path.expanduser(str(data_path)),
        demo_keys=[str(int(demo_key))],
        seed=int(seed),
        shuffle=False,
        **kwargs,
    )


def _fetch_window(dataset, index: int) -> Dict[str, Any]:
    gen = dataset.get_streamed_data(0, int(index), int(index) + 1)
    return next(gen)


def _resolve_failure_jsonl(*, run_dir: str, stage: str, top_k: Optional[int]) -> str:
    failure_dir = os.path.join(run_dir, "failure_cases")
    if top_k is None:
        if os.path.isdir(failure_dir):
            files = [f for f in os.listdir(failure_dir) if f.startswith(f"{stage}_top") and f.endswith(".jsonl")]
            files = sorted(files)
            if len(files) > 0:
                return os.path.join(failure_dir, files[-1])
        raise FileNotFoundError(f"Cannot find failure jsonl under {failure_dir}")
    return os.path.join(failure_dir, f"{stage}_top{int(top_k)}.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--stage", choices=["val", "test"], default="test")
    p.add_argument("--failure_jsonl", default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_cases", type=int, default=None)
    args = p.parse_args()

    run_dir = os.path.expanduser(str(args.run_dir))
    conf_path = os.path.join(run_dir, "conf.yaml")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"conf.yaml not found: {conf_path}")
    cfg = OmegaConf.load(conf_path)

    stage = str(args.stage)
    failure_jsonl = (
        os.path.expanduser(str(args.failure_jsonl))
        if args.failure_jsonl is not None
        else _resolve_failure_jsonl(run_dir=run_dir, stage=stage, top_k=args.top_k)
    )
    if not os.path.exists(failure_jsonl):
        raise FileNotFoundError(f"failure jsonl not found: {failure_jsonl}")

    out_dir = (
        os.path.expanduser(str(args.out_dir))
        if args.out_dir is not None
        else os.path.join(run_dir, "failure_windows", stage)
    )
    os.makedirs(out_dir, exist_ok=True)

    dataset_class_path = str(cfg.data.dataset_class)
    DatasetClass = _load_dataset_class(dataset_class_path)
    data_path = str(cfg.data.data_path)
    seed = int(cfg.seed)
    dataset_kwargs = {
        k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        for k, v in cfg.data.items()
        if k not in _DATAMODULE_KEYS
    }

    records = _load_records(failure_jsonl, args.max_cases)
    groups = _group_by_demo(records)
    for demo_key, rs in groups.items():
        dataset = _build_dataset(
            DatasetClass=DatasetClass,
            data_path=data_path,
            demo_key=int(demo_key),
            seed=seed,
            dataset_kwargs=dataset_kwargs,
        )
        for r in rs:
            idx = int(r["index"])
            sample = _fetch_window(dataset, idx)
            payload = {"record": r, "sample": sample}
            out_path = os.path.join(out_dir, f"demo_{int(demo_key)}_index_{idx}.pt")
            torch.save(payload, out_path)
            print(out_path)


if __name__ == "__main__":
    main()
