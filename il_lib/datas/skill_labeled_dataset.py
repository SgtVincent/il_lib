import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch


from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset
from omnigibson.learning.utils.array_tensor_utils import sequential_sum_balanced_partitioning
from il_lib.datas.resample_utils import build_resampled_indices, normalize_resample_weights



class SkillLabeledIterableDataset(BehaviorIterableDataset):
    def __init__(
        self,
        *args,
        label_level: str = "skill",
        label_mode: str = "id",
        skill_vocab_path: Optional[str] = None,
        skill_vocab: Optional[List[str]] = None,
        num_skills: Optional[int] = None,
        resample_group_by: Optional[str] = None,
        resample_weights: Optional[Dict[str, float]] = None,
        resample_default_weight: float = 1.0,
        resample_num_segments: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._label_level = str(label_level)
        self._label_mode = str(label_mode)
        self._skill_vocab_path = skill_vocab_path
        self._skill_vocab = list(skill_vocab) if skill_vocab is not None else None
        self._num_skills = int(num_skills) if num_skills is not None else None
        self._resample_group_by = None if resample_group_by is None else str(resample_group_by)
        self._resample_weights = normalize_resample_weights(resample_weights)
        self._resample_default_weight = float(resample_default_weight)
        self._resample_num_segments = None if resample_num_segments is None else int(resample_num_segments)

        if self._label_level not in {"skill", "primitive"}:
            raise ValueError(f"Unsupported label_level={self._label_level}")
        if self._label_mode not in {"id", "one_hot"}:
            raise ValueError(f"Unsupported label_mode={self._label_mode}")
        if self._resample_group_by not in {None, "skill_type", "skill_description"}:
            raise ValueError(f"Unsupported resample_group_by={self._resample_group_by}")
        if self._label_level != "skill" and self._resample_group_by == "skill_type":
            raise ValueError("resample_group_by=skill_type requires label_level=skill")
        if self._resample_num_segments is not None and self._resample_num_segments <= 0:
            raise ValueError(f"resample_num_segments must be positive, got {self._resample_num_segments}")

        self._label_to_id: Dict[str, int] = {}
        self._segments: List[Tuple[int, int, int, int]] = []
        self._segment_lengths: List[int] = []
        self._segment_skill_types: List[str] = []
        self._segment_skill_descs: List[str] = []

        self._build_vocab()
        self._filter_segments()

    def _build_vocab(self) -> None:
        if self._skill_vocab is not None:
            vocab = list(self._skill_vocab)
        elif self._skill_vocab_path is not None:
            with open(os.path.expanduser(self._skill_vocab_path), "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                vocab = list(sorted(loaded.keys()))
            elif isinstance(loaded, list):
                vocab = list(loaded)
            else:
                raise ValueError("skill_vocab_path must point to a json list[str] or dict[str,int]")
        else:
            vocab_set = set()
            for demo_key in self._demo_keys:
                task_id = int(demo_key) // 10000
                task_name_idx = f"task-{task_id:04d}"
                episode_name = f"episode_{demo_key}"
                ann_path = os.path.join(
                    self._data_path,
                    "2025-challenge-demos",
                    "annotations",
                    task_name_idx,
                    f"{episode_name}.json",
                )
                if not os.path.exists(ann_path):
                    continue
                with open(ann_path, "r") as f:
                    ann = json.load(f)
                anns = ann.get("skill_annotation" if self._label_level == "skill" else "primitive_annotation", [])
                for a in anns:
                    field = "skill_description" if self._label_level == "skill" else "primitive_description"
                    desc_list = a.get(field, [])
                    for desc in desc_list:
                        if isinstance(desc, str) and len(desc) > 0:
                            vocab_set.add(desc)
                            break
            vocab = sorted(list(vocab_set))

        if self._num_skills is None:
            self._num_skills = len(vocab)
        if self._num_skills <= 0:
            raise ValueError(f"num_skills must be positive, got {self._num_skills}")

        self._label_to_id = {name: i for i, name in enumerate(vocab[: self._num_skills])}

    def _filter_segments(self) -> None:
        new_segments: List[Tuple[int, int, int, int]] = []
        new_lengths: List[int] = []
        new_skill_types: List[str] = []
        new_skill_descs: List[str] = []

        field = "skill_description" if self._label_level == "skill" else "primitive_description"
        ann_key = "skill_annotation" if self._label_level == "skill" else "primitive_annotation"
        window_need = max(self._obs_window_size, self._ctx_len)

        for demo_ptr, demo_key in enumerate(self._demo_keys):
            task_id = int(demo_key) // 10000
            task_name_idx = f"task-{task_id:04d}"
            episode_name = f"episode_{demo_key}"

            ann_path = os.path.join(
                self._data_path,
                "2025-challenge-demos",
                "annotations",
                task_name_idx,
                f"{episode_name}.json",
            )
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, "r") as f:
                ann = json.load(f)

            anns = ann.get(ann_key, [])
            for a in anns:
                desc_list = a.get(field, [])
                label_name = None
                for d in desc_list:
                    if isinstance(d, str) and len(d) > 0:
                        label_name = d
                        break
                if label_name is None:
                    continue
                if label_name not in self._label_to_id:
                    continue
                label_id = int(self._label_to_id[label_name])

                start, end = a["frame_duration"]
                skill_type = ""
                if self._label_level == "skill":
                    skill_type = str(a.get("skill_type", "")) if a.get("skill_type", "") is not None else ""
                ds = self._downsample_factor
                s_idx = int(np.ceil(start / ds))
                e_idx = int(np.floor(end / ds))
                max_valid_idx = e_idx - window_need + 1
                if max_valid_idx > s_idx:
                    new_segments.append((demo_ptr, s_idx, max_valid_idx, label_id))
                    new_lengths.append(max_valid_idx - s_idx)
                    new_skill_types.append(skill_type)
                    new_skill_descs.append(str(label_name))

        self._segments = new_segments
        self._segment_lengths = new_lengths
        self._segment_skill_types = new_skill_types
        self._segment_skill_descs = new_skill_descs
        print(
            f"SkillLabeledIterableDataset(level={self._label_level}, mode={self._label_mode}): "
            f"{len(self._segments)} segments, total chunks: {sum(self._segment_lengths)}"
        )

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        global_worker_id, total_global_workers = self._get_global_worker_id()

        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._epoch + self._seed)
            indices = torch.randperm(len(self._segments), generator=g).tolist()
        else:
            indices = list(range(len(self._segments)))

        if self._resample_group_by is not None:
            group_keys = (
                self._segment_skill_types if self._resample_group_by == "skill_type" else self._segment_skill_descs
            )
            g_resample = torch.Generator()
            g_resample.manual_seed(self._epoch + self._seed + 99991)
            indices = build_resampled_indices(
                base_indices=indices,
                segment_group_keys=group_keys,
                group_weights=self._resample_weights,
                default_weight=self._resample_default_weight,
                num_samples=len(indices) if self._resample_num_segments is None else int(self._resample_num_segments),
                generator=g_resample,
            )

        lengths_shuffled = [self._segment_lengths[i] for i in indices]
        start_seg_id, start_seg_idx, end_seg_id, end_seg_idx = sequential_sum_balanced_partitioning(
            lengths_shuffled, total_global_workers, global_worker_id
        )

        for i, seg_ptr in enumerate(indices[start_seg_id : end_seg_id + 1]):
            demo_ptr, seg_start, seg_end, label_id = self._segments[seg_ptr]
            demo_key = int(self._demo_keys[demo_ptr])

            curr_start = seg_start
            curr_end = seg_end
            if i == 0:
                curr_start += start_seg_idx
            if i == end_seg_id - start_seg_id:
                curr_end = seg_start + end_seg_idx

            for local_idx, sample in enumerate(self.get_streamed_data(demo_ptr, curr_start, curr_end)):
                sample["demo_key"] = demo_key
                sample["index"] = int(curr_start + local_idx)
                obs = sample.get("obs")
                if isinstance(obs, dict):
                    L = int(self._obs_window_size)
                    if self._label_mode == "id":
                        obs["skill_id"] = torch.full((L,), label_id, dtype=torch.int64)
                    else:
                        one_hot = torch.zeros((L, int(self._num_skills)), dtype=torch.float32)
                        one_hot[:, label_id] = 1.0
                        obs["skill_one_hot"] = one_hot
                yield sample
