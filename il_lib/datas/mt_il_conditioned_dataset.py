import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset
from omnigibson.learning.utils.array_tensor_utils import sequential_sum_balanced_partitioning

from il_lib.datas.resample_utils import build_resampled_indices, normalize_resample_weights


@dataclass
class _EpisodeCache:
    key: str
    tokens: torch.Tensor


class MTILConditionedIterableDataset(BehaviorIterableDataset):
    def __init__(
        self,
        *args,
        label_level: str = "skill",
        label_mode: str = "id",
        num_skills: int = 64,
        cache_dir: Optional[str] = None,
        paligemma_token_dim: Optional[int] = None,
        cache_field: str = "paligemma_token",
        cache_ext: str = "pt",
        cache_is_downsampled: bool = True,
        strict_cache: bool = True,
        resample_group_by: Optional[str] = None,
        resample_weights: Optional[Dict[str, float]] = None,
        resample_default_weight: float = 1.0,
        resample_num_segments: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._label_level = str(label_level)
        self._label_mode = str(label_mode)
        self._num_skills = int(num_skills)

        self._cache_dir = None if cache_dir is None else os.path.expanduser(str(cache_dir))
        self._paligemma_token_dim = None if paligemma_token_dim is None else int(paligemma_token_dim)
        self._cache_field = str(cache_field)
        self._cache_ext = str(cache_ext)
        self._cache_is_downsampled = bool(cache_is_downsampled)
        self._strict_cache = bool(strict_cache)
        self._last_episode: Optional[_EpisodeCache] = None
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
        if self._num_skills <= 0:
            raise ValueError(f"num_skills must be positive, got {self._num_skills}")
        if self._cache_ext not in {"pt"}:
            raise ValueError(f"Unsupported cache_ext={self._cache_ext}")
        if (self._cache_dir is None) != (self._paligemma_token_dim is None):
            raise ValueError("cache_dir and paligemma_token_dim must be both set or both None")

        self._label_to_id: Dict[str, int] = {}
        self._segments: List[Tuple[int, int, int, int]] = []
        self._segment_lengths: List[int] = []
        self._segment_skill_types: List[str] = []
        self._segment_skill_descs: List[str] = []

        self._build_vocab()
        self._filter_segments()

    def _build_vocab(self) -> None:
        vocab_set = set()
        field = "skill_description" if self._label_level == "skill" else "primitive_description"
        ann_key = "skill_annotation" if self._label_level == "skill" else "primitive_annotation"

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
            anns = ann.get(ann_key, [])
            for a in anns:
                desc_list = a.get(field, [])
                for desc in desc_list:
                    if isinstance(desc, str) and len(desc) > 0:
                        vocab_set.add(desc)
                        break

        vocab = sorted(list(vocab_set))
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
            f"MTILConditionedIterableDataset(level={self._label_level}, mode={self._label_mode}, cache={self._cache_dir is not None}): "
            f"{len(self._segments)} segments, total chunks: {sum(self._segment_lengths)}"
        )

    def _resolve_cache_path(self, episode_name: str, task_id: int) -> str:
        assert self._cache_dir is not None
        candidates = [
            os.path.join(self._cache_dir, f"{episode_name}.{self._cache_ext}"),
            os.path.join(self._cache_dir, f"task-{int(task_id):04d}", f"{episode_name}.{self._cache_ext}"),
            os.path.join(self._cache_dir, f"{episode_name.replace('episode_', '')}.{self._cache_ext}"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"paligemma_token cache not found for {episode_name}: tried {candidates}")

    def _load_episode_tokens(self, episode_name: str, task_id: int) -> torch.Tensor:
        assert self._paligemma_token_dim is not None
        if self._last_episode is not None and self._last_episode.key == episode_name:
            return self._last_episode.tokens

        path = self._resolve_cache_path(episode_name, task_id)
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if self._cache_field not in obj:
                raise KeyError(f"cache file missing field={self._cache_field}: {path}")
            tok = obj[self._cache_field]
        else:
            tok = obj
        if not isinstance(tok, torch.Tensor):
            tok = torch.as_tensor(tok)
        tok = tok.to(dtype=torch.float32)
        if tok.ndim != 2 or tok.shape[-1] != self._paligemma_token_dim:
            raise ValueError(f"token shape mismatch: got {tuple(tok.shape)}, expected (N,{self._paligemma_token_dim})")

        self._last_episode = _EpisodeCache(key=episode_name, tokens=tok)
        return tok

    def _extract_index(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        idx = None
        if "index" in sample:
            idx = sample["index"]
        else:
            obs = sample.get("obs")
            if isinstance(obs, dict) and "index" in obs:
                idx = obs["index"]
        if idx is None:
            return None
        if isinstance(idx, (int, float)):
            return torch.tensor([int(idx)], dtype=torch.long)
        if isinstance(idx, torch.Tensor):
            if idx.ndim == 0:
                return idx.to(dtype=torch.long).view(1)
            return idx.to(dtype=torch.long).view(-1)
        return torch.as_tensor(idx, dtype=torch.long).view(-1)

    def _inject_conditions(
        self,
        sample: Dict[str, Any],
        *,
        label_id: int,
        episode_name: str,
        task_id: int,
    ) -> Dict[str, Any]:
        obs = sample.get("obs")
        if not isinstance(obs, dict):
            return sample

        L = int(self._obs_window_size)
        if self._label_mode == "id":
            obs["skill_id"] = torch.full((L,), int(label_id), dtype=torch.int64)
        else:
            one_hot = torch.zeros((L, int(self._num_skills)), dtype=torch.float32)
            one_hot[:, int(label_id)] = 1.0
            obs["skill_one_hot"] = one_hot

        if self._cache_dir is None:
            return sample

        indices = self._extract_index(sample)
        if indices is None:
            if self._strict_cache:
                raise KeyError("sample missing index for paligemma_token alignment")
            return sample

        if indices.numel() == 1:
            start = int(indices.item())
            indices = torch.arange(start, start + L, dtype=torch.long)

        if not self._cache_is_downsampled:
            indices = indices * int(self._downsample_factor)

        tok = self._load_episode_tokens(episode_name, task_id)
        if indices.max().item() >= tok.shape[0] or indices.min().item() < 0:
            if self._strict_cache:
                raise IndexError(f"index out of range: [{int(indices.min().item())},{int(indices.max().item())}] vs N={tok.shape[0]}")
            return sample

        obs[self._cache_field] = tok.index_select(0, indices)
        return sample

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
            task_id = int(demo_key) // 10000
            episode_name = f"episode_{demo_key}"

            curr_start = seg_start
            curr_end = seg_end
            if i == 0:
                curr_start += start_seg_idx
            if i == end_seg_id - start_seg_id:
                curr_end = seg_start + end_seg_idx

            for local_idx, sample in enumerate(self.get_streamed_data(demo_ptr, curr_start, curr_end)):
                sample["demo_key"] = demo_key
                sample["index"] = int(curr_start + local_idx)
                yield self._inject_conditions(
                    sample,
                    label_id=int(label_id),
                    episode_name=episode_name,
                    task_id=task_id,
                )
