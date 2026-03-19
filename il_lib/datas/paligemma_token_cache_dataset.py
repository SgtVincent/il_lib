import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Tuple

import torch

from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset
from omnigibson.learning.utils.array_tensor_utils import sequential_sum_balanced_partitioning


@dataclass
class _EpisodeCache:
    key: str
    tokens: torch.Tensor


class PaligemmaTokenCachedIterableDataset(BehaviorIterableDataset):
    def __init__(
        self,
        *args,
        cache_dir: str,
        token_dim: int,
        cache_field: str = "paligemma_token",
        cache_ext: str = "pt",
        strict: bool = True,
        cache_is_downsampled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._cache_dir = os.path.expanduser(str(cache_dir))
        self._token_dim = int(token_dim)
        self._cache_field = str(cache_field)
        self._cache_ext = str(cache_ext)
        self._strict = bool(strict)
        self._cache_is_downsampled = bool(cache_is_downsampled)
        self._last_episode: Optional[_EpisodeCache] = None

        if self._cache_ext not in {"pt"}:
            raise ValueError(f"Unsupported cache_ext={self._cache_ext}")

    def _extract_episode_key(self, sample: Dict[str, Any]) -> Tuple[str, Optional[int]]:
        if "demo_key" in sample:
            demo_key = str(sample["demo_key"])
            try:
                task_id = int(demo_key) // 10000
            except Exception:
                task_id = None
            return f"episode_{demo_key}", task_id

        episode_index = None
        if "episode_index" in sample:
            episode_index = int(sample["episode_index"])
        else:
            obs = sample.get("obs")
            if isinstance(obs, dict) and "episode_index" in obs:
                episode_index = int(obs["episode_index"])
        if episode_index is None:
            raise KeyError("Cannot find episode key: expected sample['demo_key'] or sample['episode_index']")
        return f"episode_{episode_index:08d}", None

    def _extract_window_indices(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
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

    def _resolve_cache_path(self, episode_name: str, task_id: Optional[int]) -> str:
        candidates = []
        candidates.append(os.path.join(self._cache_dir, f"{episode_name}.{self._cache_ext}"))
        if task_id is not None:
            candidates.append(
                os.path.join(self._cache_dir, f"task-{int(task_id):04d}", f"{episode_name}.{self._cache_ext}")
            )
        candidates.append(os.path.join(self._cache_dir, f"{episode_name.replace('episode_', '')}.{self._cache_ext}"))

        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"paligemma_token cache not found for {episode_name}: tried {candidates}")

    def _load_episode_tokens(self, episode_name: str, task_id: Optional[int]) -> torch.Tensor:
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
        if tok.ndim != 2 or tok.shape[-1] != self._token_dim:
            raise ValueError(f"token shape mismatch: got {tuple(tok.shape)}, expected (N,{self._token_dim})")

        self._last_episode = _EpisodeCache(key=episode_name, tokens=tok)
        return tok

    def _inject(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "obs" not in sample or not isinstance(sample["obs"], dict):
            if self._strict:
                raise KeyError("sample must contain dict obs")
            return sample

        episode_name, task_id = self._extract_episode_key(sample)
        episode_tokens = self._load_episode_tokens(episode_name, task_id)

        indices = self._extract_window_indices(sample)
        if indices is None:
            if self._strict:
                raise KeyError("sample missing index for paligemma_token alignment")
            return sample

        if indices.numel() == 1:
            start = int(indices.item())
            L = int(self._obs_window_size)
            indices = torch.arange(start, start + L, dtype=torch.long)

        if not self._cache_is_downsampled:
            indices = indices * int(self._downsample_factor)

        if indices.max().item() >= episode_tokens.shape[0] or indices.min().item() < 0:
            if self._strict:
                raise IndexError(
                    f"index out of range: [{int(indices.min().item())},{int(indices.max().item())}] vs N={episode_tokens.shape[0]}"
                )
            return sample

        sample["obs"][self._cache_field] = episode_tokens.index_select(0, indices)
        return sample

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        global_worker_id, total_global_workers = self._get_global_worker_id()
        demo_lengths_shuffled = [self._demo_lengths[i] for i in self._demo_indices]
        start_demo_id, start_demo_idx, end_demo_id, end_demo_idx = sequential_sum_balanced_partitioning(
            demo_lengths_shuffled, total_global_workers, global_worker_id
        )
        for demo_idx, demo_ptr in enumerate(self._demo_indices[start_demo_id : end_demo_id + 1]):
            start_idx = start_demo_idx if demo_idx == 0 else 0
            end_idx = end_demo_idx if demo_idx == end_demo_id - start_demo_id else self._demo_lengths[demo_ptr]
            demo_key = int(self._demo_keys[demo_ptr])
            for local_idx, sample in enumerate(self.get_streamed_data(demo_ptr, start_idx, end_idx)):
                sample["demo_key"] = demo_key
                sample["index"] = int(start_idx + local_idx)
                yield self._inject(sample)
