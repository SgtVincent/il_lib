from __future__ import annotations

from typing import Dict, List, Optional

import torch


def normalize_resample_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if weights is None:
        return {}
    return {str(k): float(v) for k, v in dict(weights).items()}


def build_resampled_indices(
    *,
    base_indices: List[int],
    segment_group_keys: List[str],
    group_weights: Dict[str, float],
    default_weight: float,
    num_samples: int,
    generator: torch.Generator,
) -> List[int]:
    if len(base_indices) == 0:
        return []
    if len(segment_group_keys) == 0:
        return list(base_indices)
    if num_samples <= 0:
        return []

    w = torch.empty((len(base_indices),), dtype=torch.float32)
    for i, seg_ptr in enumerate(base_indices):
        key = segment_group_keys[int(seg_ptr)]
        weight = float(group_weights.get(key, default_weight))
        w[i] = 0.0 if weight < 0.0 else weight

    if float(w.sum().item()) <= 0.0:
        return list(base_indices)

    sampled = torch.multinomial(w, num_samples=int(num_samples), replacement=True, generator=generator)
    return [base_indices[int(i)] for i in sampled.tolist()]
