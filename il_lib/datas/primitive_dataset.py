import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset
from omnigibson.learning.utils.array_tensor_utils import sequential_sum_balanced_partitioning


class PrimitiveIterableDataset(BehaviorIterableDataset):
    """A primitive-filtered variant of `BehaviorIterableDataset`.

    This dataset yields only windows whose observation window **and** action context
    window are fully contained within a selected primitive segment.

    Selection options:
    - `primitive_idx`: choose the primitive by index after sorting by start frame.
    - `primitive_desc`: choose primitives whose description contains this substring.
    """

    def __init__(
        self,
        *args,
        primitive_idx: Optional[int] = None,
        primitive_desc: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if primitive_idx is None and primitive_desc is None:
            raise ValueError("Must provide either primitive_idx or primitive_desc")
        self._primitive_idx = None if primitive_idx is None else int(primitive_idx)
        self._primitive_desc = primitive_desc

        self._segments: List[Tuple[int, int, int]] = []
        self._segment_lengths: List[int] = []
        self._filter_segments()

    def _filter_segments(self) -> None:
        new_segments: List[Tuple[int, int, int]] = []
        new_lengths: List[int] = []

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
                ann_data = json.load(f)

            primitives = ann_data.get("primitive_annotation", [])
            if not primitives:
                continue

            primitives = sorted(primitives, key=lambda x: x["frame_duration"][0])

            selected = []
            if self._primitive_idx is not None:
                if 0 <= self._primitive_idx < len(primitives):
                    selected = [primitives[self._primitive_idx]]
            else:
                assert self._primitive_desc is not None
                # Allow CLI-friendly tokens like "pick_up_from".
                query_variants = {self._primitive_desc}
                if "_" in self._primitive_desc and " " not in self._primitive_desc:
                    query_variants.add(self._primitive_desc.replace("_", " "))

                for prim in primitives:
                    desc_list = prim.get("primitive_description", [])
                    for desc in desc_list:
                        if not isinstance(desc, str):
                            continue
                        desc_l = desc.lower()
                        if any(q.lower() in desc_l for q in query_variants):
                            selected.append(prim)
                            break

            for prim in selected:
                start, end = prim["frame_duration"]
                ds = self._downsample_factor
                # Convert raw frames (30Hz) to downsampled indices.
                s_idx = int(np.ceil(start / ds))
                e_idx = int(np.floor(end / ds))

                # Require both obs window and ctx_len actions to be fully contained.
                window_need = max(self._obs_window_size, self._ctx_len)
                max_valid_idx = e_idx - window_need + 1
                if max_valid_idx > s_idx:
                    new_segments.append((demo_ptr, s_idx, max_valid_idx))
                    new_lengths.append(max_valid_idx - s_idx)

        self._segments = new_segments
        self._segment_lengths = new_lengths
        print(
            f"Filtered dataset for primitive idx={self._primitive_idx} desc={self._primitive_desc!r}: "
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

        lengths_shuffled = [self._segment_lengths[i] for i in indices]
        start_seg_id, start_seg_idx, end_seg_id, end_seg_idx = sequential_sum_balanced_partitioning(
            lengths_shuffled, total_global_workers, global_worker_id
        )

        for i, seg_ptr in enumerate(indices[start_seg_id : end_seg_id + 1]):
            demo_ptr, seg_start, seg_end = self._segments[seg_ptr]

            curr_start = seg_start
            curr_end = seg_end
            if i == 0:
                curr_start += start_seg_idx
            if i == end_seg_id - start_seg_id:
                curr_end = seg_start + end_seg_idx

            yield from self.get_streamed_data(demo_ptr, curr_start, curr_end)