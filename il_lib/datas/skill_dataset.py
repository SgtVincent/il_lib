import json
import os
import torch
import numpy as np
from typing import List, Any, Dict, Generator
from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset
from omnigibson.learning.utils.array_tensor_utils import sequential_sum_balanced_partitioning
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES

class SkillIterableDataset(BehaviorIterableDataset):
    def __init__(
        self,
        *args,
        skill_name: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self._skill_name = skill_name
        self._segments = [] # List of (demo_ptr, start_idx, end_idx)
        self._segment_lengths = []
        
        # Filter segments
        self._filter_segments()
        
    def _filter_segments(self):
        # Iterate over all demos and find the skill segments
        new_segments = []
        new_lengths = []
        
        for demo_ptr, demo_key in enumerate(self._demo_keys):
            task_id = int(demo_key) // 10000
            task_name_idx = f"task-{task_id:04d}"
            episode_name = f"episode_{demo_key}"
            
            # Load annotation
            ann_path = os.path.join(
                self._data_path,
                "2025-challenge-demos",
                "annotations", 
                task_name_idx, 
                f"{episode_name}.json"
            )
            
            if not os.path.exists(ann_path):
                continue
                
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
                
            # Check skill annotations
            for skill in ann_data.get("skill_annotation", []):
                # Check if skill_name matches any of the descriptions
                # skill_description is a list
                if any(self._skill_name in desc for desc in skill.get("skill_description", [])):
                    start, end = skill["frame_duration"]
                    
                    ds = self._downsample_factor
                    
                    # Convert raw frames to downsampled indices
                    s_idx = int(np.ceil(start / ds))
                    e_idx = int(np.floor(end / ds))
                    
                    # We need to ensure we have enough frames for obs_window_size
                    # The effective length of demo chunking is end_idx - start_idx - obs_window_size + 1
                    # get_streamed_data iterates from start_idx to end_idx (exclusive)
                    # and extracts window [i : i + obs_window_size]
                    # So i can go up to end_idx - obs_window_size.
                    
                    # We want to include all windows that are fully within the skill?
                    # Or just windows that start in the skill?
                    # Let's assume we want windows that start in the skill.
                    # But we must ensure i + obs_window_size <= demo_length.
                    # The parent class already calculated _demo_lengths based on full demo.
                    # We just need to ensure we don't go out of bounds of the skill segment.
                    
                    # Let's say we want to yield data for the duration of the skill.
                    # The number of chunks we can yield is roughly (end - start) / ds.
                    
                    # Let's set the range for get_streamed_data.
                    # It yields chunks starting at i.
                    # We want i >= s_idx and i < e_idx.
                    # But we also need i + obs_window_size <= full_demo_len (in downsampled space).
                    # The parent _demo_lengths stores (L - obs_window_size + 1).
                    # So max i is _demo_lengths[demo_ptr].
                    
                    # Also we need to ensure i + obs_window_size <= e_idx?
                    # If we want the observation to be fully within the skill?
                    # Usually yes.
                    
                    max_valid_idx = e_idx - self._obs_window_size + 1
                    
                    if max_valid_idx > s_idx:
                        # Ensure we don't exceed the demo length
                        # (though annotation should be within demo)
                        
                        # We store (demo_ptr, s_idx, max_valid_idx)
                        # get_streamed_data iterates range(start_idx, end_idx)
                        
                        new_segments.append((demo_ptr, s_idx, max_valid_idx))
                        new_lengths.append(max_valid_idx - s_idx)

        self._segments = new_segments
        self._segment_lengths = new_lengths
        print(f"Filtered dataset for skill '{self._skill_name}': {len(self._segments)} segments, total chunks: {sum(self._segment_lengths)}")
        
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        global_worker_id, total_global_workers = self._get_global_worker_id()
        
        # Shuffle segments if needed
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
                # end_seg_idx is the length to take from the start of the segment?
                # sequential_sum_balanced_partitioning returns end_idx as exclusive end index relative to the start of the item.
                # So if we start at seg_start, we end at seg_start + end_seg_idx.
                curr_end = seg_start + end_seg_idx
                
            yield from self.get_streamed_data(demo_ptr, curr_start, curr_end)
