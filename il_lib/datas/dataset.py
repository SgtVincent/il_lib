import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH


class DummyDataset(Dataset):
    """
    Dummy dataset for test_step().
    Does absolutely nothing since we will do online evaluation.
    """

    def __init__(self, batch_size: int=1, epoch_len: int=1):
        """
        Still set batch_size because pytorch_lightning tracks it
        """
        self.n = epoch_len
        self._batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.zeros((self._batch_size,), dtype=bool)


class SyntheticBehaviorDataset(Dataset):
    def __init__(
        self,
        *args,
        data_path: str,
        demo_keys: List[Any],
        robot_type: str = "R1Pro",
        obs_window_size: int = 2,
        ctx_len: int = 16,
        seed: int = 42,
        visual_obs_types: Optional[List[str]] = None,
        use_task_info: bool = False,
        task_info_range: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._data_path = data_path
        self._demo_keys = list(demo_keys)
        self._robot_type = robot_type
        self._obs_window_size = int(obs_window_size)
        self._ctx_len = int(ctx_len)
        self._seed = int(seed)
        self._visual_obs_types = set(visual_obs_types or [])
        self._use_task_info = bool(use_task_info)
        self._task_info_range = task_info_range
        self.epoch = 0

        if self._robot_type != "R1Pro":
            raise ValueError(f"SyntheticBehaviorDataset only supports robot_type=R1Pro, got {self._robot_type}")

    @staticmethod
    def get_all_demo_keys(data_path: str, task_name: str) -> List[int]:
        return list(range(100))

    def __len__(self) -> int:
        return len(self._demo_keys)

    def _rng(self, key: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + int(key) + 100000 * int(self.epoch))
        return g

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self._demo_keys[idx]
        g = self._rng(int(key))
        L = self._obs_window_size
        T = self._ctx_len
        N = 128

        obs: Dict[str, Any] = {}
        obs["odom"] = {"base_velocity": (torch.rand((L, 3), generator=g) * 2 - 1).float()}
        obs["qpos"] = {
            "torso": (torch.rand((L, 4), generator=g) * 2 - 1).float(),
            "left_arm": (torch.rand((L, 7), generator=g) * 2 - 1).float(),
            "left_gripper": (torch.rand((L, 1), generator=g) * 2 - 1).float(),
            "right_arm": (torch.rand((L, 7), generator=g) * 2 - 1).float(),
            "right_gripper": (torch.rand((L, 1), generator=g) * 2 - 1).float(),
        }
        obs["eef"] = {
            "left_pos": (torch.rand((L, 3), generator=g) * 2 - 1).float(),
            "left_quat": (torch.rand((L, 4), generator=g) * 2 - 1).float(),
            "right_pos": (torch.rand((L, 3), generator=g) * 2 - 1).float(),
            "right_quat": (torch.rand((L, 4), generator=g) * 2 - 1).float(),
        }
        obs["pcd"] = torch.rand((L, N, 6), generator=g).float()
        if self._use_task_info:
            obs["task"] = (torch.rand((L, 8), generator=g) * 2 - 1).float()

        actions: Dict[str, torch.Tensor] = {
            "base": (torch.rand((T, 3), generator=g) * 2 - 1).float(),
            "torso": (torch.rand((T, 4), generator=g) * 2 - 1).float(),
            "left_arm": (torch.rand((T, 7), generator=g) * 2 - 1).float(),
            "left_gripper": (torch.rand((T, 1), generator=g) * 2 - 1).float(),
            "right_arm": (torch.rand((T, 7), generator=g) * 2 - 1).float(),
            "right_gripper": (torch.rand((T, 1), generator=g) * 2 - 1).float(),
        }
        masks = torch.ones((T,), dtype=torch.float32)

        return {"obs": obs, "actions": actions, "masks": masks}


class MultiTaskSyntheticBehaviorDataset(Dataset):
    def __init__(
        self,
        *args,
        data_path: str,
        demo_keys: List[Any],
        robot_type: str = "R1Pro",
        obs_window_size: int = 2,
        ctx_len: int = 16,
        seed: int = 42,
        num_tasks: int = 8,
        task_dim: int = 46,
        num_skills: Optional[int] = None,
        skill_one_hot_dim: Optional[int] = None,
        include_skill_one_hot: bool = False,
        include_skill_id: bool = False,
        paligemma_token_dim: Optional[int] = None,
        include_paligemma_token: bool = False,
        use_action_chunks: bool = False,
        action_prediction_horizon: int = 8,
        include_rgbd: bool = True,
        rgb_height: int = 256,
        rgb_width: int = 256,
        rgb_views: Optional[List[str]] = None,
        use_task_info: bool = True,
        task_info_range: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._data_path = data_path
        self._demo_keys = list(demo_keys)
        self._robot_type = robot_type
        self._obs_window_size = int(obs_window_size)
        self._ctx_len = int(ctx_len)
        self._seed = int(seed)
        self._num_tasks = int(num_tasks)
        self._task_dim = int(task_dim)
        self._num_skills = int(num_skills) if num_skills is not None else None
        self._skill_one_hot_dim = int(skill_one_hot_dim) if skill_one_hot_dim is not None else None
        self._include_skill_one_hot = bool(include_skill_one_hot)
        self._include_skill_id = bool(include_skill_id)
        self._paligemma_token_dim = int(paligemma_token_dim) if paligemma_token_dim is not None else None
        self._include_paligemma_token = bool(include_paligemma_token)
        self._use_action_chunks = bool(use_action_chunks)
        self._action_prediction_horizon = int(action_prediction_horizon)
        self._include_rgbd = bool(include_rgbd)
        self._rgb_height = int(rgb_height)
        self._rgb_width = int(rgb_width)
        self._rgb_views = rgb_views or [
            "robot_r1::robot_r1:left_realsense_link:Camera:0",
            "robot_r1::robot_r1:right_realsense_link:Camera:0",
            "robot_r1::robot_r1:zed_link:Camera:0",
        ]
        self._use_task_info = bool(use_task_info)
        self._task_info_range = task_info_range
        self.epoch = 0

        if self._robot_type != "R1Pro":
            raise ValueError(f"MultiTaskSyntheticBehaviorDataset only supports robot_type=R1Pro, got {self._robot_type}")
        if self._num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {self._num_tasks}")
        if self._task_dim < self._num_tasks:
            raise ValueError(f"task_dim must be >= num_tasks, got task_dim={self._task_dim}, num_tasks={self._num_tasks}")
        if self._include_skill_one_hot and (self._num_skills is None and self._skill_one_hot_dim is None):
            raise ValueError("include_skill_one_hot=true requires num_skills or skill_one_hot_dim")
        if self._include_skill_id and self._num_skills is None:
            raise ValueError("include_skill_id=true requires num_skills")
        if self._include_paligemma_token and self._paligemma_token_dim is None:
            raise ValueError("include_paligemma_token=true requires paligemma_token_dim")
        if self._use_action_chunks and self._action_prediction_horizon <= 0:
            raise ValueError(f"action_prediction_horizon must be positive, got {self._action_prediction_horizon}")

    @staticmethod
    def get_all_demo_keys(data_path: str, task_name: Any) -> List[int]:
        return list(range(100))

    def __len__(self) -> int:
        return len(self._demo_keys)

    def _rng(self, key: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + int(key) + 100000 * int(self.epoch))
        return g

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = int(self._demo_keys[idx])
        g = self._rng(key)
        L = self._obs_window_size
        T = self._ctx_len
        N = 128

        task_id = int(key % self._num_tasks)
        task_vec = torch.zeros((self._task_dim,), dtype=torch.float32)
        task_vec[task_id] = 1.0
        if self._task_dim > self._num_tasks:
            task_vec[self._num_tasks:] = (torch.rand((self._task_dim - self._num_tasks,), generator=g) * 2 - 1).float() * 0.1

        if self._num_skills is not None:
            skill_id = int(key % self._num_skills)
            if self._skill_one_hot_dim is not None:
                skill_one_hot_dim = self._skill_one_hot_dim
            else:
                skill_one_hot_dim = self._num_skills
            skill_vec = torch.zeros((skill_one_hot_dim,), dtype=torch.float32)
            if skill_id < skill_one_hot_dim:
                skill_vec[skill_id] = 1.0

        obs: Dict[str, Any] = {}
        obs["odom"] = {"base_velocity": (torch.rand((L, 3), generator=g) * 2 - 1).float()}
        obs["qpos"] = {
            "torso": (torch.rand((L, 4), generator=g) * 2 - 1).float(),
            "left_arm": (torch.rand((L, 7), generator=g) * 2 - 1).float(),
            "left_gripper": (torch.rand((L, 1), generator=g) * 2 - 1).float(),
            "right_arm": (torch.rand((L, 7), generator=g) * 2 - 1).float(),
            "right_gripper": (torch.rand((L, 1), generator=g) * 2 - 1).float(),
        }
        obs["eef"] = {
            "left_pos": (torch.rand((L, 3), generator=g) * 2 - 1).float(),
            "left_quat": (torch.rand((L, 4), generator=g) * 2 - 1).float(),
            "right_pos": (torch.rand((L, 3), generator=g) * 2 - 1).float(),
            "right_quat": (torch.rand((L, 4), generator=g) * 2 - 1).float(),
        }
        obs["pcd"] = torch.rand((L, N, 6), generator=g).float()
        if self._use_task_info:
            obs["task"] = task_vec.unsqueeze(0).repeat(L, 1).contiguous()
        if self._include_skill_one_hot and self._num_skills is not None:
            obs["skill_one_hot"] = skill_vec.unsqueeze(0).repeat(L, 1).contiguous()
        if self._include_skill_id and self._num_skills is not None:
            obs["skill_id"] = torch.full((L,), skill_id, dtype=torch.int64)
        if self._include_paligemma_token and self._paligemma_token_dim is not None:
            tok = torch.zeros((self._paligemma_token_dim,), dtype=torch.float32)
            tok[0] = float(task_id) / float(max(1, self._num_tasks - 1)) * 2.0 - 1.0
            tok[1] = float(key % 997) / 996.0 * 2.0 - 1.0
            if self._paligemma_token_dim > 2:
                tok[2:] = (torch.rand((self._paligemma_token_dim - 2,), generator=g) * 2 - 1).float() * 0.05
            obs["paligemma_token"] = tok.unsqueeze(0).repeat(L, 1).contiguous()

        if self._include_rgbd:
            for view in self._rgb_views:
                obs[f"{view}::rgb"] = torch.randint(
                    low=0,
                    high=256,
                    size=(L, 3, self._rgb_height, self._rgb_width),
                    generator=g,
                    dtype=torch.uint8,
                )
                obs[f"{view}::depth_linear"] = (
                    torch.rand((L, self._rgb_height, self._rgb_width), generator=g).float()
                    * (MAX_DEPTH - MIN_DEPTH)
                    + MIN_DEPTH
                )

        action_bias = torch.linspace(-0.4, 0.4, steps=self._num_tasks, dtype=torch.float32)[task_id]
        if self._use_action_chunks:
            H = self._action_prediction_horizon
            full_action = (torch.rand((L, H, 23), generator=g) * 2 - 1).float() * 0.5 + action_bias
            full_action = torch.clamp(full_action, -1.0, 1.0)
            actions: Dict[str, torch.Tensor] = {
                "base": full_action[..., 0:3],
                "torso": full_action[..., 3:7],
                "left_arm": full_action[..., 7:14],
                "left_gripper": full_action[..., 14:15],
                "right_arm": full_action[..., 15:22],
                "right_gripper": full_action[..., 22:23],
            }
            masks = torch.ones((L, H), dtype=torch.float32)
        else:
            full_action = (torch.rand((T, 23), generator=g) * 2 - 1).float() * 0.5 + action_bias
            full_action = torch.clamp(full_action, -1.0, 1.0)
            actions = {
                "base": full_action[:, 0:3],
                "torso": full_action[:, 3:7],
                "left_arm": full_action[:, 7:14],
                "left_gripper": full_action[:, 14:15],
                "right_arm": full_action[:, 15:22],
                "right_gripper": full_action[:, 22:23],
            }
            masks = torch.ones((T,), dtype=torch.float32)

        return {"obs": obs, "actions": actions, "masks": masks}
