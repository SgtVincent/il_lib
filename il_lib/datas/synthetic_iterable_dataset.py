import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Any, Dict, Generator, List, Optional


class MTILSyntheticIterableDataset(IterableDataset):
    @staticmethod
    def get_all_demo_keys(data_path: str, task_name: Any) -> List[int]:
        return list(range(100))

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
        num_chunks_per_demo: int = 8,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self._demo_keys = [int(x) for x in demo_keys]
        self._robot_type = str(robot_type)
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
        self._num_chunks_per_demo = int(num_chunks_per_demo)
        self._shuffle = bool(shuffle)
        self.epoch = 0

        if self._robot_type != "R1Pro":
            raise ValueError(f"MTILSyntheticIterableDataset only supports robot_type=R1Pro, got {self._robot_type}")
        if self._include_skill_one_hot and (self._num_skills is None and self._skill_one_hot_dim is None):
            raise ValueError("include_skill_one_hot=true requires num_skills or skill_one_hot_dim")
        if self._include_skill_id and self._num_skills is None:
            raise ValueError("include_skill_id=true requires num_skills")
        if self._include_paligemma_token and self._paligemma_token_dim is None:
            raise ValueError("include_paligemma_token=true requires paligemma_token_dim")

    def _rng(self, demo_key: int, index: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self._seed + int(demo_key) * 100000 + int(index) + 10000000 * int(self.epoch))
        return g

    def _iter_one(self, demo_key: int, index: int) -> Dict[str, Any]:
        g = self._rng(demo_key, index)
        L = self._obs_window_size
        T = self._ctx_len

        task_id = int(demo_key % self._num_tasks)
        task_vec = torch.zeros((self._task_dim,), dtype=torch.float32)
        task_vec[task_id] = 1.0
        if self._task_dim > self._num_tasks:
            task_vec[self._num_tasks :] = (torch.rand((self._task_dim - self._num_tasks,), generator=g) * 2 - 1).float() * 0.1

        skill_id = None
        skill_vec = None
        if self._num_skills is not None:
            skill_id = int(demo_key % self._num_skills)
            k = self._skill_one_hot_dim if self._skill_one_hot_dim is not None else self._num_skills
            skill_vec = torch.zeros((int(k),), dtype=torch.float32)
            if skill_id < int(k):
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
        obs["task"] = task_vec.unsqueeze(0).repeat(L, 1).contiguous()
        if self._include_skill_one_hot and skill_vec is not None:
            obs["skill_one_hot"] = skill_vec.unsqueeze(0).repeat(L, 1).contiguous()
        if self._include_skill_id and skill_id is not None:
            obs["skill_id"] = torch.full((L,), int(skill_id), dtype=torch.int64)
        if self._include_paligemma_token and self._paligemma_token_dim is not None:
            tok = torch.zeros((self._paligemma_token_dim,), dtype=torch.float32)
            tok[0] = float(task_id) / float(max(1, self._num_tasks - 1)) * 2.0 - 1.0
            tok[1] = float(demo_key % 997) / 996.0 * 2.0 - 1.0
            if self._paligemma_token_dim > 2:
                tok[2:] = (torch.rand((self._paligemma_token_dim - 2,), generator=g) * 2 - 1).float() * 0.05
            obs["paligemma_token"] = tok.unsqueeze(0).repeat(L, 1).contiguous()

        action_bias = torch.linspace(-0.4, 0.4, steps=self._num_tasks, dtype=torch.float32)[task_id]
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

        return {"obs": obs, "actions": actions, "masks": masks, "demo_key": int(demo_key), "index": int(index)}

    def get_streamed_data(self, demo_ptr: int, start_idx: int, end_idx: int) -> Generator[Dict[str, Any], None, None]:
        demo_key = int(self._demo_keys[int(demo_ptr)])
        for idx in range(int(start_idx), int(end_idx)):
            yield self._iter_one(demo_key, idx)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1

        demo_keys = list(self._demo_keys)
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + int(self.epoch))
            demo_keys = [demo_keys[i] for i in torch.randperm(len(demo_keys), generator=g).tolist()]

        for demo_idx, demo_key in enumerate(demo_keys):
            if demo_idx % num_workers != worker_id:
                continue
            for idx in range(self._num_chunks_per_demo):
                yield self._iter_one(int(demo_key), idx)

