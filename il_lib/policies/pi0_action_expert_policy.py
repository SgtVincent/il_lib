import torch
import torch.nn.functional as F
from dataclasses import dataclass
from il_lib.nn.common import MLP
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_concat
from il_lib.utils.functional_utils import call_once
from il_lib.utils.training_utils import freeze_params, load_torch
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional


@dataclass
class _Pi0TorchConfig:
    dtype: str
    paligemma_variant: str
    action_expert_variant: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    pi05: bool = False


class Pi0ActionExpertPolicy(BasePolicy):
    is_sequence_policy = True

    def __init__(
        self,
        *args,
        prop_keys: List[str],
        task_dim: int,
        state_encoder_hidden_dim: int,
        state_encoder_hidden_depth: int,
        state_encoder_activation: str,
        state_dim: int = 32,
        action_dim: int,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        num_latest_obs: int,
        deployed_action_steps: int,
        horizon: int,
        num_flow_steps_per_inference: int,
        pi0_dtype: str = "bfloat16",
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        action_expert_name: str = "gemma_token",
        action_expert_kwargs: Optional[Dict[str, Any]] = None,
        pi0_ckpt_path: Optional[str] = None,
        freeze_pi0_vlm: bool = True,
        freeze_pi0_gemma_expert: bool = False,
        dummy_token_id: int = 0,
        dummy_image_value: float = -1.0,
        lr: float = 7e-4,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())

        self._prop_keys = list(prop_keys)
        self._task_dim = int(task_dim)
        self._state_dim = int(state_dim)
        self._num_latest_obs = int(num_latest_obs)

        self.action_dim = int(action_dim)
        self._action_keys = list(action_keys)
        self._action_key_dims = dict(action_key_dims)

        self.horizon = int(horizon)
        self.deployed_action_steps = int(deployed_action_steps)
        self.num_flow_steps_per_inference = int(num_flow_steps_per_inference)

        self._pi0_action_dim = 32
        self._pi0_action_horizon = self.horizon
        self._dummy_token_id = int(dummy_token_id)
        self._dummy_image_value = float(dummy_image_value)

        prop_dim = self._infer_prop_dim()
        state_encoder_in_dim = prop_dim + self._task_dim
        self.state_encoder = MLP(
            input_dim=state_encoder_in_dim,
            hidden_dim=int(state_encoder_hidden_dim),
            output_dim=self._state_dim,
            hidden_depth=int(state_encoder_hidden_depth),
            activation=str(state_encoder_activation),
            add_output_activation=False,
        )

        self.pi0 = self._create_pi0(
            dtype=str(pi0_dtype),
            paligemma_variant=str(paligemma_variant),
            action_expert_variant=str(action_expert_variant),
            action_expert_name=str(action_expert_name),
            action_expert_kwargs=action_expert_kwargs,
            ckpt_path=pi0_ckpt_path,
            freeze_vlm=bool(freeze_pi0_vlm),
            freeze_gemma_expert=bool(freeze_pi0_gemma_expert),
        )

        self.lr = float(lr)
        self.use_cosine_lr = bool(use_cosine_lr)
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = float(lr_layer_decay)
        self.optimizer = str(optimizer)
        self.weight_decay = float(weight_decay)
        self.save_hyperparameters(ignore=["action_expert_kwargs"])

    def _infer_prop_dim(self) -> int:
        dim = 0
        for key in self._prop_keys:
            if key.startswith("odom/"):
                dim += 3
            elif key.startswith("qpos/"):
                if key.endswith("torso"):
                    dim += 4
                elif key.endswith("left_arm") or key.endswith("right_arm"):
                    dim += 7
                elif key.endswith("left_gripper") or key.endswith("right_gripper"):
                    dim += 1
                else:
                    raise ValueError(f"Unknown qpos key: {key}")
            elif key.startswith("eef/"):
                if key.endswith("_pos"):
                    dim += 3
                elif key.endswith("_quat"):
                    dim += 4
                else:
                    raise ValueError(f"Unknown eef key: {key}")
            else:
                raise ValueError(f"Unknown prop key: {key}")
        return dim

    def _create_pi0(
        self,
        *,
        dtype: str,
        paligemma_variant: str,
        action_expert_variant: str,
        action_expert_name: str,
        action_expert_kwargs: Optional[Dict[str, Any]],
        ckpt_path: Optional[str],
        freeze_vlm: bool,
        freeze_gemma_expert: bool,
    ):
        try:
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "无法 import openpi PI0Pytorch。请确保运行时满足：\n"
                "1) PYTHONPATH 包含 openpi-comet/src\n"
                "2) 已安装 openpi-comet 依赖（含 transformers 等）\n"
                f"原始错误: {type(e).__name__}: {e}"
            ) from e

        cfg = _Pi0TorchConfig(
            dtype=dtype,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            action_dim=self._pi0_action_dim,
            action_horizon=self._pi0_action_horizon,
            max_token_len=1,
            pi05=False,
        )
        model = PI0Pytorch(
            cfg,
            action_expert_name=action_expert_name,
            action_expert_kwargs=action_expert_kwargs,
        )

        if ckpt_path is not None:
            ckpt = load_torch(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(unexpected) > 0:
                raise RuntimeError(f"Unexpected keys in pi0 checkpoint: {unexpected[:20]}")

        if freeze_vlm:
            if hasattr(model, "paligemma_with_expert") and hasattr(model.paligemma_with_expert, "paligemma"):
                freeze_params(model.paligemma_with_expert.paligemma)
        if freeze_gemma_expert:
            if hasattr(model, "paligemma_with_expert") and hasattr(model.paligemma_with_expert, "gemma_expert"):
                freeze_params(model.paligemma_with_expert.gemma_expert)

        return model

    def _extract_proprio(self, obs: dict) -> torch.Tensor:
        prop_obs = []
        for prop_key in self._prop_keys:
            group, key = prop_key.split("/")
            prop_obs.append(obs[group][key])
        prop_obs = torch.cat(prop_obs, dim=-1)
        if prop_obs.shape[1] != self._num_latest_obs:
            raise ValueError(f"obs window mismatch: got {prop_obs.shape[1]}, expected {self._num_latest_obs}")
        return prop_obs[:, -1]

    def _extract_task(self, obs: dict) -> torch.Tensor:
        if "task" not in obs:
            return torch.zeros((obs["qpos"]["torso"].shape[0], self._task_dim), device=obs["qpos"]["torso"].device)
        task = obs["task"]
        if task.shape[1] != self._num_latest_obs:
            raise ValueError(f"task window mismatch: got {task.shape[1]}, expected {self._num_latest_obs}")
        if task.shape[-1] != self._task_dim:
            raise ValueError(f"task_dim mismatch: got {task.shape[-1]}, expected {self._task_dim}")
        return task[:, -1]

    def _encode_state(self, obs: dict) -> torch.Tensor:
        prop = self._extract_proprio(obs)
        task = self._extract_task(obs)
        x = torch.cat([prop, task], dim=-1).to(dtype=torch.float32)
        state = self.state_encoder(x).to(dtype=torch.float32)
        if state.shape[-1] != self._state_dim:
            raise ValueError(f"state_dim mismatch: got {state.shape[-1]}, expected {self._state_dim}")
        return state

    def _build_pi0_observation(self, *, state: torch.Tensor) -> Any:
        B = state.shape[0]
        device = state.device
        images = {
            "base_0_rgb": torch.full((B, 3, 224, 224), self._dummy_image_value, device=device, dtype=torch.float32),
            "left_wrist_0_rgb": torch.full((B, 3, 224, 224), self._dummy_image_value, device=device, dtype=torch.float32),
            "right_wrist_0_rgb": torch.full((B, 3, 224, 224), self._dummy_image_value, device=device, dtype=torch.float32),
        }
        image_masks = {k: torch.ones((B,), device=device, dtype=torch.bool) for k in images}
        tokenized_prompt = torch.full((B, 1), self._dummy_token_id, device=device, dtype=torch.int32)
        tokenized_prompt_mask = torch.ones((B, 1), device=device, dtype=torch.bool)

        class _Obs:
            pass

        o = _Obs()
        o.images = images
        o.image_masks = image_masks
        o.state = state
        o.tokenized_prompt = tokenized_prompt
        o.tokenized_prompt_mask = tokenized_prompt_mask
        o.token_ar_mask = None
        o.token_loss_mask = None
        return o

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        state = self._encode_state(obs)
        return state

    @torch.no_grad()
    def act(self, obs: dict, policy_state=None, deterministic=None) -> torch.Tensor:
        data = self.process_data(obs, extract_action=False)
        state = self._encode_state(data)
        observation = self._build_pi0_observation(state=state)
        actions32 = self.pi0.sample_actions(
            device=self.device,
            observation=observation,
            noise=None,
            num_steps=int(self.num_flow_steps_per_inference),
        )
        actions23 = actions32[:, self._num_latest_obs - 1 :, : self.action_dim].clone().cpu()
        return self._denormalize_action(actions23)

    def reset(self) -> None:
        pass

    @call_once
    def _check_batch_shapes(self, obs: dict, actions: torch.Tensor, masks: torch.Tensor) -> None:
        if actions.shape[1] != self.horizon:
            raise ValueError(f"action horizon mismatch: got {actions.shape[1]}, expected {self.horizon}")
        if actions.shape[2] != self.action_dim:
            raise ValueError(f"action_dim mismatch: got {actions.shape[2]}, expected {self.action_dim}")
        if masks.shape[1] != self.horizon:
            raise ValueError(f"mask horizon mismatch: got {masks.shape[1]}, expected {self.horizon}")

    def policy_training_step(self, batch, batch_idx):
        batch["actions"] = any_concat([batch["actions"][k] for k in self._action_keys], dim=-1)
        actions23 = batch["actions"]
        masks = batch["masks"]
        data = self.process_data(batch, extract_action=False)
        self._check_batch_shapes(data, actions23, masks)

        B = actions23.shape[0]
        state = self._encode_state(data).to(self.device)
        observation = self._build_pi0_observation(state=state)

        actions32 = torch.zeros((B, self.horizon, self._pi0_action_dim), device=self.device, dtype=torch.float32)
        actions32[:, :, : self.action_dim] = actions23.to(self.device, dtype=torch.float32)

        noise = self.pi0.sample_noise(actions32.shape, self.device)
        time = self.pi0.sample_time(B, self.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions32
        u_t = noise - actions32

        images, img_masks, lang_tokens, lang_masks, pi0_state = self.pi0._preprocess_observation(observation, train=False)
        v_t = self.pi0.action_expert.compute_velocity_train(
            model=self.pi0,
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=pi0_state,
            x_t=x_t,
            time=time,
        )
        loss_per_dim = F.mse_loss(u_t, v_t, reduction="none")[:, :, : self.action_dim].mean(dim=-1)
        loss = (loss_per_dim * masks.to(self.device, dtype=loss_per_dim.dtype)).sum() / masks.sum()
        return loss, {"mse": loss}, masks.sum()

    def policy_evaluation_step(self, batch, batch_idx):
        batch["actions"] = any_concat([batch["actions"][k] for k in self._action_keys], dim=-1)
        actions23 = batch["actions"]
        masks = batch["masks"]
        data = self.process_data(batch, extract_action=False)
        self._check_batch_shapes(data, actions23, masks)

        B = actions23.shape[0]
        state = self._encode_state(data).to(self.device)
        observation = self._build_pi0_observation(state=state)

        actions32 = torch.zeros((B, self.horizon, self._pi0_action_dim), device=self.device, dtype=torch.float32)
        actions32[:, :, : self.action_dim] = actions23.to(self.device, dtype=torch.float32)

        noise = self.pi0.sample_noise(actions32.shape, self.device)
        time = self.pi0.sample_time(B, self.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions32
        u_t = noise - actions32

        images, img_masks, lang_tokens, lang_masks, pi0_state = self.pi0._preprocess_observation(observation, train=False)
        v_t = self.pi0.action_expert.compute_velocity_train(
            model=self.pi0,
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=pi0_state,
            x_t=x_t,
            time=time,
        )

        u_t_23 = u_t[:, :, : self.action_dim]
        v_t_23 = v_t[:, :, : self.action_dim]
        pad = masks.to(self.device, dtype=torch.float32)

        mse = F.mse_loss(u_t_23, v_t_23, reduction="none").mean(dim=-1)
        mse = (mse * pad).sum() / pad.sum()

        l1 = F.l1_loss(u_t_23, v_t_23, reduction="none").mean(dim=-1)
        l1 = (l1 * pad).sum() / pad.sum()

        return l1, {"l1": l1, "mse": mse}, 1

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        if self.use_cosine_lr:
            scheduler_kwargs = dict(
                base_value=1.0,
                final_value=self.lr_cosine_min / self.lr,
                epochs=self.lr_cosine_steps,
                warmup_start_value=self.lr_cosine_min / self.lr,
                warmup_epochs=self.lr_warmup_steps,
                steps_per_epoch=1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return ([optimizer], [{"scheduler": scheduler, "interval": "step"}])
        return optimizer

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        data = {"qpos": data_batch["obs"]["qpos"], "eef": data_batch["obs"]["eef"]}
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]
        if "task" in data_batch["obs"]:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            data.update({"actions": data_batch["actions"], "masks": data_batch["masks"]})
        return data
