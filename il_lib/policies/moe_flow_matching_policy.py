import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.nn.flow_matching import ActionBlockMoEVelocityField, ActionBlockSpec
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size, any_concat
from il_lib.utils.functional_utils import call_once
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
from omegaconf import DictConfig
from typing import Any, Dict, Optional, List


class MoEFlowMatchingPolicy(BasePolicy):
    is_sequence_policy = True

    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        feature_extractors: Dict[str, DictConfig],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        action_dim: int,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        num_latest_obs: int,
        deployed_action_steps: int,
        num_flow_steps_per_inference: int,
        horizon: int,
        moe_num_experts: int,
        moe_top_k: Optional[int] = None,
        moe_expert_hidden_dim: int = 512,
        moe_expert_hidden_depth: int = 2,
        moe_gating_hidden_dim: int = 256,
        moe_gating_hidden_depth: int = 1,
        moe_activation: str = "silu",
        moe_gating_temperature: float = 1.0,
        lr: float = 7e-4,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._prop_keys = prop_keys
        self._features = set(feature_extractors.keys())
        self.feature_extractor = SimpleFeatureFusion(
            extractors={k: instantiate(v) for k, v in feature_extractors.items()},
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )

        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())
        self.action_dim = action_dim
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims

        self.horizon = horizon
        self.num_latest_obs = num_latest_obs
        self.deployed_action_steps = deployed_action_steps
        self.num_flow_steps_per_inference = num_flow_steps_per_inference

        blocks: List[ActionBlockSpec] = []
        start = 0
        for k in self._action_keys:
            d = int(self._action_key_dims[k])
            blocks.append(ActionBlockSpec(name=str(k), start=start, dim=d))
            start += d

        cond_dim = int(feature_fusion_output_dim * num_latest_obs)
        self.velocity_field = ActionBlockMoEVelocityField(
            action_dim=self.action_dim,
            action_blocks=blocks,
            cond_dim=cond_dim,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            expert_hidden_dim=moe_expert_hidden_dim,
            expert_hidden_depth=moe_expert_hidden_depth,
            gating_hidden_dim=moe_gating_hidden_dim,
            gating_hidden_depth=moe_gating_hidden_depth,
            activation=moe_activation,
            gating_temperature=moe_gating_temperature,
        )

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, obs: dict, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)
        obs["proprioception"] = prop_obs
        obs = {k: obs[k] for k in self._features}
        self._check_forward_input_shape(obs, x)
        obs_feature = self.feature_extractor(obs)
        cond = obs_feature.reshape(obs_feature.shape[0], -1)
        v = self.velocity_field(x=x, t=t, cond=cond)
        return v

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        obs = self.process_data(obs, extract_action=False)
        B = get_batch_size(obs, strict=True)
        x = torch.randn(
            size=(B, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        n = int(self.num_flow_steps_per_inference)
        dt = 1.0 / float(n)
        for i in range(n):
            t = (float(i) + 0.5) / float(n)
            v = self.forward(obs, x, t)
            x = x + dt * v
        action = x[:, self.num_latest_obs - 1:].clone().cpu()
        return self._denormalize_action(action)

    def reset(self) -> None:
        pass

    @call_once
    def _check_forward_input_shape(self, obs, x):
        L_obs = get_batch_size(any_slice(obs, 0), strict=True)
        assert L_obs == self.num_latest_obs, f"obs must have length {self.num_latest_obs}"
        L_traj = get_batch_size(any_slice(x, 0), strict=True)
        assert L_traj == self.horizon, f"x must have length {self.horizon}"
        B_obs = get_batch_size(obs, strict=True)
        B_traj = get_batch_size(x, strict=True)
        assert B_obs == B_traj, "Batch size must match"

    def policy_training_step(self, batch, batch_idx):
        batch["actions"] = any_concat([batch["actions"][k] for k in self._action_keys], dim=-1)
        B = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        x1 = batch.pop("actions")
        x0 = torch.randn_like(x1)

        t = torch.rand((B,), device=x1.device, dtype=x1.dtype)
        t_b = t.view(B, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0

        v_pred = self.forward(obs=batch, x=x_t, t=t)
        loss = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=-1)
        loss = loss * pad_mask
        real_batch_size = pad_mask.sum()
        loss = loss.sum() / real_batch_size
        return loss, {"flow_matching_loss": loss}, real_batch_size

    def policy_evaluation_step(self, batch, batch_idx):
        batch["actions"] = any_concat([batch["actions"][k] for k in self._action_keys], dim=-1)
        batch = self.process_data(batch, extract_action=True)
        pad_mask = batch.pop("masks")
        gt_actions = batch.pop("actions")

        x = torch.randn(size=gt_actions.shape, device=self.device, dtype=self.dtype)
        n = int(self.num_flow_steps_per_inference)
        dt = 1.0 / float(n)
        for i in range(n):
            t = (float(i) + 0.5) / float(n)
            v = self.forward(batch, x, t)
            x = x + dt * v

        pred_actions = x[:, self.num_latest_obs - 1:]
        gt_actions = gt_actions[:, self.num_latest_obs - 1:]
        pad_mask = pad_mask[:, self.num_latest_obs - 1:]

        l1_full_future_horizon = F.l1_loss(pred_actions, gt_actions, reduction="none")
        l1_full_future_horizon = l1_full_future_horizon.mean(dim=-1).reshape(pad_mask.shape)
        l1_full_future_horizon = l1_full_future_horizon * pad_mask
        real_batch_size_full_future_horizon = pad_mask.sum()

        deployed_start_t = self.num_latest_obs - 1
        deployed_end_t = deployed_start_t + self.deployed_action_steps
        pred_actions_to_deploy = pred_actions[:, deployed_start_t:deployed_end_t]
        gt_actions_deploy = gt_actions[:, deployed_start_t:deployed_end_t]
        pad_mask_deploy = pad_mask[:, deployed_start_t:deployed_end_t]

        l1_deployed_steps_only = F.l1_loss(pred_actions_to_deploy, gt_actions_deploy, reduction="none")
        l1_deployed_steps_only = l1_deployed_steps_only.mean(dim=-1).reshape(pad_mask_deploy.shape)
        l1_deployed_steps_only = l1_deployed_steps_only * pad_mask_deploy
        real_batch_size_deployed_steps_only = pad_mask_deploy.sum()

        l1_full_future_horizon = l1_full_future_horizon.sum() / real_batch_size_full_future_horizon
        l1_deployed_steps_only = l1_deployed_steps_only.sum() / real_batch_size_deployed_steps_only
        return (
            l1_full_future_horizon,
            {
                "l1": l1_full_future_horizon,
                "l1_full_future_horizon": l1_full_future_horizon,
                "l1_deployed_steps_only": l1_deployed_steps_only,
            },
            1,
        )

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
        if "rgb" in self._features:
            data["rgb"] = {k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
        if "rgbd" in self._features:
            rgb = {k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
            depth = {k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) for k in data_batch["obs"] if "depth" in k}
            data["rgbd"] = {k: {"rgb": rgb[k], "depth": depth[k].unsqueeze(-3)} for k in rgb}
        if "pcd" in self._features:
            data["pcd"] = {"rgb": data_batch["obs"]["pcd"][..., :3], "xyz": data_batch["obs"]["pcd"][..., 3:]}
        if "task" in self._features:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            data.update({"actions": data_batch["actions"], "masks": data_batch["masks"]})
        return data

