import torch
import torch.nn as nn
import torch.nn.functional as F
from il_lib.nn.flow_matching import ActionBlockMoEVelocityField, ActionBlockSpec
from il_lib.nn.common import build_mlp
from il_lib.optim import default_optimizer_groups
from typing import Dict, List, Optional


class MomaBCActionExpert(nn.Module):
    def __init__(
        self,
        *,
        whole_body_decoding_order: List[str],
        action_dim_per_part: Dict[str, int],
        obs_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        hidden_depth: int = 2,
        activation: str = "relu",
        loss_type: str = "l1",
    ) -> None:
        super().__init__()
        self._order = list(whole_body_decoding_order)
        self._dims = {k: int(v) for k, v in action_dim_per_part.items()}
        self.obs_dim = int(obs_dim)
        self.action_horizon = int(action_horizon)
        self.loss_type = str(loss_type)
        if self.loss_type not in {"l1", "mse"}:
            raise ValueError(f"Unsupported loss_type={self.loss_type}")

        self._heads = nn.ModuleDict()
        for k in self._order:
            out_dim = self._dims[k] * self.action_horizon
            self._heads[k] = build_mlp(
                input_dim=self.obs_dim,
                hidden_dim=int(hidden_dim),
                output_dim=int(out_dim),
                hidden_depth=int(hidden_depth),
                activation=activation,
                weight_init="orthogonal",
                bias_init="zeros",
                norm_type=None,
                add_input_activation=False,
                add_input_norm=False,
                add_output_activation=False,
                add_output_norm=False,
            )

    def inference(self, *, obs: torch.Tensor, return_last_timestep_only: bool) -> Dict[str, torch.Tensor]:
        if obs.ndim != 3:
            raise ValueError(f"obs must have shape (B,T_obs,D), got {tuple(obs.shape)}")
        B, T_obs, D = obs.shape
        if D != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {D}, expected {self.obs_dim}")

        if return_last_timestep_only:
            x = obs[:, -1]
            parts: Dict[str, torch.Tensor] = {}
            for k in self._order:
                pred = self._heads[k](x).view(B, self.action_horizon, self._dims[k])
                parts[k] = pred
            return parts

        x = obs.reshape(B * T_obs, D)
        parts = {}
        for k in self._order:
            pred = self._heads[k](x).view(B, T_obs, self.action_horizon, self._dims[k])
            parts[k] = pred
        return parts

    def compute_loss(self, *, obs: torch.Tensor, gt_action: Dict[str, torch.Tensor]) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError(f"obs must have shape (B,T_obs,D), got {tuple(obs.shape)}")
        B, T_obs, D = obs.shape
        if D != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {D}, expected {self.obs_dim}")

        pred_parts = self.inference(obs=obs, return_last_timestep_only=False)
        total = None
        denom = 0.0
        for k in self._order:
            pred = pred_parts[k]
            gt = gt_action[k]
            if pred.shape != gt.shape:
                raise ValueError(f"gt_action[{k}] shape mismatch: got {tuple(gt.shape)}, expected {tuple(pred.shape)}")
            if self.loss_type == "l1":
                per_dim = F.l1_loss(pred, gt, reduction="none")
            else:
                per_dim = F.mse_loss(pred, gt, reduction="none")
            per_step = per_dim.mean(dim=-1)
            if total is None:
                total = per_step
            else:
                total = total + per_step
            denom += 1.0
        assert total is not None
        return total / max(denom, 1.0)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return default_optimizer_groups(self, weight_decay=weight_decay, lr_scale=lr_scale)


class MomaFlowMatchingActionExpert(nn.Module):
    def __init__(
        self,
        *,
        whole_body_decoding_order: List[str],
        action_dim_per_part: Dict[str, int],
        obs_dim: int,
        action_horizon: int,
        num_flow_steps_per_inference: int = 16,
        moe_num_experts: int = 4,
        moe_top_k: Optional[int] = None,
        moe_expert_hidden_dim: int = 512,
        moe_expert_hidden_depth: int = 2,
        moe_gating_hidden_dim: int = 256,
        moe_gating_hidden_depth: int = 1,
        moe_activation: str = "silu",
        moe_gating_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self._order = list(whole_body_decoding_order)
        self._dims = {k: int(v) for k, v in action_dim_per_part.items()}
        self.obs_dim = int(obs_dim)
        self.action_horizon = int(action_horizon)
        self.num_flow_steps_per_inference = int(num_flow_steps_per_inference)

        blocks: List[ActionBlockSpec] = []
        start = 0
        for k in self._order:
            d = int(self._dims[k])
            blocks.append(ActionBlockSpec(name=str(k), start=start, dim=d))
            start += d
        self.action_dim = int(start)

        self.velocity_field = ActionBlockMoEVelocityField(
            action_dim=self.action_dim,
            action_blocks=blocks,
            cond_dim=self.obs_dim,
            num_experts=int(moe_num_experts),
            top_k=moe_top_k,
            expert_hidden_dim=int(moe_expert_hidden_dim),
            expert_hidden_depth=int(moe_expert_hidden_depth),
            gating_hidden_dim=int(moe_gating_hidden_dim),
            gating_hidden_depth=int(moe_gating_hidden_depth),
            activation=str(moe_activation),
            gating_temperature=float(moe_gating_temperature),
        )

    def _split_parts(self, x: torch.Tensor, *, has_time_dim: bool) -> Dict[str, torch.Tensor]:
        parts = {}
        start = 0
        for k in self._order:
            d = int(self._dims[k])
            parts[k] = x[..., start : start + d]
            start += d
        return parts

    @torch.no_grad()
    def inference(self, *, obs: torch.Tensor, return_last_timestep_only: bool) -> Dict[str, torch.Tensor]:
        if obs.ndim != 3:
            raise ValueError(f"obs must have shape (B,T_obs,D), got {tuple(obs.shape)}")
        B, T_obs, D = obs.shape
        if D != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {D}, expected {self.obs_dim}")
        n = int(self.num_flow_steps_per_inference)
        if n <= 0:
            raise ValueError(f"num_flow_steps_per_inference must be positive, got {n}")
        dt = 1.0 / float(n)

        if return_last_timestep_only:
            cond = obs[:, -1]
            x = torch.randn((B, self.action_horizon, self.action_dim), device=obs.device, dtype=obs.dtype)
            for i in range(n):
                t = (float(i) + 0.5) / float(n)
                v = self.velocity_field(x=x, t=t, cond=cond)
                x = x + dt * v
            return self._split_parts(x, has_time_dim=False)

        cond = obs.reshape(B * T_obs, D)
        x = torch.randn((B * T_obs, self.action_horizon, self.action_dim), device=obs.device, dtype=obs.dtype)
        for i in range(n):
            t = (float(i) + 0.5) / float(n)
            v = self.velocity_field(x=x, t=t, cond=cond)
            x = x + dt * v
        x = x.view(B, T_obs, self.action_horizon, self.action_dim)
        return self._split_parts(x, has_time_dim=True)

    def compute_loss(self, *, obs: torch.Tensor, gt_action: Dict[str, torch.Tensor]) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError(f"obs must have shape (B,T_obs,D), got {tuple(obs.shape)}")
        B, T_obs, D = obs.shape
        if D != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {D}, expected {self.obs_dim}")

        gt = torch.cat([gt_action[k] for k in self._order], dim=-1)
        if gt.shape[:3] != (B, T_obs, self.action_horizon):
            raise ValueError(f"gt_action shape mismatch: got {tuple(gt.shape)}")
        if gt.shape[-1] != self.action_dim:
            raise ValueError(f"action_dim mismatch: got {gt.shape[-1]}, expected {self.action_dim}")

        cond = obs.reshape(B * T_obs, D)
        x1 = gt.reshape(B * T_obs, self.action_horizon, self.action_dim)
        x0 = torch.randn_like(x1)

        N = x1.shape[0]
        t = torch.rand((N,), device=x1.device, dtype=x1.dtype)
        t_b = t.view(N, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0
        v_pred = self.velocity_field(x=x_t, t=t, cond=cond)

        per_step = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=-1)
        return per_step.view(B, T_obs, self.action_horizon)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        return default_optimizer_groups(self, weight_decay=weight_decay, lr_scale=lr_scale)

