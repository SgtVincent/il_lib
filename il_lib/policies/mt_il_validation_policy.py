import heapq
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.nn.flow_matching import ActionBlockMoEVelocityField, ActionBlockSpec
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size, any_concat
from il_lib.utils.functional_utils import call_once
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional


class MTILValidationPolicy(BasePolicy):
    is_sequence_policy = True

    def __init__(
        self,
        *args,
        prop_keys: List[str],
        feature_extractors: Dict[str, DictConfig],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        action_expert_type: str,
        action_dim: int,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        num_latest_obs: int,
        deployed_action_steps: int,
        horizon: int,
        bc_hidden_dim: int = 512,
        bc_hidden_depth: int = 2,
        bc_activation: str = "relu",
        bc_loss: str = "l1",
        diffusion_backbone: Optional[DictConfig] = None,
        noise_scheduler: Optional[DictConfig] = None,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: Optional[int] = None,
        num_flow_steps_per_inference: Optional[int] = None,
        moe_num_experts: int = 4,
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
        log_skill_buckets: bool = False,
        max_skill_buckets: int = 32,
        log_failure_stats: bool = False,
        failure_top_k: int = 16,
        failure_quantile: float = 0.95,
        export_failure_cases: bool = False,
        export_failure_top_k: int = 64,
        export_failure_dir: str = "failure_cases",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._prop_keys = list(prop_keys)
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

        if sum(action_key_dims.values()) != action_dim:
            raise ValueError(f"action_dim mismatch: sum(action_key_dims)={sum(action_key_dims.values())} != {action_dim}")
        if set(action_keys) != set(action_key_dims.keys()):
            raise ValueError("action_keys must match action_key_dims keys")

        self.action_dim = int(action_dim)
        self._action_keys = list(action_keys)
        self._action_key_dims = dict(action_key_dims)

        self.horizon = int(horizon)
        self.num_latest_obs = int(num_latest_obs)
        self.deployed_action_steps = int(deployed_action_steps)

        self.action_expert_type = str(action_expert_type)
        self._cond_dim = int(feature_fusion_output_dim) * int(self.num_latest_obs)

        self.bc_loss = str(bc_loss)
        if self.bc_loss not in {"l1", "mse"}:
            raise ValueError(f"Unsupported bc_loss={self.bc_loss}")
        self.bc_head = None

        self.backbone = None
        self.noise_scheduler = None
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.num_denoise_steps_per_inference = num_denoise_steps_per_inference

        self.velocity_field = None
        self.num_flow_steps_per_inference = num_flow_steps_per_inference

        if self.action_expert_type == "bc":
            self.bc_head = self._build_mlp(
                input_dim=self._cond_dim,
                output_dim=self.horizon * self.action_dim,
                hidden_dim=int(bc_hidden_dim),
                hidden_depth=int(bc_hidden_depth),
                activation=str(bc_activation),
            )
        elif self.action_expert_type == "diffusion":
            if diffusion_backbone is None or noise_scheduler is None or num_denoise_steps_per_inference is None:
                raise ValueError("diffusion requires diffusion_backbone, noise_scheduler, num_denoise_steps_per_inference")
            self.backbone = instantiate(diffusion_backbone)
            self.noise_scheduler = instantiate(noise_scheduler)
            self.num_denoise_steps_per_inference = int(num_denoise_steps_per_inference)
        elif self.action_expert_type == "flow_matching":
            if num_flow_steps_per_inference is None:
                raise ValueError("flow_matching requires num_flow_steps_per_inference")
            blocks: List[ActionBlockSpec] = []
            start = 0
            for k in self._action_keys:
                d = int(self._action_key_dims[k])
                blocks.append(ActionBlockSpec(name=str(k), start=start, dim=d))
                start += d
            self.velocity_field = ActionBlockMoEVelocityField(
                action_dim=self.action_dim,
                action_blocks=blocks,
                cond_dim=self._cond_dim,
                num_experts=int(moe_num_experts),
                top_k=moe_top_k,
                expert_hidden_dim=int(moe_expert_hidden_dim),
                expert_hidden_depth=int(moe_expert_hidden_depth),
                gating_hidden_dim=int(moe_gating_hidden_dim),
                gating_hidden_depth=int(moe_gating_hidden_depth),
                activation=str(moe_activation),
                gating_temperature=float(moe_gating_temperature),
            )
            self.num_flow_steps_per_inference = int(num_flow_steps_per_inference)
        else:
            raise ValueError(f"Unknown action_expert_type={self.action_expert_type}")

        self.lr = float(lr)
        self.use_cosine_lr = bool(use_cosine_lr)
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = float(lr_layer_decay)
        self.optimizer = str(optimizer)
        self.weight_decay = float(weight_decay)
        self.log_skill_buckets = bool(log_skill_buckets)
        self.max_skill_buckets = int(max_skill_buckets)
        self.log_failure_stats = bool(log_failure_stats)
        self.failure_top_k = int(failure_top_k)
        self.failure_quantile = float(failure_quantile)
        self.export_failure_cases = bool(export_failure_cases)
        self.export_failure_top_k = int(export_failure_top_k)
        self.export_failure_dir = str(export_failure_dir)
        self._failure_heap: list[tuple[float, dict]] = []
        self.save_hyperparameters()

    def _reset_failure_heap(self) -> None:
        self._failure_heap = []

    def _maybe_record_failure_cases(
        self,
        *,
        stage: str,
        demo_key: Optional[Any],
        index: Optional[Any],
        skill_id: Optional[Any],
        l1_per_sample: torch.Tensor,
    ) -> None:
        if not self.export_failure_cases:
            return
        if self.trainer is None or self.trainer.sanity_checking:
            return

        k = max(1, int(self.export_failure_top_k))
        x = l1_per_sample.detach().to(device="cpu", dtype=torch.float32).view(-1)
        B = int(x.numel())

        demo_key_t = None
        if demo_key is not None:
            if isinstance(demo_key, torch.Tensor):
                demo_key_t = demo_key.detach().to(device="cpu").view(-1)
            else:
                demo_key_t = torch.as_tensor(demo_key).view(-1)
            if int(demo_key_t.numel()) not in {1, B}:
                demo_key_t = None

        index_t = None
        if index is not None:
            if isinstance(index, torch.Tensor):
                index_t = index.detach().to(device="cpu").view(-1)
            else:
                index_t = torch.as_tensor(index).view(-1)
            if int(index_t.numel()) not in {1, B}:
                index_t = None

        skill_t = None
        if skill_id is not None:
            if isinstance(skill_id, torch.Tensor):
                s = skill_id.detach().to(device="cpu")
                if s.ndim == 2:
                    s = s[:, -1]
                skill_t = s.view(-1)
            else:
                skill_t = torch.as_tensor(skill_id).view(-1)
            if int(skill_t.numel()) not in {1, B}:
                skill_t = None

        for i in range(B):
            m = float(x[i].item())
            rec = {
                "stage": str(stage),
                "epoch": int(self.current_epoch),
                "l1": m,
            }
            if demo_key_t is not None:
                rec["demo_key"] = int(demo_key_t[i if int(demo_key_t.numel()) == B else 0].item())
            if index_t is not None:
                rec["index"] = int(index_t[i if int(index_t.numel()) == B else 0].item())
            if skill_t is not None:
                rec["skill_id"] = int(skill_t[i if int(skill_t.numel()) == B else 0].item())

            if len(self._failure_heap) < k:
                heapq.heappush(self._failure_heap, (m, rec))
            else:
                if m > self._failure_heap[0][0]:
                    heapq.heapreplace(self._failure_heap, (m, rec))

    def _flush_failure_cases(self, *, stage: str) -> None:
        if not self.export_failure_cases:
            return
        if self.trainer is None or not self.trainer.is_global_zero:
            self._reset_failure_heap()
            return
        if len(self._failure_heap) == 0:
            return

        base_dir = getattr(self, "run_dir", os.getcwd())
        out_dir = os.path.join(str(base_dir), self.export_failure_dir)
        os.makedirs(out_dir, exist_ok=True)
        k = max(1, int(self.export_failure_top_k))
        path = os.path.join(out_dir, f"{stage}_top{k}.jsonl")
        rows = [rec for _, rec in sorted(self._failure_heap, key=lambda x: x[0], reverse=True)]
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._reset_failure_heap()

    def on_validation_epoch_start(self) -> None:
        self._reset_failure_heap()

    def on_validation_epoch_end(self) -> None:
        self._flush_failure_cases(stage="val")
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self._reset_failure_heap()

    def on_test_epoch_end(self) -> None:
        self._flush_failure_cases(stage="test")

    def _log_skill_buckets(self, *, batch: dict, metric_per_sample: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.log_skill_buckets:
            return {}
        if "skill_id" not in batch:
            return {}
        skill_id = batch["skill_id"]
        if not isinstance(skill_id, torch.Tensor):
            skill_id = torch.as_tensor(skill_id)
        if skill_id.ndim == 2:
            skill_id = skill_id[:, -1]
        elif skill_id.ndim != 1:
            return {}
        skill_id = skill_id.to(device=metric_per_sample.device, dtype=torch.long)

        unique = torch.unique(skill_id).detach().cpu().tolist()
        unique = sorted(int(x) for x in unique)[: max(0, self.max_skill_buckets)]
        out: Dict[str, torch.Tensor] = {}
        for sid in unique:
            m = skill_id == int(sid)
            if torch.any(m):
                out[f"l1_skill_{sid}"] = metric_per_sample[m].mean()
        return out

    def _log_failure_stats(self, *, metric_per_sample: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.log_failure_stats:
            return {}
        x = metric_per_sample.detach()
        if x.numel() == 0:
            return {}
        top_k = min(max(1, int(self.failure_top_k)), int(x.numel()))
        top_mean = torch.topk(x, k=top_k, largest=True).values.mean()
        q = float(self.failure_quantile)
        q = min(max(q, 0.0), 1.0)
        p = torch.quantile(x, q) if x.numel() > 1 else x.max()
        return {
            f"l1_p{int(q * 100)}": p,
            "l1_max": x.max(),
            f"l1_top{top_k}_mean": top_mean,
        }

    def _build_mlp(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str,
    ) -> nn.Module:
        if hidden_depth <= 0:
            return nn.Linear(input_dim, output_dim)
        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation={activation}")
        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act())
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _encode_obs(self, obs: dict) -> tuple[dict, torch.Tensor, torch.Tensor]:
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)
        obs = dict(obs)
        obs["proprioception"] = prop_obs
        obs = {k: obs[k] for k in self._features}
        obs_feature = self.feature_extractor(obs)
        cond = obs_feature.reshape(obs_feature.shape[0], -1)
        if cond.shape[-1] != self._cond_dim:
            raise ValueError(f"cond_dim mismatch: got {cond.shape[-1]}, expected {self._cond_dim}")
        return obs, obs_feature, cond

    def forward(self, obs: dict, *args, **kwargs):
        obs = self.process_data({"obs": obs}, extract_action=False)
        _, obs_feature, cond = self._encode_obs(obs)
        if self.action_expert_type == "bc":
            out = self.bc_head(cond).reshape(cond.shape[0], self.horizon, self.action_dim)
            return out
        if self.action_expert_type == "flow_matching":
            x = kwargs["x"]
            t = kwargs["t"]
            return self.velocity_field(x=x, t=t, cond=cond)
        noisy_traj = kwargs["noisy_traj"]
        diffusion_timesteps = kwargs["diffusion_timesteps"]
        return self.backbone(sample=noisy_traj, timestep=diffusion_timesteps, cond=obs_feature)

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        obs = self.process_data(obs, extract_action=False)
        _, obs_feature, cond = self._encode_obs(obs)
        if self.action_expert_type == "bc":
            traj = self.bc_head(cond).reshape(cond.shape[0], self.horizon, self.action_dim)
        elif self.action_expert_type == "flow_matching":
            x = torch.randn(size=(cond.shape[0], self.horizon, self.action_dim), device=self.device, dtype=self.dtype)
            n = int(self.num_flow_steps_per_inference)
            dt = 1.0 / float(n)
            for i in range(n):
                t = (float(i) + 0.5) / float(n)
                v = self.velocity_field(x=x, t=t, cond=cond)
                x = x + dt * v
            traj = x
        else:
            noisy_traj = torch.randn(size=(cond.shape[0], self.horizon, self.action_dim), device=self.device, dtype=self.dtype)
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(self.num_denoise_steps_per_inference)
            for t in scheduler.timesteps:
                pred = self.backbone(sample=noisy_traj, timestep=t, cond=obs_feature)
                noisy_traj = scheduler.step(pred, t, noisy_traj, **self.noise_scheduler_step_kwargs).prev_sample
            traj = noisy_traj

        action = traj[:, self.num_latest_obs - 1 :].clone().cpu()
        return self._denormalize_action(action)

    def reset(self) -> None:
        pass

    @call_once
    def _check_forward_input_shape(self, obs: dict, actions: torch.Tensor, masks: torch.Tensor) -> None:
        L_obs = get_batch_size(any_slice(obs, 0), strict=True)
        if L_obs != self.num_latest_obs:
            raise ValueError(f"obs must have length {self.num_latest_obs}, got {L_obs}")
        if actions.shape[1] != self.horizon:
            raise ValueError(f"actions must have length {self.horizon}, got {actions.shape[1]}")
        if actions.shape[2] != self.action_dim:
            raise ValueError(f"action_dim mismatch: got {actions.shape[2]}, expected {self.action_dim}")
        if masks.ndim != 2 or masks.shape[1] != self.horizon:
            raise ValueError(f"masks must have shape (B,{self.horizon}), got {tuple(masks.shape)}")

    def policy_training_step(self, batch, batch_idx):
        batch["actions"] = any_concat([batch["actions"][k] for k in self._action_keys], dim=-1)
        batch = self.process_data(batch, extract_action=True)
        masks = batch.pop("masks")
        actions = batch.pop("actions")
        self._check_forward_input_shape(batch, actions, masks)
        _, obs_feature, cond = self._encode_obs(batch)

        if self.action_expert_type == "bc":
            pred_actions = self.bc_head(cond).reshape(cond.shape[0], self.horizon, self.action_dim)
            if self.bc_loss == "l1":
                loss = F.l1_loss(pred_actions, actions, reduction="none").mean(dim=-1)
            else:
                loss = F.mse_loss(pred_actions, actions, reduction="none").mean(dim=-1)
            loss = loss * masks
            real_batch_size = masks.sum()
            loss = loss.sum() / real_batch_size
            return loss, {"bc_loss": loss}, real_batch_size

        if self.action_expert_type == "flow_matching":
            B = actions.shape[0]
            x1 = actions
            x0 = torch.randn_like(x1)
            t = torch.rand((B,), device=x1.device, dtype=x1.dtype)
            t_b = t.view(B, 1, 1)
            x_t = (1.0 - t_b) * x0 + t_b * x1
            v_target = x1 - x0
            v_pred = self.velocity_field(x=x_t, t=t, cond=cond)
            loss = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=-1)
            loss = loss * masks
            real_batch_size = masks.sum()
            loss = loss.sum() / real_batch_size
            return loss, {"flow_matching_loss": loss}, real_batch_size

        B = actions.shape[0]
        trajectories = actions
        noise = torch.randn_like(trajectories)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=trajectories.device,
        ).long()
        noisy_trajs = self.noise_scheduler.add_noise(trajectories, noise, timesteps)
        pred = self.backbone(sample=noisy_trajs, timestep=timesteps, cond=obs_feature)
        loss = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)
        loss = loss * masks
        real_batch_size = masks.sum()
        loss = loss.sum() / real_batch_size
        return loss, {"diffusion_loss": loss}, real_batch_size

    def policy_evaluation_step(self, batch, batch_idx):
        demo_key = batch.get("demo_key")
        index = batch.get("index")
        batch["actions"] = any_concat([batch["actions"][k] for k in self._action_keys], dim=-1)
        batch = self.process_data(batch, extract_action=True)
        masks = batch.pop("masks")
        gt_actions = batch.pop("actions")
        self._check_forward_input_shape(batch, gt_actions, masks)
        _, obs_feature, cond = self._encode_obs(batch)
        gt_actions_full = gt_actions
        masks_full = masks

        if self.action_expert_type == "bc":
            traj = self.bc_head(cond).reshape(cond.shape[0], self.horizon, self.action_dim)
        elif self.action_expert_type == "flow_matching":
            x = torch.randn(size=gt_actions.shape, device=self.device, dtype=self.dtype)
            n = int(self.num_flow_steps_per_inference)
            dt = 1.0 / float(n)
            for i in range(n):
                t = (float(i) + 0.5) / float(n)
                v = self.velocity_field(x=x, t=t, cond=cond)
                x = x + dt * v
            traj = x
        else:
            noisy_traj = torch.randn(size=gt_actions.shape, device=self.device, dtype=self.dtype)
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(self.num_denoise_steps_per_inference)
            for t in scheduler.timesteps:
                pred = self.backbone(sample=noisy_traj, timestep=t, cond=obs_feature)
                noisy_traj = scheduler.step(pred, t, noisy_traj, **self.noise_scheduler_step_kwargs).prev_sample
            traj = noisy_traj

        proxy_log: Dict[str, torch.Tensor] = {}
        denom_full_proxy = masks_full.sum().clamp(min=1.0)
        if self.action_expert_type == "bc":
            mse_per_step = F.mse_loss(traj, gt_actions_full, reduction="none").mean(dim=-1)
            mse_per_step = mse_per_step * masks_full
            proxy_log["mse"] = mse_per_step.sum() / denom_full_proxy
        elif self.action_expert_type == "flow_matching":
            B = gt_actions_full.shape[0]
            x1 = gt_actions_full
            x0 = torch.randn_like(x1)
            t = torch.rand((B,), device=x1.device, dtype=x1.dtype)
            t_b = t.view(B, 1, 1)
            x_t = (1.0 - t_b) * x0 + t_b * x1
            v_target = x1 - x0
            v_pred = self.velocity_field(x=x_t, t=t, cond=cond)
            fm_per_step = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=-1)
            fm_per_step = fm_per_step * masks_full
            proxy_log["flow_matching_mse"] = fm_per_step.sum() / denom_full_proxy
            proxy_log["nll_proxy"] = proxy_log["flow_matching_mse"]
        else:
            B = gt_actions_full.shape[0]
            trajectories = gt_actions_full
            noise = torch.randn_like(trajectories)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=trajectories.device,
            ).long()
            noisy_trajs = self.noise_scheduler.add_noise(trajectories, noise, timesteps)
            pred = self.backbone(sample=noisy_trajs, timestep=timesteps, cond=obs_feature)
            dm_per_step = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)
            dm_per_step = dm_per_step * masks_full
            proxy_log["diffusion_noise_mse"] = dm_per_step.sum() / denom_full_proxy
            proxy_log["nll_proxy"] = proxy_log["diffusion_noise_mse"]

        pred_actions = traj[:, self.num_latest_obs - 1 :]
        gt_actions = gt_actions[:, self.num_latest_obs - 1 :]
        masks = masks[:, self.num_latest_obs - 1 :]

        l1_full_future_horizon_per_step = F.l1_loss(pred_actions, gt_actions, reduction="none").mean(dim=-1)
        l1_full_future_horizon_per_step = l1_full_future_horizon_per_step * masks
        denom_full = masks.sum()
        l1_full_future_horizon = l1_full_future_horizon_per_step.sum() / denom_full

        denom_per_sample = masks.sum(dim=-1).clamp(min=1.0)
        l1_per_sample = l1_full_future_horizon_per_step.sum(dim=-1) / denom_per_sample

        deployed_start_t = 0
        deployed_end_t = deployed_start_t + self.deployed_action_steps
        pred_actions_to_deploy = pred_actions[:, deployed_start_t:deployed_end_t]
        gt_actions_deploy = gt_actions[:, deployed_start_t:deployed_end_t]
        masks_deploy = masks[:, deployed_start_t:deployed_end_t]

        l1_deployed = F.l1_loss(pred_actions_to_deploy, gt_actions_deploy, reduction="none").mean(dim=-1)
        l1_deployed = l1_deployed * masks_deploy
        denom_deploy = masks_deploy.sum()
        l1_deployed = l1_deployed.sum() / denom_deploy

        log_dict = {
            "l1": l1_full_future_horizon,
            "l1_full_future_horizon": l1_full_future_horizon,
            "l1_deployed_steps_only": l1_deployed,
        }
        log_dict.update(proxy_log)
        log_dict.update(self._log_skill_buckets(batch=batch, metric_per_sample=l1_per_sample))
        log_dict.update(self._log_failure_stats(metric_per_sample=l1_per_sample))
        self._maybe_record_failure_cases(
            stage="test" if self.trainer is not None and self.trainer.testing else "val",
            demo_key=demo_key,
            index=index,
            skill_id=batch.get("skill_id"),
            l1_per_sample=l1_per_sample,
        )

        return (
            l1_full_future_horizon,
            log_dict,
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
        obs_in = data_batch["obs"]
        data: Dict[str, Any] = {"qpos": obs_in["qpos"], "eef": obs_in["eef"]}
        if "odom" in obs_in:
            data["odom"] = obs_in["odom"]
        for k in ["task", "skill_one_hot", "skill_id", "paligemma_token"]:
            if k in obs_in:
                data[k] = obs_in[k]
        if extract_action:
            data.update({"actions": data_batch["actions"], "masks": data_batch["masks"]})
        return data
