import math
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from il_lib.nn.diffusion import WholeBodyUNetDiffusionHead
from il_lib.nn.features import ObsTokenizer
from il_lib.nn.transformers import GPT
from il_lib.optim import (
    CosineScheduleFunction,
    check_optimizer_groups,
    default_optimizer_groups,
)
from il_lib.policies.policy_base import BasePolicy
from il_lib.training.trainer import rank_zero_info
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size


class MomaSTAGE(BasePolicy):
    """STAGE-style policy for MoMa (Mobile Manipulation).

    This is a minimal, evaluation-ready implementation aligned with the doc in
    `Robot Learning for Mobile Manipulation.md`:
    - **HSTT-like encoder**: WB-VIMA-style tokenization + GPT, plus a lightweight
      *content-addressable* memory context computed (a) causally within the obs window
      during training and (b) from a persistent episode memory during rollout.
    - **MC-MoE-like decoder**: diffusion head split into kinematic experts:
      `mobility` (base+torso), `left` (left arm + gripper), `right` (right arm + gripper),
      with sequential conditioning (mobility -> left -> right).
    """

    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        num_latest_obs: int,
        # ====== Obs Tokenizer ======
        feature_extractors: Dict[str, DictConfig],
        use_modality_type_tokens: bool,
        # ====== Transformer ======
        xf_n_embd: int,
        xf_n_layer: int,
        xf_n_head: int,
        xf_dropout_rate: float,
        xf_use_geglu: bool,
        # ====== Action Decoding ======
        learnable_action_readout_token: bool,
        action_dim: int,
        action_prediction_horizon: int,
        diffusion_step_embed_dim: int,
        unet_down_dims: List[int],
        unet_kernel_size: int,
        unet_n_groups: int,
        unet_cond_predict_scale: bool,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        # ====== Diffusion ======
        noise_scheduler: DictConfig,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: int = 16,
        # ====== STAGE knobs ======
        use_memory_context: bool = True,
        use_window_context: bool = True,
        use_external_memory_in_act: bool = True,
        memory_size: int = 256,
        memory_top_k: int = 16,
        memory_temperature: float = 1.0,
        # ====== learning ======
        lr: float = 1e-5,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: float = 5e-6,
        lr_layer_decay: float = 1.0,
        weight_decay: float = 0.0,
        loss_on_latest_obs_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._prop_keys = prop_keys
        self._features = set(feature_extractors.keys())

        self.obs_tokenizer = ObsTokenizer(
            extractors={k: instantiate(v) for k, v in feature_extractors.items()},
            use_modality_type_tokens=use_modality_type_tokens,
            token_dim=xf_n_embd,
            token_concat_order=list(feature_extractors.keys()),
            strict=True,
        )

        self.num_latest_obs = num_latest_obs
        if learnable_action_readout_token:
            self.action_readout_token = nn.Parameter(torch.zeros(xf_n_embd))
        else:
            # Need a buffer so Lightning / .to(device) moves it.
            self.register_buffer(
                "action_readout_token",
                torch.zeros(xf_n_embd),
                persistent=False,
            )

        self.transformer = GPT(
            n_embd=xf_n_embd,
            n_layer=xf_n_layer,
            n_head=xf_n_head,
            dropout=xf_dropout_rate,
            use_geglu=xf_use_geglu,
        )

        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())
        self.action_dim = action_dim
        self.action_prediction_horizon = action_prediction_horizon
        self._action_keys = list(action_keys)
        self._action_key_dims = dict(action_key_dims)
        self._noise_scheduler_step_kwargs = noise_scheduler_step_kwargs

        # --- STAGE: memory context ---
        self.use_memory_context = use_memory_context
        self.use_window_context = use_window_context
        self.use_external_memory_in_act = use_external_memory_in_act
        self.memory_size = int(memory_size)
        self.memory_top_k = int(memory_top_k)
        self.memory_temperature = float(memory_temperature)
        self._memory_keys: deque[torch.Tensor] = deque(maxlen=self.memory_size)
        self._memory_values: deque[torch.Tensor] = deque(maxlen=self.memory_size)

        decoder_obs_dim = xf_n_embd * (2 if self.use_memory_context else 1)
        mobility_dim = self._action_key_dims["base"] + self._action_key_dims["torso"]
        left_dim = self._action_key_dims["left_arm"] + self._action_key_dims["left_gripper"]
        right_dim = self._action_key_dims["right_arm"] + self._action_key_dims["right_gripper"]
        assert mobility_dim + left_dim + right_dim == self.action_dim

        self.action_decoder = WholeBodyUNetDiffusionHead(
            whole_body_decoding_order=["mobility", "left", "right"],
            action_dim_per_part={
                "mobility": mobility_dim,
                "left": left_dim,
                "right": right_dim,
            },
            obs_dim=decoder_obs_dim,
            action_horizon=action_prediction_horizon,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            noise_scheduler=instantiate(noise_scheduler),
            noise_scheduler_step_kwargs=noise_scheduler_step_kwargs,
            inference_denoise_steps=num_denoise_steps_per_inference,
            unet_down_dims=unet_down_dims,
            unet_kernel_size=unet_kernel_size,
            unet_n_groups=unet_n_groups,
            unet_cond_predict_scale=unet_cond_predict_scale,
        )

        # Learning
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay
        self.loss_on_latest_obs_only = loss_on_latest_obs_only

        self.save_hyperparameters()

    def reset(self) -> None:
        self._memory_keys.clear()
        self._memory_values.clear()

    def forward(self, obs: dict) -> torch.Tensor:
        # construct prop obs
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)  # (B, T, Prop_dim)

        obs_to_pass_in: Dict[str, torch.Tensor] = {
            "proprioception": prop_obs,
            "pcd": obs["pcd"],
        }
        if "task" in self._features and "task" in obs:
            obs_to_pass_in["task"] = obs["task"]

        obs_tokens = self.obs_tokenizer(obs_to_pass_in)
        B, _, E = obs_tokens.shape

        action_readout_tokens = self.action_readout_token.view(1, 1, -1).expand(
            B, self.num_latest_obs, -1
        )

        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        n_total_tokens = self.num_latest_obs * n_tokens_per_step
        tokens_in = torch.zeros(
            (B, n_total_tokens, E),
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )

        # insert obs tokens (interleaved modalities)
        for j in range(self.obs_tokenizer.num_tokens_per_step):
            tokens_in[:, j::n_tokens_per_step] = obs_tokens[
                :, j :: self.obs_tokenizer.num_tokens_per_step
            ]
        # insert action readout tokens
        tokens_in[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = (
            action_readout_tokens
        )

        # attention mask: mask action readout tokens
        mask = torch.ones(B, n_total_tokens, dtype=torch.bool, device=self.device)
        mask[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = False
        mask = mask.unsqueeze(1)  # (B, 1, T)

        # position ids: all obs tokens in the same step share the same id
        position_ids = torch.zeros((B, n_total_tokens), device=self.device, dtype=torch.long)
        p_id = 0
        for t in range(self.num_latest_obs):
            obs_st = t * n_tokens_per_step
            obs_end = obs_st + self.obs_tokenizer.num_tokens_per_step
            action_readout_p = obs_st + self.obs_tokenizer.num_tokens_per_step
            position_ids[:, obs_st:obs_end] = p_id
            p_id += 1
            position_ids[:, action_readout_p] = p_id
            p_id += 1

        tokens_in = rearrange(tokens_in, "B T E -> T B E")
        tokens_out = self.transformer(
            tokens_in,
            custom_mask=mask,
            batch_first=False,
            position_ids=position_ids,
        )
        tokens_out = rearrange(tokens_out, "T B E -> B T E")
        return tokens_out

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        obs = self.process_data(data_batch=obs, extract_action=False)

        transformer_output = self.forward(obs)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)

        context = self._compute_window_context(action_readout_tokens) if self.use_window_context else None
        if self.use_external_memory_in_act:
            ext_context = self._compute_external_memory_context(action_readout_tokens)
            context = ext_context if context is None else (context + ext_context)

        fused_readouts = self._fuse_readouts(action_readout_tokens, context)

        pred_parts = self.action_decoder.inference(
            obs=fused_readouts,
            return_last_timestep_only=True,
        )
        pred_action_dict = self._parts_to_action_dict(pred_parts, has_time_dim=False)
        action = torch.cat([pred_action_dict[k][0].detach().cpu() for k in self._action_keys], dim=-1)  # (T_act, A)

        # update persistent memory *after* computing action
        self._memory_push(action_readout_tokens[:, -1])
        return self._denormalize_action(action)

    def policy_training_step(self, batch, batch_idx) -> Any:
        batch = self.process_data(data_batch=batch, extract_action=True)
        B = get_batch_size(any_slice(batch["actions"], np.s_[0]), strict=True)

        pad_mask = batch.pop("masks")  # (B, T_obs, T_act)
        target_action_dict = batch.pop("actions")

        transformer_output = self.forward(batch)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        context = self._compute_window_context(action_readout_tokens) if self.use_window_context else None
        fused_readouts = self._fuse_readouts(action_readout_tokens, context)

        gt_parts = self._action_dict_to_parts(target_action_dict)
        loss = self.action_decoder.compute_loss(obs=fused_readouts, gt_action=gt_parts)  # (B, T_obs, T_act)

        if self.loss_on_latest_obs_only:
            latest_mask = torch.zeros_like(pad_mask)
            latest_mask[:, -1] = 1
            pad_mask = pad_mask * latest_mask

        loss = loss * pad_mask
        diffusion_loss = torch.sum(loss) / pad_mask.sum()
        diffusion_loss = diffusion_loss * self.action_prediction_horizon
        return diffusion_loss, {"diffusion_loss": diffusion_loss}, B

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        batch = self.process_data(data_batch=batch, extract_action=True)
        B = get_batch_size(any_slice(batch["actions"], np.s_[0]), strict=True)

        pad_mask = batch.pop("masks")  # (B, T_obs, T_act)
        target_action_dict = batch.pop("actions")

        transformer_output = self.forward(batch)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        context = self._compute_window_context(action_readout_tokens) if self.use_window_context else None
        fused_readouts = self._fuse_readouts(action_readout_tokens, context)

        pred_parts = self.action_decoder.inference(
            obs=fused_readouts,
            return_last_timestep_only=False,
        )
        pred_action_dict = self._parts_to_action_dict(pred_parts, has_time_dim=True)

        all_l1 = {}
        for action_k in self._action_keys:
            pred = pred_action_dict[action_k]
            gt = target_action_dict[action_k]
            l1 = F.l1_loss(pred, gt, reduction="none")
            l1 = l1.sum(dim=-1).reshape(pad_mask.shape)
            if self.loss_on_latest_obs_only:
                latest_mask = torch.zeros_like(pad_mask)
                latest_mask[:, -1] = 1
                pad_mask = pad_mask * latest_mask
            all_l1[action_k] = l1 * pad_mask

        all_loss = {
            f"l1_{k}": torch.sum(v) / pad_mask.sum() * self.action_prediction_horizon
            for k, v in all_l1.items()
        }
        summed_l1 = sum(all_loss.values())
        all_loss["l1"] = summed_l1
        return summed_l1, all_loss, B

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_groups = self._get_optimizer_groups(
            weight_decay=self.weight_decay,
            lr_layer_decay=self.lr_layer_decay,
            lr_scale=1.0,
        )
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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

    def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        feature_encoder_pg, feature_encoder_pid = self.obs_tokenizer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        transformer_pg, transformer_pid = self.transformer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        action_decoder_pg, action_decoder_pid = self.action_decoder.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=["action_readout_token"],
            exclude_filter=lambda name, p: id(p)
            in feature_encoder_pid + transformer_pid + action_decoder_pid,
        )
        all_groups = feature_encoder_pg + transformer_pg + action_decoder_pg + other_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups

    def _get_action_readout_tokens(self, transformer_output: torch.Tensor) -> torch.Tensor:
        B, _, E = transformer_output.shape
        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        action_readout_tokens = transformer_output[
            :, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step
        ]  # (B, T_obs, E)
        assert action_readout_tokens.shape == (B, self.num_latest_obs, E)
        return action_readout_tokens

    def _compute_window_context(self, readouts: torch.Tensor) -> torch.Tensor:
        """Causal within-window attention over readout tokens.

        Args:
            readouts: (B, T, E)
        Returns:
            context: (B, T, E), context[:, 0] = 0
        """
        B, T, E = readouts.shape
        if T <= 1:
            return torch.zeros_like(readouts)

        # sim[b, t, s] = dot(q_t, k_s)
        sim = torch.einsum("bte,bse->bts", readouts, readouts) / math.sqrt(E)
        # mask: only attend to s < t
        t_idx = torch.arange(T, device=readouts.device)
        causal = t_idx[None, :, None] > t_idx[None, None, :]
        sim = sim.masked_fill(~causal, -1e9)
        att = torch.softmax(sim / max(self.memory_temperature, 1e-6), dim=-1)
        context = torch.einsum("bts,bse->bte", att, readouts)
        context[:, 0] = 0
        return context

    def _compute_external_memory_context(self, readouts: torch.Tensor) -> torch.Tensor:
        """Attention over persistent episode memory (used in rollout)."""
        if (not self.use_memory_context) or len(self._memory_keys) == 0:
            return torch.zeros_like(readouts)

        mem_k = torch.stack(list(self._memory_keys), dim=0)  # (M, E)
        mem_v = torch.stack(list(self._memory_values), dim=0)  # (M, E)
        B, T, E = readouts.shape
        sim = torch.einsum("bte,me->btm", readouts, mem_k) / math.sqrt(E)

        k = min(self.memory_top_k, sim.shape[-1])
        if k <= 0:
            return torch.zeros_like(readouts)
        topv, topi = torch.topk(sim, k=k, dim=-1)
        weights = torch.softmax(topv / max(self.memory_temperature, 1e-6), dim=-1)  # (B, T, k)
        selected_v = mem_v[topi]  # (B, T, k, E)
        context = torch.einsum("btk,btke->bte", weights, selected_v)
        return context

    def _memory_push(self, readout_last: torch.Tensor) -> None:
        """Push last-step readout token into memory.

        Args:
            readout_last: (B, E). Rollout uses B=1.
        """
        if not self.use_memory_context:
            return
        # store per-batch entries independently
        for b in range(readout_last.shape[0]):
            v = readout_last[b].detach()
            self._memory_keys.append(v)
            self._memory_values.append(v)

    def _fuse_readouts(self, readouts: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_memory_context:
            return readouts
        if context is None:
            context = torch.zeros_like(readouts)
        return torch.cat([readouts, context], dim=-1)

    def _action_dict_to_parts(self, action_dict: dict) -> dict[str, torch.Tensor]:
        mobility = torch.cat([action_dict["base"], action_dict["torso"]], dim=-1)
        left = torch.cat([action_dict["left_arm"], action_dict["left_gripper"]], dim=-1)
        right = torch.cat([action_dict["right_arm"], action_dict["right_gripper"]], dim=-1)
        return {"mobility": mobility, "left": left, "right": right}

    def _parts_to_action_dict(self, parts: dict[str, torch.Tensor], *, has_time_dim: bool) -> dict[str, torch.Tensor]:
        """Convert decoder parts to canonical action dict.

        Args:
            parts:
              - if has_time_dim: each value is (B, T_obs, T_act, D_part)
              - else: each value is (B, T_act, D_part)
        """
        base_dim = self._action_key_dims["base"]
        torso_dim = self._action_key_dims["torso"]
        left_arm_dim = self._action_key_dims["left_arm"]
        left_gripper_dim = self._action_key_dims["left_gripper"]
        right_arm_dim = self._action_key_dims["right_arm"]
        right_gripper_dim = self._action_key_dims["right_gripper"]

        mob = parts["mobility"]
        left = parts["left"]
        right = parts["right"]

        if has_time_dim:
            base = mob[..., :base_dim]
            torso = mob[..., base_dim : base_dim + torso_dim]
            left_arm = left[..., :left_arm_dim]
            left_gripper = left[..., left_arm_dim : left_arm_dim + left_gripper_dim]
            right_arm = right[..., :right_arm_dim]
            right_gripper = right[..., right_arm_dim : right_arm_dim + right_gripper_dim]
        else:
            # (B, T_act, D)
            base = mob[..., :base_dim]
            torso = mob[..., base_dim : base_dim + torso_dim]
            left_arm = left[..., :left_arm_dim]
            left_gripper = left[..., left_arm_dim : left_arm_dim + left_gripper_dim]
            right_arm = right[..., :right_arm_dim]
            right_gripper = right[..., right_arm_dim : right_arm_dim + right_gripper_dim]

        return {
            "base": base,
            "torso": torso,
            "left_arm": left_arm,
            "left_gripper": left_gripper,
            "right_arm": right_arm,
            "right_gripper": right_gripper,
        }

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        fused_pcd = data_batch["obs"]["pcd"]
        data: Dict[str, Any] = {
            "pcd": {
                "rgb": fused_pcd[..., :3],
                "xyz": fused_pcd[..., 3:],
            },
            "qpos": data_batch["obs"]["qpos"],
            "eef": data_batch["obs"]["eef"],
        }
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]
        if "task" in self._features and "task" in data_batch["obs"]:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            data.update(
                {
                    "actions": data_batch["actions"],
                    "masks": data_batch["masks"],
                }
            )
        return data