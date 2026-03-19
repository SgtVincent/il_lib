---
name: "moma-action-expert"
description: "指导扩展与评测 MoMaSTAGE 的 action expert（BC/Diffusion/Flow 等）并沉淀可复现配置与指标。当你要新增/替换 moma decoder 或做 ablation 时调用。"
---

# MoMa Action Expert

## 适用场景

- 需要在固定 MoMaSTAGE encoder（ObsTokenizer + GPT + memory context）条件下，替换 action expert（decoder / loss）做可比实验
- 需要新增一个新的 moma action expert（例如新的 diffusion 结构、FM 变体、MoE 设计、分块解码顺序等）
- 需要把实验矩阵、配置入口、指标口径沉淀到 `.trae/documents/`

## 约定接口

对齐 `WholeBodyUNetDiffusionHead` 的接口：

- `inference(obs: (B,T_obs,D), return_last_timestep_only: bool) -> dict[str, Tensor]`
  - `True`：每个 part 为 `(B, T_act, D_part)`
  - `False`：每个 part 为 `(B, T_obs, T_act, D_part)`
- `compute_loss(obs, gt_action_parts) -> (B, T_obs, T_act)`
- `get_optimizer_groups(weight_decay, lr_layer_decay, lr_scale)`

## 推荐工作流

1. 在 `il_lib/il_lib/nn/action_experts/` 新增或扩展 expert，实现上述接口
2. 在 `il_lib/il_lib/policies/moma_stage_policy.py` 里通过 `module.action_expert` 注入
3. 在 `il_lib/il_lib/configs/arch/` 增加独立的 arch yaml（每个 expert 一个入口）
4. 在 `.trae/documents/multitask_il_baselines.md` 更新：
   - 当前实现入口（代码/配置链接）
   - 最小可跑命令模板
   - TODO 与指标口径

## 参考入口

- Policy：[moma_stage_policy.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/policies/moma_stage_policy.py)
- Experts：[moma_action_experts.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/nn/action_experts/moma_action_experts.py)
- Arch configs（示例）：
  - [moma_stage.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/arch/moma_stage.yaml)
  - [moma_stage_action_expert_bc.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/arch/moma_stage_action_expert_bc.yaml)
  - [moma_stage_action_expert_flow.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/arch/moma_stage_action_expert_flow.yaml)

## 输出模板

输出应包含：

- “改动点摘要”（新增 expert / 改动接口 / 新增 config）
- “最小可跑命令”（cpu + fast_dev_run / limit_batches）
- “指标口径”（offline l1 / per-skill buckets / deployed-steps）

