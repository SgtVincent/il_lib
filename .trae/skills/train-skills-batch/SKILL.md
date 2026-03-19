---
name: "train-skills-batch"
description: "批量训练技能策略并生成分层策略映射提示。当你要跑 hierarchical policy 的 skills 训练/ckpt 管理时调用。"
---

# Train Skills Batch

## 适用场景

- 需要为分层策略准备一组 skill checkpoints
- 需要把 skill→ckpt 映射用于在线评测（`+eval=hierarchical_*`）

## 参考

- 批量训练脚本： [train_skills.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/train_skills.sh)
- 分层策略说明： [hierarchical_policy.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/hierarchical_policy.md)
- 评测说明： [EVALUATION.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/EVALUATION.md)

## 建议输出

- 一组可复现的训练命令（按 skill name 列表展开）
- 统一的 ckpt 命名/目录规范（便于 eval 覆盖）
- 一个可粘贴到 `configs/eval/*.yaml` 的映射片段（由用户按机器路径覆盖）
