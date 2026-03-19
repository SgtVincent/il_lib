---
name: "debug-hydra-compose"
description: "定位 Hydra 配置 compose/覆盖导致的报错。当你遇到 defaults/组缺失/_target_ 导入失败/字段不生效时调用。"
---

# Debug Hydra Compose

## 适用场景

- `Could not find ...`（组缺失、defaults 组合错误）
- `_target_` import error
- overrides 没生效或字段被意外覆盖

## 排查路径

1. 先确认入口与搜索路径
   - 训练入口： [train.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/train.py)
   - Hydra 配置入口： [base_config.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/base_config.yaml)
   - Hydra 搜索路径插件： [setup.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/setup.py)
2. 输出 compose 后配置
   - 使用 Hydra 的输出选项打印最终 cfg，确认 defaults 组合与 overrides 的最终值
3. 检查 `_target_` 指向
   - 确认模块路径可 import，且类名拼写正确
4. 缩小最小复现
   - 先用 `trainer.fast_dev_run=true` 或最小 batch 验证能跑通

## 常见修复

- 把新增配置放进正确组目录（如 `configs/arch/`、`configs/eval/`）
- 避免在 YAML 中复用同名字段造成覆盖歧义
- 将路径类配置设为可覆盖参数，避免默认值依赖本机
