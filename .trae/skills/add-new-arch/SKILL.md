---
name: "add-new-arch"
description: "新增一种 policy/arch（代码+Hydra 配置+最小训练命令）。当你要引入新模型结构或接入新策略到 train.py 时调用。"
---

# Add New Arch

## 适用场景

- 新增一个策略/模型结构，并希望能通过 `python train.py ... arch=<name>` 直接训练
- 需要把现有代码重构成可配置 `_target_` 组件

## 约定

- 代码放 `il_lib/policies/` 或 `il_lib/nn/`
- Hydra 配置放 `il_lib/configs/arch/<name>.yaml`
- 训练入口固定走 [train.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/train.py)

## 步骤

1. 选择实现落点
   - 若是完整策略（包含 loss/优化器/训练步骤），优先放到 `il_lib/policies/`
   - 若是纯网络模块，放到 `il_lib/nn/`，由现有 policy 组合调用
2. 写可导入的 Python 类（路径稳定、无隐式 IO）
3. 增加 `configs/arch/<name>.yaml`
   - 定义 `module:` 的 `_target_` 指向你的类
   - 把超参放入 YAML，避免写死在代码里
4. 用最小命令验证
   - 参考 [base_config.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/base_config.yaml) 的组合结构与 [train_commands.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/train_commands.sh) 的 overrides 风格

## 最小验证清单

- `python train.py arch=<name> task=<task> robot=<robot> trainer.fast_dev_run=true`
- cfg 能正常 compose，且 `_target_` 可 import
- 不依赖本机绝对路径默认值
