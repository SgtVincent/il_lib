---
name: "il_lib-hydra-lightning-rules"
scope: "repo"
---

# Hydra / Lightning 规则

## Hydra 组合

- 新增 arch 时：
  - 在 `il_lib/configs/arch/<name>.yaml` 定义 `module:` 的 `_target_` 与超参
  - 保持 `task/robot` 可通过 overrides 组合使用
- 新增 eval 时：
  - 在 `il_lib/configs/eval/<name>.yaml` 定义在线评测相关配置，保证可与 `serve.py` 联动

## 配置字段习惯

- `_target_` 只指向可导入的 Python 路径
- 不在 YAML 中写运行时副作用（如删除目录、下载权重）
- 任何默认路径若不可通用，默认设为空并要求命令行覆盖

## Lightning 训练接口

- 策略训练组件应与 LightningModule 接口兼容，训练流程由 [train.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/train.py) 驱动
- 训练侧封装遵循 `il_lib/training/` 的现有结构，优先复用现有 trainer 逻辑（见 [trainer.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/training/trainer.py)）

## 常见调试方法

- 仅验证配置组合：先运行 `python train.py -c job --cfg job` 或打印 cfg（避免先跑全量训练）
- 遇到 `_target_` 导入失败：检查包路径与 `setup.py` / 包结构一致（见 [setup.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/setup.py)）
