# il_lib 本地工作区规则

## 目标

本仓库是面向机器人操作的 imitation learning 模型库，核心入口为 [train.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/train.py) 与 [serve.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/serve.py)，配置系统以 Hydra 为中心（见 [base_config.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/base_config.yaml)）。

## 代码组织

- 新策略 / policy：放在 `il_lib/policies/`
- 新网络 / 模块：放在 `il_lib/nn/`
- 通用工具：放在 `il_lib/utils/`
- 新配置：放在 `il_lib/configs/` 下对应组（常见为 `arch/`、`task/`、`robot/`、`eval/`）
- 数据集： 放在  `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/`

## 配置约定
- 运行python 脚本需要先激活conda 环境 `conda activate openpi-comet-nas`
- 任何新增训练/推理组件都必须能通过 Hydra `_target_` 方式实例化
- 项目自进化：所有的新增功能都必须有对应的文档和代码注释，文档放置在 `.trae/documents` 下面，函数有docstring，新增功能和non-trivial的修改需要有注释，同时对于相对独立和有潜在多次复用可能的流程，生成skill 到 `.trae/skills` 路径下。
- Agentic Behavior: 在进行复杂任务时，需要思考对应的任务task plan，同时思考是否有合适的的MCP tool/ skill 来完成任务。

