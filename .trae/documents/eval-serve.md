# 在线评测 / 服务联调规则

## 服务端（serve.py）
- 服务端通过 Hydra 构建 policy，并用 `policy_wrapper` 包装后启动 websocket server（见 [serve.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/serve.py)）
- 新增 wrapper 时必须可通过 `_target_` 替换，并且不修改 policy 的权重加载语义

## 评测配置（+eval=...）
- `configs/eval/*.yaml` 应仅包含“如何组合策略+wrapper+评测参数”，不要包含环境实现细节
- 默认不写硬编码绝对路径；若确实需要（如技能 ckpt 映射），放在文档示例里并要求用户覆盖

## 分层策略
- 分层策略评测以 `+eval=hierarchical_*` 方式组合（参考 [EVALUATION.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/EVALUATION.md) 与 `il_lib/configs/eval/`）
