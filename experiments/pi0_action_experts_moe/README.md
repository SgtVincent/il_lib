# PI0 Action Expert 对比（Default vs MoE）— Mobile Manipulation IL

## 目标

在 `il_lib` 的 imitation learning 训练与评测链路内，对比同一个 PI0 backbone 下两种 action expert：

- Default expert：`gemma_token`
- 新设计 MoE expert：`il_moe_velocity`（实现位于 openpi-comet 的 `action_experts/il_moe_velocity_expert.py`，内部调用 `il_lib` 的 MoE 速度场）

对比任务：Mobile manipulation 轨迹（示例：`make_pizza`），输入只使用：

- proprio（qpos / eef / odom）
- task 低维向量（应包含 task obs(state) 与 primitive_index_one_hot）

不使用 sensor 输入（RGB / Depth / PCD）。由于 openpi PI0 模型接口要求三路图像与 prompt，本实验在 policy 内部构造常量 dummy images 与 dummy prompt，它们不携带任务信息。

## 关键实现

- 新增 policy：`il_lib.policies.Pi0ActionExpertPolicy`
- 新增 arch configs：
  - `arch=pi0_default_expert`：action expert = `gemma_token`
  - `arch=pi0_moe_expert`：action expert = `il_moe_velocity`

## 运行前准备

1) 使用包含 openpi-comet 依赖的 Python 环境（至少需要 `transformers` 等依赖能 import）。  
2) 确保 openpi-comet 可被 import（建议用 PYTHONPATH）：

```bash
export PYTHONPATH=/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src:$PYTHONPATH
```

3) 准备数据目录（应包含 `2025-challenge-demos` 结构，与 `il_lib/.trae/documents/dataset_readme.md` 一致）：

- `${DATA_DIR}/2025-challenge-demos/data/...`
- `${DATA_DIR}/2025-challenge-demos/annotations/...`

4) 准备 PI0 预训练权重（torch checkpoint），用于 `module.pi0_ckpt_path=...`。

## 训练（离线 IL）

### Default expert（gemma_token）

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib
python train.py \
  robot=r1pro \
  task=behavior task.name=make_pizza \
  arch=pi0_default_expert \
  data_dir=${DATA_DIR} \
  module.pi0_ckpt_path=${PI0_CKPT} \
  module.task_dim=82 \
  use_wandb=false
```

### MoE expert（il_moe_velocity）

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib
python train.py \
  robot=r1pro \
  task=behavior task.name=make_pizza \
  arch=pi0_moe_expert \
  data_dir=${DATA_DIR} \
  module.pi0_ckpt_path=${PI0_CKPT} \
  module.task_dim=82 \
  use_wandb=false
```

说明：

- `module.task_dim` 必须与数据里 `obs["task"]` 的维度一致。真实数据中该维度可能不是 82，可按需覆盖（参考 `EVALUATION.md` 里对 WBVIMA 的 override 示例）。
- 两个 arch 都设置了 `data.visual_obs_types=[]`，并且 `data.use_task_info=true`，不会从 policy wrapper 读取视觉输入。

## 启动 policy server（用于在线评测）

Default expert：

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib
python serve.py \
  robot=r1pro \
  task=behavior task.name=make_pizza \
  arch=pi0_default_expert \
  ckpt_path=${IL_CKPT}
```

MoE expert：

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib
python serve.py \
  robot=r1pro \
  task=behavior task.name=make_pizza \
  arch=pi0_moe_expert \
  ckpt_path=${IL_CKPT}
```

## SubTask (Primitive) Evaluation（BRS Protocol）

按 [EVALUATION.md](../../EVALUATION.md) 的 “SubTask (Primitive) Evaluation - BRS Protocol” 运行：

```bash
conda activate behavior
python ./OmniGibson/omnigibson/learning/subtask_eval.py \
  policy=websocket \
  task.name=make_pizza \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  demo_data_path=${DATA_DIR}/2025-challenge-demos \
  log_path=./eval_logs/subtask_eval_pi0 \
  reset_on_primitive_failure=true \
  primitive_timeout_multiplier=2.0 \
  num_demos=10
```

产出结果包括：

- `subtask_eval_{task_name}_{demo_id}.json`
- `subtask_eval_{task_name}_aggregate.json`（ST/ET success rate）

## 结果记录

把你实际跑出来的：

- 训练曲线（train/val 指标、loss）
- checkpoint 路径
- subtask eval 输出 json

汇总到同目录的 [report.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/experiments/pi0_action_experts_moe/report.md)。
