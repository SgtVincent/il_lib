---
name: "run-train-eval"
description: "生成并校验训练/服务/评测联调命令。当你要跑一套可复现实验（train→ckpt→serve→eval）时调用。"
---

# Run Train + Serve + Eval

## 适用场景

- 想快速从配置组合得到一条可运行的训练命令
- 想把某个 ckpt 拉起服务端，并给出评测侧联调提示
- 想把命令沉淀到 `scripts/` 或文档，保证可复现

## 参考入口

- 训练： [train.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/train.py)
- 服务： [serve.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/serve.py)
- 评测说明： [EVALUATION.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/EVALUATION.md)
- 命令风格： [train_commands.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/train_commands.sh)

## 输出模板

### 训练命令（最小跑通）

```bash
python train.py arch=<arch> task=<task> robot=<robot> trainer.fast_dev_run=true
```

### 正式训练命令（示例骨架）

```bash
python train.py arch=<arch> task=<task> robot=<robot> \
  data.data_dir=<DATA_DIR> \
  trainer.devices=<N> trainer.accelerator=gpu
```

### 服务端命令（加载 ckpt）

```bash
python serve.py arch=<arch> task=<task> robot=<robot> ckpt_path=<CKPT_PATH>
```

## 检查清单

- cfg compose 成功，无缺失组（`arch/task/robot`）
- `_target_` 导入无异常
- `ckpt_path` 为显式参数，不通过默认配置写死
