---
name: "mt-il-dataset-conditioning"
description: "规范 MT-IL 的条件字段（task/skill_id/skill_one_hot/paligemma_token）与数据集封装方式，并给出可复现的 Hydra 用法。当你要给真实数据加条件输入或做 cache 接入时调用。"
---

# MT-IL Dataset Conditioning

## 目标

把 multi-task 条件输入从“实验想法”固化成可复用的数据接口，保证不同 policy/action expert 都能吃到一致的条件字段。

## 条件字段规范

- `obs["task"]`: `(L, D_task)` float32，数值 task-info（来自 env 或离线生成）
- `obs["skill_id"]`: `(L,)` int64，离散 skill 标签
- `obs["skill_one_hot"]`: `(L, K)` float32，离散 skill 的 one-hot
- `obs["paligemma_token"]`: `(L, D)` float32，预提取/缓存的 VLM embedding

DataLoader collate 后：

- `task/skill_one_hot/paligemma_token`: `(B, L, D)`
- `skill_id`: `(B, L)`

## 真实数据：annotation 注入 skill label

使用 `SkillLabeledIterableDataset` 按 segment 切片并注入 `skill_id/skill_one_hot`：

- [skill_labeled_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/skill_labeled_dataset.py)

示例（skill id）：

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.skill_labeled_dataset.SkillLabeledIterableDataset \
  data.label_level=skill data.label_mode=id data.num_skills=64
```

示例（skill one-hot）：

```bash
python train.py \
  arch=mt_il_validation_s1 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.skill_labeled_dataset.SkillLabeledIterableDataset \
  data.label_level=skill data.label_mode=one_hot data.num_skills=64 \
  module.feature_extractors.skill_one_hot.input_dim=64
```

## paligemma_token cache（已实现）

建议拆成两层：

1. 离线提取：episode/frame -> embedding（落盘到 cache_dir）
2. 训练读取：dataset 读取并对齐到 `(L, D)` 的时间窗

训练侧注入已提供（注入 `obs["paligemma_token"]`，并要求 sample 能提供 `demo_key/index` 用于对齐；本仓库相关 dataset 已会注入 `demo_key/index`）：

- `PaligemmaTokenCachedIterableDataset`：[paligemma_token_cache_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/paligemma_token_cache_dataset.py)

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.paligemma_token_cache_dataset.PaligemmaTokenCachedIterableDataset \
  data.cache_dir=<CACHE_DIR> data.token_dim=2048
```

离线生成已提供（从 BEHAVIOR demos/videos 或从 frame tensor）：

- [build_paligemma_token_cache_from_behavior_demos.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/build_paligemma_token_cache_from_behavior_demos.py)
- [build_paligemma_token_cache_from_frames.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/build_paligemma_token_cache_from_frames.py)

## Setting 2 一体化数据集（skill_id + paligemma_token）

当你需要同时注入 `skill_id` 与 `paligemma_token`（Setting 2），推荐直接使用一体化 dataset（避免 dataset_class 无法叠加）：

- [mt_il_conditioned_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/mt_il_conditioned_dataset.py)

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.mt_il_conditioned_dataset.MTILConditionedIterableDataset \
  data.label_level=skill data.label_mode=id data.num_skills=64 \
  data.cache_dir=<CACHE_DIR> data.paligemma_token_dim=2048 \
  module.feature_extractors.skill_id.num_embeddings=64 \
  module.feature_extractors.paligemma_token.input_dim=2048
```

## 输出要求

- 明确写出字段 shape（训练与 rollout 两侧）
- 明确 cache 的 key 设计（episode_id、frame_idx、downsample）
- 明确与 policy 的 feature_extractors 对齐方式（Identity/MLP/Embedding）

## Failure 回放（索引 -> window 数据）

当训练开启 `module.export_failure_cases=true` 后，会在 run_dir 下生成 `failure_cases/{val|test}_topK.jsonl`。
可用脚本把对应 `(demo_key,index)` 的 window 抽取出来保存为 `pt`：

- [export_failure_windows.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/export_failure_windows.py)
