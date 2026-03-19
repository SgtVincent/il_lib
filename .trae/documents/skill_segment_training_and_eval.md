# Skill-level segment 训练与评测（MT-IL）

## 背景与目标

在 BEHAVIOR-1K 的长任务轨迹里，一个 primitive 往往包含多个更细粒度的 skill 段落（例如大量 `move to` 穿插在 pick/place/open 等操作之间）。直接按 primitive 作为多任务条件训练时，单个条件下的动作分布会更复合、更多峰，容易导致：

- 训练侧：同一 label 覆盖的状态-动作模式过多，BC / diffusion / flow-matching 都更难拟合
- 数据侧：skill 分布严重不均衡，`move to` 等高频 skill 容易主导梯度

本文件给出一条“可落地”的最小闭环：按 skill segments 训练 + 离线分桶评估，并补齐需要的 vocab 与采样设置。

## 标签定义（推荐）

训练用的离散 label 建议使用 `skill_description`（可选拼上 `skill_type`）：

- `label_scheme=description`：label = `skill_description`
- `label_scheme=type_description`：label = `skill_type:skill_description`

原因：

- label 空间小且稳定（通常几十个量级），便于 embedding / one-hot
- 便于做 per-skill bucket 的离线指标诊断

## 生成 vocab 与频次统计

从 annotations 构建 vocab，优先按频次排序（用于截断 top-K 训练）：

```bash
python scripts/build_segment_vocab.py \
  --demo_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/pcd_vid \
  --label_level skill --label_scheme description \
  --out_dir outputs/skill_vocab_desc \
  --top_k 64
```

产物：

- `vocab.json`：list[str]（按频次排序）
- `vocab_to_id.json`：dict[str,int]
- `counts.csv / counts.json`：频次统计

## Dataset：按 segment 切片并注入 skill label

使用 `SkillLabeledIterableDataset`：

- `data.label_level=skill|primitive`
- `data.label_scheme=description|type_description`
- `data.label_mode=id|one_hot`
- `data.skill_vocab_path=<vocab.json>` 或 `data.num_skills=<K>`（不传 vocab 时会扫描 demo_keys 构建）

## 训练：解决类别不均衡（推荐自动均衡采样）

`move to` 等高频 skill 会主导训练。建议开启按 group 的自动均衡重采样：

- `data.resample_group_by=label`：按最终 label（受 `label_scheme` 影响）分组
- `data.resample_auto_balance=true`：对每个 group 赋权 `1/count`，近似实现 group-uniform sampling
- `data.resample_num_segments=<N>`：控制每个 epoch 采样多少 segment（稳定 epoch 长度）

示例（Setting 2：skill_id + paligemma_token）：

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.mt_il_conditioned_dataset.MTILConditionedIterableDataset \
  data.label_level=skill data.label_scheme=description data.label_mode=id \
  data.skill_vocab_path=outputs/skill_vocab_desc/vocab.json data.num_skills=64 \
  data.resample_group_by=label data.resample_auto_balance=true data.resample_num_segments=200000 \
  module.feature_extractors.skill_id.num_embeddings=64 \
  module.log_skill_buckets=true module.max_skill_buckets=64
```

NAS 数据读取建议调大 dataloader 的 `num_workers`（单卡 16+，多卡每卡 10+）。

## 离线评测：per-skill bucket 指标

当 batch 里包含 `obs["skill_id"]` 时，可以打开按 skill 分桶统计：

- `module.log_skill_buckets=true`
- `module.max_skill_buckets=64`

用于快速判断：

- 哪些 skill 的误差显著高（模型能力问题）
- 哪些 skill 训练充分但闭环仍失败（更可能是 skill 切换 / planner 问题）

## 闭环评测的现实约束

Challenge 的端到端评测不会提供 `skill_id/primitive_id` 作为输入，因此：

- skill/primitive segment 训练更适合用于离线诊断、子任务/段落能力验证
- 若要把 `skill_id` 作为在线条件，需要额外的高层模块产生 skill 序列（planner / selector / latent gating）
