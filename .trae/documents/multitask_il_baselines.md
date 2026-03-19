# MT-IL Validation（固定 encoder，仅对比 action expert）

## 目的

在 multi-task imitation learning 场景下，**固定条件编码（encoder）路径**，只对比不同 **action expert 结构 / loss** 对轨迹学习的影响，避免 encoder 差异带来的混杂因素。

## 当前实现

- Policy：[mt_il_validation_policy.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/policies/mt_il_validation_policy.py)
- Arch config：
  - Setting 1: [mt_il_validation_s1.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/arch/mt_il_validation_s1.yaml)
  - Setting 2: [mt_il_validation_s2.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/configs/arch/mt_il_validation_s2.yaml)
- Synthetic dataset（用于连通性验证与快速 ablation）：
  - [dataset.py:MultiTaskSyntheticBehaviorDataset](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/dataset.py#L110-L260)

## 运行约束（重要）

- 运行 python / Hydra 入口前，先激活 conda 环境：`conda activate openpi-comet-nas`
- 本仓库默认启用 Hydra search-path plugin（依赖 `omnigibson`）；未激活正确环境会导致 `hydra_plugins.search_path_plugin` 导入失败

## Held-out offline validation（当前口径）

- `train.py` 会自动执行 `fit()` 然后执行 `test()`（见 [train.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/train.py)）
- `BehaviorDataModule` 当前的 held-out/test 就是 `val_split_ratio` 切分出的那一部分 demo keys，`test` 阶段复用 `val_demo_keys`（见 [BehaviorDataModule.setup](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/data_module.py#L45-L83)）

### Setting 1（state-based）

- 条件输入：`proprioception` + `task` + `skill_one_hot`
- 用途：验证单一网络学习所有 skill 的 multi-tasking 能力（state-based policy）

### Setting 2（visuomotor-lite）

- 条件输入：`proprioception` + `paligemma_token` + `skill_id`（embedding）
- 用途：假设视觉信息已被外部 VLM 编码为固定向量，仅研究 action expert 的多任务轨迹学习能力

### action expert 切换

仅通过 `module.action_expert_type` 切换（encoder 配置保持不变）：

- `bc`：直接回归整段 trajectory（`bc_loss` 支持 `l1/mse`）
- `diffusion`：DDIM 条件扩散（预测噪声 MSE）
- `flow_matching`：Flow Matching（MoE velocity field，速度场 MSE）

## Synthetic 命令（建议先跑通）

### Synthetic（IterableDataset，带 demo_key/index + failure 导出）

当你想验证 failure 导出 / cache 对齐 / window 抽取链路，推荐用可迭代版 synthetic 数据集：

- Dataset: `il_lib.datas.synthetic_iterable_dataset.MTILSyntheticIterableDataset`

### Setting 1：skill_one_hot

```bash
python train.py \
  arch=mt_il_validation_s1 task=behavior robot=r1pro \
  task.name=mt_synth_s1 data_dir=. use_wandb=false \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 \
  data.dataset_class=il_lib.datas.dataset.MultiTaskSyntheticBehaviorDataset \
  +data.num_tasks=8 +data.task_dim=46 data.use_action_chunks=false +data.include_rgbd=false \
  +data.num_skills=8 +data.include_skill_one_hot=true +data.skill_one_hot_dim=8 \
  module.feature_extractors.task.input_dim=46 module.feature_extractors.skill_one_hot.input_dim=8
```

切换 action expert：

```bash
python train.py ... module.action_expert_type=diffusion
python train.py ... module.action_expert_type=flow_matching
```

### Setting 2：PaLI-Gemma token + skill embedding

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  task.name=mt_synth_s2 data_dir=. use_wandb=false \
  trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 \
  data.dataset_class=il_lib.datas.dataset.MultiTaskSyntheticBehaviorDataset \
  +data.num_tasks=8 +data.task_dim=46 data.use_action_chunks=false +data.include_rgbd=false \
  +data.num_skills=8 +data.include_skill_id=true \
  +data.include_paligemma_token=true +data.paligemma_token_dim=2048 \
  module.feature_extractors.paligemma_token.input_dim=2048 \
  module.feature_extractors.skill_id.num_embeddings=8
```

## 实验记录（最小验证：训练 + held-out test）

说明：以下记录均为 `train.py` 自动 `fit()` + `test()`，且 `test` 复用 `val_split_ratio` 得到的 held-out keys。

### 2026-03-11

- Setting 1（state-based, BC, held-out test）
  - Run dir: `outputs/2026-03-11/mt_il_validation_s1_mt_synth_iter_s1_final_20260311-121743`
  - Test: `l1=0.299751`, `l1_deployed_steps_only=0.296973`, `mse=0.135149`
  - Artifacts: `failure_cases/test_top8.jsonl`
- Setting 2（visuomotor-lite, BC, held-out test）
  - Run dir: `outputs/2026-03-11/mt_il_validation_s2_mt_synth_iter_s2_final_20260311-121925`
  - Test: `l1=0.304319`, `l1_deployed_steps_only=0.301732`, `mse=0.139190`
  - Artifacts: `failure_cases/test_top8.jsonl`

## 真实数据：skill label 注入（已实现）

已新增 `SkillLabeledIterableDataset`，按 annotation segment（skill / primitive）切片并为每个 window 注入：

- `obs["skill_id"]`（`label_mode=id`）或 `obs["skill_one_hot"]`（`label_mode=one_hot`）
- 代码：[skill_labeled_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/skill_labeled_dataset.py)

用法示例（Skill id）：

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.skill_labeled_dataset.SkillLabeledIterableDataset \
  data.label_level=skill data.label_mode=id data.num_skills=64
```

用法示例（Skill one-hot）：

```bash
python train.py \
  arch=mt_il_validation_s1 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.skill_labeled_dataset.SkillLabeledIterableDataset \
  data.label_level=skill data.label_mode=one_hot data.num_skills=64 \
  module.feature_extractors.skill_one_hot.input_dim=64
```

## 真实数据：Setting 2（skill_id + paligemma_token）一体化数据集（已实现）

Setting 2 需要同时具备：

- `obs["skill_id"]`（用于 skill embedding）
- `obs["paligemma_token"]`（用于 VLM token 条件输入）

为避免 `dataset_class` 无法组合的问题，提供一体化 dataset：

- [mt_il_conditioned_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/mt_il_conditioned_dataset.py)

示例：

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.mt_il_conditioned_dataset.MTILConditionedIterableDataset \
  data.label_level=skill data.label_mode=id data.num_skills=64 \
  data.cache_dir=<CACHE_DIR> data.paligemma_token_dim=2048 \
  module.feature_extractors.skill_id.num_embeddings=64 \
  module.feature_extractors.paligemma_token.input_dim=2048
```

## 真实数据：paligemma_token cache 注入（已实现）

当使用 Setting 2（visuomotor-lite）时，推荐把 PaLI-Gemma embedding 离线提取并缓存，在训练侧通过 dataset 注入 `obs["paligemma_token"]`，避免训练时在线编码带来的吞吐与不稳定。

- Dataset：[paligemma_token_cache_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/paligemma_token_cache_dataset.py)
- Skill：[paligemma-token-cache](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/.trae/skills/paligemma-token-cache/SKILL.md)

示例：

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.paligemma_token_cache_dataset.PaligemmaTokenCachedIterableDataset \
  data.cache_dir=<CACHE_DIR> data.token_dim=2048
```

离线生成（从 frame tensor 构建 cache）：

- [build_paligemma_token_cache_from_frames.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/build_paligemma_token_cache_from_frames.py)

离线生成（从 BEHAVIOR demos / videos 构建 cache，推荐）：

- [build_paligemma_token_cache_from_behavior_demos.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/build_paligemma_token_cache_from_behavior_demos.py)

## TODO

- 统一离线评测指标口径：已实现 `per-step L1` / `deployed-steps L1` / `NLL/likelihood proxy`；后续把不同 action expert 的 proxy 指标统一成一套字段并固化到实验脚本
- 失败案例采样：已支持导出 top-k worst 的 `(demo_key, index, skill_id, l1)` 到 `failure_cases/{val|test}_topK.jsonl`；后续补齐“自动渲染/视频化”链路

## 已实现：按 skill 分桶的离线指标（可选开启）

当 batch 里包含 `obs["skill_id"]` 时，可开启分桶统计：

- MTILValidationPolicy：`module.log_skill_buckets=true module.max_skill_buckets=32`
- MomaSTAGE：`module.log_skill_buckets=true module.max_skill_buckets=32`

## 已实现：离线 worst-case 统计（可选开启）

对每个 batch 的 per-sample `l1` 统计 p95/max/topk mean（不包含样本导出）：

- MTILValidationPolicy：`module.log_failure_stats=true module.failure_top_k=16 module.failure_quantile=0.95`
- MomaSTAGE：`module.log_failure_stats=true module.failure_top_k=16 module.failure_quantile=0.95`

## 已实现：导出 top-k worst 样本（可选开启）

在 `val/test` epoch 结束时导出 jsonl（仅记录索引与指标，不包含图像/轨迹 tensor）：

- MTILValidationPolicy：`module.export_failure_cases=true module.export_failure_top_k=64`

## 已实现：NLL/likelihood proxy（可选开启，默认随 test 记录）

说明：不同 action expert 的真实对数似然口径不一致，因此先提供“可比较的代理指标（proxy）”，用于 ablation 排序与回归监控：

- `bc`: `mse`（回归误差）
- `diffusion`: `diffusion_noise_mse`（噪声预测 MSE），并记录 `nll_proxy=diffusion_noise_mse`
- `flow_matching`: `flow_matching_mse`（速度场 MSE），并记录 `nll_proxy=flow_matching_mse`

## 已实现：failure 窗口导出（回放/可视化入口）

`failure_cases/*.jsonl` 只包含索引与指标。可用脚本把对应 window 抽取并保存为 `pt`：

- [export_failure_windows.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/export_failure_windows.py)

```bash
python il_lib/scripts/export_failure_windows.py \
  --run_dir <RUN_DIR> \
  --stage test
```
