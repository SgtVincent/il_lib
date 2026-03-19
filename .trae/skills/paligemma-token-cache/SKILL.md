---
name: "paligemma-token-cache"
description: "生成 paligemma_token 的离线 cache 规范与接入方式（dataset 注入+配置示例）。当你要落地 VLM embedding cache 或排查对齐问题时调用。"
---

# PaLI-Gemma Token Cache

## 目标

把 “PaLI-Gemma token / embedding 作为条件输入” 变成可复用、可排查的工程流程：

- 离线提取并落盘（避免训练时在线编码）
- 训练数据读取并对齐到 `(obs_window_size, token_dim)` 的时间窗
- 明确 cache 的 key / 时间步对齐语义（30Hz vs downsampled）

## Cache 文件规范（建议）

- 目录：`<CACHE_DIR>/task-0001/episode_10001.pt` 或 `<CACHE_DIR>/episode_00010001.pt`
- 内容：torch `pt` 文件
  - 允许两种形式：
    - `Tensor[N, D]` 直接存 token 序列
    - `{"paligemma_token": Tensor[N, D]}` 字典形式
- 对齐语义：默认按 dataset 的 downsample 后 index 对齐（10Hz）；如按 30Hz 存储，需要在读取侧设置 `cache_is_downsampled=false`

## 训练侧接入（dataset 注入）

使用 `PaligemmaTokenCachedIterableDataset` 在 streaming sample 上注入 `obs["paligemma_token"]`：

- 代码：[paligemma_token_cache_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/paligemma_token_cache_dataset.py)

示例：

```bash
python train.py \
  arch=mt_il_validation_s2 task=behavior robot=r1pro \
  data.dataset_class=il_lib.datas.paligemma_token_cache_dataset.PaligemmaTokenCachedIterableDataset \
  data.cache_dir=<CACHE_DIR> data.token_dim=2048
```

## 常见对齐坑排查

- `index` 缺失：dataset 没有提供 sample 对应的时间步索引，需要先在上游输出 `index`（本仓库的 `SkillLabeledIterableDataset / MTILConditionedIterableDataset / PaligemmaTokenCachedIterableDataset` 已会注入 `demo_key/index`）
- `obs_window_size` 不一致：cache 注入默认用 `obs_window_size` 生成窗口索引
- 30Hz/10Hz 混用：cache 若按 30Hz 存，读取端必须 `cache_is_downsampled=false`

## 离线生成（从 frame tensor 构建 cache）

提供了一个最小离线脚本：输入每个 episode 的 frame tensor（`uint8`，`(N,3,H,W)` 或 `(N,H,W,3)`），输出符合本仓库读取约束的 `{"paligemma_token": (N,2048)}`：

- [build_paligemma_token_cache_from_frames.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/build_paligemma_token_cache_from_frames.py)

```bash
python il_lib/scripts/build_paligemma_token_cache_from_frames.py \
  --frames_pt <FRAMES_PT> \
  --output_pt <CACHE_PT> \
  --pi0_ckpt_path <PI0_CKPT> \
  --openpi_src /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src
```

## 离线生成（从 BEHAVIOR demos / videos 构建 cache，推荐）

按 BEHAVIOR 数据目录结构读取视频 + parquet 的长度对齐信息，生成每个 episode 的 `paligemma_token` 序列：

- [build_paligemma_token_cache_from_behavior_demos.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/scripts/build_paligemma_token_cache_from_behavior_demos.py)

```bash
python il_lib/scripts/build_paligemma_token_cache_from_behavior_demos.py \
  --data_path <B1K_DATA_PATH> \
  --task_names turning_on_radio,picking_up_trash \
  --cache_dir <CACHE_DIR> \
  --pi0_ckpt_path <PI0_CKPT> \
  --openpi_src /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src \
  --camera head \
  --downsample_factor 3
```
