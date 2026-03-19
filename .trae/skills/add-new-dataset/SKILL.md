---
name: "add-new-dataset"
description: "新增/改造数据集或采样逻辑（Dataset/DataModule/Hydra 接入）。当你要接入新数据源或做 skill-filter/filtering 时调用。"
---

# Add New Dataset

## 适用场景

- 需要支持一种新的数据格式或目录结构
- 需要新增过滤/重采样/skill 切分逻辑
- 需要让训练侧通过 `data.dataset_class=...` 快速切换数据集实现

## 参考实现

- Skill 数据集： [skill_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/skill_dataset.py)
- 数据模块：`il_lib/datas/*`

## 步骤

1. 选择接口形态
   - 迭代式数据：优先用 IterableDataset
   - 可随机访问：用标准 Dataset
2. 将实现放到 `il_lib/datas/`
3. 在配置中暴露开关
   - 通过 `data.dataset_class` / `data.datamodule_class` 或现有约定字段接入
4. 最小验证
   - `python train.py ... trainer.fast_dev_run=true`
   - 只跑 1-2 个 batch 验证字段对齐、shape 对齐

## 常见坑

- dataset 内部不要硬编码绝对路径，路径由 `cfg.data.data_dir` 或 overrides 提供
- 多进程/多 worker 下确保可序列化与可重复
