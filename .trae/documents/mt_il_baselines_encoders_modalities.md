# Multi-task IL Baselines：Encoder / 架构 / 输入模态速查

本文汇总当前仓库内 multi-task IL 常用 baseline（dp3 / diffusion / ACT / MoE Flow Matching / WBVIMA）的网络结构要点：各自的 encoder（feature extractor / backbone）、整体架构、输入模态与关键配置入口。

参考入口文档：`multitask_il_baselines.md`（同目录下）。

## DP3（arch=dp3）

* 配置入口：[../../il\_lib/configs/arch/dp3.yaml](../../il_lib/configs/arch/dp3.yaml)

* Policy 实现：[../../il\_lib/policies/diffusion\_policy.py](../../il_lib/policies/diffusion_policy.py)

* 输入模态

  * proprio（qpos/eef/odom 等，`prop_dim=37`）

  * point cloud：`pcd`（点云）

  * task：默认不使用（`data.use_task_info=false`）

* Encoder / Feature Extractors

  * proprio：`il_lib.nn.common.MLP`（37 → 512）

  * pcd：`il_lib.nn.features.UncoloredPointNet`（xyz → 512）

  * 融合：`SimpleFeatureFusion`（在 policy 内部构建）

* Action head / 轨迹生成

  * `DiffusionPolicy` + DDIM

  * Backbone：`il_lib.nn.diffusion.unet.ConditionalUnet1D`（1D UNet 条件扩散去噪）

* 一句话

  * “PointNet 编码点云 + MLP 编码本体状态 → 融合 → 1D UNet diffusion 预测动作序列”

## Diffusion Policy（DiffusionPolicy 系列）

Policy 实现：[../../il\_lib/policies/diffusion\_policy.py](../../il_lib/policies/diffusion_policy.py)

### diffusion\_rgbd\_unet（arch=diffusion\_rgbd\_unet）

* 配置入口：[../../il\_lib/configs/arch/diffusion\_rgbd\_unet.yaml](../../il_lib/configs/arch/diffusion_rgbd_unet.yaml)

* 输入模态

  * proprio（`prop_dim=37`）

  * 多视角 RGB + Depth（`data.visual_obs_types: [rgb, depth_linear]`）

  * task：默认不使用（`data.use_task_info=false`）

* Encoder / Feature Extractors

  * proprio：`MLP`（37 → 512）

  * 视觉：`il_lib.nn.features.MultiviewResNet18`（支持 `load_pretrained=true`、多视角共享 backbone、随机裁剪、`include_depth=true`）

* Action head / 轨迹生成

  * `DiffusionPolicy` + DDIM

  * Backbone：`il_lib.nn.diffusion.ConditionalUnet1D`（1D UNet 条件扩散去噪）

* 一句话

  * “多视角 ResNet18（RGBD）+ proprio MLP → 融合 → 1D UNet diffusion”

### diffusion\_state\_transformer（arch\_name=diffusion\_rgb\_transformer）

* 配置入口：[../../il\_lib/configs/arch/diffusion\_state\_transformer.yaml](../../il_lib/configs/arch/diffusion_state_transformer.yaml)

* 输入模态（以 config + policy 的实际过滤逻辑为准）

  * proprio（`prop_dim=37`）

  * task 数值向量（`feature_extractors.task`，示例里 `input_dim=46`）

  * 该 config 中 `data.visual_obs_types: [rgb]`，但 `feature_extractors` 未配置 `rgb` extractor；在 `DiffusionPolicy.forward()` 里会按 `feature_extractors.keys()` 过滤 obs，因此实际更接近 “state(+task) diffusion”，而不是视觉版。

* Encoder / Feature Extractors

  * proprio：`MLP`（37 → 512）

  * task：`MLP`（46 → 512）

* Action head / 轨迹生成

  * `DiffusionPolicy` + DDIM

  * Backbone：`il_lib.nn.diffusion.TransformerForDiffusion`（Transformer 条件扩散去噪）

* 一句话

  * “proprio+task MLP → 融合 → Transformer diffusion”

## ACT（arch=act）

* 配置入口：[../../il\_lib/configs/arch/act.yaml](../../il_lib/configs/arch/act.yaml)

* Policy 实现：[../../il\_lib/policies/act\_policy.py](../../il_lib/policies/act_policy.py)

* 输入模态

  * proprio（`prop_dim=37`）

  * 多视角 RGB + Depth（`data.visual_obs_types: [rgb, depth_linear]`，`obs_backbone.include_depth=true`）

  * task：默认不启用，但 policy 支持将 `obs["task"]` 拼到 proprio（需要 `features` 包含 `task` 且 `task_dim` 正确设置）

* Encoder

  * 视觉：`MultiviewResNet18`，且 `return_last_spatial_map=true`（输出 feature map 给 Transformer）

  * proprio：线性投影到 `hidden_dim`（`input_proj_robot_state`）

* Action head / 轨迹生成

  * DETR 风格 Transformer encoder-decoder + learned queries 输出 action chunk（`horizon=20`）

  * 训练时包含 VAE 式 latent（KL 正则）；推理时 latent 默认为 0

  * 可选 temporal ensemble 平滑执行（`temporal_ensemble=true`）

* 一句话

  * “多视角 ResNet18 feature map + proprio → Transformer（ACT）直接回归 action chunk（带 VAE 训练技巧）”

## MoE Flow Matching（arch=moe\_flow\_matching）

* 配置入口：[../../il\_lib/configs/arch/moe\_flow\_matching.yaml](../../il_lib/configs/arch/moe_flow_matching.yaml)

* Policy 实现：[../../il\_lib/policies/moe\_flow\_matching\_policy.py](../../il_lib/policies/moe_flow_matching_policy.py)

* 输入模态

  * proprio（`prop_dim=37`）

  * point cloud：`pcd`

  * task：默认不使用（`data.use_task_info=false`）

* Encoder / Feature Extractors

  * proprio：`MLP`（37 → 512）

  * pcd：`UncoloredPointNet`（xyz → 512）

  * 融合：`SimpleFeatureFusion`

* Action head / 轨迹生成

  * Flow Matching：学习速度场 `v(x,t,cond)`，推理时用 Euler 积分多步生成 action trajectory

  * head：`ActionBlockMoEVelocityField`（按 `action_keys/action_key_dims` 分块 + gating/top-k experts）

* 一句话

  * “(proprio+pcd) 编码 → MoE 速度场（Flow Matching）→ 积分生成动作序列”

## WBVIMA（arch=wbvima）

* 配置入口：[../../il\_lib/configs/arch/wbvima.yaml](../../il_lib/configs/arch/wbvima.yaml)

* Policy 实现：[../../il\_lib/policies/wbvima\_policy.py](../../il_lib/policies/wbvima_policy.py)

* 输入模态

  * proprio（`prop_dim=37`）

  * point cloud：`pcd`

  * task 数值向量：默认启用（`data.use_task_info=true`，`feature_extractors.task`）

  * 默认使用 action chunks（`data.use_action_chunks=true`）

* Encoder / Tokenizer

  * `ObsTokenizer` 把各模态编码成 tokens：

    * proprio：`MLP` → token

    * pcd：`il_lib.nn.features.PointNet`（带颜色，`n_coordinates=3, n_color=3`）→ token

    * task：`MLP` → token

* Action head / 轨迹生成

  * token-level GPT Transformer 建模多模态 token 序列

  * `WholeBodyUNetDiffusionHead` 作为动作解码器，按 base/torso/arms 分块扩散解码动作轨迹

* 一句话

  * “ObsTokenizer（proprio/pcd/task → tokens）→ GPT → 分块 diffusion head 解码 whole-body action chunk”

## 对比时最关键的 encoder 差异（快速定位混杂因素）

* 点云系（pcd）：DP3 / MoE-FM / WBVIMA

  * DP3 & MoE-FM：`UncoloredPointNet`（偏 xyz）

  * WBVIMA：`PointNet`（xyz+rgb）+ tokenization + GPT

* 视觉系（RGBD）：ACT / diffusion\_rgbd\_unet

  * 都是 `MultiviewResNet18`，但 ACT 用 feature map + Transformer 输出 chunk；diffusion\_rgbd\_unet 是编码成向量条件后走 diffusion（1D UNet）

* state(+task) 系：diffusion\_state\_transformer（按当前 config，实际不包含 rgb extractor）

