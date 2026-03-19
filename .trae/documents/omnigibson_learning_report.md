# OmniGibson `omnigibson/learning` 代码分析报告（BEHAVIOR-1K 2025 Challenge）

本文聚焦 [OmniGibson/omnigibson/learning](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning) 的评测/数据/策略对接链路，梳理：

* 模块结构与关键入口

* 指标（metric）如何计算与汇总

* primitive-level（subtask-level）评测如何工作

* 是否存在“skill success”以及如何扩展

## 1. 模块结构（高层）

`omnigibson/learning` 可以粗分为 5 块：

1. **入口脚本**

* 端到端评测入口：[eval.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py)

* subtask/primitive reset 评测入口：[eval\_subtask\_reset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py)

* 单 primitive 调试入口：[eval\_primitive.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_primitive.py)

* 批量 primitive 调试入口：[eval\_primitive\_batch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_primitive_batch.py)

1. **Hydra 配置**

* 基础配置：[base\_config.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/base_config.yaml)

* policy 组：[configs/policy](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/policy)

* primitive/subtask 评测配置：

  * [eval\_primitive\_config.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/eval_primitive_config.yaml)

  * [eval\_subtask\_reset\_config.yaml](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/eval_subtask_reset_config.yaml)

1. **Wrappers（观测注入 / 分辨率 / modality）**

* wrappers 目录：[wrappers](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers)

* 关键点：wrapper 决定 policy “看到什么 obs”（例如是否注入 `task` 特权观测）。

* [WBVIMAWrapper](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/wbvima_wrapper.py) 在 `reset/step` 注入 `obs[\"task\"]=env.task.get_obs()`，primitive 评测里专门做了“绕开 wrapper 的 get\_obs 修复”，见 [eval\_subtask\_reset.py:L261-L284](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L261-L284)。

1. **Datasets（训练/加载工具）**

* datas 目录：[datas](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/datas)

* 注意：primitive/subtask 评测读取 demos 标注与 parquet 数据，不走 `datas/`，而是在 `eval_subtask_reset.py` 中直接读文件。

1. **Utils（常量与索引）**

* `PROPRIOCEPTION_INDICES / PROPRIO_QPOS_INDICES / TASK_NAMES_TO_INDICES` 等都在 [eval\_utils.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/eval_utils.py)。

## 2. 端到端评测（eval.py）控制流

`Evaluator` 的核心路径：

* 构建环境：`load_env()` 创建 OG `Environment`，再用 Hydra `env_wrapper._target_` 包装，见 [eval.py:L95-L158](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L95-L158)

* 加载 policy：`instantiate(self.cfg.model)`，见 [eval.py:L169-L180](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L169-L180)

* 加载 metrics：默认 `AgentMetric + TaskMetric`，见 [eval.py:L181-L186](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L181-L186)

* 每步 rollout：`policy.forward(obs) → env.step(action)`，并对 `metric.step_callback(env)` 逐步累计，见 [eval.py:L187-L221](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L187-L221)

* 任务成功计数：`info[\"done\"][\"success\"]` 只用于统计 trial 数 / success trial 数，不直接参与 leaderboard q\_score 的定义，见 [eval.py:L214-L218](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L214-L218)

* 结束写盘：`metric.end_callback(env)` + `metric.gather_results()` 输出 json，见 [eval.py:L452-L494](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L452-L494)

## 3. Metric 计算与最终得分汇总

### 3.1 `q_score.final` 的定义（TaskMetric）

[TaskMetric](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/metrics/task_metric.py) 的口径是“任务 goal predicates 的完成度”：

* 若 `env.task.success`：`final_q_score = 1.0`

* 否则：对 `env.task.ground_goal_state_options` 中每个 grounding，计算“相对初始状态新增满足谓词的比例”，并取最大值作为 partial credit

实现见 [task\_metric.py:L30-L46](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/metrics/task_metric.py#L30-L46)。

这意味着：

* leaderboard 的“成功”不是靠对齐 demo 的动作段落，而是靠环境内的 goal predicates（可部分得分）

* `info["done"]["success"]` 与 `env.task.success` 在“任务完全成功”时应该一致，但 `q_score` 支持 partial credit

### 3.2 `time.normalized_time` 与 time score

`TaskMetric.gather_results()`：

* `normalized_time = human_avg_steps / agent_steps`（越大表示越接近/优于人类步数）

实现见 [task\_metric.py:L47-L55](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/metrics/task_metric.py#L47-L55)。

提交结果汇总时把它映射成分数：

* `time_score = 2 - 1 / normalized_time`

见 [score\_utils.py:L98-L101](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/score_utils.py#L98-L101)。

### 3.3 汇总脚本（per-rollout → per-task → overall）

最终 “提交打分” 由 [score\_utils.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/score_utils.py) 完成：

* 每个任务 10 个 instance，每个 instance 1 rollout

* per-task 取 10 个 rollouts 的均值，再跨 50 个任务平均

* 额外输出 `task_sr`：`q_score.final == 1` 的比例（严格成功率），见 [score\_utils.py:L114-L127](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/score_utils.py#L114-L127)

## 4. Primitive/Subtask 评测（为什么可以“单独 eval primitive”）

### 4.1 primitive 评测的输入数据

`SubTaskEvaluator` 需要 `demo_data_path` 指向 `2025-challenge-demos` 目录，并读取两类数据：

* `annotations/task-XXXX/episode_YYYYYYYY.json`：`primitive_annotation`（以及 `skill_annotation`），见 [eval\_subtask\_reset.py:L589-L626](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L589-L626)

* `data/task-XXXX/episode_YYYYYYYY.parquet`：低维 `observation.state`（用于 state-match 与 fallback 恢复），见 [eval\_subtask\_reset.py:L628-L660](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L628-L660)

### 4.2 primitive 边界状态恢复（restore-to-frame）

恢复策略按优先级：

1. raw rawdata HDF5 的 **完整世界状态**（serialized）：`og.sim.load_state(..., serialized=True)`
2. `primitive_state_cache_dir` 的 `.npz` **完整世界状态缓存**
3. parquet 的 `observation.state` **仅机器人 proprio**（robot-only）

这种设计解释了为何能把一个 long-horizon 任务拆成“按 primitive 起点恢复 → 单 primitive rollout → 判定成功”。

### 4.3 primitive 成功判定（当前实现是 state-match）

`check_primitive_success()` 的口径：

* 若环境 `terminated`：认为整任务完成（primitive 也算成功），返回 `success_env`

* 否则默认开启 `primitive_success_use_state_match`：把当前 proprio 与 demo 在该 primitive **结束帧** 的 `observation.state` 做误差，对阈值判断成功

实现见 [eval\_subtask\_reset.py:L759-L831](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L759-L831)。

误差的 slice 由 [eval\_utils.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/eval_utils.py) 的 `PROPRIOCEPTION_INDICES/PROPRIO_QPOS_INDICES` 定义。

### 4.4 单 primitive 调试入口

[eval\_primitive.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_primitive.py) 会：

* 选择一个 `demo_id + primitive_idx`

* 强制 restore 到 primitive 起点

* rollout 到 success/timeout

* 输出 `primitive_success_debug`（best/final state errors + success reason）

见 [eval\_primitive.py:L66-L227](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_primitive.py#L66-L227)。

## 5. “skill success” 是否存在？能否单独评估 skill？

### 5.1 结论（现状）

learning 的评测链路里：

* **端到端**：只认 `TaskMetric(q_score/time)` 与 agent distance 等（任务级）

* **分段评测**：只实现了 **primitive-level success**（ST/ET），且 success 判定是 task-agnostic 的 state-match

虽然 demos 的 annotation 里确实有 `skill_annotation`，并且被读取/打印（见 [eval\_subtask\_reset.py:L622-L625](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L622-L625)），但目前没有“skill success rate” 的计算、写盘与汇总。

### 5.2 技术上是否可行：可以，但要先明确 success 定义

在现有框架下做 skill-level eval 的最低成本路径是：

* 把 `skill_annotation` 当成 “更细的 segments”

* 完全复用 primitive 的 `restore-to-frame` 与 `state-match end frame` 成功判定

* 产出 `n_skill_trials / n_skill_successes` 或 per-skill-type/per-skill-desc 成功率

但要注意：当前 state-match 主要用 `std_joint_qpos_rmse / eef+gripper / joint_rmse`，这对“导航类 skill”（例如 `move to`）可能相关性很弱，因为 base 位置/朝向在标准赛道通常被视为不可用信息，且现有成功判定也没用 `base_pos/yaw` 阈值（即使阈值字段存在）。

### 5.3 为什么这会影响你的 MT-IL “输入到底是 skill 还是 primitive”

如果你的训练/推理想依赖“外部给定的离散 label（skill/primitive id）”：

* **端到端评测（challenge/leaderboard）不会提供该 label**；policy 在 eval.py 流程中拿到的是 obs（proprio/rgb + wrapper 注入的 task obs），没有 ground-truth skill/primitive 提示。

* **primitive/skill 分段评测**是你本地开发工具：它通过访问 demos 标注实现“拆段与恢复”，但这不是官方 q\_score 的组成部分。

因此，若要把 label 当作“可用输入”，你需要额外的高层模块在部署时自行产生该 label（例如：skill/primitive 识别器、基于语言/状态的 planner、或者把 label 变成 latent 由模型自发学习）。

## 6. 对 `il_lib` 的直接建议：优先用 primitive 作为离散条件输入

结合当前评测实现与数据标注结构，更稳妥的决策是：

1. **把 primitive 当作“动作专家/条件输入”的离散单位**\
   理由：

* OmniGibson learning 里已有 primitive-level eval 工具链（restore + success 判定 + ST/ET 统计）

* primitive 集合更小、更语义稳定；并且与 subtask reset 评测对齐

1. **把 skill 作为额外分析信号（或 planner/latent），而不是“评测可直接验证”的单位**\
   理由：

* 当前链路里没有 skill success；若用 state-match 做 skill success，对导航类 skill 相关性弱

* 端到端评测不会提供 skill/primitive id，单独依赖外部 label 会把问题推成“需要一个高层 label 生成器”

1. **实现落地（你这边已经具备）**

* `SkillLabeledIterableDataset` 已支持 `label_level=primitive`，见 [SkillLabeledIterableDataset](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/skill_labeled_dataset.py#L12-L41) 与 [skill\_labeled\_dataset.py:L88-L141](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/il_lib/il_lib/datas/skill_labeled_dataset.py#L88-L141)

* 对 MT-IL baseline：建议在配置里把 `data.label_level=primitive`，并把 `obs["skill_id"]` 语义上视为 “primitive id”（必要时后续可重命名成 `obs["primitive_id"]` 以减少歧义）

