# 计划：基于 BDDL 谓词的 primitive/skill segment 评测

## 目标

在 BEHAVIOR-1K / OmniGibson 的仿真评测链路中，补齐“segment（primitive / skill）级别”的评测能力，并且复用 BDDL 谓词系统（而不是仅用 `std_joint_qpos_rmse` 这类 state-match）来定义与计算 segment 的成功、进度与诊断指标。

本计划覆盖：

* BDDL 谓词集合（goal predicates）在 OmniGibson 中如何被构建、grounding、evaluate

* 现有 `eval.py`（任务级 q\_score）与 `eval_subtask_reset.py / eval_primitive.py`（primitive state-match）链路如何工作

* 如何在 segment 级别复用 BDDL 谓词：定义 segment subgoal predicates、成功判定、报告与回放

* 需要落地的代码改动点与验证方式

## Scratch（思考草稿）

### 已确认的关键事实（来自代码）

1. BDDL 条件与谓词 evaluate

* BDDL 的 compiled condition 是 `bddl.condition_evaluation.HEAD`（Expression tree），`evaluate_state()` 会逐条 `compiled_condition.evaluate()` 得到 satisfied/unsatisfied（返回 bool + dict）：[condition\_evaluation.py:evaluate\_state](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/bddl3/bddl/condition_evaluation.py#L546-L554)

* goal conditions 的 grounding（枚举可满足的组合）由 `get_ground_state_options()` 构造，返回 `List[List[HEAD]]`：每个 option 是一组更原子化的条件（用来衡量 progress / partial credit）：[condition\_evaluation.py:get\_ground\_state\_options](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/bddl3/bddl/condition_evaluation.py#L556-L585)

1. OmniGibson 如何构建 goal predicates

* `BehaviorTask` 在 `_load()` 时用 `bddl.activity.Conditions` 解析 BDDL problem，然后生成：

  * `activity_goal_conditions = get_goal_conditions(...)`

  * `ground_goal_state_options = get_ground_goal_state_options(...)`
    见：[behavior\_task.py:L323-L330](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/tasks/behavior_task.py#L323-L330)

* 任务 success 终止条件 `PredicateGoal` 每步调用 `evaluate_goal_conditions(self._goal_fcn())`，其中 `_goal_fcn` 返回 compiled goal conditions（非 ground options）；success 口径是 compiled goal 全部满足：[predicate\_goal.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/termination_conditions/predicate_goal.py#L34-L37)

1. eval.py（任务级）如何算 q\_score 与 success rate

* success rate：按 episode 结束时 `info["done"]["success"]` 计数（trial-level）：[eval.py:L214-L218](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py#L214-L218)

* q\_score：`TaskMetric.end_callback()` 若 `env.task.success` 则 1，否则对 `env.task.ground_goal_state_options` 的每个 grounding 计算新增满足谓词比例并取最大值：[task\_metric.py:L30-L46](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/metrics/task_metric.py#L30-L46)

1. eval\_subtask\_reset.py / eval\_primitive.py（primitive级）当前口径

* `eval_primitive.py` 用 `SubTaskEvaluator._try_restore_to_frame(start_frame)` 恢复状态，然后 rollout policy，用 `check_primitive_success()` 判定 primitive success；当前主判定是 state-match（`std_joint_qpos_rmse` / eef / joint rmse），不是 BDDL 谓词：[eval\_primitive.py:L130-L175](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_primitive.py#L130-L175) + [eval\_subtask\_reset.py:L791-L826](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L791-L826)

* `_try_restore_to_frame(frame_idx)` 优先 rawdata full state：`og.sim.load_state(..., serialized=True)`，因此只要 rawdata\_path 完整可用，skill 起点（任意 frame idx）也可做 full restore：[eval\_subtask\_reset.py:L321-L376](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L321-L376) 与 [eval\_subtask\_reset.py:L382-L401](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py#L382-L401)

### 设计核心问题

要做 BDDL 谓词驱动的 segment 评测，关键不是能不能 restore（rawdata 下能），而是：

1. segment 的 subgoal predicates 如何定义？
2. 如何选 grounding（`ground_goal_state_options` 里哪个 option 才对应该 demo）？
3. 如何避免把 task-level q\_score 当作 segment success？

## 方案概述（将要实现的能力）

### A. SegmentPredicateEvaluator（primitive / skill 通用）

新增一个评测脚本 / evaluator（放在 BEHAVIOR-1K 的 `OmniGibson/omnigibson/learning/` 下），支持：

* `segment_level=primitive|skill`

* `demo_id` + `segment_idx`（与 `eval_primitive.py` 类似）

* `rawdata_path / primitive_state_cache_dir / demo_data_path`（复用现有 restore）

* `success_mode`：

  * `predicate_subgoal`：本段 subgoal predicates 全部满足即成功

  * `predicate_progress`：记录进度（满足比例、delta），但不做二值成功

  * `state_match`：兼容现有 primitive success，作为 baseline

### B. segment subgoal predicates 的定义（默认策略）

默认用 demo 前后帧对比自动挖掘 subgoal：

1. 对同一个 demo\_id：

   * restore 到 segment start\_frame：对每个 grounding option 计算 predicate truth 向量 `S_start`

   * restore 到 segment end\_frame（或 end\_frame-1）：计算 `S_end`
2. 对每个 grounding option：

   * subgoal = { i | S\_start\[i]==False and S\_end\[i]==True }（该段在 demo 中新增满足的谓词索引集合）
3. 选择 grounding：

   * 优先选 subgoal 数量较大且 end 满足比例更高的 option（更可能与 demo 对齐）
4. 在 policy rollout 时：

   * success = subgoal 全部满足（允许其他 goal predicates 未满足）

   * 同时输出 q\_score\_delta（全局）与 subgoal\_progress（局部）

注意：对纯 navigation skill（move to）可能挖不到 subgoal（goal predicates 不变化）；对此默认只输出 `predicate_progress` 诊断，后续再加导航子目标（距离阈值）。

### C. 输出与对齐（诊断为主、与 leaderboard 区分）

每个 segment 输出 JSON：

* restore 方法（rawdata/cache/robot）

* segment 元信息（level、idx、desc、frame\_duration）

* 选中的 grounding id 与 subgoal predicate 列表（可含 token/对象名）

* subgoal success / progress

* q\_score\_start/end/delta（使用 TaskMetric 同口径，但只作为辅助）

## 实施步骤（执行阶段要做什么）

### Step 0：补齐 predicate / grounding 的可观测性（可读化）

* 调研 `env.task.ground_goal_state_options` 的元素类型（HEAD/Expression），如何提取更可读的 predicate 表示（token + args）

* 产出工具函数：`summarize_ground_option(option) -> list[dict]`，包含 token/args/当前 truth

### Step 1：实现 SegmentPredicateEvaluator（新脚本/新类）

* 在 `OmniGibson/omnigibson/learning/` 新增 `eval_segment.py`

* 支持 segment\_level：

  * primitive：读取 `primitive_annotation`

  * skill：读取 `skill_annotation`

* 复用 `SubTaskEvaluator`：

  * restore：`_try_restore_to_frame(frame_idx)`

  * rollout：`evaluator.step()` 循环

* subgoal 构建：

  * start/end 帧分别 restore 并 eval predicate truth（按 grounding）

  * 选 grounding + 计算 subgoal 索引集合

* predicate\_subgoal success：

  * 每步 eval 当前 grounding truth

  * subgoal 全满足 → success；超时 → fail

### Step 2：复用 TaskMetric 作为辅助输出（q\_score delta）

* 复用 `TaskMetric` 的定义：

  * segment start 记录 initial predicate states（按 grounding）

  * segment end 计算 `q_score.final`（max across groundings），并输出 delta

* 同时输出“按选中 grounding 的 progress”（便于与 subgoal 一致）

### Step 3：文档与入口（il\_lib / openpi-comet）

* `il_lib/.trae/documents/` 新增说明文档（或扩展现有）：

  * 何时用 state-match primitive eval

  * 何时用 predicate segment eval

  * rawdata\_path 的必要性与常见坑

* `il_lib/EVALUATION.md` 补命令示例（调用 BEHAVIOR-1K 的新脚本）

* `openpi-comet` 仅补“如何接入”说明，不改其 evaluator

### Step 4：验证策略（最小闭环）

* 选一个任务 + 一个 demo\_id：

  * 在 demo start/end 帧提取 subgoal predicates（校验 subgoal 非空且 end 满足）

  * 用 policy=websocket 跑 segment eval，看是否能满足 subgoal

* 对比：state-match success vs predicate\_subgoal success 的差异（用于阈值与策略诊断）

## 进度监控（执行阶段更新）

* [ ] Step 0：predicate/grounding 可读化与抽取接口

* [ ] Step 1：实现 eval\_segment.py（primitive/skill 通用 + predicate\_subgoal）

* [ ] Step 2：加入 q\_score delta 与 report 字段

* [ ] Step 3：补齐 il\_lib 文档与命令入口

* [ ] Step 4：跑通一个 demo 的单段评测并产出样例 JSON/视频

