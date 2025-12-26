# Evaluation Instructions

This document provides commands to run evaluations for the three requested scenarios.

## Prerequisites
- Ensure you are in the `baselines/il_lib` directory.
- Ensure `conda activate behavior` is active for the evaluation client.
- Set `DATA_DIR` to your data path (used in Case 3): `/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data`

## 1. Default WB-VIMA Policy

**Step 1: Start the Policy Server**
Run this in a terminal:
```bash
cd baselines/il_lib
python serve.py \
  robot=r1pro \
  task=behavior \
  task.name=picking_up_trash \
  arch=wbvima \
  data.use_task_info=true \
  module.feature_extractors.task.input_dim=82 \
  ckpt_path=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/repo/b1k-baselines/baselines/il_lib/outputs/2025-11-30/14-31-17/wbvima_picking_up_trash_20251130-143117/ckpt/last.pth
```

**Step 2: Run the Evaluator**
Run this in a **separate** terminal:
```bash
conda activate behavior
cd baselines/il_lib
python ../../OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  log_path=./eval_logs/default_wbvima
```

---

## 2. WB-VIMA Policy without Privileged Task Info

**Step 1: Start the Policy Server**
Run this in a terminal:
```bash
cd baselines/il_lib
python serve.py \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=wbvima \
  ckpt_path=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/repo/b1k-baselines/baselines/il_lib/outputs/2025-11-30/17-37-18/wbvima_picking_up_trash_task_info_False_20251130-173718/ckpt/last.pth \
  data.use_task_info=false \
  '~module.feature_extractors.task'
```

**Step 2: Run the Evaluator**
Run this in a **separate** terminal:
```bash
conda activate behavior
cd baselines/il_lib
python ../../OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  log_path=./eval_logs/no_task_info_wbvima
```

---

## 3. Hierarchical Policy with WB-VIMA Skills (Offline GT)

This runs **offline evaluation** using the Ground Truth skill sequence.

Run this in a terminal:
```bash
cd baselines/il_lib
python scripts/eval_hierarchical_gt.py \
  data_dir=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=hierarchical \
  +eval=hierarchical_gt
```

---

## 4. Hierarchical Policy with VLM Skill Selection (Online)

This runs **online evaluation** where a VLM selects the active skill based on visual observations and task info.

**Step 1: Start the Policy Server**
Run this in a terminal (ensure `ARK_API_KEY` is set or passed in config):
```bash
export ARK_API_KEY="your_api_key_here"
cd baselines/il_lib
python serve.py \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=hierarchical_vlm \
  +eval=hierarchical_vlm \
  module.query_frequency=100 \
  module.verbose=true \
  module.log_dir=eval_logs/vlm_queries
```

**Step 2: Run the Evaluator**
Run this in a **separate** terminal:
```bash
conda activate behavior
cd baselines/il_lib
python ../../OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  log_path=./eval_logs/hierarchical_vlm
```

---

## 5. SubTask (Primitive) Evaluation - BRS Protocol

This evaluation follows the BRS paper protocol for subtask-level (primitive-level) evaluation:
- Evaluate each primitive in sequence
- If a primitive fails (timeout), reset the robot to the start of the next primitive
- Report both SubTask (ST) success rates and End-to-End (ET) success rates

This is useful for more granular evaluation of policy performance on long-horizon tasks.

**Step 1: Start the Policy Server**
Run this in a terminal:
```bash
cd baselines/il_lib
python serve.py \
  robot=r1pro \
  task=behavior \
  task.name=turning_on_radio \
  arch=wbvima \
  ckpt_path=/path/to/checkpoint.pth
```

**Step 2: Run SubTask Evaluation**
Run this in a **separate** terminal:
```bash
conda activate behavior
# in ./BEHAVIOR-1K folder
python ./OmniGibson/omnigibson/learning/subtask_eval.py \
  policy=websocket \
  task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  demo_data_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
  log_path=./eval_logs/subtask_eval \
  reset_on_primitive_failure=true \
  primitive_timeout_multiplier=2.0 \
  num_demos=10
```

**Options:**
- `demo_data_path`: Path to the `2025-challenge-demos` folder containing `annotations` and `data` (required)
- `rawdata_path`: Optional path to raw HDF5 data for more accurate state restoration
- `reset_on_primitive_failure`: If true, reset to next primitive start when current fails (default: true)
- `primitive_timeout_multiplier`: Timeout = demo_primitive_duration * multiplier (default: 2.0)
- `num_demos`: Number of demos to evaluate (null = all available)

**Output:**
The evaluation produces:
- Per-primitive results JSON files: `subtask_eval_{task_name}_{demo_id}.json`
- Aggregate metrics: `subtask_eval_{task_name}_aggregate.json` containing:
  - `primitive_success_rate`: ST success rate
  - `endtoend_success_rate`: ET success rate
  - `per_demo_results`: Detailed results per demo
