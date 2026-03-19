#!/usr/bin/env bash
set -euo pipefail

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:?set BEHAVIOR_ROOT}"
DATA_DIR="${DATA_DIR:?set DATA_DIR}"

cd "${BEHAVIOR_ROOT}"

python ./OmniGibson/omnigibson/learning/subtask_eval.py \
  policy=websocket \
  task.name=make_pizza \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  demo_data_path="${DATA_DIR}/2025-challenge-demos" \
  log_path="./eval_logs/subtask_eval_pi0" \
  reset_on_primitive_failure=true \
  primitive_timeout_multiplier=2.0 \
  num_demos=10
