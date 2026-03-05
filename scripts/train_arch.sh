#!/usr/bin/env bash
set -euo pipefail

# Train a single architecture config under il_lib/configs/arch.
#
# Usage:
#   DATA_DIR=/path/to/data \
#   ./scripts/train_arch.sh wbvima turning_on_radio \
#     gpus=auto trainer.precision=32
#
# Notes:
# - For WB-VIMA privileged task info dim, set WBVIMA_TASK_INFO_DIM=82.
# - To disable WB-VIMA task info, set WBVIMA_USE_TASK_INFO=false.

ARCH="${1:-}"
TASK_NAME="${2:-}"

if [[ -z "${ARCH}" || -z "${TASK_NAME}" ]]; then
  echo "Usage: $0 <arch> <task.name> [hydra_overrides...]" >&2
  echo "Example: $0 wbvima turning_on_radio gpus=auto trainer.precision=32" >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-${DATA_PATH:-}}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "Error: set DATA_DIR (or DATA_PATH) to your dataset root." >&2
  exit 2
fi

ROBOT="${ROBOT:-r1pro}"
TASK_CFG="${TASK_CFG:-behavior}"
GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
PRECISION="${PRECISION:-32}"

EXTRA_ARGS=("${@:3}")

WBVIMA_USE_TASK_INFO="${WBVIMA_USE_TASK_INFO:-}"
WBVIMA_TASK_INFO_DIM="${WBVIMA_TASK_INFO_DIM:-}"

WBVIMA_EXTRA=()
if [[ "${ARCH}" == "wbvima" ]]; then
  # wbvima.yaml defaults to data.use_task_info=true.
  # Some datasets expose a larger privileged task tensor (often 82); allow overriding.
  if [[ -n "${WBVIMA_TASK_INFO_DIM}" ]]; then
    WBVIMA_EXTRA+=("module.feature_extractors.task.input_dim=${WBVIMA_TASK_INFO_DIM}")
    WBVIMA_EXTRA+=("data.use_task_info=true")
  fi
  if [[ "${WBVIMA_USE_TASK_INFO}" == "false" || "${WBVIMA_USE_TASK_INFO}" == "0" ]]; then
    WBVIMA_EXTRA+=("data.use_task_info=false")
    # Quote to avoid any accidental shell tilde expansion.
    WBVIMA_EXTRA+=("~module.feature_extractors.task")
  fi
fi

set -x
python train.py \
  "data_dir=${DATA_DIR}" \
  "robot=${ROBOT}" \
  "task=${TASK_CFG}" \
  "task.name=${TASK_NAME}" \
  "arch=${ARCH}" \
  "gpus=${GPUS}" \
  "num_nodes=${NUM_NODES}" \
  "trainer.precision=${PRECISION}" \
  "${WBVIMA_EXTRA[@]}" \
  "${EXTRA_ARGS[@]}"