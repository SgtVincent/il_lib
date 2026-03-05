#!/usr/bin/env bash
set -euo pipefail

# Train all *trainable* architectures under il_lib/configs/arch.
# (Hierarchical policies are eval-only and intentionally skipped.)
#
# Usage:
#   DATA_DIR=/path/to/data \
#   ./scripts/train_all_arches.sh turning_on_radio \
#     gpus=auto trainer.precision=32
#
# To customize WB-VIMA privileged task info:
#   WBVIMA_TASK_INFO_DIM=82 ./scripts/train_all_arches.sh turning_on_radio
#
# To disable WB-VIMA task info:
#   WBVIMA_USE_TASK_INFO=false ./scripts/train_all_arches.sh turning_on_radio

TASK_NAME="${1:-}"
if [[ -z "${TASK_NAME}" ]]; then
  echo "Usage: $0 <task.name> [hydra_overrides...]" >&2
  echo "Example: $0 turning_on_radio gpus=auto trainer.precision=32" >&2
  exit 2
fi

EXTRA_ARGS=("${@:2}")

ARCHES=(
  act
  bcrnn_rgbd
  diffusion_rgbd_unet
  diffusion_state_transformer
  dp3
  wbvima
)

for ARCH in "${ARCHES[@]}"; do
  echo "============================================================"
  echo "[train] arch=${ARCH} task.name=${TASK_NAME}"
  echo "============================================================"
  ./scripts/train_arch.sh "${ARCH}" "${TASK_NAME}" "${EXTRA_ARGS[@]}"
done