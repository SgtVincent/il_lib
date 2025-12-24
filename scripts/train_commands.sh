export DATA_PATH=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data
python train.py \
  data_dir=$DATA_PATH \
  robot=r1pro \
  task=behavior task.name=picking_up_trash \
  arch=wbvima \
  gpus=auto \
  trainer.precision=32

# To resume from a previous run, add appropriate `resume.ckpt_path=...` and `resume.full_state=true` arguments to the command above.
# with task info 
conda activate behavior
export DATA_PATH=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data
python train.py \
  data_dir=$DATA_PATH \
  robot=r1pro \
  task=behavior task.name=picking_up_trash \
  arch=wbvima \
  trainer.devices=4 \
  trainer.strategy=ddp \
  trainer.precision=32 \
  data.batch_size=64 \
  data.dataloader_num_workers=16 \
  data.use_task_info=true \
  module.feature_extractors.task.input_dim=82 \
  resume.ckpt_path=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/repo/b1k-baselines/baselines/il_lib/outputs/2025-11-29/20-33-07/wbvima_picking_up_trash_20251129-203307/ckpt/last.pth \
  resume.full_state=true \
  2>&1 | tee train_behavior_with_task_info.log

# without task info
conda activate behavior
export DATA_PATH=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data
python train.py \
  data_dir=$DATA_PATH \
  robot=r1pro \
  task=behavior task.name=picking_up_trash \
  arch=wbvima \
  trainer.devices=4 \
  trainer.strategy=ddp \
  trainer.precision=32 \
  data.batch_size=64 \
  data.dataloader_num_workers=16 \
  data.use_task_info=false \
  ~module.feature_extractors.task