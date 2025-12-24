#!/bin/bash


# source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
# source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
source /mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh  # To ensure
conda activate behavior

export LD_LIBRARY_PATH=/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/miniconda3/envs/behavior/lib:$LD_LIBRARY_PATH

# Define paths
# Adjust DATA_PATH to point to your actual dataset location
DATA_PATH="/mnt/bn/robot-mllm-data-hl/mlx/users/chenjunting/data"
# DATA_PATH="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data"
TASK="turning_on_radio"
ROBOT="r1pro"
ARCH="wbvima"

# Function to train a skill
train_skill() {
    local skill_name=$1
    
    echo "=================================================="
        echo "Training skill: $skill_name"
        echo "=================================================="
        
        python train.py \
            data_dir=$DATA_PATH \
            robot=$ROBOT \
            task=behavior \
            task.name=$TASK \
            arch=$ARCH \
            gpus=1 \
            data.dataset_class=il_lib.datas.skill_dataset.SkillIterableDataset \
            data.skill_name="$skill_name" \
            run_name="${ARCH}_${TASK}_${skill_name// /_}" \
            trainer.max_epochs=500 \
            trainer.check_val_every_n_epoch=10
}

# Train skills for turning_on_radio
# Skills: "move to", "pick up from", "press", "place on"

# train_skill "move to" &
train_skill "pick up from" &
# train_skill "press" &
# train_skill "place on" 
echo "All skills trained."
