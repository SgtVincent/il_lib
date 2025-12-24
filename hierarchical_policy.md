# Hierarchical Policy Training and Evaluation

This tutorial describes how to train and evaluate a hierarchical policy for BEHAVIOR-1K tasks. The hierarchical policy consists of a high-level policy (GT or learned) that dispatches tasks to low-level skill policies.

## 1. Training Skill Policies

To train a hierarchical policy, you first need to train individual policies for each skill. We provide `SkillIterableDataset` to filter the demonstration data by skill annotations.

### 1.1 Configuration

You can use the existing `train.py` script but override the dataset class and specify the skill name.

### 1.2 Training Command

For each skill (e.g., "move to", "pick up", "press", "place"), run the training command:

```bash
# Train 'move to' skill
python train.py \
    data_dir=$DATA_PATH \
    robot=r1pro \
    task=behavior \
    task.name=turning_on_radio \
    arch=wbvima \
    data.dataset_class=il_lib.datas.skill_dataset.SkillIterableDataset \
    data.skill_name="move to" \
    run_name=wbvima_turning_on_radio_move_to

# Train 'pick up' skill
python train.py \
    data_dir=$DATA_PATH \
    robot=r1pro \
    task=behavior \
    task.name=turning_on_radio \
    arch=wbvima \
    data.dataset_class=il_lib.datas.skill_dataset.SkillIterableDataset \
    data.skill_name="pick up" \
    run_name=wbvima_turning_on_radio_pick_up

# ... repeat for other skills
```

Ensure that the `skill_name` matches the descriptions in the annotation JSON files (partial match is supported).

## 2. Hierarchical Policy Evaluation

After training the skill policies, you can combine them into a `HierarchicalPolicy`.

### 2.1 Policy Definition

The `HierarchicalPolicy` is defined in `il_lib/policies/hierarchical_policy.py`. It loads multiple checkpoints and dispatches the observation to the active skill policy.

### 2.2 Configuration

Create a new config file or override parameters to use `HierarchicalPolicy`.

Example config structure (e.g., `il_lib/configs/arch/hierarchical.yaml`):

```yaml
# @package _global_
arch_name: hierarchical

module:
  _target_: il_lib.policies.hierarchical_policy.HierarchicalPolicy
  skills:
    "move to": "/path/to/move_to_checkpoint.ckpt"
    "pick up": "/path/to/pick_up_checkpoint.ckpt"
    "press": "/path/to/press_checkpoint.ckpt"
    "place": "/path/to/place_checkpoint.ckpt"
  
  # ... other args if needed
```

### 2.3 Implementing High-Level Logic

The current `HierarchicalPolicy` requires a `get_active_skill(obs)` method to determine which skill to execute. You need to implement this logic based on the task state (available in `obs['task']` or `obs['task_info']`).

Modify `il_lib/policies/hierarchical_policy.py`:

```python
    def get_active_skill(self, obs):
        # Implement your GT logic here
        # Example for turning_on_radio:
        # Check distance to radio, grasp state, etc.
        pass
```

### 2.4 Running Evaluation

Run the evaluation using `serve.py` or `eval.py` with the hierarchical architecture.

```bash
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

Then run the OmniGibson evaluation wrapper as usual.
