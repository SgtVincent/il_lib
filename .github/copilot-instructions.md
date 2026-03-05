# Copilot instructions — il_lib 🔧

Purpose: give an AI-coding agent the minimal, high-value knowledge to be productive in this repo.

## Quickstart (most common tasks) ✅
- Train (example):

  python train.py data_dir=$DATA_PATH robot=r1pro task=behavior task.name=picking_up_trash arch=wbvima gpus=auto trainer.precision=32

  For a fast sanity check use:

  python train.py data_dir=$DATA_PATH robot=r1pro task=behavior task.name=turning_on_radio arch=wbvima trainer.fast_dev_run=true +eval=behavior headless=false

- Serve a trained model (starts a websocket server on 0.0.0.0:8000):

  python serve.py robot=r1pro task=behavior task.name=turning_on_radio arch=wbvima ckpt_path=/path/to/ckpt.pth

- Evaluate (OmniGibson eval connects to the websocket server):

  python ../../OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper log_path=./eval_logs/...

- Test VLM queries (debug OpenAI/Ark API):

  python il_lib/scripts/test_vlm_query.py --api_key $ARK_API_KEY --model ep-20250826131655-jxxss --image /path/to/frame.jpg --verbose

## Big-picture architecture (read these files first) 📂
- Top-level entry points: `train.py` (Hydra + Trainer wrapper), `serve.py` (instantiate policy, open websocket).  See `il_lib/training/trainer.py` for run setup and logging.
- Config-driven design: Hydra config tree under `il_lib/configs/`.
  - `base_config.yaml` (global defaults)
  - `configs/arch/*` (e.g., `wbvima.yaml`, `hierarchical.yaml`, `hierarchical_vlm.yaml`) specifies `_target_` for models
- Major components:
  - Policies: `il_lib/policies/` (WBVIMA, Hierarchical, HierarchicalVLM)
  - Data: `il_lib/datas/` (BehaviorDataModule + optional `SkillIterableDataset`) — dataset selection done via `data.dataset_class` in config
  - Training: `il_lib/training/trainer.py` (wraps Lightning, creates run dir, loggers, ckpt behavior)
  - Serving: `serve.py` → `WebsocketPolicyServer` (OmniGibson network util)

## Hydra / config conventions (practical examples) ⚙️
- Typical Hydra CLI overrides: `robot=r1pro task=behavior task.name=turning_on_radio arch=wbvima`
- Add nodes with `+node=val` and remove a node with `~node.path` (examples in `scripts/train_arch.sh` and `run_eval_instructions.sh`).
- Resume training:
  - `resume.ckpt_path=...` and `resume.full_state=true` (restores optimizer & trainer state) or `false` (model-only load)
  - Trainer expands paths via `il_lib.utils.file_utils.f_expand` and writes `conf.yaml` to run dir
- Output layout (where to find runs): `outputs/<YYYY-MM-DD>/<HH-MM-SS>/<run_name>/ckpt/last.pth`, `logs/`, `tb/` (see `Trainer` creation of run directories)
- Model checkpoints use `.pth` extension (set in Trainer). 

## Project-specific patterns & gotchas 🔍
- Policies expect specific obs keys and formats:
  - `PolicyWrapper` uses camera keys like `<robot>::<camera>::rgb`, `pcd`, etc. See `il_lib/policies/policy_base.py`.
  - When `pcd` is used: fused point cloud is expected with RGB in first 3 channels and XYZ in subsequent channels. WBVIMA splits fused_pcd[..., :3] for RGB and fused_pcd[..., 3:] for xyz (see `WBVIMA.process_data`).
- Temporal behavior: `policy_wrapper.deployed_action_steps` and `module.action_prediction_horizon` control how often inference runs and how action trajectories are consumed.
- Hierarchical policies:
  - `module.skills` maps skill names to checkpoint paths. `HierarchicalPolicy` loads these and dispatches using `get_active_skill()`.
  - `HierarchicalVLMWBVIMAPolicy` queries a VLM (ARK/OpenAI): set `ARK_API_KEY` env var or `module.api_key` to use it; logs saved to `module.log_dir`.
  - Use `scripts/test_vlm_query.py` to debug API connectivity/payloads.
- Dataset conventions:
  - Behavior demos and annotations expected under `data_dir` (SkillIterableDataset reads `2025-challenge-demos/annotations/task-XXXX/episode_*.json`).
  - To train per-skill: set `data.dataset_class=il_lib.datas.skill_dataset.SkillIterableDataset` and `data.skill_name="move to"` (see `scripts/train_skills.sh`).

## Logging & experiment hygiene 🧹
- WandB integration: set `use_wandb=true`; defaults in `base_config.yaml` set `wandb_project=B1K`.
- CSV logging is always enabled; run directories are created under the run dir.
- Clean empty/dummy runs using `scripts/remove_dummy_wandb_runs.py`.

## Developer workflows & debugging tips 🛠️
- Fast checks: `trainer.fast_dev_run=true +eval=behavior headless=false` spawns a short training + online eval sanity check (tutorial shows this).
- Cluster runs: `scripts/train_behavior.sbatch.sh` is an example SLURM job; `scripts/train_arch.sh` wraps common hydra overrides.
- If a policy requires `openai`/Ark client and import fails, `HierarchicalVLMWBVIMAPolicy` will warn / error — install `openai` or ensure Ark client is available.
- When debugging evaluation: run `serve.py` first (server prints listening info), then run OmniGibson `eval.py` with `policy=websocket`.

## Files & places to look (quick map) 🗺️
- Core: `train.py`, `serve.py`, `setup.py`
- Configs: `il_lib/configs/base_config.yaml`, `il_lib/configs/arch/*.yaml`
- Policies: `il_lib/policies/*.py` (WBVIMA, hierarchical, hierarchical_vlm_...)
- Data: `il_lib/datas/*` (`BehaviorDataModule`, `SkillIterableDataset`)
- Scripts & examples: `scripts/*.sh`, `run_eval_instructions.sh`, `tutorials/il_lib.md`, `scripts/test_vlm_query.py`

---

If anything above is unclear or you want more examples (e.g., a step-by-step debug recipe for VLM failures or a sample `module.skills` mapping), tell me which part to expand and I will iterate. 💬