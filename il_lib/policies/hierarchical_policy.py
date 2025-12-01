import torch
import torch.nn as nn
from il_lib.policies.policy_base import BasePolicy
from il_lib.policies.wbvima_policy import WBVIMA
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HierarchicalPolicy(BasePolicy):
    def __init__(self, skills: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        
        self.skills_config = skills
        self.policies = nn.ModuleDict()
        
        # Load policies
        for skill_name, ckpt_path in self.skills_config.items():
            logger.info(f"Loading skill '{skill_name}' from {ckpt_path}")
            # Load checkpoint. We assume the checkpoint contains the model.
            # We use WBVIMA class.
            # Note: load_from_checkpoint is a Lightning method.
            # It requires the class to be the same as the one saved.
            # If the saved model is WBVIMA, this works.
            # We map to cpu to avoid GPU OOM during loading, Lightning handles moving to device
            self.policies[skill_name] = WBVIMA.load_from_checkpoint(ckpt_path, map_location='cpu')
            self.policies[skill_name].eval()
            self.policies[skill_name].freeze()
            
    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        skill_name = self.get_active_skill(obs)
        return self.policies[skill_name](obs, *args, **kwargs)

    @torch.no_grad()
    def act(self, obs, policy_state=None, deterministic=None) -> torch.Tensor:
        skill_name = self.get_active_skill(obs)
        # We assume sub-policies are stateless or handle state internally/via wrapper
        return self.policies[skill_name].act(obs, policy_state, deterministic)
    
    def reset(self) -> None:
        for policy in self.policies.values():
            policy.reset()

    def policy_training_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError("HierarchicalPolicy is for evaluation only.")

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        # We could implement evaluation if we have ground truth skills?
        pass

    def configure_optimizers(self):
        return None

    def get_active_skill(self, obs):
        # Placeholder for GT logic
        # User should implement this based on task info
        # For turning_on_radio, we have skills: move_to, pick_up, press, place
        
        # Example logic (pseudo-code):
        # task_info = obs['obs']['task'] # (B, L, D)
        # radio_pos = task_info[..., 0:3]
        # robot_pos = obs['obs']['eef']['right_pos'] # or base pos
        # dist = torch.norm(radio_pos - robot_pos, dim=-1)
        
        # if dist > 0.5: return "move_to"
        # elif not is_grasped: return "pick_up"
        # elif not is_pressed: return "press"
        # else: return "place"
        
        # For now, return a default skill to avoid crash
        # In a real scenario, this should be implemented based on the specific task
        if len(self.policies) > 0:
            return list(self.policies.keys())[0]
        return None
