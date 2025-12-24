import torch
import torch.nn as nn
from il_lib.policies.hierarchical_policy import HierarchicalPolicy
import logging
from typing import Dict, List, Any, Optional
import os
import base64
import io
import numpy as np
from PIL import Image
import json
import glob
import time

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    logger.warning("openai module not found. HierarchicalVLMWBVIMAPolicy will fail if used.")
    OpenAI = None

class HierarchicalVLMWBVIMAPolicy(HierarchicalPolicy):
    def __init__(
        self, 
        skills: Dict[str, str], 
        task_obs_keys: List[str] = [], 
        task_meta_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "ep-20250826131655-jxxss",
        query_frequency: int = 100,
        verbose: bool = False,
        log_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Hierarchical Policy that uses a VLM to select the active skill based on 
        visual observation and ground truth task info.
        
        Args:
            skills: Dictionary mapping skill names to checkpoint paths.
            task_obs_keys: List of keys corresponding to the values in the task info tensor.
            task_meta_path: Path to directory containing episode metadata json files. Used if task_obs_keys is empty.
            api_key: API key for the VLM service. If None, looks for ARK_API_KEY env var.
            model_name: Name of the VLM model to use.
            query_frequency: Frequency of VLM queries in steps.
            verbose: Whether to print verbose output.
            log_dir: Directory to save VLM I/O logs.
        """
        # Filter kwargs to remove arguments not accepted by BasePolicy/LightningModule
        # BasePolicy accepts: online_eval, policy_wrapper, robot_type
        # We ignore training params like lr, optimizer settings, etc. inherited from WBVIMA config
        valid_base_args = ['online_eval', 'policy_wrapper', 'robot_type']
        base_kwargs = {k: v for k, v in kwargs.items() if k in valid_base_args}
        
        super().__init__(skills=skills, **base_kwargs)
        
        self.task_obs_keys = task_obs_keys
        self.query_frequency = query_frequency
        self.verbose = verbose
        self.log_dir = log_dir
        self.step_counter = 0
        
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        if not self.task_obs_keys and task_meta_path:
            try:
                # Find first json file
                json_files = glob.glob(os.path.join(task_meta_path, "*.json"))
                if json_files:
                    # Sort to ensure deterministic selection (e.g. episode_00000000.json)
                    json_files.sort()
                    target_file = json_files[0]
                    logger.info(f"Loading task_obs_keys from {target_file}")
                    with open(target_file, 'r') as f:
                        meta = json.load(f)
                        self.task_obs_keys = meta.get("task_obs_keys", [])
                    logger.info(f"Loaded {len(self.task_obs_keys)} task_obs_keys")
                else:
                    logger.warning(f"No json files found in {task_meta_path}")
            except Exception as e:
                logger.error(f"Failed to load task_obs_keys from {task_meta_path}: {e}")

        self.model_name = model_name
        
        if OpenAI is None:
            raise ImportError("Please install openai package to use HierarchicalVLMWBVIMAPolicy: pip install openai")
            
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        # self.api_key = ""
        if not self.api_key:
            logger.warning("No API key provided for VLM. Please set ARK_API_KEY environment variable or pass api_key.")
            
        self.client = OpenAI(
            base_url="https://ark-cn-beijing.bytedance.net/api/v3",
            api_key=self.api_key or "dummy_key", # Prevent crash on init if key missing, will fail on call
        )
        
        # Cache the last selected skill to avoid flickering or excessive API calls?
        # For now, we query every time as requested, but in practice one might want to hold a skill until completion.
        self.last_skill = None

    def get_active_skill(self, obs: Dict[str, Any]) -> str:
        """
        Query VLM to determine the active skill.
        """
        # Rate limiting: only query if it's the first step or every query_frequency steps
        if self.last_skill is not None and self.step_counter % self.query_frequency != 0:
            self.step_counter += 1
            return self.last_skill

        # 1. Extract Observation Data
        # obs['obs'] is expected to contain camera RGB (e.g. 'robot_r1::head_camera::rgb') 
        # and 'task' (if use_task_info=True)
        
        observation = obs.get('obs', {})
        
        # Find RGB tensor (prefer head camera)
        rgb_tensor = None
        for key in observation:
            if "::rgb" in key:
                if "head" in key:
                    rgb_tensor = observation[key]
                    if self.verbose:
                        logger.info(f"Found head camera RGB: {key}, shape: {rgb_tensor.shape}")
                    break
                if rgb_tensor is None:
                    rgb_tensor = observation[key]
                    if self.verbose:
                        logger.info(f"Found RGB: {key}, shape: {rgb_tensor.shape}")
        
        task_tensor = observation.get('task')
        
        # 2. Prepare Image
        img_str = self._process_image(rgb_tensor)
        if img_str is None:
            logger.warning("Could not process RGB image for VLM. Defaulting to first skill.")
            self.step_counter += 1
            return self._default_skill()

        # 3. Prepare Task Info
        task_info_str = self._process_task_info(task_tensor)
        
        # 4. Construct Prompt
        skills_list = list(self.policies.keys())
        prompt_text = (
            f"You are a robot control policy. Available skills are: {', '.join(skills_list)}.\n"
            f"Current Task Information (Ground Truth):\n{task_info_str}\n"
            "Based on the visual observation and task information, which skill should be executed next? "
            "Return ONLY the skill name from the available list. Do not add any explanation."
        )
        
        if self.verbose:
            print(f"\n[VLM Policy] Step {self.step_counter}: Querying VLM...")
            print(f"[VLM Policy] Prompt:\n{prompt_text}")

        # 5. Call VLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=20,
                extra_headers={"X-TT-LOGID": "hierarchical_policy_vlm_query"},
            )
            raw_content = response.choices[0].message.content
            content = raw_content.strip() if raw_content else ""
            
            if self.verbose:
                print(f"[VLM Policy] Response: {content}")

            # 6. Parse Response
            # Find the skill name in the response
            selected_skill = None
            # Exact match check first
            if content in skills_list:
                selected_skill = content
            else:
                # Fuzzy match: check if skill name is in content
                for skill in skills_list:
                    if skill in content:
                        selected_skill = skill
                        break
            
            # Save I/O logs
            if self.log_dir:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                log_file = os.path.join(self.log_dir, f"query_{self.step_counter}_{timestamp}.json")
                log_data = {
                    "step": self.step_counter,
                    "prompt": prompt_text,
                    "image_base64": img_str,
                    "response_raw": content,
                    "selected_skill": selected_skill
                }
                try:
                    with open(log_file, "w") as f:
                        json.dump(log_data, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save VLM log: {e}")

            if selected_skill:
                if selected_skill != self.last_skill:
                    logger.info(f"VLM switched skill to: {selected_skill}")
                    self.last_skill = selected_skill
                self.step_counter += 1
                return selected_skill
            else:
                logger.warning(f"VLM returned '{content}', which does not match any known skill. Using default.")
                self.step_counter += 1
                return self._default_skill()

        except Exception as e:
            logger.error(f"Error calling VLM: {e}")
            self.step_counter += 1
            return self._default_skill()

    def _process_image(self, rgb_tensor: Optional[torch.Tensor]) -> Optional[str]:
        if rgb_tensor is None:
            return None
            
        # Handle dimensions: (B, T, C, H, W) or (B, C, H, W)
        # We take the first element of the batch, and the last frame if temporal
        try:
            if rgb_tensor.ndim == 5: 
                img_t = rgb_tensor[0, -1]
            elif rgb_tensor.ndim == 4:
                img_t = rgb_tensor[0]
            else:
                return None
                
            # Convert to numpy (C, H, W) -> (H, W, C)
            img_np = img_t.detach().cpu().numpy()
            if img_np.shape[0] in [1, 3]:
                img_np = np.transpose(img_np, (1, 2, 0))
                
            # Normalize if needed (assuming float 0-1 or int 0-255)
            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            
            pil_img = Image.fromarray(img_np)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

    def _process_task_info(self, task_tensor: Optional[torch.Tensor]) -> str:
        if task_tensor is None:
            return "No task info available."
            
        try:
            # Handle dimensions: (B, T, D) or (B, D)
            if task_tensor.ndim == 3:
                t_vec = task_tensor[0, -1]
            elif task_tensor.ndim == 2:
                t_vec = task_tensor[0]
            else:
                t_vec = task_tensor
                
            t_vec = t_vec.detach().cpu().numpy()
            
            if not self.task_obs_keys:
                # Fallback if no keys provided
                return f"Values: {t_vec.tolist()}"
                
            pairs = []
            for i, key in enumerate(self.task_obs_keys):
                if i < len(t_vec):
                    pairs.append(f"{key}: {t_vec[i]:.4f}")
            return "\n".join(pairs)
        except Exception as e:
            logger.error(f"Task info processing error: {e}")
            return "Error processing task info."

    def _default_skill(self) -> str:
        if self.last_skill:
            return self.last_skill
        if len(self.policies) > 0:
            return list(self.policies.keys())[0]
        # Fallback: should never happen if policies are loaded
        raise RuntimeError("No skills available and no last_skill cached.")
