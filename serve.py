import hydra
from hydra.utils import instantiate
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from omegaconf import OmegaConf
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
import logging

# fix pytorch 2.6 weights_only = True by default issue
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg):
    register_omegaconf_resolvers()
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    policy = instantiate(cfg.module, _recursive_=False)
    if cfg.ckpt_path:
        ckpt = load_torch(
            cfg.ckpt_path,
            map_location="cpu",
        )
        load_state_dict(
            policy,
            ckpt["state_dict"],
            strict=True
        )
    else:
        logger = logging.getLogger(__name__)
        logger.info("No ckpt_path provided in config; skipping loading state_dict for the top-level policy.")
    policy = policy.to("cuda")
    policy.eval()
    # instantiate wrapper for policy
    policy_wrapper = instantiate(cfg.policy_wrapper)
    policy_wrapper.policy = policy
    server = WebsocketPolicyServer(
        policy=policy_wrapper,
        host="0.0.0.0",
        port=8000,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
