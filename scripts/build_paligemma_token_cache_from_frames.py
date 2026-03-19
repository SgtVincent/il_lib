import argparse
import os
import sys
from dataclasses import dataclass

import torch


@dataclass
class _Pi0TorchConfig:
    dtype: str
    paligemma_variant: str
    action_expert_variant: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    pi05: bool = False


def _load_frames(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "frames" in obj:
        obj = obj["frames"]
    if not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    if obj.ndim == 4 and obj.shape[-1] == 3:
        obj = obj.permute(0, 3, 1, 2).contiguous()
    if obj.ndim != 4 or obj.shape[1] != 3:
        raise ValueError(f"frames must have shape (N,3,H,W) or (N,H,W,3), got {tuple(obj.shape)}")
    if obj.dtype != torch.uint8:
        obj = obj.to(dtype=torch.uint8)
    return obj


def _to_model_image(frames_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = frames_u8.to(device=device, dtype=torch.float32)
    x = x / 255.0 * 2.0 - 1.0
    return x


def _pool_tokens(tokens: torch.Tensor, pool: str) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B,T,D), got {tuple(tokens.shape)}")
    if pool == "mean":
        return tokens.mean(dim=1)
    if pool == "first":
        return tokens[:, 0]
    raise ValueError(f"Unsupported pool={pool}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_pt", required=True)
    p.add_argument("--output_pt", required=True)
    p.add_argument("--pi0_ckpt_path", required=True)
    p.add_argument("--openpi_src", default=None)
    p.add_argument("--paligemma_variant", default="gemma_2b")
    p.add_argument("--action_expert_variant", default="gemma_300m")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pool", choices=["mean", "first"], default="mean")
    args = p.parse_args()

    if args.openpi_src is not None:
        sys.path.insert(0, os.path.abspath(args.openpi_src))

    from il_lib.utils.training_utils import load_torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _Pi0TorchConfig(
        dtype=str(args.dtype),
        paligemma_variant=str(args.paligemma_variant),
        action_expert_variant=str(args.action_expert_variant),
        action_dim=32,
        action_horizon=50,
        max_token_len=1,
        pi05=False,
    )
    model = PI0Pytorch(cfg, action_expert_name="gemma_token", action_expert_kwargs=None)
    ckpt = load_torch(args.pi0_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device=device)

    frames_u8 = _load_frames(args.frames_pt)
    N = int(frames_u8.shape[0])
    out = torch.zeros((N, 2048), dtype=torch.float32)

    bs = int(args.batch_size)
    for start in range(0, N, bs):
        end = min(N, start + bs)
        batch = _to_model_image(frames_u8[start:end], device=device)
        with torch.no_grad():
            tok = model.paligemma_with_expert.embed_image(batch)
            emb = _pool_tokens(tok, pool=str(args.pool)).to(dtype=torch.float32, device="cpu")
        out[start:end] = emb

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)), exist_ok=True)
    torch.save({"paligemma_token": out}, args.output_pt)


if __name__ == "__main__":
    main()

