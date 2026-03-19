import argparse
import os
import sys
from dataclasses import dataclass

import pandas as pd
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


def _load_expected_downsampled_len(*, data_path: str, demo_key: str, downsample_factor: int) -> int:
    task_id = int(demo_key) // 10000
    parquet_path = os.path.join(
        data_path,
        "2025-challenge-demos",
        "data",
        f"task-{task_id:04d}",
        f"episode_{demo_key}.parquet",
    )
    df = pd.read_parquet(parquet_path)
    n = len(df["observation.state"][:: int(downsample_factor)])
    return int(n)


def _iter_downsampled_rgb(
    *,
    data_path: str,
    demo_key: str,
    camera: str,
    output_size: tuple[int, int],
    downsample_factor: int,
    expected_len: int,
    batch_size: int,
):
    from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP

    task_id = int(demo_key) // 10000
    cam_id = str(camera)
    loader = OBS_LOADER_MAP["rgb"](
        data_path=os.path.join(data_path, "2025-challenge-demos"),
        task_id=task_id,
        camera_id=cam_id,
        demo_id=str(demo_key),
        batch_size=1,
        stride=1,
        start_idx=0,
        end_idx=None,
        output_size=output_size,
    )

    buf = []
    raw_i = 0
    out_i = 0
    try:
        for frame in loader:
            if out_i >= expected_len:
                break
            if raw_i % int(downsample_factor) == 0:
                buf.append(frame[0].to(dtype=torch.uint8))
                out_i += 1
                if len(buf) >= int(batch_size):
                    yield torch.stack(buf, dim=0)
                    buf = []
            raw_i += 1
    finally:
        loader.close()

    if len(buf) > 0:
        yield torch.stack(buf, dim=0)


def _save_cache(*, cache_dir: str, demo_key: str, tokens: torch.Tensor) -> str:
    task_id = int(demo_key) // 10000
    out_dir = os.path.join(os.path.expanduser(cache_dir), f"task-{task_id:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"episode_{demo_key}.pt")
    torch.save({"paligemma_token": tokens.to(dtype=torch.float32, device="cpu")}, out_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--task_names", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--pi0_ckpt_path", required=True)
    p.add_argument("--openpi_src", default=None)
    p.add_argument("--camera", default="head", choices=["head", "left_wrist", "right_wrist"])
    p.add_argument("--output_h", type=int, default=256)
    p.add_argument("--output_w", type=int, default=256)
    p.add_argument("--downsample_factor", type=int, default=1)
    p.add_argument("--max_demos", type=int, default=None)
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
    from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset

    task_names = [x for x in str(args.task_names).split(",") if len(x) > 0]
    demo_keys = BehaviorIterableDataset.get_all_demo_keys(os.path.expanduser(args.data_path), task_names)
    if args.max_demos is not None:
        demo_keys = demo_keys[: int(args.max_demos)]

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

    for demo_key in demo_keys:
        demo_key = str(demo_key)
        expected_len = _load_expected_downsampled_len(
            data_path=os.path.expanduser(args.data_path),
            demo_key=demo_key,
            downsample_factor=int(args.downsample_factor),
        )
        tokens = torch.zeros((expected_len, 2048), dtype=torch.float32)
        offset = 0
        for frames_u8 in _iter_downsampled_rgb(
            data_path=os.path.expanduser(args.data_path),
            demo_key=demo_key,
            camera=str(args.camera),
            output_size=(int(args.output_h), int(args.output_w)),
            downsample_factor=int(args.downsample_factor),
            expected_len=expected_len,
            batch_size=int(args.batch_size),
        ):
            batch = _to_model_image(frames_u8, device=device)
            with torch.no_grad():
                tok = model.paligemma_with_expert.embed_image(batch)
                emb = _pool_tokens(tok, pool=str(args.pool)).to(dtype=torch.float32, device="cpu")
            end = min(expected_len, offset + int(emb.shape[0]))
            tokens[offset:end] = emb[: end - offset]
            offset = end
            if offset >= expected_len:
                break
        out_path = _save_cache(cache_dir=str(args.cache_dir), demo_key=demo_key, tokens=tokens)
        print(f"saved {out_path} ({tuple(tokens.shape)})")


if __name__ == "__main__":
    main()

