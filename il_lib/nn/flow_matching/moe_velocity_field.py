import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ActionBlockSpec:
    name: str
    start: int
    dim: int

    @property
    def end(self) -> int:
        return self.start + self.dim


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(act())
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionBlockMoEVelocityField(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        action_blocks: List[ActionBlockSpec],
        cond_dim: int,
        num_experts: int,
        top_k: Optional[int] = None,
        expert_hidden_dim: int = 512,
        expert_hidden_depth: int = 2,
        gating_hidden_dim: int = 256,
        gating_hidden_depth: int = 1,
        activation: str = "silu",
        gating_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(action_blocks) == 0:
            raise ValueError("action_blocks must be non-empty")
        if sum(b.dim for b in action_blocks) != action_dim:
            raise ValueError("sum(block.dim) must equal action_dim")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k is not None and (top_k <= 0 or top_k > num_experts):
            raise ValueError("top_k must be in [1, num_experts]")
        if gating_temperature <= 0:
            raise ValueError("gating_temperature must be positive")

        self.action_dim = int(action_dim)
        self.action_blocks = action_blocks
        self.cond_dim = int(cond_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k) if top_k is not None else None
        self.gating_temperature = float(gating_temperature)

        self._block_names = [b.name for b in action_blocks]
        self._block_slices = [(b.start, b.end) for b in action_blocks]

        self.gates = nn.ModuleDict()
        self.experts = nn.ModuleDict()

        for b in action_blocks:
            self.gates[b.name] = _MLP(
                input_dim=cond_dim,
                hidden_dim=gating_hidden_dim,
                output_dim=num_experts,
                hidden_depth=gating_hidden_depth,
                activation=activation,
            )
            self.experts[b.name] = nn.ModuleList(
                [
                    _MLP(
                        input_dim=b.dim + cond_dim + 1,
                        hidden_dim=expert_hidden_dim,
                        output_dim=b.dim,
                        hidden_depth=expert_hidden_depth,
                        activation=activation,
                    )
                    for _ in range(num_experts)
                ]
            )

    def _prepare_t(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        t = t.to(device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.view(1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        elif t.ndim == 1:
            t = t.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        elif t.ndim == 2:
            t = t.view(x.shape[0], -1, 1)
        elif t.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported t shape: {t.shape}")
        if t.shape[0] != x.shape[0] or t.shape[1] != x.shape[1] or t.shape[2] != 1:
            raise ValueError(f"t must broadcast to (B,H,1), got {t.shape} vs x {x.shape}")
        return t

    def forward(self, *, x: torch.Tensor, t: torch.Tensor | float, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be (B,H,A), got {x.shape}")
        if cond.ndim != 2:
            raise ValueError(f"cond must be (B,D), got {cond.shape}")
        if x.shape[-1] != self.action_dim:
            raise ValueError(f"action_dim mismatch: {x.shape[-1]} vs {self.action_dim}")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond_dim mismatch: {cond.shape[-1]} vs {self.cond_dim}")

        B, H, _ = x.shape
        t = self._prepare_t(x, t)
        cond_h = cond.unsqueeze(1).expand(B, H, self.cond_dim)
        v = x.new_zeros((B, H, self.action_dim))

        for name, (s, e) in zip(self._block_names, self._block_slices):
            x_b = x[:, :, s:e]
            gate_logits = self.gates[name](cond) / self.gating_temperature
            if self.top_k is not None and self.top_k < self.num_experts:
                topv, topi = torch.topk(gate_logits, k=self.top_k, dim=-1)
                sparse = gate_logits.new_full(gate_logits.shape, float("-inf"))
                sparse.scatter_(dim=-1, index=topi, src=topv)
                gate_logits = sparse
            w = F.softmax(gate_logits, dim=-1)

            inp = torch.cat([x_b, cond_h, t], dim=-1).reshape(B * H, -1)
            out = 0.0
            for j, expert in enumerate(self.experts[name]):
                out = out + expert(inp).view(B, H, -1) * w[:, j].view(B, 1, 1)
            v[:, :, s:e] = out

        return v

