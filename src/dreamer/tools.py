"""Utility helpers for DreamerV3 / R2Dreamer.

Trimmed copy of r2dreamer/tools.py — only the bits required by the other
dreamer modules (distributions, networks, rssm, dreamer, train).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def to_i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.int32)


def weight_init_(m: nn.Module, fan_type: str = "in") -> None:
    # RMSNorm: initialize scale to 1.
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None or weight.numel() == 0:
        return

    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)

    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


class Every:
    """Fires every N steps and returns how many intervals elapsed."""

    def __init__(self, every: float) -> None:
        self._every = float(every)
        self._last: float | None = None

    def __call__(self, step: float) -> int:
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self) -> None:
        self._once = True

    def __call__(self) -> bool:
        if self._once:
            self._once = False
            return True
        return False


def tensorstats(tensor: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
    }


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_rms(tensors) -> torch.Tensor:
    flattened = torch.cat([t.reshape(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)


def compute_global_norm(tensors) -> torch.Tensor:
    flattened = torch.cat([t.reshape(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2)


def rpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x
