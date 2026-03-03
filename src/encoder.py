"""Shared CNN encoder used by both DQN and Rainbow."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """CNN encoder shared between DQN and Rainbow for fair comparison.

    Args:
        in_channels: Number of input channels (typically ``stack_size``).
    """

    OUT_FEATURES: int = 64 * 7 * 7  # 3136

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)  # (B, OUT_FEATURES)
