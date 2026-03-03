"""Branching DQN model."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.encoder import Encoder


class DQNetwork(nn.Module):
    """Branching DQN: shared encoder -> hidden FC -> one linear head per action branch.

    Args:
        in_channels: Number of stacked frames.
        action_bins: Discrete bins per action dimension.
        num_branches: Number of independent action dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        action_bins: int,
        num_branches: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.action_bins = action_bins
        self.encoder = Encoder(in_channels)
        self.fc = nn.Sequential(
            nn.Linear(Encoder.OUT_FEATURES, 512),
            nn.ReLU(),
        )
        self.branches = nn.Linear(512, num_branches * action_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values of shape ``(B, branches, bins)``."""
        z = self.fc(self.encoder(x))
        return self.branches(z).view(-1, self.num_branches, self.action_bins)
