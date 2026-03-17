import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import Encoder


class Actor(nn.Module):
    """Simple MLP representing the PPO actor. Actions are continous"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Actor forward method. Outputs a gaussian distribution for each action dim.

        Args:
            x: Input vector (Encoder latent space)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output distributions.
                Tensors are [mean, log_std], each of shape [B, action_dim]
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)

        log_std = self.log_std.expand_as(mean)

        return mean, log_std


class Critic(nn.Module):
    """Simple MLP representing the PPO Critic"""

    def __init__(self, state_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Critic forward method. Outputs a given state's value.

        Args:
            x: Input vector (Encoder latent space)

        Returns:
            torch.Tensor: State value
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPO(nn.Module):
    def __init__(self, in_channels: int, action_dim: int):
        super().__init__()

        self.encoder = Encoder(in_channels)

        self.actor = Actor(self.encoder.OUT_FEATURES, action_dim)
        self.critic = Critic(self.encoder.OUT_FEATURES)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO Forward. Action outputs are represented as Gaussian distributions.

        Args:
            x: _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Action means (shape: [B, action_dim]),
                Action log stds (shape: [B, action_dim]),
                and state value (shape: [B, 1])
        """
        latent_representation = self.encoder(x)

        means, log_stds = self.actor(latent_representation)
        value = self.critic(latent_representation)

        return means, log_stds, value
