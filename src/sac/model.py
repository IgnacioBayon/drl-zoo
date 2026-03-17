"""Model definitions for SAC with image observations."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.encoder import Encoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# exp(log_std) bounds for numerical stability and reasonable exploration in continuous action spaces
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------
def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers: list[nn.Module] = []
    dims = [input_dim] + hidden_dims

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """Gaussian policy for continuous control from image observations.

    Args:
        in_channels: Number of stacked input frames.
        action_dim: Action dimensionality.
        hidden_dims: Hidden layer widths after the encoder.
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        # Extracts visual features
        self.encoder = Encoder(in_channels)
        # Further processes latent features before outputting Gaussian parameters
        self.trunk = build_mlp(
            Encoder.OUT_FEATURES,
            hidden_dims,
            hidden_dims[-1],
        )
        # Outputs mean and log std for each action dimension
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns Gaussian parameters ``(mu, log_std)`` from image observations."""
        # 1. Encode visual input to latent features
        z = self.encoder(x)
        # 2. Process latent features through trunk MLP
        h = self.trunk(z)
        # 3. Output mean and log std for Gaussian policy
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        # Clamp log std to prevent numerical issues and ensure reasonable exploration
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples a tanh-squashed action and returns ``(action, log_prob)``."""
        # Compute Gaussian parameters from input
        mu, log_std = self(x)
        std = log_std.exp()

        dist = Normal(mu, std)
        # Reparameterised sample for backpropagation: u = mu + std * eps, where eps ~ N(0, 1)
        u = dist.rsample()
        # Apply tanh squashing to ensure actions are in (-1, 1)
        action = torch.tanh(u)

        # Compute log probability of the sampled action, accounting for tanh transformation
        log_prob = dist.log_prob(u)
        # When you sample from the Gaussian and then squash with tanh, the final action is no longer
        # Gaussian-distributed, so this corrects the log-probability after the transformation.
        # (See Appendix C of SAC paper for details on this correction)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Returns an action for environment interaction."""
        # Add batch dimension if input is a single observation (C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        mu, log_std = self(x)

        if deterministic:
            action = torch.tanh(mu)
        else:
            std = log_std.exp()
            dist = Normal(mu, std)
            u = dist.rsample()
            action = torch.tanh(u)

        return action.squeeze(0)


# ---------------------------------------------------------------------------
# Critic head
# ---------------------------------------------------------------------------
class CriticHead(nn.Module):
    """Single Q-head mapping ``(latent, action) -> Q``."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.q = build_mlp(latent_dim + action_dim, hidden_dims, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns scalar Q-values with shape ``(B, 1)``."""
        x = torch.cat([z, action], dim=-1)
        return self.q(x)


# ---------------------------------------------------------------------------
# Double critic
# ---------------------------------------------------------------------------
class DoubleCritic(nn.Module):
    """Double Q-network for image observations.

    Encodes the observation once, then applies two independent Q-heads.
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.q1 = CriticHead(Encoder.OUT_FEATURES, action_dim, hidden_dims)
        self.q2 = CriticHead(Encoder.OUT_FEATURES, action_dim, hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(q1, q2)``."""
        z = self.encoder(x)
        return self.q1(z, action), self.q2(z, action)
