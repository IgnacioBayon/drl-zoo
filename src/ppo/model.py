import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.encoder import Encoder


class Actor(nn.Module):
    """Diagonal-Gaussian actor head for PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, int] = (512, 256),
    ):
        super().__init__()

        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)

        self.mean = nn.Linear(h2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Actor forward method. Outputs a gaussian distribution for each action dim.

        Args:
            x: Input vector (Encoder latent space)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output distributions parameters.
                Tensors are [mean, log_std], each of shape [B, action_dim]
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)

        log_std = self.log_std.expand_as(mean).clamp(-5, 2)

        return mean, log_std


class Critic(nn.Module):
    """Value-function head for PPO."""

    def __init__(self, state_dim: int, hidden_sizes: tuple[int, int] = (512, 256)):
        super().__init__()

        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Critic forward method. Outputs a given state's value.

        Args:
            x: Input vector (Encoder latent space)

        Returns:
            torch.Tensor: State value
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x).squeeze(-1)


class PPO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        share_encoder: bool = True,
        actor_hidden_sizes: tuple[int, int] = (512, 256),
        critic_hidden_sizes: tuple[int, int] = (512, 256),
        action_low: list[float] | torch.Tensor | None = None,
        action_high: list[float] | torch.Tensor | None = None,
    ):
        super().__init__()

        self.share_encoder = share_encoder

        if share_encoder:
            self.actor_encoder = Encoder(in_channels)
            self.critic_encoder = self.actor_encoder
        else:
            self.actor_encoder = Encoder(in_channels)
            self.critic_encoder = Encoder(in_channels)

        self.actor = Actor(
            self.actor_encoder.OUT_FEATURES,
            action_dim,
            hidden_sizes=actor_hidden_sizes,
        )
        self.critic = Critic(
            self.critic_encoder.OUT_FEATURES,
            hidden_sizes=critic_hidden_sizes,
        )

        if action_low is None or action_high is None:
            action_low_t = torch.full((action_dim,), -1.0, dtype=torch.float32)
            action_high_t = torch.full((action_dim,), 1.0, dtype=torch.float32)
        else:
            action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
            action_high_t = torch.as_tensor(action_high, dtype=torch.float32)

        if action_low_t.shape != (action_dim,) or action_high_t.shape != (action_dim,):
            raise ValueError(
                "action_low/action_high must have shape (action_dim,), "
                f"got {tuple(action_low_t.shape)} and {tuple(action_high_t.shape)}"
            )

        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO Forward. Action outputs are represented as Gaussian distributions.

        Args:
            x: Input vector (Observation image)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Action means (shape: [B, action_dim]),
                Action log stds (shape: [B, action_dim]),
                and state value (shape: [B])
        """
        actor_latent = self.actor_encoder(x)
        critic_latent = self.critic_encoder(x)

        means, log_stds = self.actor(actor_latent)
        value = self.critic(critic_latent)

        return means, log_stds, value

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from a diagonal Gaussian policy for PPO rollouts."""
        mean, log_std, value = self.forward(x)
        dist = Normal(mean, log_std.exp())

        action = dist.sample()
        action = torch.clamp(action, self.action_low, self.action_high)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value

    @torch.no_grad()
    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self.forward(x)
        return torch.clamp(mean, self.action_low, self.action_high)

    def evaluate(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored rollout actions under the current Gaussian policy."""
        mean, log_std, value = self.forward(x)
        dist = Normal(mean, log_std.exp())

        clipped_actions = torch.clamp(actions, self.action_low, self.action_high)
        log_prob = dist.log_prob(clipped_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy
