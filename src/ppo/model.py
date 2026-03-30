import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.encoder import Encoder


def _init_orthogonal(layer: nn.Module, gain: float) -> None:
    """Apply orthogonal initialization with zero bias to linear/conv layers."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    """Diagonal-Gaussian actor head for PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        _init_orthogonal(self.fc1, gain=2**0.5)
        _init_orthogonal(self.fc2, gain=2**0.5)
        _init_orthogonal(self.mean, gain=0.01)

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

    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        _init_orthogonal(self.fc1, gain=2**0.5)
        _init_orthogonal(self.fc2, gain=2**0.5)
        _init_orthogonal(self.fc3, gain=1.0)

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
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 512,
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
            hidden_dim=actor_hidden_dim,
        )
        self.critic = Critic(
            self.critic_encoder.OUT_FEATURES,
            hidden_dim=critic_hidden_dim,
        )

        self.actor_encoder.apply(lambda m: _init_orthogonal(m, gain=2**0.5))
        if not self.share_encoder:
            self.critic_encoder.apply(lambda m: _init_orthogonal(m, gain=2**0.5))

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
        actor_latent = self.actor_encoder(x)
        critic_latent = self.critic_encoder(x)

        means, log_stds = self.actor(actor_latent)
        value = self.critic(critic_latent)

        return means, log_stds, value

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(x)
        dist = Normal(mean, log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        # Clip to action bounds without log-prob correction (standard PPO)
        action = action.clamp(
            self.action_low.unsqueeze(0), self.action_high.unsqueeze(0)
        )
        return action, log_prob, value

    @torch.no_grad()
    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self.forward(x)

        squashed = torch.tanh(mean)
        return self.action_low + 0.5 * (self.action_high - self.action_low) * (
            squashed + 1.0
        )

    def evaluate(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(x)
        dist = Normal(mean, log_std.exp())
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy
