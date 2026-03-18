import torch
import torch.nn as nn
from torch.distributions import Normal

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
            tuple[torch.Tensor, torch.Tensor]: Output distributions parameters.
                Tensors are [mean, log_std], each of shape [B, action_dim]
        """

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        mean = self.mean(x)

        log_std = self.log_std.expand_as(mean).clamp(-20, 2)

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

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return self.fc3(x).squeeze(-1)


class PPO(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, share_encoder: bool = True):
        super().__init__()

        self.share_encoder = share_encoder

        if share_encoder:
            self.actor_encoder = Encoder(in_channels)
            self.critic_encoder = self.actor_encoder
        else:
            self.actor_encoder = Encoder(in_channels)
            self.critic_encoder = Encoder(in_channels)

        self.actor = Actor(self.actor_encoder.OUT_FEATURES, action_dim)
        self.critic = Critic(self.critic_encoder.OUT_FEATURES)

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
        """Sample an action from the policy. Used during rollout collection.

        Samples from the Gaussian policy and computes the log probability of
        the sampled action (summed over action dimensions).

        Args:
            x: Observation image, shape [B, in_channels, H, W].

        Returns:
            action: Sampled action, shape [B, action_dim].
            log_prob: Log probability of the sampled action, shape [B].
                Summed over action dimensions (factored Gaussian assumption).
            value: State value estimate, shape [B].
        """
        mean, log_std, value = self.forward(x)
        dist = Normal(mean, log_std.exp())

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value

    def evaluate(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored actions under the current policy. Used during the update step.

        Re-computes log probabilities and entropy for actions that were collected
        during a previous rollout, which is needed to calculate the PPO clipped
        surrogate loss and the entropy bonus.

        Args:
            x: Observation images, shape [B, in_channels, H, W].
            actions: Previously sampled actions to evaluate, shape [B, action_dim].

        Returns:
            log_prob: Log probability of the given actions under the current policy,
                shape [B]. Summed over action dimensions.
            value: State value estimates under the current critic, shape [B].
            entropy: Differential entropy of the current policy distribution,
                shape [B]. Summed over action dimensions. Used as a bonus in the
                loss to encourage exploration.
        """
        mean, log_std, value = self.forward(x)
        dist = Normal(mean, log_std.exp())
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy
