import torch

from .loss import PPOLoss
from .model import PPO


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Calculate advantages using Generalized Advantage Estimation (GAE)

    Args:
        rewards: Rewards received at each timestep [B, T]
        values: Value estimates for each state [B, T]
        next_values: Value estimates for the next states [B, T]
        dones: Binary indicators of episode termination [B, T]
        gamma: Discount factor. Defaults to 0.99.
        lam: GAE lambda parameter. Defaults to 0.95.

    Returns:
        torch.Tensor: Advantage estimates [B, T]
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values

    advantages = torch.zeros_like(deltas)
    gae = 0
    for t in reversed(range(deltas.size(1))):
        gae = deltas[:, t] + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae

    return advantages
