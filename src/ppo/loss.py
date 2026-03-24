import torch
import torch.nn as nn
import torch.nn.functional as F


def _clip_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Loss function for PPO training

    Args:
        log_probs: Log probabilities of the current policy
        old_log_probs: Log probabilities of the old policy
        advantages: Advantage estimates
        epsilon: Clipping parameter. Defaults to 0.2.

    Returns:
        torch.Tensor: Clipped policy loss
    """
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    policy_ratio = torch.exp(log_probs - old_log_probs)

    clipped_ratio = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)

    loss = -torch.min(policy_ratio * advantages, clipped_ratio * advantages).mean()

    return loss


class PPOLoss(nn.Module):
    """Computes the clipped surrogate policy loss (L^CLIP) from the PPO paper.

    Attributes:
        c1: Coefficient for value loss
        c2: Coefficient for entropy loss
        epsilon: Clipping parameter for policy loss
    """

    def __init__(self, c1: float, c2: float, epsilon: float = 0.2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.epsilon = epsilon

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        clip_loss = _clip_loss(log_probs, old_log_probs, advantages, self.epsilon)
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        total_loss = clip_loss + self.c1 * value_loss + self.c2 * entropy_loss

        return total_loss
