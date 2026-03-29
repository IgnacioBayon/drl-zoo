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

    log_ratio = (log_probs - old_log_probs).clamp(-10.0, 10.0)
    policy_ratio = torch.exp(log_ratio)

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

    def __init__(
        self,
        c1: float,
        c2: float,
        epsilon: float = 0.2,
        value_clip: float | None = 0.2,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.epsilon = epsilon
        self.value_clip = value_clip

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, total_loss = self.compute_terms(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            values=values,
            old_values=old_values,
            entropy=entropy,
        )
        return total_loss

    def compute_terms(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return actor, critic, entropy and total PPO losses."""
        clip_loss = _clip_loss(log_probs, old_log_probs, advantages, self.epsilon)
        if self.value_clip is None:
            value_loss = F.huber_loss(values, returns)
        else:
            unclipped = F.huber_loss(values, returns, reduction="none")
            clipped_values = old_values + torch.clamp(
                values - old_values,
                -self.value_clip,
                self.value_clip,
            )
            clipped = F.huber_loss(clipped_values, returns, reduction="none")
            value_loss = torch.max(unclipped, clipped).mean()
        entropy_loss = -entropy.mean()

        total_loss = clip_loss + self.c1 * value_loss + self.c2 * entropy_loss

        return clip_loss, value_loss, entropy_loss, total_loss
