"""Rainbow DQN model: Dueling + NoisyNets + C51 distributional."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import Encoder

# ---------------------------------------------------------------------------
# NoisyLinear (Fortunato et al., 2018)
# ---------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer."""

    def __init__(
        self, in_features: int, out_features: int, sigma0: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / in_features**0.5
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

        init_sigma = sigma0 / in_features**0.5
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), init_sigma)
        )
        self.bias_sigma = nn.Parameter(torch.full((out_features,), init_sigma))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.register_buffer("_noise_in", torch.zeros(in_features))
        self.register_buffer("_noise_out", torch.zeros(out_features))
        self.reset_noise()

    @staticmethod
    def _factored_noise_(x: torch.Tensor) -> torch.Tensor:
        """In-place f(x) = sign(x) * sqrt(|x|) on a pre-filled N(0,1) buffer."""
        s = x.sign()
        return x.abs_().sqrt_().mul_(s)

    def reset_noise(self) -> None:
        self._noise_in.normal_()
        self._noise_out.normal_()
        self._factored_noise_(self._noise_in)
        self._factored_noise_(self._noise_out)
        self.weight_epsilon.copy_(self._noise_out.outer(self._noise_in))
        self.bias_epsilon.copy_(self._noise_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# Dueling head (per-branch)
# ---------------------------------------------------------------------------


class DuelingHead(nn.Module):
    """Dueling distributional head for a single action branch (Wang et al., 2016).

    Outputs ``(B, action_bins, atoms)`` logits for C51.

    Args:
        latent_dim: Encoder output dimensionality.
        action_bins: Number of discrete actions in this branch.
        atoms: Number of atoms for the categorical distribution.
        hidden_dim: Hidden layer width inside each stream.
        sigma0: NoisyLinear initial sigma.
    """

    def __init__(
        self,
        latent_dim: int,
        action_bins: int,
        atoms: int,
        hidden_dim: int,
        sigma0: float,
    ) -> None:
        super().__init__()
        self.action_bins = action_bins
        self.atoms = atoms

        self.v1 = NoisyLinear(latent_dim, hidden_dim, sigma0=sigma0)
        self.v2 = NoisyLinear(hidden_dim, atoms, sigma0=sigma0)

        self.a1 = NoisyLinear(latent_dim, hidden_dim, sigma0=sigma0)
        self.a2 = NoisyLinear(hidden_dim, action_bins * atoms, sigma0=sigma0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, action_bins, atoms)`` logits."""
        v = self.v2(F.relu(self.v1(z)))  # (B, atoms)
        a = self.a2(F.relu(self.a1(z))).view(z.size(0), self.action_bins, self.atoms)
        return v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)  # (B, bins, atoms)


# ---------------------------------------------------------------------------
# RainbowDQN — full model
# ---------------------------------------------------------------------------


class RainbowDQN(nn.Module):
    """Rainbow DQN with branching for discretised continuous control.

    Combines: Dueling architecture, NoisyNets, and C51 distributional output.
    Uses the same :class:`Encoder` as DQNetwork for fair comparison.

    Args:
        in_channels: Number of stacked frames.
        action_bins: Discrete bins per action dimension.
        num_branches: Number of independent action dimensions.
        atoms: C51 number of atoms.
        vmin: Minimum return support value.
        vmax: Maximum return support value.
        noisy_sigma0: NoisyLinear initial sigma.
        head_hidden_dim: Hidden layer width in dueling streams.
    """

    def __init__(
        self,
        in_channels: int,
        action_bins: int,
        num_branches: int,
        atoms: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        noisy_sigma0: float = 0.5,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.action_bins = action_bins
        self.atoms = atoms

        self.encoder = Encoder(in_channels)

        self.heads = nn.ModuleList(
            [
                DuelingHead(
                    Encoder.OUT_FEATURES,
                    action_bins,
                    atoms,
                    head_hidden_dim,
                    noisy_sigma0,
                )
                for _ in range(num_branches)
            ]
        )

        self.register_buffer("support", torch.linspace(vmin, vmax, atoms))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with ``logits``, ``probs``, ``q`` — each ``(B, branches, bins, ...)``."""
        z = self.encoder(x)
        logits = torch.stack(
            [h(z) for h in self.heads], dim=1
        )  # (B, branches, bins, atoms)
        probs = F.softmax(logits, dim=-1)
        q = (probs * self.support).sum(dim=-1)  # (B, branches, bins)
        return {"logits": logits, "probs": probs, "q": q}

    def reset_noise(self) -> None:
        """Resample noise for all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
