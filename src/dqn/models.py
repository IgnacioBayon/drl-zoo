"""Model architectures for DQN and Rainbow DQN with shared encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared encoder — identical architecture for both DQN and Rainbow
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """CNN encoder shared between DQN and Rainbow for fair comparison.

    Architecture: 3-layer CNN (Nature DQN) → flatten → FC 512.

    Args:
        in_channels: Number of input channels (typically ``stack_size``).
        in_resolution: Spatial resolution ``(H, W)`` of each frame.
    """

    OUT_FEATURES: int = 512

    def __init__(self, in_channels: int, in_resolution: tuple[int, ...]) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, in_channels, *in_resolution)).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_out, self.OUT_FEATURES), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# DQN — branching Q-network
# ---------------------------------------------------------------------------


class DQNetwork(nn.Module):
    """Branching DQN: shared encoder → one linear head per action branch.

    Args:
        in_channels: Number of stacked frames.
        in_resolution: ``(H, W)`` spatial resolution.
        action_bins: Discrete bins per action dimension.
        num_branches: Number of independent action dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        in_resolution: tuple[int, ...],
        action_bins: int,
        num_branches: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.action_bins = action_bins
        self.encoder = Encoder(in_channels, in_resolution)
        self.branches = nn.Linear(Encoder.OUT_FEATURES, num_branches * action_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values of shape ``(B, branches, bins)``."""
        z = self.encoder(x)
        return self.branches(z).view(-1, self.num_branches, self.action_bins)


# ---------------------------------------------------------------------------
# Rainbow DQN components
# ---------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """Factorised NoisyNet linear layer (Fortunato et al., 2017).

    Samples independent Gaussian noise per forward call during training;
    uses mean weights at eval time.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        sigma0: Initial noise standard deviation scaling factor.
    """

    def __init__(
        self, in_features: int, out_features: int, sigma0: float = 0.5
    ) -> None:
        super().__init__()
        bound = 1.0 / (in_features**0.5)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

        init_sigma = sigma0 / (in_features**0.5)
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), init_sigma)
        )
        self.bias_sigma = nn.Parameter(torch.full((out_features,), init_sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            b = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class DuelingHead(nn.Module):
    """Dueling distributional head for a single action branch (Wang et al., 2016).

    Outputs ``(B, action_bins, atoms)`` logits used for C51 distributional RL.

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

        # Value stream → (B, atoms)
        self.v1 = NoisyLinear(latent_dim, hidden_dim, sigma0)
        self.v2 = NoisyLinear(hidden_dim, atoms, sigma0)

        # Advantage stream → (B, action_bins * atoms)
        self.a1 = NoisyLinear(latent_dim, hidden_dim, sigma0)
        self.a2 = NoisyLinear(hidden_dim, action_bins * atoms, sigma0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, action_bins, atoms)`` logits."""
        v = self.v2(F.relu(self.v1(z)))  # (B, atoms)
        a = self.a2(F.relu(self.a1(z))).view(z.size(0), self.action_bins, self.atoms)
        return v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)  # (B, bins, atoms)


# ---------------------------------------------------------------------------
# Rainbow DQN — full model
# ---------------------------------------------------------------------------


class RainbowDQN(nn.Module):
    """Rainbow DQN with branching for discretised continuous control (Hessel et al., 2017).

    Combines: Dueling architecture, NoisyNets, and C51 distributional output.
    Uses the same :class:`Encoder` as :class:`DQNetwork` for fair comparison.

    Args:
        in_channels: Number of stacked frames.
        in_resolution: ``(H, W)`` spatial resolution.
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
        in_resolution: tuple[int, ...],
        action_bins: int,
        num_branches: int,
        atoms: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        noisy_sigma0: float = 0.5,
        head_hidden_dim: int = 256,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.action_bins = action_bins
        self.atoms = atoms

        self.encoder = Encoder(in_channels, in_resolution)

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
        """Returns dict with ``logits``, ``probs``, ``q`` — each ``(B, branches, bins, …)``."""
        z = self.encoder(x)
        # Each head outputs (B, bins, atoms)
        logits = torch.stack(
            [h(z) for h in self.heads], dim=1
        )  # (B, branches, bins, atoms)
        probs = F.softmax(logits, dim=-1)
        q = (probs * self.support).sum(dim=-1)  # (B, branches, bins)
        return {"logits": logits, "probs": probs, "q": q}
