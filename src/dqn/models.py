from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def stack_to_channels(x: torch.Tensor) -> torch.Tensor:
    """
    Input:  (B, S, H, W, C)  (your current format)
    Output: (B, S*C, H, W)   (Torch Conv2d expects NCHW)
    """
    if x.ndim != 5:
        raise ValueError(f"Expected 5D (B,S,H,W,C), got shape {tuple(x.shape)}")

    b, s, h, w, c = x.shape
    # (B, S, H, W, C) -> (B, H, W, S, C) -> (B, H, W, S*C) -> (B, S*C, H, W)
    x = x.permute(0, 2, 3, 1, 4).contiguous().view(b, h, w, s * c)
    x = x.permute(0, 3, 1, 2).contiguous()
    # .contiguous() is often needed after permute before view/reshape
    # to ensure memory layout is correct for the new shape.
    return x


class EncoderDQN(nn.Module):
    """Torch-native encoder for DQN: conv stack + flatten + MLP stack."""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # ---- Conv stack ----
        in_ch = int(cfg.encoder.input_channels) * int(cfg.encoder.input_channels)
        conv_layers = []
        for layer in cfg.encoder.conv:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=int(layer.out_channels),
                    kernel_size=int(layer.kernel_size),
                    stride=int(layer.stride),
                    padding=int(layer.padding),
                )
            )
            # TODO: is there a ReLU in the original implementation?
            conv_layers.append(nn.ReLU())
            in_ch = int(layer.out_channels)
        self.conv = nn.Sequential(*conv_layers)

        # --- infer flatten dim --- E.g. not having to calculate 32 * 9 * 9
        with torch.no_grad():
            h = int(cfg.encoder.input_height)
            w = int(cfg.encoder.input_width)
            in_ch = int(cfg.encoder.input_channels)

            dummy_input = torch.zeros(1, in_ch, h, w)
            conv_out = self.conv(dummy_input)
            flat_dim = conv_out.view(1, -1).shape[1]

        # ---- MLP stack ----
        mlp_layers = []
        prev = flat_dim
        for layer in cfg.encoder.mlp:
            out = int(layer.out_features)
            mlp_layers.append(nn.Linear(prev, out))
            mlp_layers.append(nn.ReLU())
            prev = out
            last_out = out

        self.mlp = nn.Sequential(*mlp_layers)
        self.latent_dim = int(last_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = stack_to_channels(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)  # flatten all but batch dimension
        x = self.mlp(x)
        return x


# --- DQN ---


class DQN(nn.Module):
    """DQN implementation with a shared encoder and a simple linear head for Q-values."""

    def __init__(self, cfg: DictConfig, num_actions: int):
        super().__init__()

        self.encoder = EncoderDQN(cfg=cfg)
        self.head = nn.Linear(self.encoder.latent_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


# --- RAINBOW DQN ---


class NoisyLinear(nn.Module):
    """Torch-native NoisyLinear layer as described in "Noisy Networks for Exploration" (Fortunato et al., 2017)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float):
        super().__init__()

        # Learnable parameters for mean and sigma of weights and biases
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-0.1, 0.1),
        )
        self.bias_mu = nn.Parameter(
            torch.empty(out_features).uniform_(-0.1, 0.1),
        )

        init_sigma = sigma0 / (in_features**0.5)
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), init_sigma),
        )
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), init_sigma),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            b = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class DuelingHead(nn.Module):
    """
    Dueling DQN head that outputs (B, A, atoms) logits for the distributional version.
    Implemented as described in "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016).
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        atoms: int,
        hidden_dim: int,
        sigma0: float,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.atoms = atoms

        # Value stream: (B, atoms)
        self.v1 = NoisyLinear(latent_dim, hidden_dim, sigma0)
        self.v2 = NoisyLinear(hidden_dim, atoms, sigma0)

        # Advantage stream: (B, A*atoms) -> reshape to (B, A, atoms)
        self.a1 = NoisyLinear(latent_dim, hidden_dim, sigma0)
        self.a2 = NoisyLinear(hidden_dim, num_actions * atoms, sigma0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.v1(z))
        v = self.v2(v)  # (B, atoms)

        a = F.relu(self.a1(z))
        a = self.a2(a)  # (B, A*atoms)
        a = a.view(z.size(0), self.num_actions, self.atoms)  # (B, A, atoms)

        v = v.unsqueeze(1)  # (B, 1, atoms)
        logits = v + (a - a.mean(dim=1, keepdim=True))  # dueling combine
        return logits  # (B, A, atoms)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN implementation that combines all the model improvements: dueling architecture and noisy nets.
    Implemented as described in "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017).
    """

    def __init__(self, cfg: DictConfig, num_actions: int):
        super().__init__()

        self.encoder = EncoderDQN(cfg=cfg)

        atoms = int(cfg.rainbow.atoms)
        vmin = float(cfg.rainbow.vmin)
        vmax = float(cfg.rainbow.vmax)

        sigma0 = float(cfg.rainbow.noisy_init_sigma)
        hidden_dim = int(cfg.head.hidden_dim)

        self.head = DuelingHead(
            latent_dim=self.encoder.latent_dim,
            num_actions=num_actions,
            atoms=atoms,
            hidden_dim=hidden_dim,
            sigma0=sigma0,
        )

        # Support for distributional DQN  # TODO: ???
        self.register_buffer(
            "support",
            torch.linspace(vmin, vmax, atoms),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        logits = self.head(z)  # (B, A, atoms)
        probs = F.softmax(logits, dim=-1)  # (B, A, atoms)
        # Expected Q-values: sum over atoms of (probability * support)
        # probs has shape (B, A, atoms) and support has shape (atoms,),
        # so we need to align dimensions for broadcasting.
        q = torch.sum(probs * self.support.unsqueeze(0).unsqueeze(0), dim=-1)  # (B, A)
        return {"logits": logits, "probs": probs, "q": q}
