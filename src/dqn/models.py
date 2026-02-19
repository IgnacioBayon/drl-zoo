import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig


def stack_to_channels(x: jnp.ndarray) -> jnp.ndarray:
    """
    Stacks the S (stacked frames) dimension into channels.

    Args:
      x: (B, S, H, W, C) input tensor
    Returns:
      (B, H, W, S*C) tensor suitable for convolution
    """
    B, S, H, W, C = x.shape
    return x.transpose(0, 2, 3, 1, 4).reshape(B, H, W, S * C)


class EncoderDQN(nnx.Module):
    """Shared Encoder for both DQN and Rainbow DQN. Configurable conv + MLP stack."""

    def __init__(self, cfg: DictConfig, *, rngs: nnx.Rngs):
        # ---- Conv stack ----
        in_ch = int(cfg.encoder.in_features)
        self.convs: nnx.List[nnx.Conv] = nnx.List()
        for layer_cfg in cfg.encoder.conv_layers:
            out_ch = int(layer_cfg.out_features)
            self.convs.append(
                nnx.Conv(
                    in_features=in_ch,
                    out_features=out_ch,
                    kernel_size=(layer_cfg.kernel_size, layer_cfg.kernel_size),
                    strides=(layer_cfg.strides, layer_cfg.strides),
                    padding="VALID",
                    rngs=rngs,
                )
            )
            in_ch = out_ch

        # ---- MLP stack ----
        self.fcs: nnx.List[nnx.Linear] = nnx.List()
        for layer_cfg in cfg.encoder.fc_layers:
            fc_in = int(layer_cfg.in_features)
            out = int(layer_cfg.out_features)
            self.fcs.append(nnx.Linear(in_features=fc_in, out_features=out, rngs=rngs))

    def __call__(self, x):
        x = stack_to_channels(x)

        for conv in self.convs:
            x = nnx.relu(conv(x))

        x = x.reshape((x.shape[0], -1))

        for fc in self.fcs:
            x = nnx.relu(fc(x))

        return x


class DQN(nnx.Module):
    """DQN implementation with a shared encoder and a simple linear head for Q-values."""

    def __init__(self, cfg: DictConfig, num_actions: int, *, rngs: nnx.Rngs):
        self.encoder = EncoderDQN(cfg=cfg, rngs=rngs)
        self.head = nnx.Linear(
            in_features=int(cfg.encoder.fc_layers[-1].out_features),
            out_features=num_actions,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.encoder(x)
        return self.head(x)


# --- RAINBOW DQN ---


class NoisyLinear(nnx.Module):
    """Noisy linear layer as described in "Noisy Networks for Exploration" (Fortunato et al., 2017)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        noisy_init_sigma: float,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs

        # Learnable parameters
        self.weight_mu = nnx.Param(
            jax.random.uniform(
                rngs.params(), (out_features, in_features), minval=-0.1, maxval=0.1
            )
        )
        self.bias_mu = nnx.Param(
            jax.random.uniform(rngs.params(), (out_features,), minval=-0.1, maxval=0.1)
        )

        init_sigma = noisy_init_sigma / jnp.sqrt(in_features)
        self.weight_sigma = nnx.Param(jnp.full((out_features, in_features), init_sigma))
        self.bias_sigma = nnx.Param(jnp.full((out_features,), init_sigma))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight_noise = jax.random.normal(self.rngs.params(), self.weight_mu.shape)
        bias_noise = jax.random.normal(self.rngs.params(), self.bias_mu.shape)

        # Create noisy weights and biases
        weight = self.weight_mu + self.weight_sigma * weight_noise
        bias = self.bias_mu + self.bias_sigma * bias_noise

        return x @ weight.T + bias


class DuelingHead(nnx.Module):
    """
    Dueling DQN head that outputs (B, A, atoms) logits for the distributional version.
    Implemented as described in "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016).
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        atoms: int,
        noisy_init_sigma: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.atoms = atoms

        hidden = latent_dim  # you can make this configurable

        # Value stream: (B, atoms)
        self.v1 = NoisyLinear(latent_dim, hidden, noisy_init_sigma, rngs=rngs)
        self.v2 = NoisyLinear(hidden, atoms, noisy_init_sigma, rngs=rngs)

        # Advantage stream: (B, A*atoms) -> reshape to (B, A, atoms)
        self.a1 = NoisyLinear(latent_dim, hidden, noisy_init_sigma, rngs=rngs)
        self.a2 = NoisyLinear(hidden, num_actions * atoms, noisy_init_sigma, rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        v = nnx.relu(self.v1(z))
        v = self.v2(v)  # (B, atoms)

        a = nnx.relu(self.a1(z))
        a = self.a2(a)  # (B, A*atoms)
        a = a.reshape((a.shape[0], self.num_actions, self.atoms))  # (B, A, atoms)

        v = v[:, None, :]  # (B, 1, atoms)
        logits = v + (a - a.mean(axis=1, keepdims=True))  # dueling combine
        return logits  # (B, A, atoms)


class RainbowDQN(nnx.Module):
    """
    Rainbow DQN implementation that combines all the model improvements: dueling architecture and noisy nets.
    Implemented as described in "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017).
    """

    def __init__(self, cfg: DictConfig, num_actions: int, *, rngs: nnx.Rngs):
        self.encoder = EncoderDQN(cfg=cfg, rngs=rngs)
        latent_dim = int(cfg.encoder.fc_layers[-1].out_features)

        self.head = DuelingHead(
            latent_dim, num_actions, cfg.atoms, cfg.noisy_init_sigma, rngs=rngs
        )
        self.support = jnp.linspace(cfg.vmin, cfg.vmax, cfg.atoms, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray):
        z = self.encoder(x)
        logits = self.head(z)  # (B, A, atoms)

        probs = jax.nn.softmax(logits, axis=-1)
        q = jnp.sum(probs * self.support[None, None, :], axis=-1)  # (B, A)

        return {"logits": logits, "q": q}
