import jax
import jax.numpy as jnp
from flax import nnx


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


class DQNEncoder(nnx.Module):
    def __init__(self, in_features: int, num_actions: int, *, rngs: nnx.Rngs):
        self.num_actions = num_actions

        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=16,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=16,
            out_features=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
        )

        self.fc = nnx.Linear(in_features=32 * 9 * 9, out_features=256, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1(x)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x = jax.nn.relu(x)

        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc(x)
        x = jax.nn.relu(x)

        return x


class DQN(nnx.Module):
    def __init__(self, num_actions: int, *, rngs: nnx.Rngs):

        self.encoder = DQNEncoder(in_features=12, num_actions=num_actions, rngs=rngs)

        self.head = nnx.Linear(in_features=256, out_features=num_actions, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            - x: (B, S, H, W, C) input tensor
        Returns:
            - Q-values: (B, num_actions)
        """
        x = stack_to_channels(x)  # -> (B,H,W,S*C)

        x = x.astype(jnp.float32)

        x = self.encoder(x)
        q = self.head(x)
        return q


# --- Minimal usage example ---
if __name__ == "__main__":
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    model = DQN(num_actions=6, rngs=rngs)

    x = jnp.zeros((32, 4, 84, 84, 3), dtype=jnp.float32)
    q = model(x)
    print(q.shape)  # (32, 6)
