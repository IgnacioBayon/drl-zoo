import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

# import hydra
# @hydra.main(version_base=None, config_path="config/algorithms", config_name="dqn")
# def main(cfg: DictConfig) -> None:
#     rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
#     model = DQN(cfg=cfg, num_actions=6, rngs=rngs)

#     x = jnp.zeros((32, 4, 84, 84, 3), dtype=jnp.float32)
#     q = model(x)
#     print(q.shape)


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


class PaperDQN(nnx.Module):
    def __init__(self, cfg: DictConfig, num_actions: int, *, rngs: nnx.Rngs):
        self.num_actions = num_actions

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

        # HEAD stack
        self.head = nnx.Linear(in_features=out, out_features=num_actions, rngs=rngs)

    def __call__(self, x):
        x = stack_to_channels(x)

        for conv in self.convs:
            x = nnx.relu(conv(x))

        x = x.reshape((x.shape[0], -1))

        for fc in self.fcs:
            x = nnx.relu(fc(x))

        return self.head(x)


# --- Minimal usage example ---
@hydra.main(version_base=None, config_path="../../config/models", config_name="dqn")
def main(cfg: DictConfig):
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    model = PaperDQN(cfg, num_actions=6, rngs=rngs)

    x = jnp.zeros((32, 4, 84, 84, 3), dtype=jnp.float32)
    q = model(x)
    print(q.shape)  # (32, 6)


if __name__ == "__main__":
    main()
