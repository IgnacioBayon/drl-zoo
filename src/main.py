from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

from src.dqn.models import DQN, RainbowDQN
from src.utils import prepare_run_dirs


@hydra.main(version_base=None, config_path="pkg://config", config_name="config")
def main(cfg: DictConfig):
    run_dir = Path(cfg.paths.work_dir)
    prepare_run_dirs(cfg, run_dir)

    rngs = nnx.Rngs(params=jax.random.PRNGKey(cfg.seed))
    dqn = DQN(cfg.dqn, num_actions=6, rngs=rngs)
    rainbow = RainbowDQN(cfg.rainbow, num_actions=6, rngs=rngs)

    x = jnp.zeros((32, 4, 84, 84, 3), dtype=jnp.float32)
    q = dqn(x)
    q_rainbow = rainbow(x)
    print(q.shape)  # (32, 6)
    print(q_rainbow["q"].shape)  # (32, 6)


if __name__ == "__main__":
    main()
