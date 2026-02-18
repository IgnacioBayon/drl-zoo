# training algorightms file. To be implemented
import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

from src.dqn.models import PaperDQN


def prepare_run_dirs(cfg: DictConfig, run_dir: Path):
    # TODO: modify created folders
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cfg.paths.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    run_dir = Path(cfg.paths.work_dir)
    prepare_run_dirs(cfg, run_dir)

    rngs = nnx.Rngs(params=jax.random.PRNGKey(cfg.seed))
    model = PaperDQN(cfg=cfg.models, num_actions=6, rngs=rngs)

    x = jnp.zeros((32, 4, 84, 84, 3), dtype=jnp.float32)
    q = model(x)
    print(q.shape)  # (32, 6)


if __name__ == "__main__":
    main()
