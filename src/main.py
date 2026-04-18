import random

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.dqn.train import train_dqn
from src.rainbow.train import train_rainbow
from src.sac.train import train_sac
from src.ppo.train import train_ppo
from src.dreamer.train import train_dreamer

_TRAINERS = {
    "dqn": train_dqn,
    "rainbow": train_rainbow,
    "sac": train_sac,
    "ppo": train_ppo,
    "dreamer": train_dreamer,
}


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point: seed all RNGs and dispatch to the right trainer."""
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    algorithm: str = HydraConfig.get().runtime.choices["train"]
    trainer = _TRAINERS.get(algorithm)
    if trainer is None:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: {list(_TRAINERS)}"
        )
    trainer(cfg)


if __name__ == "__main__":
    main()
