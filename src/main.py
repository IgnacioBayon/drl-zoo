import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.dqn.train_dqn import train_dqn

_TRAINERS = {
    "dqn": train_dqn,
    # "rainbow": train_rainbow,  # TODO: implement
}


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point: seed all RNGs and dispatch to the right trainer."""
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name: str = cfg.model.name
    trainer = _TRAINERS.get(model_name)
    if trainer is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(_TRAINERS)}"
        )
    trainer(cfg)


if __name__ == "__main__":
    main()
