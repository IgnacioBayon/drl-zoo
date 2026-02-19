from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.dqn.models import DQN, RainbowDQN
from src.utils import prepare_run_dirs


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


@hydra.main(version_base="1.3", config_path="pkg://config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir: Path = prepare_run_dirs(cfg)
    print("Run dir:", run_dir)

    device = get_device(str(cfg.train.device))
    torch.manual_seed(int(cfg.seed))

    num_actions = 6

    # Smoke test input (same shape as before): (B, S, H, W, C)
    x = torch.zeros((32, 4, 84, 84, 3), dtype=torch.float32, device=device)

    if cfg.model.name == "dqn":
        model = DQN(cfg.model, num_actions=num_actions).to(device)
        q = model(x)
        print("DQN q:", tuple(q.shape))  # (32, 6)

    elif cfg.model.name == "rainbow":
        model = RainbowDQN(cfg.model, num_actions=num_actions).to(device)
        out = model(x)
        print("Rainbow logits:", tuple(out["logits"].shape))  # (32, 6, atoms)
        print("Rainbow q:", tuple(out["q"].shape))            # (32, 6)

    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.model.name} params: {n_params:,}")


if __name__ == "__main__":
    main()
