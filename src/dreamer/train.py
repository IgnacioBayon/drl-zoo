import atexit
import pathlib
import warnings

import torch
from omegaconf import DictConfig

import src.dreamer.tools as tools
from src.dreamer.buffer import Buffer
from src.dreamer.dreamer import Dreamer
from src.dreamer.trainer import OnlineTrainer
from src.environment import build_from_config

warnings.filterwarnings("ignore")
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _resolve_device(device_cfg: object) -> torch.device:
    if str(device_cfg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_cfg))


def train_dreamer(config: DictConfig) -> None:
    train_cfg = config.train
    device = _resolve_device(
        train_cfg.device if "device" in train_cfg else config.device
    )
    config.device = str(device)
    train_cfg.device = str(device)
    tools.set_seed_everywhere(config.seed)
    if config.get("deterministic_run", False):
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.paths.run_dir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(train_cfg.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = build_from_config(
        config.env,
        train_cfg,
        mode="train",
    )

    print("Simulate agent.")
    agent = Dreamer(
        train_cfg.model,
        obs_space,
        act_space,
    ).to(device)

    policy_trainer = OnlineTrainer(
        train_cfg.trainer, replay_buffer, logger, logdir, train_envs, eval_envs
    )
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    raise SystemExit("Run Dreamer through src.main with train=dreamer")
