"""Shared training utilities: checkpointing, evaluation, and recording.

These helpers are algorithm-agnostic — they accept a generic ``action_fn``
callable so they work with DQN, Rainbow, and future PPO / SAC trainers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import imageio
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config

log = logging.getLogger(__name__)

# obs tensor (1, C, H, W) on device → action numpy (n_joints,)
ActionFn = Callable[[torch.Tensor], np.ndarray]


# ---------------------------------------------------------------------------
# Device / loss helpers
# ---------------------------------------------------------------------------


def get_device(device_cfg: str) -> torch.device:
    """Return the appropriate torch device based on a config string."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def get_loss_fn(loss_fn_cfg: str) -> nn.Module:
    """Return the appropriate loss function based on a config string."""
    if loss_fn_cfg == "mse":
        return nn.MSELoss()
    elif loss_fn_cfg == "huber":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss function '{loss_fn_cfg}'. Choose from: mse, huber")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    checkpoint_dir: str,
    name: str = "",
) -> None:
    """Save model and optimizer state to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = name or f"ckpt_{global_step}.pt"
    torch.save(
        {
            "global_step": global_step,
            "policy": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(checkpoint_dir, filename),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_eval_episode(
    action_fn: ActionFn,
    env_cfg: DictConfig,
    device: torch.device,
    record: bool = False,
    seed: int | None = None,
    discretize_actions: bool = True,
    env: object | None = None,
) -> tuple[float, float, int, list[np.ndarray]]:
    """Run one greedy episode and optionally capture full-resolution frames.

    Args:
        env: Optional pre-built eval env to reuse. When provided the env
            is **not** closed on return — the caller owns its lifecycle.

    Returns:
        total_reward, final_x, episode_steps, frames.
    """
    owns_env = env is None
    if owns_env:
        env = build_from_config(
            env_cfg,
            mode="eval",
            discretize_actions=discretize_actions,
        )

    obs, _ = env.reset(seed=seed)  # type: ignore[union-attr]
    frames: list[np.ndarray] = []
    total_reward = 0.0
    ep_steps = 0
    done = False

    with torch.no_grad():
        while not done:
            if record:
                frames.append(env.render())  # type: ignore[union-attr]

            state_t = (
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                / 255.0
            )
            action = action_fn(state_t)
            obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore[union-attr]
            total_reward += float(reward)
            ep_steps += 1
            done = terminated or truncated

    final_x = float(env.unwrapped.data.qpos[0])  # type: ignore[union-attr]

    if owns_env:
        env.close()  # type: ignore[union-attr]
    return total_reward, final_x, ep_steps, frames


def evaluate_and_record(
    policy: nn.Module,
    action_fn: ActionFn,
    step: int,
    env_cfg: DictConfig,
    video_dir: str,
    device: torch.device,
    writer: SummaryWriter,
    n_episodes: int,
    discretize_actions: bool = True,
) -> tuple[float, float, float]:
    """Run *n_episodes* greedy evaluations and record the best episode.

    Toggles the *policy* to eval mode (disabling NoisyNets / dropout) and
    restores train mode before returning.

    Returns:
        mean_reward, std_reward, mean_final_x.
    """
    policy.eval()

    eval_env = build_from_config(
        env_cfg,
        mode="eval",
        discretize_actions=discretize_actions,
    )

    returns = np.empty(n_episodes, dtype=np.float64)
    final_xs = np.empty(n_episodes, dtype=np.float64)
    ep_steps = np.empty(n_episodes, dtype=np.int64)
    for i in range(n_episodes):
        returns[i], final_xs[i], ep_steps[i], _ = run_eval_episode(
            action_fn,
            env_cfg,
            device,
            record=False,
            seed=int(step) + i,
            discretize_actions=discretize_actions,
            env=eval_env,
        )

    mean_r, std_r = float(returns.mean()), float(returns.std())
    best_idx = int(returns.argmax())

    # Always record the best episode
    _, _, _, frames = run_eval_episode(
        action_fn,
        env_cfg,
        device,
        record=True,
        seed=int(step) + best_idx,
        discretize_actions=discretize_actions,
        env=eval_env,
    )
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimsave(os.path.join(video_dir, f"eval_step_{step}.mp4"), frames, fps=30)

    eval_env.close()
    policy.train()

    writer.add_scalar("eval/mean_reward", mean_r, step)
    writer.add_scalar("eval/final_x", float(final_xs.mean()), step)
    avg_speeds = final_xs / (ep_steps * 0.008)  # dt = 0.008
    writer.add_scalar("eval/avg_speed", float(avg_speeds.mean()), step)
    return mean_r, std_r, float(final_xs.mean())
