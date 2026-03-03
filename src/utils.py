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
    env: object | None = None,
) -> tuple[float, float, float, list[np.ndarray]]:
    """Run one greedy episode and optionally capture full-resolution frames.

    Recording grabs 480×480 RGB frames via ``env.render()`` (the
    underlying MuJoCo renderer) each step.  The agent still receives the
    downscaled grayscale observations produced by ``ImageObsWrapper``.

    Args:
        env: Optional pre-built eval env to reuse. When provided the env
            is **not** closed on return — the caller owns its lifecycle.

    Returns:
        total_reward, final_torso_x, final_com_x, frames.
    """
    owns_env = env is None
    if owns_env:
        env = build_from_config(env_cfg, mode="eval")

    obs, _ = env.reset(seed=seed)  # type: ignore[union-attr]
    frames: list[np.ndarray] = []
    total_reward = 0.0
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
            done = terminated or truncated

    mj_data = env.unwrapped.data  # type: ignore[union-attr]
    mj_model = env.unwrapped.model  # type: ignore[union-attr]
    torso_id: int = mj_model.body("torso").id
    final_torso_x = float(mj_data.qpos[0])
    final_com_x = float(mj_data.subtree_com[torso_id][0])

    if owns_env:
        env.close()  # type: ignore[union-attr]
    return total_reward, final_torso_x, final_com_x, frames


def evaluate_and_record(
    policy: nn.Module,
    action_fn: ActionFn,
    step: int,
    env_cfg: DictConfig,
    video_dir: str,
    device: torch.device,
    writer: SummaryWriter,
    n_episodes: int,
    best_mean_reward: float = float("-inf"),
) -> tuple[float, float, float, str]:
    """Run *n_episodes* greedy evaluations; conditionally record a video.

    Toggles the *policy* to eval mode (disabling NoisyNets / dropout) and
    restores train mode before returning.

    Returns:
        mean, std, max of episode rewards and the path to the saved video
        (empty string when no improvement).
    """
    policy.eval()

    # Reuse a single eval env so the rendering context (OpenGL state) is
    # identical across episodes and the optional re-recording pass.
    eval_env = build_from_config(env_cfg, mode="eval")

    returns = np.empty(n_episodes, dtype=np.float64)
    final_torso_xs = np.empty(n_episodes, dtype=np.float64)
    final_com_xs = np.empty(n_episodes, dtype=np.float64)
    for i in range(n_episodes):
        returns[i], final_torso_xs[i], final_com_xs[i], _ = run_eval_episode(
            action_fn,
            env_cfg,
            device,
            record=False,
            seed=int(step) + i,
            env=eval_env,
        )

    mean_r, std_r = float(returns.mean()), float(returns.std())
    best_idx = int(returns.argmax())
    max_r = float(returns[best_idx])

    filename = ""
    if mean_r > best_mean_reward:
        reward, _, _, frames = run_eval_episode(
            action_fn,
            env_cfg,
            device,
            record=True,
            seed=int(step) + best_idx,
            env=eval_env,
        )
        if abs(reward - max_r) > 0.5:
            log.warning(
                "Re-eval reward %.2f differs from first-pass max %.2f "
                "(possible rendering non-determinism)",
                reward,
                max_r,
            )
        os.makedirs(video_dir, exist_ok=True)
        filename = os.path.join(video_dir, f"eval_step_{step}_reward_{reward:.2f}.mp4")
        imageio.mimsave(filename, frames, fps=30)

    eval_env.close()
    policy.train()

    writer.add_scalar("eval/mean_reward", mean_r, step)
    writer.add_scalar("eval/std_reward", std_r, step)
    writer.add_scalar("eval/max_reward", max_r, step)
    writer.add_scalars(
        "eval/final_x_position",
        {"torso": float(final_torso_xs.mean()), "com": float(final_com_xs.mean())},
        step,
    )
    return mean_r, std_r, max_r, filename
