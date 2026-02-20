"""DQN training loop with branching Q-network for discretised continuous control."""

from __future__ import annotations

import math
import os
from time import perf_counter

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.environment import build_from_config
from src.utils import get_device

from .models import DQNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_and_record(
    policy: DQNetwork,
    step: int,
    env_cfg: DictConfig,
    train_resolution: tuple[int, int],
    video_dir: str,
    device: torch.device,
) -> None:
    """Run one episode at display resolution, save an MP4, and print the reward."""
    print(f"\n--- Testing & Recording Video at Step {step} ---")
    env = build_from_config(env_cfg, mode="eval")

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    done = False

    policy.eval()
    with torch.no_grad():
        while not done:
            frames.append(env.unwrapped.render())

            state_tensor = (
                torch.from_numpy(np.array(obs))
                .unsqueeze(0)
                .to(device, dtype=torch.float32)
                / 255.0
            )
            state_tensor = F.interpolate(
                state_tensor,
                size=train_resolution,
                mode="bilinear",
                align_corners=False,
            )
            logits = policy(state_tensor)
            action = logits.argmax(dim=2).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

    env.close()
    policy.train()

    os.makedirs(video_dir, exist_ok=True)
    filename = os.path.join(video_dir, f"hopper_step_{step}.mp4")
    imageio.mimsave(filename, frames, fps=30)
    print(f"Test Episode | Reward: {total_reward:.2f} | Video -> {filename}\n")


def _train_step(
    buffer: dict[str, torch.Tensor],
    policy: DQNetwork,
    target_policy: DQNetwork,
    optimizer: torch.optim.Optimizer,
    buf_filled: int,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    """Sample a mini-batch from the replay buffer and perform one gradient step."""
    indices = torch.randint(0, buf_filled, (batch_size,))

    states = buffer["obs"][indices].to(device, dtype=torch.float32) / 255.0
    next_states = buffer["next_obs"][indices].to(device, dtype=torch.float32) / 255.0
    actions = buffer["actions"][indices].to(device, dtype=torch.long)
    rewards = buffer["rewards"][indices].to(device, dtype=torch.float32)
    dones = buffer["dones"][indices].to(device, dtype=torch.float32)

    # Q(s, a)
    q_values = policy(states)  # (B, Branches, Bins)
    q_taken = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, Branches)

    # Target Q(s', a')
    with torch.no_grad():
        max_target_q = target_policy(next_states).max(dim=2).values  # (B, Branches)
        targets = rewards.unsqueeze(1) + gamma * max_target_q * (1 - dones.unsqueeze(1))

    loss = F.smooth_l1_loss(q_taken, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_loop(
    cfg: DictConfig,
    envs,
    policy: DQNetwork,
    target_policy: DQNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    """Run the ε-greedy collect → train → log loop.

    Args:
        cfg: Full Hydra config (reads ``cfg.env``, ``cfg.train``, ``cfg.paths``).
        envs: Vectorised gymnasium environment.
        policy: Online Q-network.
        target_policy: Target Q-network (hard-copied periodically).
        optimizer: Optimiser for *policy*.
        buffer: Pre-allocated CPU replay buffer dict.
        device: Torch device for training tensors.
    """
    tcfg = cfg.train
    ecfg = cfg.env
    num_envs: int = ecfg.num_envs
    train_resolution = tuple(ecfg.train_resolution)

    total_frames = int(tcfg.total_frames)
    steps = math.ceil(total_frames / num_envs)

    buf_idx = 0
    buf_filled = 0
    global_step = 0

    obs, _ = envs.reset()

    print(f"Running {steps} steps ({total_frames} frames) ...")
    start = perf_counter()

    for step in range(1, steps + 1):
        # -- epsilon-greedy action selection -----------------------------------
        epsilon = max(
            tcfg.epsilon_end,
            tcfg.epsilon_start
            - (tcfg.epsilon_start - tcfg.epsilon_end)
            * global_step
            / tcfg.epsilon_anneal_frames,
        )
        if np.random.rand() < epsilon:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device, dtype=torch.float32) / 255.0
                actions = policy(obs_t).argmax(dim=2).cpu().numpy()

        next_obs, rewards, terminations, truncations, _ = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        # -- store transition --------------------------------------------------
        end_idx = buf_idx + num_envs
        indices = np.arange(buf_idx, end_idx) % tcfg.buffer_size
        buffer["obs"][indices] = torch.from_numpy(np.array(obs))
        buffer["next_obs"][indices] = torch.from_numpy(np.array(next_obs))
        buffer["actions"][indices] = torch.from_numpy(actions.astype(np.uint8))
        buffer["rewards"][indices] = torch.from_numpy(rewards.astype(np.float32))
        buffer["dones"][indices] = torch.from_numpy(dones.astype(np.uint8))

        buf_idx = end_idx % tcfg.buffer_size
        buf_filled = min(buf_filled + num_envs, tcfg.buffer_size)
        obs = next_obs
        global_step += num_envs

        # -- gradient updates --------------------------------------------------
        if buf_filled >= tcfg.batch_size:
            if num_envs >= tcfg.train_every:
                for _ in range(num_envs // tcfg.train_every):
                    _train_step(
                        buffer,
                        policy,
                        target_policy,
                        optimizer,
                        buf_filled,
                        tcfg.batch_size,
                        tcfg.gamma,
                        device,
                    )
            elif global_step % (tcfg.train_every // num_envs) < num_envs:
                _train_step(
                    buffer,
                    policy,
                    target_policy,
                    optimizer,
                    buf_filled,
                    tcfg.batch_size,
                    tcfg.gamma,
                    device,
                )

        # -- hard target update ------------------------------------------------
        if global_step % tcfg.target_update_frames < num_envs:
            target_policy.load_state_dict(policy.state_dict())

        # -- periodic video logging --------------------------------------------
        if global_step % tcfg.video_log_frames < num_envs:
            _test_and_record(
                policy,
                global_step,
                ecfg,
                train_resolution,
                cfg.paths.video_dir,
                device,
            )

        # -- console log -------------------------------------------------------
        elapsed = perf_counter() - start
        fps = global_step / elapsed
        print(
            f"Step {step}/{steps} | Frames {global_step} | "
            f"eps {epsilon:.3f} | FPS {fps:.0f}",
            end="\r",
        )

    envs.close()
    elapsed = perf_counter() - start
    return elapsed, global_step / elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train_dqn(cfg: DictConfig) -> None:
    """Build envs, networks, buffer and launch the training loop.

    Args:
        cfg: Full Hydra config with ``env``, ``train``, ``model``, ``paths`` groups.
    """
    device = get_device(cfg.train.device)

    # -- environment -----------------------------------------------------------
    envs = build_from_config(cfg.env, mode="train")
    num_branches: int = (
        envs.single_action_space.shape[0]
        if hasattr(envs.single_action_space, "nvec")
        else 1
    )

    print(
        f"Training DQN on {cfg.env.name} with config:\n{cfg}\n"
        f"Device: {device} | Envs: {cfg.env.num_envs} | "
        f"Buffer: {cfg.train.buffer_size} | Batch: {cfg.train.batch_size} | "
        f"Branches: {num_branches}"
    )

    # -- networks & optimiser --------------------------------------------------
    policy = instantiate(cfg.model, num_branches=num_branches).to(device)
    target_policy = instantiate(cfg.model, num_branches=num_branches).to(device)
    target_policy.load_state_dict(policy.state_dict())

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr)

    # -- CPU replay buffer (uint8 to save RAM) ---------------------------------
    res_h, res_w = tuple(cfg.env.train_resolution)
    buf_sz = int(cfg.train.buffer_size)
    stk = cfg.env.stack_size
    buffer: dict[str, torch.Tensor] = {
        "obs": torch.zeros((buf_sz, stk, res_h, res_w), dtype=torch.uint8),
        "next_obs": torch.zeros((buf_sz, stk, res_h, res_w), dtype=torch.uint8),
        "actions": torch.zeros((buf_sz, num_branches), dtype=torch.uint8),
        "rewards": torch.zeros((buf_sz,), dtype=torch.float32),
        "dones": torch.zeros((buf_sz,), dtype=torch.uint8),
    }

    elapsed, avg_fps = _train_loop(
        cfg, envs, policy, target_policy, optimizer, buffer, device
    )
    print(f"\nDone -- {elapsed:.1f}s | avg FPS {avg_fps:.0f}")
