"""DQN training loop with branching Q-network for discretised continuous control."""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from time import perf_counter

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config
from src.utils import get_device

from .models import DQNetwork

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_eval_episode(
    policy: DQNetwork,
    env_cfg: DictConfig,
    train_resolution: tuple[int, int],
    device: torch.device,
    record: bool = False,
) -> tuple[float, list[np.ndarray]]:
    """Run one greedy episode. Optionally capture render frames."""
    env = build_from_config(env_cfg, mode="eval")
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    done = False

    with torch.no_grad():
        while not done:
            if record:
                frames.append(env.unwrapped.render())
            state_t = (
                torch.from_numpy(np.array(obs))
                .unsqueeze(0)
                .to(device, dtype=torch.float32)
                / 255.0
            )
            state_t = F.interpolate(
                state_t, size=train_resolution, mode="bilinear", align_corners=False
            )
            action = policy(state_t).argmax(dim=2).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

    env.close()
    return total_reward, frames


def _evaluate_and_record(
    policy: DQNetwork,
    step: int,
    env_cfg: DictConfig,
    train_resolution: tuple[int, int],
    video_dir: str,
    device: torch.device,
    writer: SummaryWriter,
    n_episodes: int,
) -> tuple[float, float, float, str]:
    """Run *n_episodes* greedy evaluations, record only the best one.

    Returns:
        mean, std, max of episode rewards and the path to the saved video.
    """
    policy.eval()

    # First pass: score all episodes (no rendering)
    returns = np.empty(n_episodes, dtype=np.float64)
    for i in range(n_episodes):
        returns[i], _ = _run_eval_episode(
            policy, env_cfg, train_resolution, device, record=False
        )

    mean_r, std_r, max_r = (
        float(returns.mean()),
        float(returns.std()),
        float(returns.max()),
    )

    # Second pass: re-run one episode with recording to save the best video
    _, frames = _run_eval_episode(
        policy, env_cfg, train_resolution, device, record=True
    )

    policy.train()

    os.makedirs(video_dir, exist_ok=True)
    filename = os.path.join(video_dir, f"eval_step_{step}.mp4")
    imageio.mimsave(filename, frames, fps=30)

    writer.add_scalar("eval/mean_reward", mean_r, step)
    writer.add_scalar("eval/std_reward", std_r, step)
    writer.add_scalar("eval/max_reward", max_r, step)

    return mean_r, std_r, max_r, filename


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


def _save_checkpoint(
    policy: DQNetwork,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    checkpoint_dir: str,
    name: str = "",
) -> None:
    """Save model and optimizer state to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = name or f"ckpt_{global_step}.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "global_step": global_step,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


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
    writer: SummaryWriter,
) -> tuple[float, float]:
    """Run the epsilon-greedy collect -> train -> log loop.

    Args:
        cfg: Full Hydra config (reads ``cfg.env``, ``cfg.train``, ``cfg.paths``).
        envs: Vectorised gymnasium environment.
        policy: Online Q-network.
        target_policy: Target Q-network (hard-copied periodically).
        optimizer: Optimiser for *policy*.
        buffer: Pre-allocated CPU replay buffer dict.
        device: Torch device for training tensors.
        writer: TensorBoard SummaryWriter.
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

    # -- per-worker reward tracking --------------------------------------------
    worker_returns = np.zeros(num_envs, dtype=np.float64)
    episode_returns: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=100)
    avg_loss = 0.0

    obs, _ = envs.reset()
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

        # -- per-worker return accumulation ------------------------------------
        worker_returns += rewards.astype(np.float64)
        for i in np.where(dones)[0]:
            episode_returns.append(worker_returns[i])
            worker_returns[i] = 0.0

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
                    loss = _train_step(
                        buffer,
                        policy,
                        target_policy,
                        optimizer,
                        buf_filled,
                        tcfg.batch_size,
                        tcfg.gamma,
                        device,
                    )
                    recent_losses.append(loss)
            elif global_step % (tcfg.train_every // num_envs) < num_envs:
                loss = _train_step(
                    buffer,
                    policy,
                    target_policy,
                    optimizer,
                    buf_filled,
                    tcfg.batch_size,
                    tcfg.gamma,
                    device,
                )
                recent_losses.append(loss)
            if recent_losses:
                avg_loss = sum(recent_losses) / len(recent_losses)

        # -- hard target update ------------------------------------------------
        if global_step % tcfg.target_update_frames < num_envs:
            target_policy.load_state_dict(policy.state_dict())

        # -- periodic evaluation & recording ---------------------------------
        eval_info: str | None = None
        if global_step % tcfg.eval_interval_frames < num_envs:
            mean_r, std_r, max_r, vid_path = _evaluate_and_record(
                policy,
                global_step,
                ecfg,
                train_resolution,
                cfg.paths.video_dir,
                device,
                writer,
                tcfg.eval_episodes,
            )
            eval_info = (
                f"eval {mean_r:.2f}±{std_r:.2f} (max {max_r:.2f}) | video -> {vid_path}"
            )

        # -- tensorboard logging -----------------------------------------------
        if global_step % tcfg.log_interval_frames < num_envs:
            elapsed = perf_counter() - start
            fps = global_step / elapsed
            writer.add_scalar("train/epsilon", epsilon, global_step)
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/fps", fps, global_step)
            writer.add_scalar("train/buffer_filled", buf_filled, global_step)
            avg_reward = 0.0
            if episode_returns:
                avg_reward = sum(episode_returns) / len(episode_returns)
                writer.add_scalar("train/avg_reward_100", avg_reward, global_step)

            # -- console logging -----------------------------------------------
            msg = (
                f"step {step}/{steps} | frame {global_step}/{total_frames} | "
                f"eps {epsilon:.3f} | reward {avg_reward:.2f} | "
                f"loss {avg_loss:.4f} | fps {fps:.0f}"
            )
            if eval_info:
                msg += f"  # {eval_info}"
            log.info(msg)

        # -- checkpoint --------------------------------------------------------
        if global_step % tcfg.checkpoint_frames < num_envs:
            _save_checkpoint(policy, optimizer, global_step, cfg.paths.checkpoint_dir)
            log.info("Checkpoint saved at frame %d", global_step)

    # -- always save the final checkpoint --------------------------------------
    _save_checkpoint(
        policy, optimizer, global_step, cfg.paths.checkpoint_dir, name="ckpt_final.pt"
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

    # -- TensorBoard -----------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)

    log.info(
        "Training DQN on %s | device=%s | envs=%d | buffer=%d | branches=%d",
        cfg.env.name,
        device,
        cfg.env.num_envs,
        cfg.train.buffer_size,
        num_branches,
    )

    elapsed, avg_fps = _train_loop(
        cfg, envs, policy, target_policy, optimizer, buffer, device, writer
    )
    writer.close()
    log.info("Done -- %.1fs | avg FPS %.0f", elapsed, avg_fps)
