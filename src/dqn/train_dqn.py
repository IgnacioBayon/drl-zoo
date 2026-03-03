"""DQN training loop with branching Q-network for discretised continuous control."""

from __future__ import annotations

import logging
import math
from collections import deque
from time import perf_counter

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config
from src.utils import evaluate_and_record, get_device, get_loss_fn, save_checkpoint

from .model import DQNetwork

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_step(
    buffer: dict[str, torch.Tensor],
    policy: DQNetwork,
    target_policy: DQNetwork,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    buf_filled: int,
    batch_size: int,
    gamma: float,
    max_grad_norm: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample a mini-batch from the replay buffer and perform one gradient step.

    Returns:
        loss: Detached scalar loss tensor (no GPU sync — caller decides when to read).
    """
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

    loss = loss_fn(q_taken, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_loop(
    cfg: DictConfig,
    envs,
    policy: DQNetwork,
    target_policy: DQNetwork,
    loss_fn: torch.nn.Module,
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
    multidiscrete: bool = ecfg.action_multidiscrete

    total_frames = int(tcfg.total_frames)
    steps = math.ceil(total_frames / num_envs)

    buf_idx = 0
    buf_filled = 0
    global_step = 0

    # -- per-worker reward tracking --------------------------------------------
    worker_returns = np.zeros(num_envs, dtype=np.float64)
    episode_returns: deque[float] = deque(maxlen=100)
    reward_window_sum = 0.0
    # Accumulate detached loss tensors between log events; sync once per interval.
    step_losses: list[torch.Tensor] = []
    avg_loss = 0.0
    best_mean_reward = float("-inf")

    obs, _ = envs.reset(seed=int(cfg.seed))
    start = perf_counter()

    start_train_after: int = int(tcfg.start_train_after)

    for step in range(1, steps + 1):
        # -- epsilon-greedy action selection -----------------------------------
        # Epsilon stays at epsilon_start during warm-up (pure random collection)
        frames_since_train = max(0, global_step - start_train_after)
        epsilon = max(
            tcfg.epsilon_end,
            tcfg.epsilon_start
            - (tcfg.epsilon_start - tcfg.epsilon_end)
            * frames_since_train
            / tcfg.epsilon_anneal_frames,
        )
        if np.random.rand() < epsilon:
            actions = envs.action_space.sample()
            if not multidiscrete:
                actions = actions[:, np.newaxis]  # (E,) → (E, 1)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
                actions = policy(obs_t).argmax(dim=2).cpu().numpy()

        # Env expects (E,) scalars for Discrete, (E, branches) for MultiDiscrete
        actions_env = actions if multidiscrete else actions.squeeze(-1)
        next_obs, rewards, terminations, truncations, _ = envs.step(actions_env)
        dones = np.logical_or(terminations, truncations)

        # -- per-worker return accumulation ------------------------------------
        worker_returns += rewards.astype(np.float64)
        for i in np.where(dones)[0]:
            if len(episode_returns) == episode_returns.maxlen:
                reward_window_sum -= episode_returns[0]
            ret = float(worker_returns[i])
            reward_window_sum += ret
            episode_returns.append(ret)
            worker_returns[i] = 0.0

        # -- store transition --------------------------------------------------
        # With gymnasium 1.2+ NEXT_STEP auto-reset, next_obs[i] is the *final*
        # observation when dones[i]=True (reset happens on the next step call).
        # Storing next_obs explicitly avoids cross-episode corruption.
        indices = np.arange(buf_idx, buf_idx + num_envs) % tcfg.buffer_size
        buffer["obs"][indices] = torch.as_tensor(obs)
        buffer["next_obs"][indices] = torch.as_tensor(next_obs)
        buffer["actions"][indices] = torch.as_tensor(actions, dtype=torch.uint8)
        buffer["rewards"][indices] = torch.as_tensor(rewards, dtype=torch.float32)
        buffer["dones"][indices] = torch.as_tensor(terminations, dtype=torch.uint8)

        buf_idx = (buf_idx + num_envs) % tcfg.buffer_size
        buf_filled = min(buf_filled + num_envs, tcfg.buffer_size)
        obs = next_obs
        prev_step = global_step
        global_step += num_envs

        # -- gradient updates --------------------------------------------------
        if num_envs >= tcfg.train_every:
            n_updates = num_envs // tcfg.train_every
        else:
            n_updates = int(
                global_step // tcfg.train_every != prev_step // tcfg.train_every
            )

        if buf_filled >= max(start_train_after, tcfg.batch_size) and n_updates > 0:
            for _ in range(n_updates):
                step_losses.append(
                    _train_step(
                        buffer,
                        policy,
                        target_policy,
                        optimizer,
                        loss_fn,
                        buf_filled,
                        tcfg.batch_size,
                        tcfg.gamma,
                        tcfg.max_grad_norm,
                        device,
                    )
                )

        # -- target update ------------------------------------------------
        if (
            global_step // tcfg.target_update_frames
            != prev_step // tcfg.target_update_frames
        ):
            target_policy.load_state_dict(policy.state_dict())

        # -- periodic evaluation & conditional best-save --------------------
        eval_info: str | None = None
        if (
            global_step // tcfg.eval_interval_frames
            != prev_step // tcfg.eval_interval_frames
        ):

            def action_fn(obs_t: torch.Tensor) -> np.ndarray | int:
                act = policy(obs_t).argmax(dim=2).squeeze(0).cpu().numpy()
                return act if multidiscrete else int(act.item())

            mean_r, std_r, max_r, _ = evaluate_and_record(
                policy,
                action_fn,
                global_step,
                ecfg,
                cfg.paths.video_dir,
                device,
                writer,
                tcfg.eval_episodes,
                best_mean_reward,
            )
            eval_info = f"eval {mean_r:.2f}±{std_r:.2f} (max {max_r:.2f})"

            # Save a checkpoint whenever we achieve a new best mean reward.
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                save_checkpoint(
                    policy,
                    optimizer,
                    global_step,
                    cfg.paths.checkpoint_dir,
                    name=f"ckpt_best_mean_reward_{mean_r:.2f}.pt",
                )

        # -- tensorboard logging -----------------------------------------------
        if (
            global_step // tcfg.log_interval_frames
            != prev_step // tcfg.log_interval_frames
        ):
            # Sync all accumulated loss tensors in one shot instead of per-step.
            if step_losses:
                avg_loss = torch.stack(step_losses).mean().item()
                step_losses.clear()
            elapsed = perf_counter() - start
            fps = global_step / elapsed
            writer.add_scalar("train/epsilon", epsilon, global_step)
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/fps", fps, global_step)
            writer.add_scalar("train/buffer_filled", buf_filled, global_step)
            avg_reward = 0.0
            if episode_returns:
                avg_reward = reward_window_sum / len(episode_returns)
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

    # -- always save the final checkpoint --------------------------------------
    save_checkpoint(
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
    multidiscrete: bool = cfg.env.action_multidiscrete
    if multidiscrete:
        num_branches = envs.single_action_space.shape[0]
        action_bins = int(cfg.env.action_bins)
    else:
        num_branches = 1
        action_bins = int(envs.single_action_space.n)  # bins ** num_joints

    # -- networks & optimiser --------------------------------------------------
    policy = instantiate(
        cfg.model, num_branches=num_branches, action_bins=action_bins
    ).to(device)
    target_policy = instantiate(
        cfg.model, num_branches=num_branches, action_bins=action_bins
    ).to(device)
    target_policy.load_state_dict(policy.state_dict())

    loss_fn = get_loss_fn(cfg.train.loss_fn)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr)

    # -- CPU replay buffer (uint8 to save RAM) ---------------------------------
    res_h = res_w = cfg.env.obs_size
    buf_sz = int(cfg.train.buffer_size)
    stk = cfg.env.stack_size
    obs_shape = (buf_sz, stk, res_h, res_w)
    buffer: dict[str, torch.Tensor] = {
        "obs": torch.zeros(obs_shape, dtype=torch.uint8),
        "next_obs": torch.zeros(obs_shape, dtype=torch.uint8),
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
        cfg, envs, policy, target_policy, loss_fn, optimizer, buffer, device, writer
    )
    writer.close()
    log.info("Done -- %.1fs | avg FPS %.0f", elapsed, avg_fps)
