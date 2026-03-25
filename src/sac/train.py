"""Soft Actor-Critic training loop for image observations."""

from __future__ import annotations

import logging
import math
from collections import deque
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config
from src.utils import evaluate_and_record, get_device, save_checkpoint

from .buffer import ReplayBuffer
from .model import Actor, DoubleCritic

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft target update
# ---------------------------------------------------------------------------
def _soft_update(
    online: torch.nn.Module,
    target: torch.nn.Module,
    tau: float,
) -> None:
    """Polyak averaging update: ``target <- tau*online + (1-tau)*target``."""
    for p, p_targ in zip(online.parameters(), target.parameters()):
        p_targ.data.mul_(1.0 - tau)
        p_targ.data.add_(tau * p.data)


# ---------------------------------------------------------------------------
# SAC update step
# ---------------------------------------------------------------------------


def _train_step(
    actor: Actor,
    critic: DoubleCritic,
    critic_target: DoubleCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    alpha: float,
    log_alpha: torch.Tensor | None,
    alpha_opt: torch.optim.Optimizer | None,
    target_entropy: float | None,
    max_grad_norm: float | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Sample a batch and perform one SAC critic+actor update."""
    batch = replay_buffer.sample(batch_size)

    obs = batch["obs"].to(device, dtype=torch.float32).div_(255.0)
    nxt = batch["next_obs"].to(device, dtype=torch.float32).div_(255.0)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device).unsqueeze(1)
    dones = batch["dones"].to(device).unsqueeze(1)

    # -- critic target --------------------------------------------------------
    if log_alpha is not None:
        alpha_t = log_alpha.exp().detach()
    else:
        alpha_t = torch.tensor(alpha, device=device)

    with torch.no_grad():
        next_actions, next_logp = actor.sample(nxt)
        q1_next, q2_next = critic_target(nxt, next_actions)
        q_next = torch.min(q1_next, q2_next)
        target_q = rewards + gamma * (1.0 - dones) * (q_next - alpha_t * next_logp)

    # -- critic update --------------------------------------------------------
    q1, q2 = critic(obs, actions)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
    critic_opt.step()

    # -- actor update ---------------------------------------------------------
    for p in critic.parameters():
        p.requires_grad = False

    new_actions, logp = actor.sample(obs)
    q1_pi, q2_pi = critic(obs, new_actions)
    q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (alpha_t * logp - q_pi).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
    actor_opt.step()

    for p in critic.parameters():
        p.requires_grad = True

    # -- target update --------------------------------------------------------
    _soft_update(critic, critic_target, tau)

    alpha_loss: torch.Tensor | None = None
    if log_alpha is not None and alpha_opt is not None and target_entropy is not None:
        alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()
        alpha_value = float(log_alpha.exp().item())
    else:
        alpha_value = float(alpha_t.item())

    return (
        actor_loss.detach(),
        critic_loss.detach(),
        alpha_loss.detach() if alpha_loss is not None else None,
        alpha_value,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def _train_loop(
    cfg: DictConfig,
    envs: object,
    actor: Actor,
    critic: DoubleCritic,
    critic_target: DoubleCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    log_alpha: torch.Tensor | None,
    alpha_opt: torch.optim.Optimizer | None,
    target_entropy: float | None,
    fixed_alpha: float,
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[float, float]:
    """SAC collect -> train -> log loop."""
    tcfg = cfg.train
    ecfg = cfg.env
    num_envs: int = ecfg.num_envs

    total_frames = int(tcfg.total_frames)
    steps = math.ceil(total_frames / num_envs)

    global_step = 0
    worker_returns = np.zeros(num_envs, dtype=np.float64)
    worker_steps = np.zeros(num_envs, dtype=np.int64)
    episode_returns: deque[float] = deque(maxlen=100)
    reward_window_sum = 0.0
    actor_losses: list[torch.Tensor] = []
    critic_losses: list[torch.Tensor] = []
    alpha_losses: list[torch.Tensor] = []
    alpha_values: list[float] = []

    start_train_after: int = int(tcfg.start_train_after)

    obs, _ = envs.reset(seed=int(cfg.seed))
    start = perf_counter()

    for step in range(1, steps + 1):
        # -- action selection --------------------------------------------------
        if len(replay_buffer) < start_train_after:
            actions = np.stack(
                [envs.single_action_space.sample() for _ in range(num_envs)],
                axis=0,
            ).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).div_(
                    255.0
                )
                actions = actor.act(obs_t, deterministic=False).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        # -- per-worker return tracking ---------------------------------------
        worker_returns += rewards.astype(np.float64)
        worker_steps += 1

        for i in np.where(dones)[0]:
            if len(episode_returns) == episode_returns.maxlen:
                reward_window_sum -= episode_returns[0]

            ret = float(worker_returns[i])
            reward_window_sum += ret
            episode_returns.append(ret)

            final_x = float(infos["x_position"][i]) if "x_position" in infos else 0.0
            ep_time = int(worker_steps[i]) * 0.008
            avg_speed = final_x / ep_time if ep_time > 0 else 0.0

            writer.add_scalar("episode/final_x", final_x, global_step)
            writer.add_scalar("episode/avg_speed", avg_speed, global_step)
            writer.add_scalar("episode/reward", ret, global_step)

            worker_returns[i] = 0.0
            worker_steps[i] = 0

        # -- add transitions to replay buffer ---------------------------------
        replay_buffer.add_batch(
            torch.as_tensor(obs, dtype=torch.uint8),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(next_obs, dtype=torch.uint8),
            torch.as_tensor(dones.astype(np.float32), dtype=torch.float32),
        )

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

        total_updates = n_updates * int(tcfg.gradient_steps)
        if (
            len(replay_buffer) >= max(start_train_after, tcfg.batch_size)
            and total_updates > 0
        ):
            for _ in range(total_updates):
                actor_loss, critic_loss, alpha_loss, alpha_value = _train_step(
                    actor=actor,
                    critic=critic,
                    critic_target=critic_target,
                    actor_opt=actor_opt,
                    critic_opt=critic_opt,
                    replay_buffer=replay_buffer,
                    batch_size=int(tcfg.batch_size),
                    gamma=float(tcfg.gamma),
                    tau=float(tcfg.tau),
                    alpha=fixed_alpha,
                    log_alpha=log_alpha,
                    alpha_opt=alpha_opt,
                    target_entropy=target_entropy,
                    max_grad_norm=tcfg.max_grad_norm,
                    device=device,
                )
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                alpha_values.append(alpha_value)
                if alpha_loss is not None:
                    alpha_losses.append(alpha_loss)

        # -- periodic evaluation & checkpoint ---------------------------------
        eval_info: str | None = None
        if (
            global_step // cfg.eval_interval_frames
            != prev_step // cfg.eval_interval_frames
        ):

            def action_fn(obs_t: torch.Tensor) -> np.ndarray:
                return actor.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()

            mean_r, std_r, mean_fx = evaluate_and_record(
                actor,
                action_fn,
                global_step,
                ecfg,
                tcfg,
                cfg.paths.video_dir,
                device,
                writer,
                int(cfg.eval_episodes),
                force_discretize_actions=False,
            )
            eval_info = f"eval {mean_r:.2f}+/-{std_r:.2f}"

            save_checkpoint(
                actor,
                actor_opt,
                global_step,
                cfg.paths.checkpoint_dir,
                name=f"actor_{global_step}.pt",
            )
            save_checkpoint(
                critic,
                critic_opt,
                global_step,
                cfg.paths.checkpoint_dir,
                name=f"critic_{global_step}.pt",
            )

        # -- tensorboard / console logging ------------------------------------
        if (
            global_step // cfg.log_interval_frames
            != prev_step // cfg.log_interval_frames
        ):
            avg_actor_loss = (
                torch.stack(actor_losses).mean().item() if actor_losses else 0.0
            )
            avg_critic_loss = (
                torch.stack(critic_losses).mean().item() if critic_losses else 0.0
            )
            avg_alpha_loss = (
                torch.stack(alpha_losses).mean().item() if alpha_losses else 0.0
            )
            avg_alpha = float(np.mean(alpha_values)) if alpha_values else fixed_alpha
            actor_losses.clear()
            critic_losses.clear()
            alpha_losses.clear()
            alpha_values.clear()

            elapsed = perf_counter() - start
            fps = global_step / elapsed

            writer.add_scalar("train/actor_loss", avg_actor_loss, global_step)
            writer.add_scalar("train/critic_loss", avg_critic_loss, global_step)
            writer.add_scalar("train/alpha", avg_alpha, global_step)
            if alpha_opt is not None:
                writer.add_scalar("train/alpha_loss", avg_alpha_loss, global_step)
            writer.add_scalar("train/fps", fps, global_step)
            writer.add_scalar("train/buffer_size", len(replay_buffer), global_step)

            avg_reward = 0.0
            if episode_returns:
                avg_reward = reward_window_sum / len(episode_returns)
                writer.add_scalar("train/avg_reward_100", avg_reward, global_step)

            msg = (
                f"step {step}/{steps} | frame {global_step}/{total_frames} | "
                f"reward {avg_reward:.2f} | actor {avg_actor_loss:.4f} | "
                f"critic {avg_critic_loss:.4f} | alpha {avg_alpha:.4f} | fps {fps:.0f}"
            )
            if eval_info:
                msg += f"  # {eval_info}"
            log.info(msg)

    save_checkpoint(
        actor,
        actor_opt,
        global_step,
        cfg.paths.checkpoint_dir,
        name="actor_final.pt",
    )
    save_checkpoint(
        critic,
        critic_opt,
        global_step,
        cfg.paths.checkpoint_dir,
        name="critic_final.pt",
    )

    envs.close()
    elapsed = perf_counter() - start
    return elapsed, global_step / elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def train_sac(cfg: DictConfig) -> None:
    """Build envs, networks, replay buffer and launch SAC training."""
    device = get_device(cfg.train.device)

    if bool(cfg.train.get("discretize_actions", False)):
        raise ValueError(
            "SAC requires continuous actions: set train.discretize_actions=false."
        )

    # -- environment ----------------------------------------------------------
    envs = build_from_config(
        cfg.env,
        cfg.train,
        mode="train",
        force_discretize_actions=False,
    )

    if not isinstance(envs.single_action_space, Box):
        raise ValueError("SAC requires a continuous Box action space.")

    action_dim = int(envs.single_action_space.shape[0])

    # -- networks & optimiser -------------------------------------------------
    actor = instantiate(
        cfg.train.model.actor,
        in_channels=cfg.env.stack_size,
        action_dim=action_dim,
        action_low=envs.single_action_space.low.tolist(),
        action_high=envs.single_action_space.high.tolist(),
    ).to(device)

    critic = instantiate(
        cfg.train.model.critic,
        in_channels=cfg.env.stack_size,
        action_dim=action_dim,
    ).to(device)

    critic_target = instantiate(
        cfg.train.model.critic,
        in_channels=cfg.env.stack_size,
        action_dim=action_dim,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(cfg.train.actor_lr))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=float(cfg.train.critic_lr))

    auto_alpha = bool(cfg.train.get("auto_alpha", False))
    fixed_alpha = float(cfg.train.alpha)
    log_alpha: torch.Tensor | None = None
    alpha_opt: torch.optim.Optimizer | None = None
    target_entropy: float | None = None
    if auto_alpha:
        target_entropy = float(
            cfg.train.target_entropy
            if cfg.train.target_entropy is not None
            else -action_dim
        )
        log_alpha = torch.tensor(
            np.log(max(fixed_alpha, 1e-6)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        alpha_opt = torch.optim.Adam([log_alpha], lr=float(cfg.train.alpha_lr))

    # -- replay buffer --------------------------------------------------------
    obs_shape = (cfg.env.stack_size, cfg.env.obs_size, cfg.env.obs_size)
    replay_buffer = ReplayBuffer(
        capacity=int(cfg.train.buffer_size),
        obs_shape=obs_shape,
        action_dim=action_dim,
    )

    # -- TensorBoard ----------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)

    log.info(
        "Training SAC on %s | device=%s | envs=%d | buffer=%d | action_dim=%d",
        cfg.env.name,
        device,
        cfg.env.num_envs,
        cfg.train.buffer_size,
        action_dim,
    )

    elapsed, avg_fps = _train_loop(
        cfg=cfg,
        envs=envs,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        replay_buffer=replay_buffer,
        log_alpha=log_alpha,
        alpha_opt=alpha_opt,
        target_entropy=target_entropy,
        fixed_alpha=fixed_alpha,
        device=device,
        writer=writer,
    )
    writer.close()
    log.info("Done -- %.1fs | avg FPS %.0f", elapsed, avg_fps)
