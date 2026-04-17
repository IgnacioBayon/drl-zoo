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

from .loss import PPOLoss
from .model import PPO

log = logging.getLogger(__name__)


def _random_crop_obs(obs: torch.Tensor, padding: int = 4) -> torch.Tensor:
    if obs.ndim != 4 or padding <= 0:
        return obs

    batch_size, channels, height, width = obs.shape
    padded = F.pad(obs, (padding, padding, padding, padding), mode="replicate")
    max_offset = 2 * padding

    top = torch.randint(0, max_offset + 1, (batch_size,), device=obs.device)
    left = torch.randint(0, max_offset + 1, (batch_size,), device=obs.device)

    # Vectorized advanced indexing
    batch_idx = torch.arange(batch_size, device=obs.device).view(-1, 1, 1, 1)
    c_idx = torch.arange(channels, device=obs.device).view(1, -1, 1, 1)
    y_idx = torch.arange(height, device=obs.device).view(1, 1, -1, 1) + top.view(
        -1, 1, 1, 1
    )
    x_idx = torch.arange(width, device=obs.device).view(1, 1, 1, -1) + left.view(
        -1, 1, 1, 1
    )

    return padded[batch_idx, c_idx, y_idx, x_idx]


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Calculate advantages using Generalized Advantage Estimation (GAE)

    Args:
        rewards: Rewards received at each timestep [B, T]
        values: Value estimates for each state [B, T]
        next_values: Value estimates for the next states [B, T]
        dones: Binary indicators of episode termination [B, T]
        gamma: Discount factor. Defaults to 0.99.
        lam: GAE lambda parameter. Defaults to 0.95.

    Returns:
        torch.Tensor: Advantage estimates [B, T]
    """
    deltas = rewards + gamma * next_values * (1 - dones) - values

    advantages = torch.zeros_like(deltas)
    gae = 0
    for t in reversed(range(deltas.size(1))):
        gae = deltas[:, t] + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae

    return advantages


def _train_step(
    policy: PPO,
    loss_fn: PPOLoss,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    max_grad_norm: float | None = 0.5,
) -> dict[str, torch.Tensor]:
    """Perform a single training step for PPO

    Args:
        policy: PPO policy to be trained
        loss_fn: Loss function for PPO training
        optimizer: Optimizer for updating model parameters
        batch: Batch of training data containing:
            - log_probs: Log probabilities of the current policy [B, T]
            - old_log_probs: Log probabilities of the old policy [B, T]
            - advantages: Advantage estimates [B, T]
            - returns: Discounted returns [B, T]
            - values: Value estimates [B, T]
            - entropy: Entropy of the policy [B, T]

    Returns:
        torch.Tensor: Loss value for the current training step
    """
    policy.train()

    obs = batch["obs"].to(dtype=torch.float32).div_(255.0)
    obs = _random_crop_obs(obs)
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    old_values = batch["old_values"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    log_probs, values, entropy = policy.evaluate(obs, actions)

    actor_loss, critic_loss, entropy_loss, loss = loss_fn.compute_terms(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        values=values,
        old_values=old_values,
        entropy=entropy,
    )

    ratio = torch.exp(log_probs.detach() - old_log_probs.detach())
    approx_kl = ((ratio - 1.0) - (log_probs.detach() - old_log_probs.detach())).mean()

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": loss.detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "kl": approx_kl.detach(),
        }

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": loss.detach(),
        "actor_loss": actor_loss.detach(),
        "critic_loss": critic_loss.detach(),
        "entropy_loss": entropy_loss.detach(),
        "kl": approx_kl.detach(),
    }


def _train_loop(
    cfg: DictConfig,
    envs: object,  # gymnasium.vector.VectorEnv
    policy: PPO,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[float, float]:
    """Main PPO training loop using vectorized environments.

    Args:
        cfg: Hydra config containing training hyperparameters.
        envs: Vectorized gymnasium environment (num_envs parallel envs)
        policy: PPO model
        optimizer: Optimizer for policy parameters
        device: Torch device
        writer: TensorBoard SummaryWriter

    Returns:
        tuple[float, float]: (mean_eval_reward, total_training_time_seconds)
    """
    tcfg = cfg.train
    ecfg = cfg.env
    clip_epsilon = float(tcfg.get("clip_ratio", 0.2))
    gae_lambda = float(tcfg.get("gae_lambda", 0.95))
    update_epochs = int(tcfg.get("update_epochs", 4))
    minibatch_size = int(tcfg.get("batch_size", 64))
    target_kl = float(tcfg.get("target_kl", 0.015))

    loss_fn = PPOLoss(
        c1=tcfg.c1,
        c2=tcfg.c2,
        epsilon=clip_epsilon,
        value_clip=float(tcfg.get("value_clip", 0.2)),
    )

    num_envs = envs.num_envs
    T = tcfg.rollout_steps
    total_frames = tcfg.total_frames
    num_updates = math.ceil(total_frames / (num_envs * T))

    # --- Initialize environment ---
    obs, _ = envs.reset()
    obs = torch.as_tensor(obs, dtype=torch.uint8, device=device)

    worker_returns = np.zeros(num_envs, dtype=np.float64)
    worker_steps = np.zeros(num_envs, dtype=np.int64)
    episode_returns: deque[float] = deque(maxlen=100)
    reward_window_sum = 0.0
    global_step = 0
    mean_eval_reward = float("-inf")
    t_start = perf_counter()

    for update in range(1, num_updates + 1):
        prev_step = global_step
        eval_info: str | None = None

        # ------------------------------------------------------------------ #
        # 1. ROLLOUT COLLECTION                                                #
        # ------------------------------------------------------------------ #
        # Storage tensors: shape [T, num_envs, ...]
        all_obs = torch.zeros(
            T,
            num_envs,
            *obs.shape[1:],
            device=device,
            dtype=torch.uint8,
        )
        action_shape = tuple(envs.single_action_space.shape)
        if len(action_shape) == 0:
            all_actions = torch.zeros(T, num_envs, device=device)
        else:
            all_actions = torch.zeros(T, num_envs, *action_shape, device=device)
        all_log_probs = torch.zeros(T, num_envs, device=device)
        all_values = torch.zeros(T, num_envs, device=device)
        all_rewards = torch.zeros(T, num_envs, device=device)
        all_terminated = torch.zeros(T, num_envs, device=device, dtype=torch.bool)
        all_truncated = torch.zeros(T, num_envs, device=device, dtype=torch.bool)
        # FIX: stores the correct next-state value at each timestep,
        # overriding with the true final-state value when an env is truncated.
        all_next_values = torch.zeros(T, num_envs, device=device)

        policy.eval()
        with torch.no_grad():
            for t in range(T):
                all_obs[t] = obs

                obs_input = obs.to(dtype=torch.float32).div(255.0)
                actions, log_probs, value = policy.act(obs_input)

                all_actions[t] = actions
                all_log_probs[t] = log_probs
                all_values[t] = value.squeeze(-1)

                # Step environments
                obs_np, rewards, terminated, truncated, infos = envs.step(
                    actions.cpu().numpy()
                )
                dones = terminated | truncated

                all_rewards[t] = torch.tensor(
                    rewards, dtype=torch.float32, device=device
                )
                all_terminated[t] = torch.tensor(
                    terminated, dtype=torch.bool, device=device
                )
                all_truncated[t] = torch.tensor(
                    truncated, dtype=torch.bool, device=device
                )

                # Compute V(s_{t+1}) from the next observation returned by env.step().
                # For truncated envs, override with V(final_observation) before reset.
                next_obs_t = torch.as_tensor(
                    obs_np, dtype=torch.float32, device=device
                ).div(255.0)
                _, _, next_vals_t = policy(next_obs_t)
                next_vals_t = next_vals_t.squeeze(-1)

                for i in np.where(truncated)[0]:
                    final_obs_i = infos["final_observation"][i]
                    fo = (
                        torch.as_tensor(final_obs_i, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                        .div(255.0)
                    )
                    _, _, v_final = policy(fo)
                    next_vals_t[i] = v_final.squeeze()

                # For terminated transitions, no bootstrap
                for i in np.where(terminated)[0]:
                    next_vals_t[i] = 0.0

                all_next_values[t] = next_vals_t

                worker_returns += rewards.astype(np.float64)
                worker_steps += 1

                for i in np.where(dones)[0]:
                    if len(episode_returns) == episode_returns.maxlen:
                        reward_window_sum -= episode_returns[0]

                    ret = float(worker_returns[i])
                    reward_window_sum += ret
                    episode_returns.append(ret)

                    final_info = None
                    if "final_info" in infos and infos["final_info"] is not None:
                        final_info = infos["final_info"][i]

                    if final_info is not None and "x_position" in final_info:
                        final_x = float(final_info["x_position"])
                    elif "x_position" in infos:
                        final_x = float(infos["x_position"][i])
                    else:
                        final_x = 0.0

                    ep_time = int(worker_steps[i]) * 0.008
                    avg_speed = final_x / ep_time if ep_time > 0 else 0.0

                    writer.add_scalar("episode/final_x", final_x, global_step)
                    writer.add_scalar("episode/avg_speed", avg_speed, global_step)
                    writer.add_scalar("episode/reward", ret, global_step)

                    worker_returns[i] = 0.0
                    worker_steps[i] = 0

                obs = torch.as_tensor(obs_np, dtype=torch.uint8, device=device)
                global_step += num_envs

            # Bootstrap value for the last step
            obs_input = obs.to(dtype=torch.float32).div(255.0)
            _, _, next_value = policy(obs_input)
            next_value = next_value.squeeze(-1)  # [num_envs]

        # ------------------------------------------------------------------ #
        # 2. COMPUTE ADVANTAGES & RETURNS                                      #
        # ------------------------------------------------------------------ #
        # Shapes expected by GAE: [B, T] — we use [num_envs, T]
        rewards_bt = all_rewards.T  # [num_envs, T]
        values_bt = all_values.T  # [num_envs, T]
        dones_bt = (all_terminated | all_truncated).float().T  # [num_envs, T]
        next_values_bt = all_next_values.T  # [num_envs, T]

        advantages_bt = generalized_advantage_estimation(
            rewards=rewards_bt,
            values=values_bt,
            next_values=next_values_bt,
            dones=dones_bt,
            gamma=tcfg.gamma,
            lam=gae_lambda,
        )
        returns_bt = advantages_bt + values_bt  # [num_envs, T]

        # Normalise advantages for training stability
        advantages_bt = (advantages_bt - advantages_bt.mean()) / (
            advantages_bt.std() + 1e-8
        )

        # ------------------------------------------------------------------ #
        # 3. FLATTEN for mini-batch sampling: [num_envs * T, ...]             #
        # ------------------------------------------------------------------ #
        N = num_envs * T
        flat_obs = all_obs.view(N, *obs.shape[1:])
        flat_actions = all_actions.view(N, -1)
        flat_log_probs = all_log_probs.view(N)
        flat_values = all_values.view(N)
        flat_advantages = advantages_bt.T.reshape(N)
        flat_returns = returns_bt.T.reshape(N)

        # ------------------------------------------------------------------ #
        # 4. PPO UPDATE EPOCHS                                                 #
        # ------------------------------------------------------------------ #
        total_loss_accum = 0.0
        actor_loss_accum = 0.0
        critic_loss_accum = 0.0
        entropy_loss_accum = 0.0
        kl_accum = 0.0
        total_minibatches = 0
        early_stop_epoch: int | None = None
        early_stop_kl: float | None = None

        for epoch in range(update_epochs):
            indices = torch.randperm(N, device=device)
            stop_update = False

            for mb_start in range(0, N, minibatch_size):
                mb_idx = indices[mb_start : mb_start + minibatch_size]

                batch = {
                    "obs": flat_obs[mb_idx],
                    "actions": flat_actions[mb_idx],
                    "old_log_probs": flat_log_probs[mb_idx],
                    "old_values": flat_values[mb_idx],
                    "advantages": flat_advantages[mb_idx],
                    "returns": flat_returns[mb_idx],
                }

                metrics = _train_step(
                    policy=policy,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    batch=batch,
                    max_grad_norm=tcfg.max_grad_norm,
                )
                total_loss_accum += metrics["loss"].item()
                actor_loss_accum += metrics["actor_loss"].item()
                critic_loss_accum += metrics["critic_loss"].item()
                entropy_loss_accum += metrics["entropy_loss"].item()
                kl_accum += metrics["kl"].item()
                total_minibatches += 1

                if metrics["kl"].item() > target_kl:
                    early_stop_epoch = epoch + 1
                    early_stop_kl = metrics["kl"].item()
                    stop_update = True
                    break

            if stop_update:
                break

        # ------------------------------------------------------------------ #
        # 5. LOGGING                                                           #
        # ------------------------------------------------------------------ #
        avg_loss = total_loss_accum / max(1, total_minibatches)
        avg_actor_loss = actor_loss_accum / max(1, total_minibatches)
        avg_critic_loss = critic_loss_accum / max(1, total_minibatches)
        avg_entropy_loss = entropy_loss_accum / max(1, total_minibatches)
        avg_kl = kl_accum / max(1, total_minibatches)
        if (
            global_step // cfg.log_interval_frames
            != prev_step // cfg.log_interval_frames
        ):
            elapsed = perf_counter() - t_start
            fps = global_step / elapsed
            avg_reward = 0.0
            if episode_returns:
                avg_reward = reward_window_sum / len(episode_returns)

            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/actor_loss", avg_actor_loss, global_step)
            writer.add_scalar("train/critic_loss", avg_critic_loss, global_step)
            writer.add_scalar("train/entropy_loss", avg_entropy_loss, global_step)
            writer.add_scalar("train/approx_kl", avg_kl, global_step)
            writer.add_scalar("train/global_step", global_step, update)
            writer.add_scalar("train/fps", fps, global_step)
            if episode_returns:
                writer.add_scalar("train/avg_reward_100", avg_reward, global_step)

            msg = (
                f"step {update}/{num_updates} | frame {global_step}/{total_frames} | "
                f"reward {avg_reward:.2f} | loss {avg_loss:.4f} | "
                f"actor {avg_actor_loss:.4f} | critic {avg_critic_loss:.4f} | "
                f"entropy {avg_entropy_loss:.4f} | kl {avg_kl:.6f} | fps {fps:.0f}"
            )
            if early_stop_epoch is not None and early_stop_kl is not None:
                msg += (
                    f" | early_stop_kl {early_stop_kl:.6f} "
                    f"(epoch {early_stop_epoch}/{update_epochs})"
                )
            if eval_info:
                msg += f"  # {eval_info}"
            log.info(msg)

        # ------------------------------------------------------------------ #
        # 6. EVALUATION & CHECKPOINTING                                        #
        # ------------------------------------------------------------------ #
        def action_fn(obs: torch.Tensor) -> np.ndarray:
            """Helper to evaluate current policy on eval envs."""
            with torch.no_grad():
                actions = policy.deterministic_action(obs)

            return actions.squeeze(0).cpu().numpy()

        if (
            global_step // cfg.eval_interval_frames
            != prev_step // cfg.eval_interval_frames
        ):
            mean_eval_reward, std_eval_reward, _ = evaluate_and_record(
                policy=policy,
                action_fn=action_fn,
                step=global_step,
                env_cfg=ecfg,
                train_cfg=tcfg,
                video_dir=cfg.paths.video_dir,
                device=device,
                writer=writer,
                n_episodes=int(cfg.eval_episodes),
                force_discretize_actions=False,
            )
            eval_info = f"eval {mean_eval_reward:.2f}+/-{std_eval_reward:.2f}"
            save_checkpoint(
                model=policy,
                optimizer=optimizer,
                global_step=global_step,
                checkpoint_dir=cfg.paths.checkpoint_dir,
                name=f"ppo_step_{global_step}.pt",
            )

    total_time = perf_counter() - t_start
    envs.close()
    log.info(f"Training complete in {total_time:.1f}s")
    writer.close()

    return mean_eval_reward, total_time


def train_ppo(cfg: DictConfig) -> None:
    """Build envs, PPO network, optimizer and launch training.

    Args:
        cfg: Full Hydra config (``env``, ``train``, ``loss``, ``model``, ``paths``).
    """
    device = get_device(cfg.train.device)

    if bool(cfg.train.get("discretize_actions", False)):
        raise ValueError(
            "PPO requires continuous actions: set train.discretize_actions=false."
        )

    # -- environment -----------------------------------------------------------
    envs = build_from_config(
        cfg.env,
        cfg.train,
        mode="train",
        force_discretize_actions=False,
    )

    if not isinstance(envs.single_action_space, Box):
        raise ValueError("PPO requires a continuous Box action space.")

    num_envs: int = cfg.env.num_envs
    action_dim = int(envs.single_action_space.shape[0])

    # -- network & optimizer ---------------------------------------------------
    policy = instantiate(
        cfg.train.model,
        action_dim=action_dim,
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.train.lr, eps=1e-5)

    # -- TensorBoard -----------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)

    log.info(
        "Training PPO on %s | device=%s | envs=%d | rollout_steps=%d | "
        "epochs=%d | minibatch=%d | action_dim=%d",
        cfg.env.name,
        device,
        num_envs,
        cfg.train.rollout_steps,
        int(cfg.train.get("update_epochs", cfg.train.get("num_epochs", 4))),
        int(cfg.train.get("batch_size", cfg.train.get("minibatch_size", 64))),
        action_dim,
    )

    mean_eval_reward, total_time = _train_loop(
        cfg, envs, policy, optimizer, device, writer
    )
    avg_fps = (cfg.train.total_frames) / total_time
    writer.close()
    log.info(
        "Done -- %.1fs | avg FPS %.0f | Mean Eval Reward %.2f",
        total_time,
        avg_fps,
        mean_eval_reward,
    )
