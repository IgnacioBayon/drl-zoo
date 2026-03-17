import logging
import math
from collections import deque
from time import perf_counter

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config
from src.utils import evaluate_and_record, get_device, save_checkpoint

from .loss import PPOLoss
from .model import PPO

log = logging.getLogger(__name__)


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
    model: PPO,
    loss_fn: PPOLoss,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    max_grad_norm=0.5,
) -> torch.Tensor:
    """Perform a single training step for PPO

    Args:
        model: PPO model to be trained
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
    model.train()

    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    action_means, action_log_stds, values = model(obs)
    action_stds = torch.exp(action_log_stds)
    action_dist = Normal(action_means, action_stds)

    log_probs = action_dist.log_prob(actions).sum(dim=-1)
    entropy = action_dist.entropy().sum(dim=-1)

    loss = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        values=values.squeeze(-1),
        entropy=entropy,
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss.detach()


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

    loss_fn = PPOLoss(
        c1=tcfg.c1,
        c2=tcfg.c2,
        epsilon=tcfg.epsilon,
    )

    num_envs = envs.num_envs
    T = tcfg.rollout_steps
    total_frames = tcfg.total_frames
    num_updates = math.ceil(total_frames / (num_envs * T))

    # --- Initialize environment ---
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    episode_returns = deque(maxlen=100)
    global_step = 0
    mean_eval_reward = float("-inf")
    t_start = perf_counter()

    for update in range(1, num_updates + 1):
        # ------------------------------------------------------------------ #
        # 1. ROLLOUT COLLECTION                                                #
        # ------------------------------------------------------------------ #
        # Storage tensors: shape [T, num_envs, ...]
        all_obs = torch.zeros(T, num_envs, *obs.shape[1:], device=device)
        try:
            num_actions = envs.single_action_space.shape[0]
        except IndexError:
            num_actions = int(envs.single_action_space.n)

        all_actions = torch.zeros(T, num_envs, num_actions, device=device)
        all_log_probs = torch.zeros(T, num_envs, device=device)
        all_values = torch.zeros(T, num_envs, device=device)
        all_rewards = torch.zeros(T, num_envs, device=device)
        all_dones = torch.zeros(T, num_envs, device=device)

        policy.eval()
        with torch.no_grad():
            for t in range(T):
                all_obs[t] = obs

                means, log_stds, value = policy(obs)
                stds = torch.exp(log_stds)
                dist = Normal(means, stds)

                actions = dist.sample()
                # Sum log_prob over action dims → scalar per env
                log_probs = dist.log_prob(actions).sum(dim=-1)

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
                all_dones[t] = torch.tensor(dones, dtype=torch.float32, device=device)

                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
                global_step += num_envs

                # Track completed episode returns
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info is not None and "episode" in info:
                            episode_returns.append(info["episode"]["r"])

            # Bootstrap value for the last step
            _, _, next_value = policy(obs)
            next_value = next_value.squeeze(-1)  # [num_envs]

        # ------------------------------------------------------------------ #
        # 2. COMPUTE ADVANTAGES & RETURNS                                      #
        # ------------------------------------------------------------------ #
        # Shapes expected by GAE: [B, T] — we use [num_envs, T]
        rewards_bt = all_rewards.T  # [num_envs, T]
        values_bt = all_values.T  # [num_envs, T]
        dones_bt = all_dones.T  # [num_envs, T]

        # next_values[b, t] = values[b, t+1], with bootstrap at t=T
        next_values_bt = torch.cat(
            [values_bt[:, 1:], next_value.unsqueeze(1)], dim=1
        )  # [num_envs, T]

        advantages_bt = generalized_advantage_estimation(
            rewards=rewards_bt,
            values=values_bt,
            next_values=next_values_bt,
            dones=dones_bt,
            gamma=tcfg.gamma,
            lam=tcfg.lam,
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
        flat_advantages = advantages_bt.T.reshape(N)
        flat_returns = returns_bt.T.reshape(N)

        # ------------------------------------------------------------------ #
        # 4. PPO UPDATE EPOCHS                                                 #
        # ------------------------------------------------------------------ #
        total_loss_accum = 0.0
        num_minibatches = max(1, N // tcfg.minibatch_size)

        for epoch in range(tcfg.num_epochs):
            indices = torch.randperm(N, device=device)

            for mb_start in range(0, N, tcfg.minibatch_size):
                mb_idx = indices[mb_start : mb_start + tcfg.minibatch_size]

                batch = {
                    "obs": flat_obs[mb_idx],
                    "actions": flat_actions[mb_idx],
                    "old_log_probs": flat_log_probs[mb_idx],
                    "advantages": flat_advantages[mb_idx],
                    "returns": flat_returns[mb_idx],
                }

                loss = _train_step(
                    model=policy, loss_fn=loss_fn, optimizer=optimizer, batch=batch
                )
                total_loss_accum += loss.item()

                # Gradient clipping (applied inside _train_step — see note below)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), tcfg.max_grad_norm)

        # ------------------------------------------------------------------ #
        # 5. LOGGING                                                           #
        # ------------------------------------------------------------------ #
        avg_loss = total_loss_accum / (tcfg.num_epochs * num_minibatches)
        writer.add_scalar("train/loss", avg_loss, global_step)
        writer.add_scalar("train/global_step", global_step, update)

        if len(episode_returns) > 0:
            mean_return = np.mean(episode_returns)
            writer.add_scalar("train/mean_episode_return", mean_return, global_step)
            log.info(
                f"Update {update}/{num_updates} | step {global_step} | "
                f"loss {avg_loss:.4f} | mean_return {mean_return:.2f}"
            )
        else:
            log.info(
                f"Update {update}/{num_updates} | step {global_step} | loss {avg_loss:.4f}"
            )

        # ------------------------------------------------------------------ #
        # 6. EVALUATION & CHECKPOINTING                                        #
        # ------------------------------------------------------------------ #
        def action_fn(obs: np.ndarray) -> np.ndarray:
            """Helper to evaluate current policy on eval envs."""
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                means, log_stds, _ = policy(obs_tensor)

            return means.cpu().numpy()  # Use mean action for evaluation (deterministic)

        if update % tcfg.eval_interval == 0:
            mean_eval_reward = evaluate_and_record(
                policy=policy,
                action_fn=action_fn,
                step=global_step,
                env_cfg=ecfg,
                video_dir=cfg.paths.video_dir,
                device=device,
                writer=writer,
                n_episodes=tcfg.eval_episodes,
            )

        if update % tcfg.checkpoint_interval == 0:
            save_checkpoint(
                model=policy,
                optimizer=optimizer,
                global_step=global_step,
                checkpoint_dir=cfg.paths.checkpoint_dir,
                name=f"ppo_step_{global_step}.pt",
            )

    total_time = perf_counter() - t_start
    log.info(f"Training complete in {total_time:.1f}s")
    writer.close()

    return mean_eval_reward, total_time


def train_ppo(cfg: DictConfig) -> None:
    """Build envs, PPO network, optimizer and launch training.

    Args:
        cfg: Full Hydra config (``env``, ``train``, ``loss``, ``model``, ``paths``).
    """
    device = get_device(cfg.train.device)

    # -- environment -----------------------------------------------------------
    envs = build_from_config(cfg.env, mode="train")
    num_envs: int = cfg.env.num_envs

    # -- network & optimizer ---------------------------------------------------
    policy = instantiate(cfg.model).to(device)

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
        cfg.train.num_epochs,
        cfg.train.minibatch_size,
        cfg.env.action_bins,
    )

    elapsed, avg_fps = _train_loop(cfg, envs, policy, optimizer, device, writer)
    writer.close()
    log.info("Done -- %.1fs | avg FPS %.0f", elapsed, avg_fps)
