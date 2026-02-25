"""Rainbow DQN training loop combining all six improvements.

Improvements:
    1. **Double Q-Learning** — online net selects action, target net evaluates.
    2. **Prioritized Experience Replay** — proportional PER with sum/min trees.
    3. **Dueling Networks** — value + advantage streams (in model).
    4. **Multi-step Learning** — n-step bootstrapped returns.
    5. **Distributional RL (C51)** — categorical cross-entropy on atom distributions.
    6. **Noisy Networks** — NoisyLinear replaces ε-greedy (in model).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config
from src.utils import evaluate_and_record, get_device, save_checkpoint

from .models import RainbowDQN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment trees for Prioritized Experience Replay
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """Proportional PER with flat arrays."""

    def __init__(
        self, capacity: int, obs_shape: tuple[int, ...], num_branches: int, alpha: float
    ) -> None:
        self.capacity = capacity
        self._alpha = alpha
        self._pos = 0
        self._size = 0

        self._obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self._next_obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self._actions = torch.empty((capacity, num_branches), dtype=torch.int64)
        self._rewards = torch.empty((capacity,), dtype=torch.float32)
        self._dones = torch.empty((capacity,), dtype=torch.float32)
        self._gamma_ns = torch.empty((capacity,), dtype=torch.float32)
        self._priorities = torch.empty((capacity,), dtype=torch.float32)

    def __len__(self) -> int:
        return self._size

    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        gamma_ns: torch.Tensor,
    ) -> None:
        """Insert a batch of ``n`` transitions, handling circular wrap."""
        n = obs.shape[0]
        max_prio = (
            self._priorities[: self._size].max().item() if self._size > 0 else 1.0
        )

        end = self._pos + n
        if end <= self.capacity:
            s = slice(self._pos, end)
            self._obs[s].copy_(obs)
            self._next_obs[s].copy_(next_obs)
            self._actions[s].copy_(actions)
            self._rewards[s].copy_(rewards)
            self._dones[s].copy_(dones)
            self._gamma_ns[s].copy_(gamma_ns)
            self._priorities[s] = max_prio
        else:
            first = self.capacity - self._pos
            self._obs[self._pos :].copy_(obs[:first])
            self._next_obs[self._pos :].copy_(next_obs[:first])
            self._actions[self._pos :].copy_(actions[:first])
            self._rewards[self._pos :].copy_(rewards[:first])
            self._dones[self._pos :].copy_(dones[:first])
            self._gamma_ns[self._pos :].copy_(gamma_ns[:first])
            self._priorities[self._pos :] = max_prio

            rest = n - first
            self._obs[:rest].copy_(obs[first:])
            self._next_obs[:rest].copy_(next_obs[first:])
            self._actions[:rest].copy_(actions[first:])
            self._rewards[:rest].copy_(rewards[first:])
            self._dones[:rest].copy_(dones[first:])
            self._gamma_ns[:rest].copy_(gamma_ns[first:])
            self._priorities[:rest] = max_prio

        self._pos = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(
        self, batch_size: int, beta: float
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample a batch weighted by priority."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        probs = self._priorities[: self._size].pow(self._alpha)
        probs_sum = probs.sum().item()
        if probs_sum == 0:
            probs.fill_(1.0 / self._size)
        else:
            probs.div_(probs_sum)

        indices = torch.multinomial(probs, batch_size, replacement=True)
        weights = (self._size * probs[indices]).pow(-beta)
        weights.div_(weights.max())

        batch = {
            "obs": self._obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_obs[indices],
            "dones": self._dones[indices],
            "gamma_ns": self._gamma_ns[indices],
        }
        return batch, indices, weights

    def update_priorities(
        self, indices: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        """Set new priorities (clamped to a small ε) for sampled indices."""
        self._priorities[indices] = priorities.cpu().clamp(min=1e-6)


# ---------------------------------------------------------------------------
# N-step return helper
# ---------------------------------------------------------------------------


class VectorizedNStepAccumulator:
    """Accumulates per-worker n-step transitions.

    Each parallel worker maintains a FIFO of ``(obs, action, reward)`` tuples.
    When the FIFO reaches *n* entries a full n-step transition is emitted; when
    an episode ends, all remaining partial transitions are flushed.

    Args:
        num_envs: Number of parallel environment workers.
        n_step: Lookahead horizon.
        gamma: Discount factor.
    """

    def __init__(
        self,
        num_envs: int,
        n_step: int,
        gamma: float,
    ) -> None:
        self._num_envs = num_envs
        self._n = n_step
        self._gamma = gamma
        self._fifos: list[deque[tuple[np.ndarray, np.ndarray, float]]] = [
            deque() for _ in range(num_envs)
        ]

    def _discounted_return(
        self, entries: list[tuple[np.ndarray, np.ndarray, float]]
    ) -> float:
        """Compute discounted return ``r_0 + γ·r_1 + … + γ^{k-1}·r_{k-1}``."""
        R = 0.0
        for _, _, r in reversed(entries):
            R = r + self._gamma * R
        return R

    def append(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
    ) -> dict[str, torch.Tensor] | None:
        """Process one vectorised env step and return completed transitions.

        Returns:
            A dict of batched tensors ``{obs, actions, rewards, next_obs,
            dones, gamma_ns}`` for every transition that became ready this
            step, or ``None`` when no transition is complete yet.
        """
        dones = np.logical_or(terminations, truncations)

        out_obs: list[np.ndarray] = []
        out_act: list[np.ndarray] = []
        out_rew: list[float] = []
        out_nxt: list[np.ndarray] = []
        out_done: list[float] = []
        out_gn: list[float] = []

        for i in range(self._num_envs):
            fifo = self._fifos[i]
            fifo.append((obs[i], actions[i], float(rewards[i])))

            if dones[i]:
                # Episode ended — flush every remaining FIFO entry as a
                # partial (or full) n-step transition.
                entries = list(fifo)
                K = len(entries)
                for j in range(K):
                    out_obs.append(entries[j][0])
                    out_act.append(entries[j][1])
                    out_rew.append(self._discounted_return(entries[j:]))
                    out_nxt.append(next_obs[i])
                    # Truncation should still bootstrap; only true
                    # termination zeroes out the target value.
                    out_done.append(float(terminations[i]))
                    out_gn.append(self._gamma ** (K - j))
                fifo.clear()

            elif len(fifo) == self._n:
                # Full n-step window — emit the oldest transition.
                entries = list(fifo)
                out_obs.append(entries[0][0])
                out_act.append(entries[0][1])
                out_rew.append(self._discounted_return(entries))
                out_nxt.append(next_obs[i])
                out_done.append(0.0)
                out_gn.append(self._gamma**self._n)
                fifo.popleft()

        if not out_obs:
            return None

        return {
            "obs": torch.as_tensor(np.stack(out_obs), dtype=torch.uint8),
            "actions": torch.as_tensor(np.stack(out_act), dtype=torch.int64),
            "rewards": torch.tensor(out_rew, dtype=torch.float32),
            "next_obs": torch.as_tensor(np.stack(out_nxt), dtype=torch.uint8),
            "dones": torch.tensor(out_done, dtype=torch.float32),
            "gamma_ns": torch.tensor(out_gn, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# C51 distributional loss (with Double-DQN action selection)
# ---------------------------------------------------------------------------


def _c51_loss(
    online: RainbowDQN,
    target: RainbowDQN,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute per-sample cross-entropy loss for the C51 distributional update.

    Uses **Double DQN**: online net selects the greedy action, target net
    provides the distribution to project.

    Returns:
        Per-sample loss tensor of shape ``(B,)`` (detach before using as PER
        priority; this tensor retains grad for backprop).
    """
    obs = batch["obs"].to(device, dtype=torch.float32).div_(255.0)
    nxt = batch["next_obs"].to(device, dtype=torch.float32).div_(255.0)
    actions = batch["actions"].to(device)  # (B, branches)
    rewards = batch["rewards"].to(device)  # (B,)
    dones = batch["dones"].to(device)  # (B,)
    gamma_ns = batch["gamma_ns"].to(device)  # (B,)

    atoms = online.atoms
    support = online.support  # (atoms,)
    vmin, vmax = support[0].item(), support[-1].item()
    delta_z = (vmax - vmin) / (atoms - 1)

    # --- Online network: log-probabilities for chosen actions -----------------

    out = online(obs)
    log_p = F.log_softmax(out["logits"], dim=-1)  # (B, br, bins, atoms)
    # Gather along bins dim:  actions (B, br) → (B, br, 1, atoms)
    act_idx = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, atoms)
    log_p_a = log_p.gather(2, act_idx).squeeze(2)  # (B, br, atoms)

    with torch.no_grad():
        # --- Double DQN: online selects, target evaluates ---------------------
        nxt_q = online(nxt)["q"]  # (B, br, bins)
        nxt_actions = nxt_q.argmax(dim=-1)  # (B, br)

        nxt_probs = F.softmax(target(nxt)["logits"], dim=-1)  # (B, br, bins, atoms)
        nxt_idx = nxt_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, atoms)
        p_target = nxt_probs.gather(2, nxt_idx).squeeze(2)  # (B, br, atoms)

        # --- C51 projection ---------------------------------------------------
        # Tz = r + γ^n * (1 - done) * z_j,  clipped to [vmin, vmax]
        tz = (
            rewards.view(-1, 1, 1)
            + gamma_ns.view(-1, 1, 1) * (1.0 - dones.view(-1, 1, 1)) * support
        )
        tz.clamp_(vmin, vmax)

        b = (tz - vmin) / delta_z  # (B, 1, atoms) → broadcasts to (B, br, atoms)
        lo = b.floor().long().clamp(0, atoms - 1)
        hi = b.ceil().long().clamp(0, atoms - 1)

        # Expand to branches
        n_branches = p_target.shape[1]
        tz = tz.expand(-1, n_branches, -1)
        b = b.expand(-1, n_branches, -1)
        lo = lo.expand(-1, n_branches, -1)
        hi = hi.expand(-1, n_branches, -1)

        # Distribute probability mass to neighbouring atoms
        d_lo = (hi.float() - b) * p_target  # mass for lower atom
        d_hi = (b - lo.float()) * p_target  # mass for upper atom
        # When lo == hi (b is integer), assign full mass to that atom
        eq = lo == hi
        d_lo = torch.where(eq, p_target, d_lo)

        projected = torch.zeros_like(p_target)
        projected.scatter_add_(2, lo, d_lo)
        projected.scatter_add_(2, hi, d_hi)

    # Cross-entropy per sample: -Σ_j m_j log p_j,  averaged over branches
    loss = -(projected * log_p_a).sum(dim=-1).mean(dim=1)  # (B,)
    return loss


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_step(
    online: RainbowDQN,
    target: RainbowDQN,
    optimizer: torch.optim.Optimizer,
    per_buffer: PrioritizedReplayBuffer,
    batch_size: int,
    beta: float,
    device: torch.device,
    max_grad_norm: float,
) -> torch.Tensor:
    """Sample from PER, compute C51 loss, update priorities."""
    batch, indices, weights = per_buffer.sample(batch_size, beta)
    weights = weights.to(device)

    losses = _c51_loss(online, target, batch, device)  # (B,)
    loss = (losses * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
    optimizer.step()

    # Update PER priorities to |δ| (per-sample loss)
    per_buffer.update_priorities(indices, losses.detach().abs())

    return loss.detach()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _train_loop(
    cfg: DictConfig,
    envs: object,
    online: RainbowDQN,
    target: RainbowDQN,
    optimizer: torch.optim.Optimizer,
    per_buffer: PrioritizedReplayBuffer,
    n_step_acc: VectorizedNStepAccumulator,
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[float, float]:
    """Rainbow collect → train → log loop."""
    tcfg = cfg.train
    ecfg = cfg.env
    num_envs: int = ecfg.num_envs
    multidiscrete: bool = ecfg.action_multidiscrete

    total_frames = int(tcfg.total_frames)
    steps = math.ceil(total_frames / num_envs)

    global_step = 0
    worker_returns = np.zeros(num_envs, dtype=np.float64)
    episode_returns: deque[float] = deque(maxlen=100)
    reward_window_sum = 0.0
    step_losses: list[torch.Tensor] = []
    avg_loss = 0.0
    best_mean_reward = float("-inf")
    epsilon = 0.0  # only meaningful for eps_greedy

    use_noisy: bool = bool(tcfg.exploration == "noisy")
    beta_start: float = float(tcfg.per_beta_start)
    beta_frames: int = int(tcfg.per_beta_frames)
    start_train_after: int = int(tcfg.start_train_after)

    obs, _ = envs.reset(seed=int(cfg.seed))
    start = perf_counter()

    for step in range(1, steps + 1):
        # -- action selection --------------------------------------------------
        if use_noisy:
            # NoisyNet exploration: resample noise, then act greedily
            online.reset_noise()
            target.reset_noise()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
                q = online(obs_t)["q"]  # (num_envs, branches, bins)
                actions = q.argmax(dim=-1).cpu().numpy()  # (num_envs, branches)
        else:
            # ε-greedy exploration with linear decay
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
                    obs_t = (
                        torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
                    )
                    q = online(obs_t)["q"]  # (num_envs, branches, bins)
                    actions = q.argmax(dim=-1).cpu().numpy()

        # Env expects (E,) scalars for Discrete, (E, branches) for MultiDiscrete
        actions_env = actions if multidiscrete else actions.squeeze(-1)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions_env)
        dones = np.logical_or(terminations, truncations)

        # -- per-worker return tracking ----------------------------------------
        worker_returns += rewards.astype(np.float64)
        for i in np.where(dones)[0]:
            if len(episode_returns) == episode_returns.maxlen:
                reward_window_sum -= episode_returns[0]
            ret = float(worker_returns[i])
            reward_window_sum += ret
            episode_returns.append(ret)
            worker_returns[i] = 0.0

        # -- n-step accumulation → PER buffer ----------------------------------
        transitions = n_step_acc.append(
            obs, actions, rewards, next_obs, terminations, truncations
        )
        if transitions is not None:
            per_buffer.add_batch(
                transitions["obs"],
                transitions["actions"],
                transitions["rewards"],
                transitions["next_obs"],
                transitions["dones"],
                transitions["gamma_ns"],
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

        if len(per_buffer) >= max(start_train_after, tcfg.batch_size) and n_updates > 0:
            # Anneal PER β from beta_start → 1.0 over beta_frames
            frac = min(1.0, max(0, global_step - start_train_after) / beta_frames)
            beta = beta_start + frac * (1.0 - beta_start)

            for _ in range(n_updates):
                step_losses.append(
                    _train_step(
                        online,
                        target,
                        optimizer,
                        per_buffer,
                        tcfg.batch_size,
                        beta,
                        device,
                        tcfg.max_grad_norm,
                    )
                )

        # -- target update -----------------------------------------------------
        if (
            global_step // tcfg.target_update_frames
            != prev_step // tcfg.target_update_frames
        ):
            target.load_state_dict(online.state_dict())

        # -- periodic evaluation -----------------------------------------------
        eval_info: str | None = None
        if (
            global_step // tcfg.eval_interval_frames
            != prev_step // tcfg.eval_interval_frames
        ):

            def action_fn(obs_t: torch.Tensor) -> np.ndarray | int:
                act = online(obs_t)["q"].argmax(dim=-1).squeeze(0).cpu().numpy()
                return act if multidiscrete else int(act.item())

            mean_r, std_r, max_r, _ = evaluate_and_record(
                online,
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
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                save_checkpoint(
                    online,
                    optimizer,
                    global_step,
                    cfg.paths.checkpoint_dir,
                    name=f"ckpt_best_mean_reward_{mean_r:.2f}.pt",
                )

        # -- tensorboard / console logging -------------------------------------
        if (
            global_step // tcfg.log_interval_frames
            != prev_step // tcfg.log_interval_frames
        ):
            if step_losses:
                avg_loss = torch.stack(step_losses).mean().item()
                step_losses.clear()
            elapsed = perf_counter() - start
            fps = global_step / elapsed
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/fps", fps, global_step)
            writer.add_scalar("train/buffer_size", len(per_buffer), global_step)
            if not use_noisy:
                writer.add_scalar("train/epsilon", epsilon, global_step)
            writer.add_scalar(
                "train/beta",
                beta_start
                + min(1.0, max(0, global_step - start_train_after) / beta_frames)
                * (1.0 - beta_start),
                global_step,
            )
            avg_reward = 0.0
            if episode_returns:
                avg_reward = reward_window_sum / len(episode_returns)
                writer.add_scalar("train/avg_reward_100", avg_reward, global_step)
            msg = (
                f"step {step}/{steps} | frame {global_step}/{total_frames} | "
                f"reward {avg_reward:.2f} | loss {avg_loss:.4f} | fps {fps:.0f}"
            )
            if eval_info:
                msg += f"  # {eval_info}"
            log.info(msg)

    save_checkpoint(
        online, optimizer, global_step, cfg.paths.checkpoint_dir, name="ckpt_final.pt"
    )
    envs.close()
    elapsed = perf_counter() - start
    return elapsed, global_step / elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train_rainbow(cfg: DictConfig) -> None:
    """Build envs, networks, PER buffer, n-step accumulator and launch training.

    Args:
        cfg: Full Hydra config (``env``, ``train``, ``model``, ``paths``).
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

    # -- networks & optimiser -------------------------------------------------
    online = instantiate(
        cfg.model, num_branches=num_branches, action_bins=action_bins
    ).to(device)
    target = instantiate(
        cfg.model, num_branches=num_branches, action_bins=action_bins
    ).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()  # target net is not trained, only updated via hard copies

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.train.lr, eps=1.5e-4)

    # -- PER buffer ------------------------------------------------------------
    res_h, res_w = tuple(cfg.env.train_resolution)
    stk = cfg.env.stack_size
    obs_shape = (stk, res_h, res_w)

    per_buffer = PrioritizedReplayBuffer(
        capacity=int(cfg.train.buffer_size),
        obs_shape=obs_shape,
        num_branches=num_branches,
        alpha=float(cfg.train.per_alpha),
    )

    n_step_acc = VectorizedNStepAccumulator(
        num_envs=cfg.env.num_envs,
        n_step=int(cfg.train.n_step),
        gamma=float(cfg.train.gamma),
    )

    # -- TensorBoard -----------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)

    log.info(
        "Training Rainbow DQN on %s | device=%s | envs=%d | buffer=%d | "
        "branches=%d | atoms=%d | n_step=%d | exploration=%s",
        cfg.env.name,
        device,
        cfg.env.num_envs,
        cfg.train.buffer_size,
        num_branches,
        cfg.model.atoms,
        cfg.train.n_step,
        cfg.train.exploration,
    )

    elapsed, avg_fps = _train_loop(
        cfg, envs, online, target, optimizer, per_buffer, n_step_acc, device, writer
    )
    writer.close()
    log.info("Done -- %.1fs | avg FPS %.0f", elapsed, avg_fps)
