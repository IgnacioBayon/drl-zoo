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


class SumSegmentTree:
    """Binary segment tree storing prefix sums for O(log n) proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self._cap
        self._tree[idx] = val
        while idx > 1:
            idx >>= 1
            self._tree[idx] = self._tree[2 * idx] + self._tree[2 * idx + 1]

    def __getitem__(self, idx: int) -> float:
        return float(self._tree[idx + self._cap])

    @property
    def total(self) -> float:
        return float(self._tree[1])

    def find_prefixsum_idx(self, prefix_sum: float) -> int:
        """Return the highest index *i* such that ``sum(tree[0:i]) <= prefix_sum``."""
        idx = 1
        while idx < self._cap:
            left = 2 * idx
            if self._tree[left] > prefix_sum:
                idx = left
            else:
                prefix_sum -= self._tree[left]
                idx = left + 1
        return idx - self._cap


class MinSegmentTree:
    """Binary segment tree storing running minimums."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._tree = np.full(2 * capacity, float("inf"), dtype=np.float64)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self._cap
        self._tree[idx] = val
        while idx > 1:
            idx >>= 1
            self._tree[idx] = min(self._tree[2 * idx], self._tree[2 * idx + 1])

    @property
    def min(self) -> float:
        return float(self._tree[1])


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer with n-step support
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """Ring buffer with proportional PER and per-sample n-step discount storage.

    Args:
        capacity: Maximum number of transitions stored.
        obs_shape: Observation tensor shape, e.g. ``(stack, H, W)``.
        n_branches: Number of action branches.
        alpha: PER prioritisation exponent (0 = uniform, 1 = full PER).
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        n_branches: int,
        alpha: float,
    ) -> None:
        # Round capacity to next power-of-two for the segment trees
        self._tree_cap = 1
        while self._tree_cap < capacity:
            self._tree_cap <<= 1

        self._alpha = alpha
        self._max_priority = 1.0
        self._sum = SumSegmentTree(self._tree_cap)
        self._min = MinSegmentTree(self._tree_cap)

        self._capacity = capacity
        self._obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self._next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self._actions = torch.zeros((capacity, n_branches), dtype=torch.int64)
        self._rewards = torch.zeros(capacity, dtype=torch.float32)
        self._dones = torch.zeros(capacity, dtype=torch.float32)
        self._gamma_ns = torch.zeros(capacity, dtype=torch.float32)

        self._ptr = 0
        self._size = 0

    # -- storage ---------------------------------------------------------------

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: float,
        gamma_n: float,
    ) -> None:
        """Append one n-step transition with maximal priority."""
        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = done
        self._gamma_ns[i] = gamma_n

        pa = self._max_priority**self._alpha
        self._sum[i] = pa
        self._min[i] = pa

        self._ptr = (i + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    # -- sampling --------------------------------------------------------------

    def sample(
        self, batch_size: int, beta: float
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, torch.Tensor]:
        """Stratified sampling proportional to stored priorities.

        Returns:
            batch: Dict of tensors (CPU).
            indices: Array of sampled buffer indices for later priority updates.
            weights: Importance-sampling weights ``(B,)``.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        total = self._sum.total
        segment = total / batch_size

        # Max IS weight for normalisation
        min_prob = self._min.min / total
        max_weight = (self._size * min_prob) ** (-beta) if min_prob > 0 else 1.0

        weights = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx = self._sum.find_prefixsum_idx(s)
            idx = min(idx, self._size - 1)
            indices[i] = idx
            prob = self._sum[idx] / total
            weights[i] = (self._size * prob) ** (-beta) / max_weight

        batch = {
            "obs": self._obs[indices],
            "next_obs": self._next_obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "dones": self._dones[indices],
            "gamma_ns": self._gamma_ns[indices],
        }
        return batch, indices, torch.as_tensor(weights, dtype=torch.float32)

    # -- priority update -------------------------------------------------------

    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor) -> None:
        """Set new priorities for previously sampled transitions."""
        for idx, p in zip(indices, priorities.detach().cpu().numpy()):
            p = max(float(p), 1e-8)
            self._max_priority = max(self._max_priority, p)
            pa = p**self._alpha
            self._sum[int(idx)] = pa
            self._min[int(idx)] = pa

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# N-step return helper
# ---------------------------------------------------------------------------


class NStepAccumulator:
    """Accumulates per-worker n-step transitions and flushes them into a PER buffer.

    For each vectorised environment worker, maintains a short FIFO of raw
    ``(obs, action, reward)`` tuples.  When the FIFO reaches *n* entries (or
    an episode terminates), the discounted n-step return is computed from the
    first entry to the last, and the resulting transition is pushed into the
    replay buffer.

    Args:
        num_envs: Number of parallel environment workers.
        n_step: Lookahead horizon.
        gamma: Discount factor.
        buffer: Target :class:`PrioritizedReplayBuffer` to push into.
    """

    def __init__(
        self,
        num_envs: int,
        n_step: int,
        gamma: float,
        buffer: PrioritizedReplayBuffer,
    ) -> None:
        self._n = n_step
        self._gamma = gamma
        self._buf = buffer
        self._fifos: list[list[tuple[torch.Tensor, torch.Tensor, float]]] = [
            [] for _ in range(num_envs)
        ]

    def append(
        self, obs, actions, rewards, next_obs, terminations, truncations, infos
    ) -> None:
        num_envs = len(rewards)
        for i in range(num_envs):
            self._fifos[i].append(
                (
                    torch.as_tensor(obs[i]),
                    torch.as_tensor(actions[i], dtype=torch.int64),
                    float(rewards[i]),
                )
            )

            if terminations[i] or truncations[i]:
                nxt = torch.as_tensor(infos["final_observation"][i])
                is_term = float(terminations[i])  # 1.0 if dead, 0.0 if truncated
                while self._fifos[i]:
                    self._flush_front(i, nxt, done=is_term)
            elif len(self._fifos[i]) == self._n:
                self._flush_front(i, torch.as_tensor(next_obs[i]), done=0.0)

    def _flush_front(self, worker: int, next_obs: torch.Tensor, done: float) -> None:
        fifo = self._fifos[worker]
        k = len(fifo)
        R = 0.0
        for j in range(k - 1, -1, -1):
            R = fifo[j][2] + self._gamma * R
        first_obs, first_action, _ = fifo[0]
        self._buf.add(first_obs, first_action, R, next_obs, done, self._gamma**k)
        fifo.pop(0)


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

    # Sample noise for NoisyNets
    online.reset_noise()
    target.reset_noise()

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
    n_step_acc: NStepAccumulator,
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[float, float]:
    """Rainbow collect → train → log loop."""
    tcfg = cfg.train
    ecfg = cfg.env
    num_envs: int = ecfg.num_envs

    total_frames = int(tcfg.total_frames)
    steps = math.ceil(total_frames / num_envs)

    global_step = 0
    worker_returns = np.zeros(num_envs, dtype=np.float64)
    episode_returns: deque[float] = deque(maxlen=100)
    reward_window_sum = 0.0
    step_losses: list[torch.Tensor] = []
    avg_loss = 0.0
    best_mean_reward = float("-inf")

    beta_start: float = float(tcfg.per_beta_start)
    beta_frames: int = int(tcfg.per_beta_frames)
    start_train_after: int = int(tcfg.start_train_after)

    obs, _ = envs.reset(seed=int(cfg.seed))
    start = perf_counter()

    for step in range(1, steps + 1):
        # -- action selection (noisy nets — no ε-greedy) -----------------------
        online.reset_noise()  # Resample noise for each action selection
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
            q = online(obs_t)["q"]  # (num_envs, branches, bins)
            actions = q.argmax(dim=-1).cpu().numpy()  # (num_envs, branches)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
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
        n_step_acc.append(
            obs, actions, rewards, next_obs, terminations, truncations, infos
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

            def action_fn(obs_t: torch.Tensor) -> np.ndarray:
                return online(obs_t)["q"].argmax(dim=-1).squeeze(0).cpu().numpy()

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
    num_branches: int = (
        envs.single_action_space.shape[0]
        if hasattr(envs.single_action_space, "nvec")
        else 1
    )

    # -- networks & optimiser --------------------------------------------------
    online = instantiate(cfg.model, num_branches=num_branches).to(device)
    target = instantiate(cfg.model, num_branches=num_branches).to(device)
    target.load_state_dict(online.state_dict())

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.train.lr, eps=1.5e-4)

    # -- PER buffer ------------------------------------------------------------
    res_h, res_w = tuple(cfg.env.train_resolution)
    stk = cfg.env.stack_size
    obs_shape = (stk, res_h, res_w)

    per_buffer = PrioritizedReplayBuffer(
        capacity=int(cfg.train.buffer_size),
        obs_shape=obs_shape,
        n_branches=num_branches,
        alpha=float(cfg.train.per_alpha),
    )

    n_step_acc = NStepAccumulator(
        num_envs=cfg.env.num_envs,
        n_step=int(cfg.train.n_step),
        gamma=float(cfg.train.gamma),
        buffer=per_buffer,
    )

    # -- TensorBoard -----------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)

    log.info(
        "Training Rainbow DQN on %s | device=%s | envs=%d | buffer=%d | "
        "branches=%d | atoms=%d | n_step=%d",
        cfg.env.name,
        device,
        cfg.env.num_envs,
        cfg.train.buffer_size,
        num_branches,
        cfg.model.atoms,
        cfg.train.n_step,
    )

    elapsed, avg_fps = _train_loop(
        cfg, envs, online, target, optimizer, per_buffer, n_step_acc, device, writer
    )
    writer.close()
    log.info("Done -- %.1fs | avg FPS %.0f", elapsed, avg_fps)
