"""Prioritized Experience Replay buffer and N-step return accumulator."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


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
        """Set new priorities (clamped to a small epsilon) for sampled indices."""
        self._priorities[indices] = priorities.cpu().clamp(min=1e-6)


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
        """Compute discounted return ``r_0 + gamma*r_1 + ... + gamma^{k-1}*r_{k-1}``."""
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
                entries = list(fifo)
                K = len(entries)
                for j in range(K):
                    out_obs.append(entries[j][0])
                    out_act.append(entries[j][1])
                    out_rew.append(self._discounted_return(entries[j:]))
                    out_nxt.append(next_obs[i])
                    out_done.append(float(terminations[i]))
                    out_gn.append(self._gamma ** (K - j))
                fifo.clear()

            elif len(fifo) == self._n:
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
