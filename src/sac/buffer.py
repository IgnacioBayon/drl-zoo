"""Uniform replay buffer for SAC."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Uniform replay buffer with flat preallocated tensors."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
    ) -> None:
        self.capacity = capacity
        self._pos = 0
        self._size = 0

        self._obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self._next_obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self._actions = torch.empty((capacity, action_dim), dtype=torch.float32)
        self._rewards = torch.empty((capacity,), dtype=torch.float32)
        self._dones = torch.empty((capacity,), dtype=torch.float32)

    def __len__(self) -> int:
        return self._size

    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Insert a batch of transitions, handling circular wrap."""
        n = obs.shape[0]
        end = self._pos + n

        if end <= self.capacity:
            s = slice(self._pos, end)
            self._obs[s].copy_(obs)
            self._next_obs[s].copy_(next_obs)
            self._actions[s].copy_(actions)
            self._rewards[s].copy_(rewards)
            self._dones[s].copy_(dones)
        else:
            first = self.capacity - self._pos

            self._obs[self._pos :].copy_(obs[:first])
            self._next_obs[self._pos :].copy_(next_obs[:first])
            self._actions[self._pos :].copy_(actions[:first])
            self._rewards[self._pos :].copy_(rewards[:first])
            self._dones[self._pos :].copy_(dones[:first])

            rest = n - first

            self._obs[:rest].copy_(obs[first:])
            self._next_obs[:rest].copy_(next_obs[first:])
            self._actions[:rest].copy_(actions[first:])
            self._rewards[:rest].copy_(rewards[first:])
            self._dones[:rest].copy_(dones[first:])

        self._pos = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    # Optional convenience method for adding a single transition
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float | torch.Tensor,
        next_obs: torch.Tensor,
        done: float | torch.Tensor,
    ) -> None:
        """Insert a single transition."""
        self.add_batch(
            obs.unsqueeze(0),
            action.unsqueeze(0),
            torch.as_tensor([reward], dtype=torch.float32),
            next_obs.unsqueeze(0),
            torch.as_tensor([done], dtype=torch.float32),
        )

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch uniformly at random."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        indices = torch.randint(0, self._size, (batch_size,))

        return {
            "obs": self._obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_obs[indices],
            "dones": self._dones[indices],
        }
