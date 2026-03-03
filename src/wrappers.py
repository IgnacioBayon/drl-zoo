"""Custom Gymnasium wrappers."""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete, MultiDiscrete


class DiscretizeAction(gym.ActionWrapper[ObsType, int, ActType]):
    """Discretizes a continuous Box action space using linspace endpoints.

    Unlike gymnasium's built-in DiscretizeAction which uses bin centers
    (and thus never reaches the interval extremes), this wrapper uses
    ``np.linspace(low, high, bins)`` so that the first and last discrete
    actions map exactly to the action-space boundaries.

    With 3 bins on [-1, 1] this produces actions {-1, 0, 1} instead of
    {-0.667, 0, 0.667}.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        bins: int | tuple[int, ...],
        multidiscrete: bool = False,
    ) -> None:
        """Construct the discretize-action wrapper.

        Args:
            env: The environment to wrap.
            bins: Number of discrete values per action dimension.
            multidiscrete: If ``True``, expose a ``MultiDiscrete`` action
                space instead of flattening to a single ``Discrete`` space.
        """
        if not isinstance(env.action_space, Box):
            raise TypeError(
                "DiscretizeAction requires a Box action space, "
                f"got {type(env.action_space).__name__}"
            )

        super().__init__(env)

        low = env.action_space.low
        high = env.action_space.high
        n_dims = low.shape[0]

        if np.any(np.isinf(low)) or np.any(np.isinf(high)):
            raise ValueError(
                f"Discretization requires finite bounds. low={low}, high={high}"
            )

        bins_arr = (
            np.full(n_dims, bins, dtype=int)
            if isinstance(bins, int)
            else np.asarray(bins, dtype=int)
        )
        assert bins_arr.shape == (n_dims,), (
            f"bins length mismatch: expected {n_dims}, got {bins_arr.shape[0]}"
        )

        # Pre-compute the action values for each dimension (includes extremes)
        self._values = [
            np.linspace(low[i], high[i], bins_arr[i]) for i in range(n_dims)
        ]
        self._bins = bins_arr
        self._n_dims = n_dims
        self._multidiscrete = multidiscrete

        self.action_space = (
            MultiDiscrete(bins_arr)
            if multidiscrete
            else Discrete(int(np.prod(bins_arr)))
        )

    def action(self, act: int | np.ndarray) -> np.ndarray:
        """Map a discrete action index to a continuous action vector."""
        if self._multidiscrete:
            indices = np.asarray(act, dtype=int)
        else:
            indices = np.unravel_index(int(act), self._bins)
        return np.array(
            [self._values[i][idx] for i, idx in enumerate(indices)],
            dtype=self.env.action_space.dtype,
        )

    def revert_action(self, action: np.ndarray) -> int | np.ndarray:
        """Find the closest discrete action for a continuous action vector."""
        indices = tuple(
            int(np.argmin(np.abs(self._values[i] - action[i])))
            for i in range(self._n_dims)
        )
        if self._multidiscrete:
            return np.array(indices, dtype=int)
        return int(np.ravel_multi_index(indices, self._bins))


class CustomReward(gym.Wrapper):
    """Replaces default reward with center of mass forward velocity and reduced coefficients."""

    def __init__(
        self,
        env: gym.Env,
        use_com: bool = True,
    ) -> None:
        super().__init__(env)
        self._use_com = use_com
        self._torso_id = self.unwrapped.model.body("torso").id

    def step(self, action):
        if self._use_com:
            pos_before = float(self.unwrapped.data.subtree_com[self._torso_id][0])
        else:
            pos_before = float(self.unwrapped.data.qpos[0])
        obs, og_reward, terminated, truncated, info = self.env.step(action)
        if self._use_com:
            pos_after = float(self.unwrapped.data.subtree_com[self._torso_id][0])
        else:
            pos_after = float(self.unwrapped.data.qpos[0])

        # x_vel = obs[6]
        # forward_reward = (x_vel >= 0) * 1.0
        # healthy_reward = self.unwrapped._healthy_reward if og_reward > 0 else 0.0
        x_vel = (pos_after - pos_before) / self.unwrapped.dt
        forward_reward = self.unwrapped._forward_reward_weight * x_vel
        healthy_reward = self.unwrapped._healthy_reward if not terminated else 0.0
        ctrl_cost = self.unwrapped._ctrl_cost_weight * float(np.sum(np.square(action)))

        reward = forward_reward + healthy_reward - ctrl_cost
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = -ctrl_cost
        info["reward_healthy"] = healthy_reward
        return obs, reward, terminated, truncated, info


class ImageObsWrapper(gym.ObservationWrapper):
    """Render, downscale, and grayscale in a single wrapper.

    The env must have ``render_mode='rgb_array'``.  Each step the wrapper
    calls ``env.render()`` to obtain a full-resolution RGB frame, then
    downscales with ``cv2.INTER_AREA`` (anti-aliased) and converts to
    single-channel grayscale — replacing the original observation.
    """

    def __init__(self, env: gym.Env, obs_size: int = 64) -> None:
        super().__init__(env)
        self._obs_size = obs_size
        self.observation_space = gym.spaces.Box(
            0, 255, (obs_size, obs_size), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        frame = self.env.render()  # full-resolution RGB
        frame = cv2.resize(
            frame, (self._obs_size, self._obs_size), interpolation=cv2.INTER_AREA
        )
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
