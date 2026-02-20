import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.wrappers import (
    AddRenderObservation,
    DiscretizeAction,
    FrameStackObservation,
)


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
            pos_before = float(self.data.qpos[0])
        obs, _, terminated, truncated, info = self.env.step(action)
        if self._use_com:
            pos_after = float(self.unwrapped.data.subtree_com[self._torso_id][0])
        else:
            pos_after = float(self.data.qpos[0])

        dt = self.unwrapped.dt
        forward_reward = (pos_after - pos_before) / dt
        ctrl_cost = self._ctrl_cost_weight * float(np.sum(np.square(action)))
        healthy_reward = self._healthy_reward if not terminated else 0.0

        reward = forward_reward + healthy_reward - ctrl_cost
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = -ctrl_cost
        info["reward_healthy"] = healthy_reward
        return obs, reward, terminated, truncated, info


class GrayScalePixelObs(gym.ObservationWrapper):
    """Converts RGB pixel obs to single-channel (H, W) using grayscale or luminance."""

    # BT.601 luminance coefficients
    _LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def __init__(self, env: gym.Env, use_luminance: bool) -> None:
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(0, 255, (h, w), dtype=np.uint8)
        self._luminance = use_luminance

    def observation(self, obs):
        if self._luminance:
            return (obs @ self._LUMA_WEIGHTS).astype(np.uint8)
        return np.mean(obs, axis=-1).astype(np.uint8)


def build_envs(
    env_name: str,
    num_envs: int,
    render_mode: str,
    bins: int,
    stack_size: int,
    use_luminance: bool,
    resolution: tuple[int, int],
    use_com_reward: bool,
    multidiscrete: bool,
    env_kwargs: dict,
):
    def make_wrapper(env: gym.Env) -> gym.Env:
        env = CustomReward(env, use_com_reward)
        env = DiscretizeAction(env, bins, multidiscrete)
        env = AddRenderObservation(env)
        env = GrayScalePixelObs(env, use_luminance)
        env = FrameStackObservation(env, stack_size)
        return env

    env = gym.make_vec(
        env_name,
        num_envs,
        "async",
        render_mode=render_mode,
        width=resolution[0],
        height=resolution[1],
        wrappers=[make_wrapper],
        **env_kwargs,  # health, noise and reward kwargs
    )
    return env
