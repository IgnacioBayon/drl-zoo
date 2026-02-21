import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AddRenderObservation,
    DiscretizeAction,
    FrameStackObservation,
)
from omegaconf import DictConfig


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
        obs, _, terminated, truncated, info = self.env.step(action)
        if self._use_com:
            pos_after = float(self.unwrapped.data.subtree_com[self._torso_id][0])
        else:
            pos_after = float(self.unwrapped.data.qpos[0])

        dt = self.unwrapped.dt
        forward_reward = (
            self.unwrapped._forward_reward_weight * (pos_after - pos_before) / dt
        )
        ctrl_cost = self.unwrapped._ctrl_cost_weight * float(np.sum(np.square(action)))
        healthy_reward = self.unwrapped._healthy_reward if not terminated else 0.0

        reward = forward_reward + healthy_reward - ctrl_cost
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = -ctrl_cost
        info["reward_healthy"] = healthy_reward
        return obs, reward, terminated, truncated, info


class GrayScalePixelObs(gym.ObservationWrapper):
    """Converts RGB pixel obs to single-channel (H, W) using grayscale or luminance."""

    # BT.601 luminance coefficients
    _LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # Equal-weight mean — same dot-product path as luminance for speed.
    _MEAN_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

    def __init__(self, env: gym.Env, use_luminance: bool) -> None:
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(0, 255, (h, w), dtype=np.uint8)
        self._weights = self._LUMA_WEIGHTS if use_luminance else self._MEAN_WEIGHTS

    def observation(self, obs):
        return (obs @ self._weights).astype(np.uint8)


def build_envs(
    env_name: str = "Hopper-v5",
    num_envs: int = 1,
    render_mode: str = "rgb_array",
    bins: int = 5,
    stack_size: int = 4,
    use_luminance: bool = False,
    resolution: tuple[int, int] = (480, 480),
    use_com_reward: bool = False,
    multidiscrete: bool = False,
    vectorized: bool = True,
    **env_kwargs: dict,
):
    def wrap_env(env: gym.Env) -> gym.Env:
        env = CustomReward(env, use_com_reward)
        env = DiscretizeAction(env, bins, multidiscrete)
        if render_mode != "human":
            env = AddRenderObservation(env)
            env = GrayScalePixelObs(env, use_luminance)
        env = FrameStackObservation(env, stack_size)
        return env

    all_env_kwargs = {
        "id": env_name,
        "num_envs": num_envs,
        "vectorization_mode": "sync",
        "render_mode": render_mode,
        "width": resolution[0],
        "height": resolution[1],
        "wrappers": [wrap_env],
        **env_kwargs,
    }
    # Remove num_envs and vectorization_mode for non-vectorized case
    if vectorized:
        return gym.make_vec(**all_env_kwargs)
    else:
        all_env_kwargs.pop("num_envs")
        all_env_kwargs.pop("vectorization_mode")
        all_env_kwargs.pop("wrappers")
        return wrap_env(gym.make(**all_env_kwargs))


def build_env(*args, **kwargs):
    kwargs["num_envs"] = 1
    kwargs["vectorized"] = False
    return build_envs(*args, **kwargs)


def build_from_config(env_cfg: DictConfig, mode: str = "train") -> gym.Env:
    resolution = (
        tuple(env_cfg.train_resolution)
        if mode == "train"
        else tuple(env_cfg.full_resolution)
    )
    kwargs: dict = dict(
        env_name=env_cfg.name,
        num_envs=env_cfg.num_envs,
        # observation space
        render_mode="rgb_array",
        resolution=resolution,
        stack_size=env_cfg.stack_size,
        use_luminance=env_cfg.use_luminance,
        # action space discretisation
        bins=env_cfg.action_bins,
        multidiscrete=env_cfg.action_multidiscrete,
        # reward shaping
        use_com_reward=env_cfg.use_com_reward,
        forward_reward_weight=env_cfg.forward_reward_weight,
        healthy_reward=env_cfg.healthy_reward,
        ctrl_cost_weight=env_cfg.ctrl_cost_weight,
        # env-specific init
        healthy_z_range=tuple(env_cfg.healthy_z_range),
        healthy_angle_range=tuple(env_cfg.healthy_angle_range),
        reset_noise_scale=env_cfg.reset_noise_scale,
        # vectorised envs
        vectorized=mode == "train",
    )
    if env_cfg.xml_file is not None:
        kwargs["xml_file"] = env_cfg.xml_file
    return build_envs(**kwargs)


if __name__ == "__main__":
    # Create a single environment and run a simulation in human render mode
    env = build_env(
        env_name="Hopper-v5",
        render_mode="human",
        healthy_z_range=(0.7, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        xml_file=r".venv\Lib\site-packages\gymnasium\envs\mujoco\assets\hopper.xml",
    )

    obs, info = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
