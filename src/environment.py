from typing import Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from omegaconf import DictConfig
from tensordict import TensorDict

from src.wrappers import (
    DiscretizeAction,
    FreezeJointsWrapper,
    ImageObsWrapper,
    SmoothHopperWrapper,
)

# MuJoCo render resolution — always full; ImageObsWrapper downscales for the agent.
_RENDER_RES = 480


class DreamerEnvBatch:
    """Batched environment adapter expected by Dreamer trainer.

    Exposes:
    - env_num
    - step(action_batch, done_batch) -> (TensorDict transition, done_batch)
    """

    def __init__(self, envs: list[gym.Env]):
        if not envs:
            raise ValueError("DreamerEnvBatch requires at least one environment.")
        self.envs = envs
        self.env_num = len(envs)
        self.action_space = self._annotate_action_space(envs[0].action_space)
        self.observation_space = self._build_observation_space(
            envs[0].observation_space
        )

    @staticmethod
    def _annotate_action_space(action_space: gym.Space) -> gym.Space:
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            setattr(action_space, "multi_discrete", True)
        elif isinstance(action_space, gym.spaces.Discrete):
            setattr(action_space, "discrete", True)
        return action_space

    @staticmethod
    def _obs_to_image(obs: np.ndarray) -> np.ndarray:
        image = np.asarray(obs)
        if image.ndim == 2:
            image = image[..., None]
        elif image.ndim == 3:
            # FrameStackObservation returns (T, H, W); Dreamer expects (H, W, C).
            if image.shape[0] <= 16 and image.shape[1] == image.shape[2]:
                image = np.transpose(image, (1, 2, 0))
        return image.astype(np.uint8, copy=False)

    def _build_observation_space(self, base_space: gym.Space) -> gym.spaces.Dict:
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError(
                "DreamerEnvBatch requires Box observations after wrappers, "
                f"got {type(base_space).__name__}."
            )
        image_shape = self._obs_to_image(
            np.zeros(base_space.shape, dtype=base_space.dtype)
        ).shape
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=image_shape,
                    dtype=np.uint8,
                ),
                "reward": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "is_first": gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
                "is_last": gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
                "is_terminal": gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            }
        )

    @staticmethod
    def _format_action(action: np.ndarray, action_space: gym.Space):
        if isinstance(action_space, gym.spaces.Discrete):
            a = np.asarray(action)
            if a.ndim == 0:
                return int(a)
            if a.size == action_space.n:
                return int(np.argmax(a))
            return int(a.reshape(-1)[0])
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            a = np.asarray(action)
            if a.shape == action_space.shape:
                return np.clip(np.rint(a), 0, action_space.nvec - 1).astype(np.int64)
            return np.zeros_like(action_space.nvec, dtype=np.int64)
        if isinstance(action_space, gym.spaces.Box):
            return np.asarray(action, dtype=action_space.dtype)
        raise TypeError(f"Unsupported action space: {type(action_space).__name__}")

    def step(self, action_batch: torch.Tensor, done_batch: torch.Tensor):
        actions = action_batch.detach().cpu().numpy()
        done = done_batch.detach().cpu().numpy().astype(bool)

        images = []
        rewards = []
        is_first = []
        is_last = []
        is_terminal = []
        next_done = np.zeros(self.env_num, dtype=bool)

        for i, env in enumerate(self.envs):
            if done[i]:
                obs, _ = env.reset()
                rew = 0.0
                terminated = False
                truncated = False
                first = True
            else:
                action = self._format_action(actions[i], env.action_space)
                obs, rew, terminated, truncated, _ = env.step(action)
                first = False

            last = bool(terminated or truncated)
            next_done[i] = last
            images.append(self._obs_to_image(obs))
            rewards.append([float(rew)])
            is_first.append(first)
            is_last.append(last)
            is_terminal.append(bool(terminated))

        trans = TensorDict(
            {
                "image": torch.as_tensor(np.stack(images), dtype=torch.uint8),
                "reward": torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
                "is_first": torch.as_tensor(np.asarray(is_first), dtype=torch.bool),
                "is_last": torch.as_tensor(np.asarray(is_last), dtype=torch.bool),
                "is_terminal": torch.as_tensor(
                    np.asarray(is_terminal), dtype=torch.bool
                ),
            },
            batch_size=(self.env_num,),
        )
        return trans, torch.as_tensor(next_done, dtype=torch.bool)

    def close(self):
        for env in self.envs:
            env.close()


def build_envs(
    env_name: str = "Hopper-v5",
    num_envs: int = 1,
    render_mode: str = "rgb_array",
    bins: int = 5,
    obs_size: int = 64,
    stack_size: int = 4,
    multidiscrete: bool = False,
    target_velocity: float = None,
    frozen_joints: list[int] | None = None,
    vectorized: bool = True,
    discretize_actions: bool = False,
    **env_kwargs: dict,
):
    def wrap_env(env: gym.Env) -> gym.Env:
        if frozen_joints:
            env = FreezeJointsWrapper(env, frozen_joints)
        if target_velocity is not None:
            env_id = getattr(getattr(env, "spec", None), "id", "")
            if "Hopper" in env_id:
                env = SmoothHopperWrapper(env, target_velocity)
        if discretize_actions:
            env = DiscretizeAction(env, bins, multidiscrete)
        if render_mode != "human":
            env = ImageObsWrapper(env, obs_size=obs_size)
        env = FrameStackObservation(env, stack_size)
        return env

    all_env_kwargs = {
        "id": env_name,
        "num_envs": num_envs,
        "vectorization_mode": "async",
        "render_mode": render_mode,
        "width": _RENDER_RES,
        "height": _RENDER_RES,
        "wrappers": [wrap_env],
        "exclude_current_positions_from_observation": False,
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


def build_from_config(
    env_cfg: DictConfig,
    train_cfg: DictConfig,
    mode: str = "train",
    eval: bool = False,
    force_discretize_actions: bool | None = None,
) -> Union[
    gym.Env, tuple[DreamerEnvBatch, DreamerEnvBatch, gym.spaces.Dict, gym.Space]
]:
    use_discretize_actions = (
        bool(train_cfg.get("discretize_actions", False))
        if force_discretize_actions is None
        else bool(force_discretize_actions)
    )

    kwargs: dict = dict(
        env_name=env_cfg.name,
        num_envs=env_cfg.num_envs,
        # observation space
        render_mode="rgb_array",
        obs_size=env_cfg.obs_size,
        stack_size=env_cfg.stack_size,
        # action space discretisation
        bins=int(train_cfg.get("action_bins", 0)),
        multidiscrete=bool(train_cfg.get("action_multidiscrete", False)),
        discretize_actions=use_discretize_actions,
        # vectorised envs
        vectorized=mode == "train",
    )

    if use_discretize_actions and train_cfg.get("action_bins") is None:
        raise ValueError(
            "Action discretization is enabled but train.action_bins is missing."
        )
    # Reward shaping
    if env_cfg.get("forward_reward_weight") is not None:
        kwargs["forward_reward_weight"] = env_cfg.forward_reward_weight
    if env_cfg.get("healthy_reward") is not None:
        kwargs["healthy_reward"] = env_cfg.healthy_reward
    if env_cfg.get("ctrl_cost_weight") is not None:
        kwargs["ctrl_cost_weight"] = env_cfg.ctrl_cost_weight
    if env_cfg.get("contact_cost_weight") is not None:
        kwargs["contact_cost_weight"] = env_cfg.contact_cost_weight
    if env_cfg.get("target_velocity") is not None:
        kwargs["target_velocity"] = env_cfg.target_velocity
    # Env-specific init
    if env_cfg.get("healthy_z_range") is not None:
        kwargs["healthy_z_range"] = tuple(env_cfg.healthy_z_range)
    if env_cfg.get("healthy_angle_range") is not None:
        kwargs["healthy_angle_range"] = tuple(env_cfg.healthy_angle_range)
    if env_cfg.get("reset_noise_scale") is not None:
        kwargs["reset_noise_scale"] = env_cfg.reset_noise_scale
    if env_cfg.get("frame_skip") is not None:
        kwargs["frame_skip"] = int(env_cfg.frame_skip)
    if env_cfg.get("xml_file") is not None:
        kwargs["xml_file"] = env_cfg.xml_file
    if env_cfg.get("max_episode_steps") is not None:
        kwargs["max_episode_steps"] = int(env_cfg.max_episode_steps)
    if env_cfg.get("frozen_joints") is not None:
        kwargs["frozen_joints"] = list(env_cfg.frozen_joints)
    if not eval:
        return build_envs(**kwargs)

    train_envs = DreamerEnvBatch(
        [
            build_envs(
                **{**kwargs, "num_envs": 1, "vectorized": False},
            )
            for _ in range(int(env_cfg.num_envs))
        ]
    )

    eval_envs = DreamerEnvBatch(
        [
            build_envs(
                **{**kwargs, "num_envs": 1, "vectorized": False},
            )
            for _ in range(max(1, int(train_cfg.trainer.eval_episode_num)))
        ]
    )

    return (
        train_envs,
        eval_envs,
        train_envs.observation_space,
        train_envs.action_space,
    )
