import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from omegaconf import DictConfig

from src.wrappers import (
    CustomReward,
    DiscretizeAction,
    ImageObsWrapper,
    SmoothHopperWrapper,
)

# MuJoCo render resolution — always full; ImageObsWrapper downscales for the agent.
_RENDER_RES = 480


def build_envs(
    env_name: str = "Hopper-v5",
    num_envs: int = 1,
    render_mode: str = "rgb_array",
    bins: int = 5,
    obs_size: int = 64,
    stack_size: int = 4,
    multidiscrete: bool = False,
    target_velocity: float = None,
    vectorized: bool = True,
    **env_kwargs: dict,
):
    def wrap_env(env: gym.Env) -> gym.Env:
        # env = CustomReward(env, use_com_reward)
        env = SmoothHopperWrapper(env, target_velocity)
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


def build_from_config(env_cfg: DictConfig, mode: str = "train") -> gym.Env:
    kwargs: dict = dict(
        env_name=env_cfg.name,
        num_envs=env_cfg.num_envs,
        # observation space
        render_mode="rgb_array",
        obs_size=env_cfg.obs_size,
        stack_size=env_cfg.stack_size,
        # action space discretisation
        bins=env_cfg.action_bins,
        multidiscrete=env_cfg.action_multidiscrete,
        # vectorised envs
        vectorized=mode == "train",
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
    if env_cfg.get("xml_file") is not None:
        kwargs["xml_file"] = env_cfg.xml_file
    if env_cfg.get("max_episode_steps") is not None:
        kwargs["max_episode_steps"] = int(env_cfg.max_episode_steps)
    return build_envs(**kwargs)
