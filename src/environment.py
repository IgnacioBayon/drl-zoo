import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from omegaconf import DictConfig

from src.wrappers import (
    DiscretizeAction,
    FreezeJointsWrapper,
    ImageObsWrapper,
    NormalizeActions,
    RGBImageObsWrapper,
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
    frozen_joints: list[int] | None = None,
    vectorized: bool = True,
    discretize_actions: bool = False,
    image_format: str = "gray",
    normalize_actions: bool = False,
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
        elif normalize_actions:
            env = NormalizeActions(env)
        if render_mode != "human":
            if image_format == "rgb":
                env = RGBImageObsWrapper(env, obs_size=obs_size)
            else:
                env = ImageObsWrapper(env, obs_size=obs_size)
        if stack_size and stack_size > 1:
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
    force_discretize_actions: bool | None = None,
) -> gym.Env:
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
        # image format: "gray" (default) or "rgb" for pixel world models
        image_format=str(train_cfg.get("image_format", "gray")),
        normalize_actions=bool(train_cfg.get("normalize_actions", False)),
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
    return build_envs(**kwargs)
