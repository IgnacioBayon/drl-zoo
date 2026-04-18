"""DreamerV3 / R2Dreamer training loop for gym-vec MuJoCo envs.

Follows the same Hydra + TensorBoard + checkpoint/video conventions as the
other trainers in this repo (see ``src/sac/train.py``). The world model and
actor-critic updates come verbatim from the R2Dreamer reference implementation
(see ``r2dreamer/`` sibling directory); only the environment interaction,
replay-buffer bookkeeping and evaluation glue are re-implemented here to work
with Gymnasium vector envs instead of the original parallel env pool.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from time import perf_counter

import gymnasium as gym
import imageio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from src.environment import build_from_config
from src.utils import get_device, save_checkpoint

from .buffer import Buffer
from .dreamer import Dreamer
from .tools import Every

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Obs-space helper
# ---------------------------------------------------------------------------
def _build_obs_space(obs_shape: tuple[int, int, int]) -> gym.spaces.Dict:
    """Construct the dict observation space Dreamer's encoder expects."""
    h, w, c = obs_shape
    return gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, (h, w, c), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        }
    )


def _make_trans(
    obs: np.ndarray,
    is_first: np.ndarray,
    is_last: np.ndarray,
    is_terminal: np.ndarray,
    reward: np.ndarray,
    device: torch.device,
    num_envs: int,
) -> TensorDict:
    """Pack current step observation + flags into a Dreamer TensorDict."""
    return TensorDict(
        {
            "image": torch.as_tensor(obs, device=device),
            "is_first": torch.as_tensor(
                is_first[:, None], device=device, dtype=torch.bool
            ),
            "is_last": torch.as_tensor(
                is_last[:, None], device=device, dtype=torch.bool
            ),
            "is_terminal": torch.as_tensor(
                is_terminal[:, None], device=device, dtype=torch.bool
            ),
            "reward": torch.as_tensor(
                reward[:, None], device=device, dtype=torch.float32
            ),
        },
        batch_size=(num_envs,),
    )


# ---------------------------------------------------------------------------
# Evaluation (stateful, uses Dreamer's RSSM; single-env)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(
    agent: Dreamer,
    env_cfg: DictConfig,
    train_cfg: DictConfig,
    device: torch.device,
    n_episodes: int,
    step: int,
    video_dir: str,
    writer: SummaryWriter,
) -> tuple[float, float, float]:
    """Run greedy eval episodes, record best, log reward/final_x/avg_speed."""
    agent.eval()
    env = build_from_config(env_cfg, train_cfg, mode="eval")

    returns = np.empty(n_episodes, dtype=np.float64)
    final_xs = np.empty(n_episodes, dtype=np.float64)
    ep_steps = np.empty(n_episodes, dtype=np.int64)

    def run_episode(
        seed: int, record: bool
    ) -> tuple[float, float, int, list[np.ndarray]]:
        obs, _ = env.reset(seed=seed)
        agent_state = agent.get_initial_state(1)
        frames: list[np.ndarray] = []
        total_r = 0.0
        steps_ep = 0
        done = False
        is_first = True

        while not done:
            if record:
                frames.append(env.render())
            trans = TensorDict(
                {
                    "image": torch.as_tensor(obs, device=device).unsqueeze(0),
                    "is_first": torch.as_tensor(
                        [[is_first]], device=device, dtype=torch.bool
                    ),
                    "is_last": torch.zeros(1, 1, dtype=torch.bool, device=device),
                    "is_terminal": torch.zeros(1, 1, dtype=torch.bool, device=device),
                    "reward": torch.zeros(1, 1, dtype=torch.float32, device=device),
                },
                batch_size=(1,),
            )
            act, agent_state = agent.act(trans, agent_state, eval=True)
            action_np = act.squeeze(0).detach().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            total_r += float(reward)
            steps_ep += 1
            done = terminated or truncated
            is_first = False

        final_x = float(env.unwrapped.data.qpos[0])
        return total_r, final_x, steps_ep, frames

    for i in range(n_episodes):
        returns[i], final_xs[i], ep_steps[i], _ = run_episode(
            seed=int(step) + i, record=False
        )

    mean_r, std_r = float(returns.mean()), float(returns.std())
    best_idx = int(returns.argmax())
    _, _, _, frames = run_episode(seed=int(step) + best_idx, record=True)
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimsave(
        os.path.join(video_dir, f"eval_step_{step}.mp4"), frames, fps=30
    )

    env.close()
    agent.train()

    writer.add_scalar("eval/mean_reward", mean_r, step)
    writer.add_scalar("eval/final_x", float(final_xs.mean()), step)
    avg_speeds = final_xs / (ep_steps * 0.008)
    writer.add_scalar("eval/avg_speed", float(avg_speeds.mean()), step)
    return mean_r, std_r, float(final_xs.mean())


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def train_dreamer(cfg: DictConfig) -> None:
    """Build envs, replay buffer, Dreamer agent and launch online training."""
    device = get_device(cfg.train.device)

    # -- force dreamer-friendly env overrides ---------------------------------
    # DreamerV3 expects RGB (H, W, 3) obs w/o frame stacking, stateful recurrence.
    OmegaConf.set_struct(cfg, False)
    cfg.env.num_envs = int(cfg.train.num_envs)
    cfg.env.stack_size = 1
    cfg.train.image_format = "rgb"
    # Resolve "auto" so that every ${train.device} interpolation below
    # (rssm / encoder / decoder / actor / critic / buffer) gets a real string.
    cfg.train.device = str(device)
    OmegaConf.set_struct(cfg, True)

    # -- envs -----------------------------------------------------------------
    envs = build_from_config(cfg.env, cfg.train, mode="train")
    num_envs = int(cfg.env.num_envs)
    assert num_envs == int(cfg.train.num_envs)

    obs_shape = tuple(envs.single_observation_space.shape)  # (H, W, 3)
    if len(obs_shape) != 3 or obs_shape[-1] != 3:
        raise ValueError(
            f"DreamerV3 expects RGB (H, W, 3) observations; got {obs_shape}."
        )
    if obs_shape[0] % 16 != 0 or obs_shape[1] % 16 != 0:
        raise ValueError(
            "obs_size must be divisible by 16 for the default Dreamer CNN "
            f"(4 pooling stages); got {obs_shape[:2]}."
        )

    action_space = envs.single_action_space
    action_dim = int(action_space.shape[0])

    obs_space = _build_obs_space(obs_shape)

    # -- agent + buffer -------------------------------------------------------
    agent = Dreamer(cfg.train.model, obs_space, action_space).to(device)
    replay_buffer = Buffer(cfg.train.buffer)

    # -- TensorBoard ----------------------------------------------------------
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)

    # -- training loop bookkeeping -------------------------------------------
    total_frames = int(cfg.train.total_frames)
    batch_length = int(cfg.train.buffer.batch_length)
    batch_size = int(cfg.train.buffer.batch_size)
    batch_steps = batch_size * batch_length
    train_ratio = float(cfg.train.train_ratio)
    updates_needed = Every(batch_steps / train_ratio)
    prefill = max(int(cfg.train.prefill), batch_length + 2)

    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)
    reward_window: deque[float] = deque(maxlen=100)
    reward_window_sum = 0.0

    train_metrics: dict[str, float] = {}
    metric_log_every = int(cfg.train.log_train_every)
    metric_every = Every(metric_log_every)

    # -- init envs ------------------------------------------------------------
    obs, _ = envs.reset(seed=int(cfg.seed))
    is_first = np.ones(num_envs, dtype=bool)
    is_last = np.zeros(num_envs, dtype=bool)
    is_terminal = np.zeros(num_envs, dtype=bool)
    reward = np.zeros(num_envs, dtype=np.float32)

    agent_state = agent.get_initial_state(num_envs)
    episode_ids = torch.arange(num_envs, dtype=torch.int32, device=device)

    global_step = 0
    update_count = 0
    start = perf_counter()

    log.info(
        "Training R2Dreamer on %s | device=%s | envs=%d | obs=%s | action_dim=%d",
        cfg.env.name,
        device,
        num_envs,
        obs_shape,
        action_dim,
    )

    while global_step < total_frames:
        # -- build trans for current obs ---------------------------------------
        trans = _make_trans(obs, is_first, is_last, is_terminal, reward, device, num_envs)

        # -- policy inference --------------------------------------------------
        act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

        # Env is wrapped with NormalizeActions (see cfg.train.normalize_actions),
        # so the action passed to the env is already in its expected [-1, 1] range.
        action_np = act.detach().cpu().numpy().astype(np.float32)
        done_t = torch.as_tensor(is_last, dtype=torch.bool, device=device)
        trans["action"] = act * (~done_t.unsqueeze(-1))
        trans["stoch"] = agent_state["stoch"]
        trans["deter"] = agent_state["deter"]
        trans["episode"] = episode_ids

        replay_buffer.add_transition(trans.detach())

        # -- env step ----------------------------------------------------------
        next_obs, step_reward, term, trunc, infos = envs.step(action_np)
        done = np.logical_or(term, trunc)

        # -- episode bookkeeping ----------------------------------------------
        # Reward for this transition lands at NEXT iter's trans (see is_first logic below).
        # But we also need to accumulate episode returns NOW (before flags are overwritten).
        for e in range(num_envs):
            if is_first[e]:
                # Start of new episode (reset obs) — reset counters.
                ep_returns[e] = 0.0
                ep_lengths[e] = 0
            ep_returns[e] += float(step_reward[e])
            ep_lengths[e] += 1
            if done[e]:
                # Episode just ended — log.
                ret = ep_returns[e]
                if len(reward_window) == reward_window.maxlen:
                    reward_window_sum -= reward_window[0]
                reward_window.append(ret)
                reward_window_sum += ret

                final_x = 0.0
                if "x_position" in infos:
                    final_x = float(infos["x_position"][e])
                ep_time = float(ep_lengths[e]) * 0.008
                avg_speed = final_x / ep_time if ep_time > 0 else 0.0

                writer.add_scalar("episode/reward", ret, global_step)
                writer.add_scalar("episode/final_x", final_x, global_step)
                writer.add_scalar("episode/avg_speed", avg_speed, global_step)
                writer.add_scalar("episode/length", int(ep_lengths[e]), global_step)

        # -- compute flags for next iter --------------------------------------
        was_last = is_last.copy()
        # NEXT_STEP autoreset: after `done` at current step, envs.step will return the
        # reset obs at the NEXT iter (action ignored, reward=0, done=False). So:
        #   - envs where was_last=True → current next_obs is a reset obs → is_first=True next iter, reward=0
        #   - envs where done=True (this step) → next_obs is the terminal obs → is_last=True next iter
        new_is_first = was_last.copy()
        new_is_last = np.where(was_last, False, done)
        new_is_terminal = np.where(was_last, False, term)
        new_reward = np.where(was_last, 0.0, step_reward.astype(np.float32))

        obs = next_obs
        is_first = new_is_first
        is_last = new_is_last
        is_terminal = new_is_terminal
        reward = new_reward

        global_step += num_envs

        # -- world-model / actor-critic updates -------------------------------
        buf_count = replay_buffer.count()
        if buf_count > prefill:
            n_updates = updates_needed(global_step)
            for _ in range(n_updates):
                mets = agent.update(replay_buffer)
                update_count += 1
                for k, v in mets.items():
                    if isinstance(v, torch.Tensor):
                        v = float(v.detach().cpu().item())
                    train_metrics[k] = float(v)

        # -- periodic logging --------------------------------------------------
        if metric_every(global_step):
            avg_reward = (
                reward_window_sum / len(reward_window) if reward_window else 0.0
            )
            elapsed = perf_counter() - start
            fps = global_step / elapsed if elapsed > 0 else 0.0
            writer.add_scalar("train/fps", fps, global_step)
            writer.add_scalar("train/buffer_size", buf_count, global_step)
            writer.add_scalar("train/updates", update_count, global_step)
            writer.add_scalar("train/avg_reward_100", avg_reward, global_step)
            for name, value in train_metrics.items():
                tag = name if "/" in name else f"train/{name}"
                writer.add_scalar(tag, value, global_step)

            msg = (
                f"frame {global_step}/{total_frames} | "
                f"reward {avg_reward:.2f} | buffer {buf_count} | "
                f"updates {update_count} | fps {fps:.0f}"
            )
            for key in ("loss/dyn", "loss/rep", "loss/policy", "loss/value", "loss/rew"):
                if key in train_metrics:
                    msg += f" | {key} {train_metrics[key]:.3f}"
            log.info(msg)

        # -- periodic evaluation + checkpoint ---------------------------------
        eval_now = (
            (global_step // int(cfg.eval_interval_frames))
            != ((global_step - num_envs) // int(cfg.eval_interval_frames))
        )
        if eval_now and buf_count > prefill:
            mean_r, std_r, _ = _evaluate(
                agent=agent,
                env_cfg=cfg.env,
                train_cfg=cfg.train,
                device=device,
                n_episodes=int(cfg.eval_episodes),
                step=global_step,
                video_dir=cfg.paths.video_dir,
                writer=writer,
            )
            log.info(
                "eval @ %d frames: mean_reward %.2f +/- %.2f",
                global_step,
                mean_r,
                std_r,
            )

            os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "global_step": global_step,
                    "agent": agent.state_dict(),
                },
                os.path.join(
                    cfg.paths.checkpoint_dir, f"dreamer_{global_step}.pt"
                ),
            )

    # -- final checkpoint -----------------------------------------------------
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    torch.save(
        {"global_step": global_step, "agent": agent.state_dict()},
        os.path.join(cfg.paths.checkpoint_dir, "dreamer_final.pt"),
    )

    writer.close()
    envs.close()
    elapsed = perf_counter() - start
    avg_fps = global_step / elapsed if elapsed > 0 else 0.0
    log.info("Done -- %.1fs | avg FPS %.0f | total updates %d", elapsed, avg_fps, update_count)
