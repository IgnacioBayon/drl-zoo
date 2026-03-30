import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


OUTPUTS_DIR = Path("outputs")

REWARD_TAG_CANDIDATES = [
    "eval/episode_reward",
    "eval/mean_reward",
    "rollout/ep_rew_mean",
    "train/episode_reward",
    "train/ep_rew_mean",
    "episode_reward",
    "reward",
]


def find_dqn_event_files(outputs_dir: Path):
    event_files = []
    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if "_dqn" not in run_dir.name:
            continue

        logs_dir = run_dir / "logs"
        if not logs_dir.exists():
            continue

        for f in logs_dir.iterdir():
            if f.is_file() and f.name.startswith("events.out.tfevents"):
                event_files.append((run_dir.name, f))
    return sorted(event_files)


def load_event_accumulator(event_file: Path):
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    return ea


def choose_reward_tag(tags):
    scalar_tags = tags.get("scalars", [])
    for candidate in REWARD_TAG_CANDIDATES:
        if candidate in scalar_tags:
            return candidate
    return None


def extract_scalar_series(ea, tag):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def moving_average(y, window=25):
    if len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    return y_smooth


def shorten_run_name(run_name):
    # Example:
    # run_20260306_173046_Humanoid-v5_dqn -> 20260306_173046
    m = re.match(r"run_(\d{8}_\d{6})_.*", run_name)
    return m.group(1) if m else run_name


def main():
    event_files = find_dqn_event_files(OUTPUTS_DIR)

    if not event_files:
        print("No DQN event files found.")
        return

    plt.figure(figsize=(12, 7))

    found_any = False
    window = 30  # try 20, 30, 50

    for run_name, event_file in event_files:
        ea = load_event_accumulator(event_file)
        reward_tag = choose_reward_tag(ea.Tags())

        if reward_tag is None:
            print(f"[WARNING] No known reward tag found in {run_name}")
            print("Available scalar tags:", ea.Tags().get("scalars", []))
            continue

        steps, rewards = extract_scalar_series(ea, reward_tag)

        if len(steps) == 0:
            continue

        label = shorten_run_name(run_name)

        # Raw curve, faint
        plt.plot(steps, rewards, alpha=0.15, linewidth=1)

        # Smoothed curve
        smooth_rewards = moving_average(rewards, window=window)
        if len(rewards) >= window:
            smooth_steps = steps[window - 1:]
        else:
            smooth_steps = steps

        plt.plot(smooth_steps, smooth_rewards, linewidth=2, label=label)
        found_any = True

    if not found_any:
        print("No reward series could be plotted.")
        return

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"DQN training rewards (moving average, window={window})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Run", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("dqn_rewards.pdf")
    plt.show()


if __name__ == "__main__":
    main()