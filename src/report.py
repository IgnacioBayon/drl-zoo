import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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


def plot_dqn_training_rewards():
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
    plt.savefig("report/images/dqn_rewards.pdf")
    plt.show()


def plot_eval_rewards():
    from tensorboard.backend.event_processing import event_accumulator
    import matplotlib.pyplot as plt

    log_path = "outputs/run_20260305_130907_Humanoid-v5_rainbow/logs"

    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    # Print available tags
    # print(ea.Tags()["scalars"])

    # Extract data
    tag = "eval/mean_reward"  # <- change if needed
    events = ea.Scalars(tag)

    steps = [e.step for e in events]
    values = [e.value for e in events]

    # ---------------------------
    # ICML-style settings
    # ---------------------------
    plt.rcParams.update({
        "figure.figsize": (3.3, 2.4),   # single column width
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---------------------------
    # Moving average
    # ---------------------------
    def moving_average(x, w=20):
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # ---------------------------
    # Replace with your data
    # ---------------------------
    # steps = np.array(steps)
    # values = np.array(values)

    values_s = moving_average(values, w=20)
    steps_s = steps[len(steps) - len(values_s):]

    # ---------------------------
    # Plot
    # ---------------------------
    fig, ax = plt.subplots()

    # Raw curve (faint)
    ax.plot(steps, values, alpha=0.25, linewidth=1.0)

    # Smoothed curve (main)
    ax.plot(steps_s, values_s, linewidth=1.8)

    # Labels (short and clean)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval. mean reward")

    # Optional small title (can remove if you prefer pure ICML style)
    # ax.set_title("Rainbow on Humanoid-v5\n20-episode moving average", pad=4)

    # ---------------------------
    # Format x-axis in millions
    # ---------------------------
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x/1e6:.0f}M")
    )

    # ---------------------------
    # Clean look (ICML style)
    # ---------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # ---------------------------
    # Save as vector PDF
    # ---------------------------
    plt.savefig("report/images/rainbow_results.pdf", bbox_inches="tight")


def main():
    # plot_dqn_training_rewards()
    plot_eval_rewards()



if __name__ == "__main__":
    main()