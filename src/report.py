import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
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


def set_report_style():
    plt.rcParams.update(
        {
            "figure.figsize": (3.3, 2.4),  # single-column report figure
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def moving_average(y, window=20):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y.copy()
    return np.convolve(y, np.ones(window) / window, mode="valid")


def exponential_moving_average(y, alpha=0.1):
    y = np.asarray(y, dtype=float)
    ema = np.zeros_like(y)
    ema[0] = y[0]
    for t in range(1, len(y)):
        ema[t] = alpha * y[t] + (1 - alpha) * ema[t - 1]
    return ema


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


def shorten_run_name(run_name):
    m = re.match(r"run_(\d{8}_\d{6})_.*", run_name)
    return m.group(1) if m else run_name


def format_steps_in_millions(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M"))


def style_report_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_eval_rewards():
    set_report_style()

    log_path = "outputs/run_20260305_130907_Humanoid-v5_rainbow/logs"
    tag = "eval/mean_reward"
    # smooth_window = 20

    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)

    # values_s = moving_average(values, window=20)
    values_s = exponential_moving_average(values, alpha=0.05)
    steps_s = steps[len(steps) - len(values_s) :]

    fig, ax = plt.subplots()

    ax.plot(steps, values, alpha=0.25, linewidth=1.0)
    ax.plot(steps_s, values_s, linewidth=1.8)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval. mean reward")

    format_steps_in_millions(ax)
    style_report_axes(ax)

    plt.tight_layout()
    plt.savefig("report/images/rainbow_results.pdf", bbox_inches="tight")


def plot_dqn_training_rewards():
    set_report_style()

    event_files = find_dqn_event_files(OUTPUTS_DIR)
    if not event_files:
        print("No DQN event files found.")
        return

    n = len(event_files)
    colors = matplotlib.colormaps["tab10"].resampled(n)

    fig, ax = plt.subplots()

    found_any = False

    for i, (run_name, event_file) in enumerate(event_files):
        if "dqn" not in run_name.lower():
            continue

        color = colors(i)
        ax.set_prop_cycle(color=[color])

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

        # rewards_s = moving_average(rewards, window=20)
        rewards_s = exponential_moving_average(rewards, alpha=0.05)
        steps_s = steps[len(steps) - len(rewards_s) :]

        # Raw curve (faint)
        ax.plot(steps, rewards, alpha=0.10, linewidth=0.4, color=color)

        # Smoothed curve (main)
        ax.plot(steps_s, rewards_s, linewidth=0.7, label=label, color=color)

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
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

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
    plt.rcParams.update(
        {
            "figure.figsize": (3.3, 2.4),  # single column width
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
        }
    )

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
    steps_s = steps[len(steps) - len(values_s) :]

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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M"))

    # ---------------------------
    # Clean look (ICML style)
    # ---------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("report/images/dqn_rewards.pdf", bbox_inches="tight")


def plot_eval_reward_speed_final_x() -> None:
    """Plot evaluation reward, speed, and final x position with exponential smoothing."""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

    def exponential_moving_average(
        values: list[float] | np.ndarray, alpha: float = 0.1
    ) -> np.ndarray:
        """Return exponentially smoothed values with the same length as the input."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr

        smoothed = np.empty_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, arr.size):
            smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
        return smoothed

    log_path = "outputs/hopper_dqn"
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    # Extract reward
    reward_tag = "episode/reward"
    reward_events = ea.Scalars(reward_tag)
    reward_steps = np.array([e.step for e in reward_events], dtype=float)
    reward_values = np.array([e.value for e in reward_events], dtype=float)

    # Extract speed
    speed_tag = "episode/avg_speed"
    speed_events = ea.Scalars(speed_tag)
    speed_steps = np.array([e.step for e in speed_events], dtype=float)
    speed_values = np.array([e.value for e in speed_events], dtype=float)

    # Extract final x position
    x_tag = "episode/final_x"
    x_events = ea.Scalars(x_tag)
    x_steps = np.array([e.step for e in x_events], dtype=float)
    x_values = np.array([e.value for e in x_events], dtype=float)

    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 2.4),
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
        }
    )

    reward_values_s = exponential_moving_average(reward_values)
    speed_values_s = exponential_moving_average(speed_values)
    x_values_s = exponential_moving_average(x_values)

    values_dict = {
        "Reward": (reward_steps, reward_values, reward_values_s),
        "Speed": (speed_steps, speed_values, speed_values_s),
        "Final x": (x_steps, x_values, x_values_s),
    }

    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.4))

    for i, metric in enumerate(["Speed", "Reward", "Final x"]):
        steps, values, values_s = values_dict[metric]
        axs[i].plot(steps, values, alpha=0.25, linewidth=1.0)
        axs[i].plot(steps, values_s, linewidth=1.8)
        axs[i].set_xlabel("Environment steps")
        axs[i].set_ylabel(metric)
        axs[i].set_ylim(np.percentile(values, 5), np.percentile(values, 95))
        axs[i].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M")
        )
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[0].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M")
        )
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("report/images/dqn_hopper_results.pdf", bbox_inches="tight")


def plot_eval_reward_speed_final_x() -> None:
    """Plot evaluation reward, speed, and final x position with exponential smoothing."""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

    def exponential_moving_average(
        values: list[float] | np.ndarray, alpha: float = 0.1
    ) -> np.ndarray:
        """Return exponentially smoothed values with the same length as the input."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr

        smoothed = np.empty_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, arr.size):
            smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
        return smoothed

    log_path = "outputs/hopper_dqn"
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    # Extract reward
    reward_tag = "episode/reward"
    reward_events = ea.Scalars(reward_tag)
    reward_steps = np.array([e.step for e in reward_events], dtype=float)
    reward_values = np.array([e.value for e in reward_events], dtype=float)

    # Extract speed
    speed_tag = "episode/avg_speed"
    speed_events = ea.Scalars(speed_tag)
    speed_steps = np.array([e.step for e in speed_events], dtype=float)
    speed_values = np.array([e.value for e in speed_events], dtype=float)

    # Extract final x position
    x_tag = "episode/final_x"
    x_events = ea.Scalars(x_tag)
    x_steps = np.array([e.step for e in x_events], dtype=float)
    x_values = np.array([e.value for e in x_events], dtype=float)

    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 2.4),
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
        }
    )

    reward_values_s = exponential_moving_average(reward_values)
    speed_values_s = exponential_moving_average(speed_values)
    x_values_s = exponential_moving_average(x_values)

    values_dict = {
        "Reward": (reward_steps, reward_values, reward_values_s),
        "Speed": (speed_steps, speed_values, speed_values_s),
        "Final x": (x_steps, x_values, x_values_s),
    }

    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.4))

    for i, metric in enumerate(["Speed", "Reward", "Final x"]):
        steps, values, values_s = values_dict[metric]
        axs[i].plot(steps, values, alpha=0.25, linewidth=1.0)
        axs[i].plot(steps, values_s, linewidth=1.8)
        axs[i].set_xlabel("Environment steps")
        axs[i].set_ylabel(metric)
        axs[i].set_ylim(np.percentile(values, 5), np.percentile(values, 95))
        axs[i].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M")
        )
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[0].xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M")
        )
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("report/images/dqn_hopper_results.pdf", bbox_inches="tight")


def main():
    # plot_dqn_training_rewards()
    plot_eval_rewards()


if __name__ == "__main__":
    main()
