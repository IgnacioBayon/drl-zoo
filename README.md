# RL Benchmarking for Pixel-Based Continuous Control

A high-performance implementation and benchmarking suite of Reinforcement Learning algorithms for Mujoco environments using pixel-based observations. This project explores the effectiveness of various algorithms on complex continuous control tasks like `Hopper-v5`.

## 🚀 Key Features

- **Diverse Algorithm Suite**: Implementation of both value-based and policy-based algorithms.
- **Pixel-Based Observations**: Learning directly from visual inputs (rendered Mujoco frames).
- **Common Hydra Interface**: Unified configuration and execution interface across all algorithms.
- **Optimized Environment**: Uses [uv](https://github.com/astral-sh/uv) for fast and reproducible package management.

## 🧠 Supported Algorithms

| Algorithm       | Status         | Type         | Notes                                                                   |
| :-------------- | :------------- | :----------- | :---------------------------------------------------------------------- |
| **DQN**         | ✅ Implemented | Value-Based  | Discretized actions for continuous control.                             |
| **Rainbow DQN** | ✅ Implemented | Value-Based  | Includes all 6 standard improvements.                                   |
| **PPO**         | 🚧 In Progress | Policy-Based | On-policy algorithm designed for continuous spaces.                     |
| **SAC**         | 🚧 In Progress | Actor-Critic | Off-policy Maximum Entropy RL, highly efficient for continuous control. |

> **Benchmark Note**: While DQN and Rainbow are optimized for discrete actions (implemented here via discretization), PPO and SAC are expected to achieve superior performance as they natively handle the continuous action spaces of Mujoco environments.

## 🛠️ Installation

This project uses `uv` for dependency management.

1.  **Install uv**:
    ```powershell
    # On Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Or on macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone and Sync**:
    ```bash
    git clone https://github.com/Gomi/rainbow-dqn.git
    cd rainbow-dqn
    uv sync
    ```

## 🏃 Running the Algorithms

The project uses [Hydra](https://hydra.cc/) to manage configurations. Use overrides to switch between algorithms and environments.

### 1. Run DQN / Rainbow

```bash
# Standard DQN (default)
uv run python -m src.main

# Rainbow DQN
uv run python -m src.main train=rainbow
```

### 2. Run PPO / SAC (Upcoming)

```bash
# PPO
uv run python -m src.main train=ppo

# SAC
uv run python -m src.main train=sac
```

### 3. Hyperparameter Overrides

You can override any parameter defined in the YAML configurations directly from the CLI:

```bash
# Change environment and seed
uv run python -m src.main env=hopper seed=42

# Modify training settings
uv run python -m src.main train.total_frames=1_000_000 train.lr=3e-4
```

## ⚙️ Configuration Structure

Configurations are organized in the `config/` directory:

- `config.yaml`: Main configuration entry (defaults, paths).
- `train/`: Algorithm-specific training parameters and model definitions (DQN, Rainbow, SAC...).
- `env/`: Environment-specific settings (resolution, reward shaping).

The `outputs/` folder stores logs, checkpoints, and videos for each run.

## 📊 Results and Logging

TensorBoard logs are automatically generated for every run. To visualize them:

```bash
uv run tensorboard --logdir outputs/
```

Training outputs are stored in `outputs/run_<timestamp>_<env>_<model>/`:

- `logs/`: TensorBoard events.
- `checkpoints/`: Model checkpoints (`.pt`).
- `videos/`: Video recordings of evaluation episodes.
