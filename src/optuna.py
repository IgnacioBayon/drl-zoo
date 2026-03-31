from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "optuna"


def _sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    rollout_steps = trial.suggest_categorical(
        "train.rollout_steps", [256, 512, 1024, 2048]
    )
    batch_size = trial.suggest_categorical("train.batch_size", [64, 128, 256, 512])
    if batch_size > rollout_steps:
        batch_size = rollout_steps

    return {
        "train.lr": trial.suggest_float("train.lr", 1e-5, 3e-3, log=True),
        "train.gamma": 1.0
        - trial.suggest_float("train.one_minus_gamma", 1e-4, 5e-2, log=True),
        "train.gae_lambda": trial.suggest_float("train.gae_lambda", 0.9, 0.99),
        "train.clip_ratio": trial.suggest_float("train.clip_ratio", 0.1, 0.35),
        "train.c1": trial.suggest_float("train.c1", 0.2, 1.0),
        "train.c2": trial.suggest_float("train.c2", 1e-4, 5e-2, log=True),
        "train.rollout_steps": rollout_steps,
        "train.batch_size": batch_size,
        "train.update_epochs": trial.suggest_int("train.update_epochs", 3, 10),
        "train.target_kl": trial.suggest_float("train.target_kl", 0.003, 0.05),
        "train.max_grad_norm": trial.suggest_float("train.max_grad_norm", 0.3, 2.0),
    }


def _sample_dqn_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "train.lr": trial.suggest_float("train.lr", 1e-5, 1e-3, log=True),
        "train.gamma": 1.0
        - trial.suggest_float("train.one_minus_gamma", 1e-4, 5e-2, log=True),
        "train.batch_size": trial.suggest_categorical(
            "train.batch_size", [32, 64, 128, 256]
        ),
        "train.buffer_size": trial.suggest_categorical(
            "train.buffer_size", [100_000, 200_000, 300_000, 500_000]
        ),
        "train.train_every": trial.suggest_categorical("train.train_every", [4, 8, 10]),
        "train.target_update_frames": trial.suggest_categorical(
            "train.target_update_frames", [5_000, 10_000, 20_000]
        ),
        "train.epsilon_end": trial.suggest_float("train.epsilon_end", 0.01, 0.1),
        "train.epsilon_anneal_frames": trial.suggest_categorical(
            "train.epsilon_anneal_frames", [500_000, 1_000_000, 2_000_000]
        ),
        "train.max_grad_norm": trial.suggest_float("train.max_grad_norm", 5.0, 20.0),
    }


def _sample_rainbow_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "train.lr": trial.suggest_float("train.lr", 1e-5, 1e-3, log=True),
        "train.gamma": 1.0
        - trial.suggest_float("train.one_minus_gamma", 1e-4, 5e-2, log=True),
        "train.batch_size": trial.suggest_categorical(
            "train.batch_size", [32, 64, 128, 256]
        ),
        "train.buffer_size": trial.suggest_categorical(
            "train.buffer_size", [100_000, 200_000, 300_000, 500_000]
        ),
        "train.n_step": trial.suggest_categorical("train.n_step", [1, 3, 5]),
        "train.per_alpha": trial.suggest_float("train.per_alpha", 0.4, 0.8),
        "train.per_beta_start": trial.suggest_float("train.per_beta_start", 0.2, 0.6),
        "train.max_grad_norm": trial.suggest_float("train.max_grad_norm", 5.0, 20.0),
    }


def _sample_sac_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "train.actor_lr": trial.suggest_float("train.actor_lr", 1e-5, 1e-3, log=True),
        "train.critic_lr": trial.suggest_float("train.critic_lr", 1e-5, 1e-3, log=True),
        "train.alpha_lr": trial.suggest_float("train.alpha_lr", 1e-5, 1e-3, log=True),
        "train.gamma": 1.0
        - trial.suggest_float("train.one_minus_gamma", 1e-4, 5e-2, log=True),
        "train.tau": trial.suggest_float("train.tau", 1e-3, 2e-2, log=True),
        "train.batch_size": trial.suggest_categorical(
            "train.batch_size", [64, 128, 256]
        ),
        "train.gradient_steps": trial.suggest_categorical(
            "train.gradient_steps", [1, 2, 3, 4]
        ),
        "train.start_train_after": trial.suggest_categorical(
            "train.start_train_after", [10_000, 20_000, 50_000]
        ),
    }


def _sample_params(algorithm: str, trial: optuna.Trial) -> dict[str, Any]:
    samplers = {
        "ppo": _sample_ppo_params,
        "dqn": _sample_dqn_params,
        "rainbow": _sample_rainbow_params,
        "sac": _sample_sac_params,
    }
    if algorithm not in samplers:
        raise ValueError(f"Unsupported algorithm '{algorithm}'.")
    return samplers[algorithm](trial)


def _read_best_metric(log_dir: Path) -> float:
    if not log_dir.exists():
        return float("nan")

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])

    for tag in ("eval/mean_reward", "train/avg_reward_100"):
        if tag in tags:
            values = [event.value for event in accumulator.Scalars(tag)]
            if values:
                return float(max(values))
    return float("nan")


def _build_trial_command(
    args: argparse.Namespace,
    trial: optuna.Trial,
    trial_dir: Path,
) -> list[str]:
    sampled = _sample_params(args.algorithm, trial)
    eval_interval = max(1, args.total_frames // args.n_evaluations)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "src.main",
        f"train={args.algorithm}",
        f"env={args.env}",
        f"seed={args.seed + trial.number}",
        f"train.total_frames={args.total_frames}",
        f"eval_interval_frames={eval_interval}",
        f"eval_episodes={args.eval_episodes}",
        f"hydra.run.dir={trial_dir.as_posix()}",
    ]

    for key, value in sampled.items():
        if isinstance(value, float):
            if math.isfinite(value):
                cmd.append(f"{key}={value:.12g}")
            else:
                cmd.append(f"{key}={value}")
        else:
            cmd.append(f"{key}={value}")

    return cmd


def _objective(args: argparse.Namespace, output_root: Path):
    def objective(trial: optuna.Trial) -> float:
        study_prefix = args.study_name if args.study_name else args.algorithm
        trial_dir = output_root / study_prefix / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_trial_command(args, trial, trial_dir)
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

        (trial_dir / "stdout.log").write_text(process.stdout or "", encoding="utf-8")
        (trial_dir / "stderr.log").write_text(process.stderr or "", encoding="utf-8")

        if process.returncode != 0:
            raise optuna.exceptions.TrialPruned(
                f"Training failed with exit code {process.returncode}."
            )

        score = _read_best_metric(trial_dir / "logs")
        if not math.isfinite(score):
            raise optuna.exceptions.TrialPruned(
                "No valid score found in TensorBoard logs."
            )

        trial.report(score, step=0)
        return score

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna studies against the Hydra training pipeline."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["dqn", "rainbow", "ppo", "sac"],
        help="Algorithm config group to optimize.",
    )
    parser.add_argument(
        "--env", type=str, default="hopper", help="Environment config group."
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="Number of random trials before TPE exploitation.",
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=2,
        help="Number of evaluation checkpoints during each training run.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Evaluation episodes at each checkpoint.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=200_000,
        help="Training frames budget per trial.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Global timeout for Optuna optimization in seconds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Parallel Optuna workers."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (required if storage is shared).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where trial outputs and logs are stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=max(1, args.n_evaluations // 3),
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    objective = _objective(args, output_root)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
        )
    except KeyboardInterrupt:
        pass

    if study.best_trial is not None:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best score: {study.best_trial.value:.4f}")
        print("Best params:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
