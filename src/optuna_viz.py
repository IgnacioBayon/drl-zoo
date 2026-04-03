from __future__ import annotations

import argparse

from optuna.visualization import plot_optimization_history, plot_param_importances

import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an Optuna study from the shared SQLite database."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Name of the Optuna study to load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    storage = "sqlite:///optuna.db"
    print(f"Loading study '{args.study_name}' from {storage} ...")

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage,
    )

    n_complete = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    print(f"Loaded {len(study.trials)} trials ({n_complete} complete).")

    if n_complete == 0:
        print("No completed trials found — cannot generate plots.")
        return

    print("Plotting optimization history...")
    fig_history = plot_optimization_history(study)
    fig_history.show()

    print("Plotting hyperparameter importances...")
    fig_importance = plot_param_importances(study)
    fig_importance.show()


if __name__ == "__main__":
    main()
