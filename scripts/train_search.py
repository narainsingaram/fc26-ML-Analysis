#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.model_search import run_model_search  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian/Optuna sweeps for multiple tabular models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path, help="Path to EA FC26 CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir, help="Where to write artifacts.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_CONFIG.test_size, help="Validation split fraction.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_CONFIG.random_state, help="Random seed.")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per booster (if optuna installed).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    df_raw = load_player_data(config.data_path, config.age_bins)
    results, best = run_model_search(df_raw, config, n_trials=args.trials, test_size=args.test_size)

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    for res in results:
        artifact_name = f"{res.name.lower()}_model.joblib"
        joblib.dump(res.pipeline, output / artifact_name)
        leaderboard.append(
            {
                "model": res.name,
                "mae": res.holdout_metrics["mae"],
                "rmse": res.holdout_metrics["rmse"],
                "r2": res.holdout_metrics["r2"],
                "params": res.params,
                "artifact": artifact_name,
            }
        )

    # Persist best model under the standard path for the API/UI.
    best_path = output / "model.joblib"
    joblib.dump(best.pipeline, best_path)

    # Save metrics snapshot for the best candidate.
    (output / "metrics.json").write_text(json.dumps(best.holdout_metrics, indent=2))

    # Save leaderboard for UI.
    lb_path = output / "model_leaderboard.json"
    lb_payload = {"best_model": best.name, "leaderboard": leaderboard}
    lb_path.write_text(json.dumps(lb_payload, indent=2))

    # Save a compact meta file for quick lookups.
    meta_path = output / "model_meta.json"
    meta = {
        "best_model": best.name,
        "mae": best.holdout_metrics["mae"],
        "rmse": best.holdout_metrics["rmse"],
        "r2": best.holdout_metrics["r2"],
        "trials": args.trials,
        "random_state": args.random_state,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Best model: {best.name}")
    print(f"MAE: {best.holdout_metrics['mae']:.4f} | RMSE: {best.holdout_metrics['rmse']:.4f} | R2: {best.holdout_metrics['r2']:.4f}")
    print(f"Artifacts saved to: {output.resolve()}")
    print(f"Leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
