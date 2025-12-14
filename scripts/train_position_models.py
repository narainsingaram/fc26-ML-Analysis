#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.position_model import (  # noqa: E402
    POSITION_GROUPS,
    load_position_ensemble,
    save_position_ensemble,
    train_position_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train position-conditioned models (GK/DEF/MID/ATT).")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path, help="Path to EA FC26 CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/position_ensemble"), help="Where to save models.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_CONFIG.test_size, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_CONFIG.random_state, help="Random seed.")
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
    ensemble, reports = train_position_ensemble(df_raw, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "ensemble.joblib"
    save_position_ensemble(str(model_path), ensemble, reports)

    # Summarize to console and markdown
    print("Position-conditioned models trained.")
    for group, report in reports.items():
        metrics = report["metrics"]
        print(
            f"{group}: MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f} "
            f"(train={report['n_train']}, test={report['n_test']})"
        )

    md_lines = [
        "# Position-Conditioned Models",
        "",
        "| Group | MAE | RMSE | R² | Train N | Test N |",
        "|---|---|---|---|---|---|",
    ]
    for group in POSITION_GROUPS.keys():
        if group not in reports:
            continue
        metrics = reports[group]["metrics"]
        md_lines.append(
            f"| {group} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} | {metrics['r2']:.3f} | "
            f"{reports[group]['n_train']} | {reports[group]['n_test']} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(md_lines))
    print(f"Saved ensemble to {model_path} and report to {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
