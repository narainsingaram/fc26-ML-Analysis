#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.evaluation import regression_metrics, summarize_predictions  # noqa: E402
from rating_engine.explain import permutation_importance_summary  # noqa: E402
from rating_engine.features import (  # noqa: E402
    build_preprocessor,
    clean_dataframe,
    infer_feature_types,
    split_features_target,
)
from rating_engine.model import build_model  # noqa: E402


POSITION_GROUPS = {
    "GK": {"GK"},
    "DEF": {"CB", "RB", "LB"},
    "MID": {"CM", "CDM", "CAM", "LM", "RM"},
    "ATT": {"ST", "RW", "LW"},
}


def map_position_to_group(position: str) -> str:
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return "OTHER"


@dataclass
class GroupResult:
    name: str
    metrics: Dict[str, float]
    top_features: List[Tuple[str, float]]
    n_train: int
    n_test: int
    model_path: Path
    importance_path: Path
    metrics_path: Path


def run_group_training(df: pd.DataFrame, group_name: str, config: TrainingConfig, output_dir: Path) -> GroupResult:
    group_df = df[df["pos_group"] == group_name].copy()
    group_df = group_df.drop(columns=["pos_group"])

    cleaned = clean_dataframe(group_df, config)
    numeric_cols, categorical_cols = infer_feature_types(cleaned, config)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_model(preprocessor, random_state=config.random_state)

    X, y = split_features_target(cleaned, config.target)
    stratify_labels = cleaned["Position"] if cleaned["Position"].nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_labels,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    base_metrics = regression_metrics(y_test, y_pred)
    base_metrics.update(summarize_predictions(y_test, y_pred))

    group_dir = output_dir / group_name.lower()
    group_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = group_dir / "metrics.json"
    pd.Series(base_metrics).to_json(metrics_path, indent=2)

    importances_df = permutation_importance_summary(model, X_test, y_test, n_jobs=1)
    importance_path = group_dir / "feature_importance.csv"
    importances_df.to_csv(importance_path, index=False)

    model_path = group_dir / "model.joblib"
    joblib.dump(model, model_path)

    top_features = list(zip(importances_df["feature"].head(5), importances_df["importance_mean"].head(5)))

    return GroupResult(
        name=group_name,
        metrics=base_metrics,
        top_features=top_features,
        n_train=len(X_train),
        n_test=len(X_test),
        model_path=model_path,
        importance_path=importance_path,
        metrics_path=metrics_path,
    )


def build_markdown(results: List[GroupResult], output_path: Path) -> None:
    lines = []
    lines.append("# Position-Specific OVR Models")
    lines.append("")
    lines.append("Separate regressors per role reduce averaging across very different responsibilities.")
    lines.append("")
    lines.append("## Metrics by Group")
    lines.append("| Group | MAE | RMSE | R² | Train N | Test N |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.metrics['mae']:.3f} | {r.metrics['rmse']:.3f} | {r.metrics['r2']:.3f} | {r.n_train} | {r.n_test} |"
        )
    lines.append("")
    lines.append("## Top Features per Group (Permutation Importance)")
    for r in results:
        lines.append(f"### {r.name}")
        for feat, score in r.top_features:
            lines.append(f"- {feat}: {score:.3f}")
        lines.append(f"_Files_: metrics → `{r.metrics_path}`, importances → `{r.importance_path}`")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train position-specific OVR models and compare results.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path, help="Path to EA FC26 CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/position_models"), help="Where to store artifacts.")
    parser.add_argument("--report-path", type=Path, default=Path("experiments/position_comparison.md"), help="Markdown summary path.")
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
    df_raw["pos_group"] = df_raw["Position"].apply(map_position_to_group)
    df = df_raw[df_raw["pos_group"].isin(POSITION_GROUPS.keys())].copy()

    results: List[GroupResult] = []
    for group_name in ["GK", "DEF", "MID", "ATT"]:
        results.append(run_group_training(df, group_name, config, args.output_dir))

    build_markdown(results, args.report_path)
    print(f"Completed position-specific training. Report written to {args.report_path}")


if __name__ == "__main__":
    main()
