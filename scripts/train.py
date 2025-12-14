#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.bias import collect_bias_reports, save_bias_reports  # noqa: E402
from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.evaluation import (  # noqa: E402
    regression_metrics,
    save_metrics,
    summarize_predictions,
)
from rating_engine.explain import permutation_importance_summary, save_importances  # noqa: E402
from rating_engine.features import (  # noqa: E402
    build_preprocessor,
    clean_dataframe,
    infer_feature_types,
    split_features_target,
)
from rating_engine.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OVR prediction model with bias and explainability reports.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path, help="Path to EA FC26 CSV.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir, help="Where to write artifacts and reports."
    )
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
    df = clean_dataframe(df_raw, config)

    numeric_cols, categorical_cols = infer_feature_types(df, config)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_model(preprocessor, random_state=config.random_state)

    X, y = split_features_target(df, config.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["Position"],
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)
    metrics.update(summarize_predictions(y_test, y_pred))
    save_metrics(metrics, config.output_dir / "metrics.json")

    importances = permutation_importance_summary(model, X_test, y_test, n_jobs=1)
    save_importances(importances, config.output_dir / "feature_importance.csv", top_k=40)

    groups = {
        "gender": df.loc[y_test.index, "GENDER"],
        "position": df.loc[y_test.index, "Position"],
        "league": df.loc[y_test.index, "League"],
    }
    if "age_bucket" in df.columns:
        groups["age_bucket"] = df.loc[y_test.index, "age_bucket"].astype(str)

    bias_reports = collect_bias_reports(df.loc[y_test.index], y_pred, groups, target=config.target)
    save_bias_reports(bias_reports, config.output_dir / "bias")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "model.joblib"
    joblib.dump(model, model_path)

    print("Training complete.")
    print(f"Artifacts saved to: {config.output_dir.resolve()}")
    print(f"Metrics: {metrics}")
    print("Top feature importances:")
    print(importances.head(10))


if __name__ == "__main__":
    main()
