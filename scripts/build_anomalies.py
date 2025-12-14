#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig
from rating_engine.data import load_player_data
from rating_engine.features import clean_dataframe


def main():
    parser = argparse.ArgumentParser(description="Compute overrated/underrated residuals using trained model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/anomalies.json"))
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Run scripts/train.py first.")

    cfg = TrainingConfig(data_path=args.data_path)
    df_raw = load_player_data(cfg.data_path, cfg.age_bins)
    model = joblib.load(args.model_path)

    df = clean_dataframe(df_raw, cfg, drop_null_only=True)
    X = df.drop(columns=[cfg.target], errors="ignore")
    preds = model.predict(X)

    residuals = df_raw[cfg.target].to_numpy() - preds
    df_out = pd.DataFrame(
        {
            "id": df_raw["ID"],
            "name": df_raw["Name"],
            "position": df_raw["Position"],
            "league": df_raw["League"],
            "team": df_raw["Team"],
            "age": df_raw["Age"],
            "ovr": df_raw[cfg.target],
            "predicted": preds,
            "residual": residuals,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_json(args.output_path, orient="records")
    print(f"Wrote anomalies to {args.output_path} (rows={len(df_out)})")


if __name__ == "__main__":
    main()
